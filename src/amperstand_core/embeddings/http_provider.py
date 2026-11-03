"""HTTP embedding providers: OpenAI-compatible hosts, TEI/Infinity sidecars,
and the BYOK vendors with their own request shapes (Voyage, Jina, Cohere).

All of them share `_JsonClient`: httpx, a bearer header, a timeout, and a
retry loop that uses `classify_failure` so an exhausted balance or a bad
key fails on the first attempt while a 5xx/timeout gets a few retries.
Every error surfaces as `EmbeddingError` with the HTTP status and the
provider's message, which the API layer turns into a 503.

Shapes (verified against the providers' OpenAPI specs / docs, Oct 2026):

  openai_compatible  POST {base}/embeddings   {model,input,encoding_format:"float"[,dimensions]}
                     → {data:[{index,embedding}]}        (OpenAI, Together, DeepInfra,
                                                           Fireworks, Mistral*, TEI /v1, Infinity)
  tei                POST {base}/embed        {inputs:[..],normalize,truncate[,dimensions]}
                     → [[..],[..]]                        (Hugging Face TEI native)
  voyage             POST {base}/embeddings   {model,input,input_type:query|document[,output_dimension]}
                     → {data:[{index,embedding}]}
  jina               POST {base}/embeddings   {model,input,task:retrieval.query|retrieval.passage[,dimensions]}
                     → {data:[{index,embedding}]}
  cohere             POST {base}/embed        {model,texts,input_type:search_query|search_document,
                                               embedding_types:["float"][,output_dimension]}
                     → {embeddings:{float:[[..]]}}

* Mistral names its MRL knob `output_dimension`; use AMPERSTAND_EMBED_MATRYOSHKA=1
  (client-side slice) rather than REQUEST_DIM for hosts that reject `dimensions`.
"""

from __future__ import annotations

import logging
import time
from typing import Any

import httpx

from amperstand_core.embeddings.base import (
    KIND_QUERY,
    MAX_RETRIES,
    BaseEmbedder,
    EmbeddingError,
    classify_failure,
)

logger = logging.getLogger(__name__)

DEFAULT_TIMEOUT_S = 30.0
VOYAGE_BASE_URL = "https://api.voyageai.com/v1"
JINA_BASE_URL = "https://api.jina.ai/v1"
COHERE_BASE_URL = "https://api.cohere.com/v2"


class _JsonClient:
    """POST JSON with bearer auth, timeout, and classified retries."""

    def __init__(
        self,
        *,
        base_url: str,
        api_key: str | None,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
        max_retries: int = MAX_RETRIES,
        extra_headers: dict[str, str] | None = None,
        label: str = "provider",
    ) -> None:
        if not base_url:
            raise EmbeddingError(f"{label}: base URL is not configured")
        self.base_url = base_url.rstrip("/")
        headers = {"Content-Type": "application/json", "Accept": "application/json"}
        if api_key:
            headers["Authorization"] = f"Bearer {api_key}"
        if extra_headers:
            headers.update(extra_headers)
        self._client = httpx.Client(
            headers=headers, timeout=timeout_s, transport=transport,
        )
        self.max_retries = max(1, max_retries)
        self.label = label

    def post(self, path: str, body: dict[str, Any]) -> Any:
        url = f"{self.base_url}/{path.lstrip('/')}"
        delay = 1.0
        last: str = ""
        for attempt in range(self.max_retries):
            try:
                resp = self._client.post(url, json=body)
            except httpx.HTTPError as exc:
                status, last = None, f"{type(exc).__name__}: {exc}"
            else:
                if resp.status_code < 300:
                    try:
                        return resp.json()
                    except ValueError as exc:
                        raise EmbeddingError(
                            f"{self.label}: non-JSON response from {url}: {exc}"
                        ) from exc
                status = resp.status_code
                last = f"HTTP {status}: {resp.text[:300]}"
            if classify_failure(status, last) == "fatal":
                raise EmbeddingError(f"{self.label} rejected the request — {last}")
            if attempt == self.max_retries - 1:
                break
            logger.warning(
                "%s attempt %d failed (%s); retrying in %.1fs", self.label, attempt + 1, last, delay,
            )
            time.sleep(delay)
            delay *= 2
        raise EmbeddingError(f"{self.label} failed after {self.max_retries} attempts — {last}")


def _openai_shaped_vectors(payload: Any, label: str) -> list[list[float]]:
    """Extract `data[].embedding` ordered by `index` (OpenAI/Voyage/Jina shape)."""
    if not isinstance(payload, dict) or not isinstance(payload.get("data"), list):
        raise EmbeddingError(f"{label}: unexpected response shape (no data[])")
    rows = sorted(payload["data"], key=lambda d: d.get("index", 0))
    out: list[list[float]] = []
    for row in rows:
        emb = row.get("embedding")
        if not isinstance(emb, list):
            raise EmbeddingError(f"{label}: embedding is not a float list (encoding_format?)")
        out.append(emb)
    return out


class _HttpEmbedder(BaseEmbedder):
    """Common constructor for the HTTP providers."""

    default_base_url: str = ""

    def __init__(
        self,
        *,
        model: str,
        dim: int,
        base_url: str | None = None,
        api_key: str | None = None,
        request_dim: bool = False,
        timeout_s: float = DEFAULT_TIMEOUT_S,
        transport: httpx.BaseTransport | None = None,
        **kwargs,
    ) -> None:
        super().__init__(model=model, dim=dim, **kwargs)
        self.request_dim = request_dim
        self._http = _JsonClient(
            base_url=base_url or self.default_base_url,
            api_key=api_key,
            timeout_s=timeout_s,
            transport=transport,
            label=f"{self.provider} embeddings",
        )

    @property
    def base_url(self) -> str:
        return self._http.base_url


class OpenAICompatibleEmbedder(_HttpEmbedder):
    """Any host that speaks `POST /embeddings` the OpenAI way: Together,
    DeepInfra, Fireworks, Mistral, Infinity, TEI's `/v1/embeddings`, or
    OpenAI itself without the SDK."""

    provider = "openai_compatible"

    def _embed_raw(self, texts: list[str], *, kind: str) -> list[list[float]]:
        body: dict[str, Any] = {"model": self.model, "input": texts, "encoding_format": "float"}
        if self.request_dim:
            body["dimensions"] = self.dim
        return _openai_shaped_vectors(self._http.post("/embeddings", body), self._http.label)


class TEIEmbedder(_HttpEmbedder):
    """Hugging Face text-embeddings-inference native `/embed`.

    TEI serves one model per process, so `model` is only recorded in the
    index signature (set it to the model id you launched TEI with).
    `truncate: true` lets TEI cut inputs at the model's context length;
    `normalize: true` gives unit vectors so cosine == dot."""

    provider = "tei"

    def _embed_raw(self, texts: list[str], *, kind: str) -> list[list[float]]:
        body: dict[str, Any] = {"inputs": texts, "normalize": True, "truncate": True}
        if self.request_dim:
            body["dimensions"] = self.dim
        payload = self._http.post("/embed", body)
        if not isinstance(payload, list) or (payload and not isinstance(payload[0], list)):
            raise EmbeddingError(f"{self._http.label}: unexpected /embed response shape")
        return payload


class VoyageEmbedder(_HttpEmbedder):
    """Voyage AI `/v1/embeddings` with asymmetric `input_type`."""

    provider = "voyage"
    default_base_url = VOYAGE_BASE_URL

    def _embed_raw(self, texts: list[str], *, kind: str) -> list[list[float]]:
        body: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "input_type": "query" if kind == KIND_QUERY else "document",
        }
        if self.request_dim:
            body["output_dimension"] = self.dim
        return _openai_shaped_vectors(self._http.post("/embeddings", body), self._http.label)


class JinaEmbedder(_HttpEmbedder):
    """Jina AI `/v1/embeddings` with the retrieval `task` LoRA selector."""

    provider = "jina"
    default_base_url = JINA_BASE_URL

    def _embed_raw(self, texts: list[str], *, kind: str) -> list[list[float]]:
        body: dict[str, Any] = {
            "model": self.model,
            "input": texts,
            "task": "retrieval.query" if kind == KIND_QUERY else "retrieval.passage",
        }
        if self.request_dim:
            body["dimensions"] = self.dim
        return _openai_shaped_vectors(self._http.post("/embeddings", body), self._http.label)


class CohereEmbedder(_HttpEmbedder):
    """Cohere `/v2/embed` (`texts`, `input_type`, `embedding_types`)."""

    provider = "cohere"
    default_base_url = COHERE_BASE_URL

    def _embed_raw(self, texts: list[str], *, kind: str) -> list[list[float]]:
        body: dict[str, Any] = {
            "model": self.model,
            "texts": texts,
            "input_type": "search_query" if kind == KIND_QUERY else "search_document",
            "embedding_types": ["float"],
        }
        if self.request_dim:
            body["output_dimension"] = self.dim
        payload = self._http.post("/embed", body)
        vectors = (payload or {}).get("embeddings", {}).get("float") if isinstance(payload, dict) else None
        if not isinstance(vectors, list):
            raise EmbeddingError(f"{self._http.label}: unexpected /embed response shape")
        return vectors
