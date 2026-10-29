# Pluggable embedding and rerank providers

Status: design + implementation on branch `feat/pluggable-embed-rerank`,
2026-10-28. Nothing deployed.

## 1. Summary

Today `/vault/search/semantic` and `/vault/search/hybrid` depend on one
OpenAI key for both query embeddings and the LLM rerank. On 2026-10-24
the OpenAI balance hit zero: embeddings failed with `429
insufficient_quota`, the server retried for ~14 s and answered a bare
`500`; rerank failed soft and silently returned the fused order. FTS and
stored-vector endpoints kept working. Cost was never the problem (a full
re-embed is ~35M tokens ≈ $0.70); the problem is a hard dependency on one
vendor balance plus an error that hides its cause.

The change makes the embedder and the reranker **providers** chosen by
environment variables, in two first-class setups:

- **BYOK** — a hosted API called with the operator's own key.
  Embeddings: OpenAI (default, unchanged), Voyage, Jina, Cohere, any
  OpenAI-compatible host (Together, DeepInfra, Fireworks, Mistral …).
  Rerank: the vendors' native rerank APIs (Voyage, Jina, Cohere) or the
  existing LLM grader against OpenAI / an OpenAI-compatible chat host.
- **Nearby** — self-hosted open models; defaults
  `ibm-granite/granite-embedding-97m-multilingual-r2` (Apache-2.0, 384-d,
  32k context, 200+ languages) and
  `cross-encoder/mmarco-mMiniLMv2-L12-H384-v1` (Apache-2.0, 118M). Three
  deployment modes behind the same configuration: (a) a sidecar on the
  droplet, (b) a separate small box — both the same HTTP client to a
  TEI/Infinity-style `/embed` + `/rerank` server — and (c) in-process ONNX,
  an optional extra (`pip install 'amperstand-core[onnx]'`) that is imported
  lazily so the base install never pulls onnxruntime.

Independently of the provider, the index now records which
provider/model/dim it was built with and refuses (503, not a reset) to
serve a different one; every provider failure is a 503 with the cause; and
hybrid search falls back to BM25 only when the caller opts in, with the
fallback flagged in the response.

**Recommendation** (details in §10): run **Nearby, in-process ONNX** on
the droplet — granite-97m for embeddings and mmarco-mMiniLMv2 as the
cross-encoder, together ~0.7–0.8 GB resident inside uvicorn (§5) —
and keep **BYOK OpenAI** as the documented alternative. This removes the
vendor-balance dependency entirely for ~$0/month, at the price of one
droplet resize to 2 GB (the current 961 MB box has ~300 MB free and is
already 500 MB into swap). Without a resize, the
second-best option is BYOK with a vendor whose balance is prepaid and
monitored, plus `allow_bm25_fallback=true` in clients that can live with keyword-only results. The
one decision needed: **Nearby + resize to 2 GB, or BYOK as-is**.

## 2. What changed in the code

Paths under `amperstand-core/src/amperstand_core/`.

| File | Purpose |
|---|---|
| `embeddings/base.py` | `BaseEmbedder` contract (`embed()` for queries, `embed_batch()` for documents), `EmbedSignature(provider, model, dim)`, `EmbeddingError`, shared truncation / prefixes / batching / dimension guard, `classify_failure()` (quota and auth errors are fatal, 5xx and rate limits retry). |
| `embeddings/openai_provider.py` | `OpenAIEmbedder` — the pre-existing SDK path, same model/dim/batching/backoff. `Embedder` is an alias of it; `Embedder.from_env()` still works. Only change: a `429 insufficient_quota` is no longer retried. |
| `embeddings/http_provider.py` | `OpenAICompatibleEmbedder` (`POST /embeddings`), `TEIEmbedder` (`POST /embed`), `VoyageEmbedder`, `JinaEmbedder`, `CohereEmbedder` — httpx, bearer auth, classified retries, no SDKs. |
| `embeddings/onnx_provider.py` | `ONNXEmbedder` — in-process onnxruntime + tokenizers, CLS/mean pooling, one thread by default. Optional extra. |
| `embeddings/config.py` | `EmbedConfig`, `embed_config_from_env()`, `build_embedder()` — the env-driven factory. With nothing set it builds exactly the old OpenAI embedder. |
| `embeddings/__init__.py` | Back-compat exports: `Embedder`, `EMBED_DIM`, `EmbeddingError`, plus `build_embedder_from_env`, `EmbedSignature`, `LEGACY_SIGNATURE`. |
| `search/vec_index.py` | `VectorIndex(path, signature=…)` stores `embed_provider` / `embed_model` / `embed_dim` in `vec_meta_kv`; a different configured signature sets `index.mismatch` and `search_query()` / `upsert_doc_sections()` raise `IndexSignatureMismatch`; `reset()` re-stamps. An unstamped, populated index (production's `vec.db`) is treated as `openai/text-embedding-3-small@512`. |
| `search/rerank.py` | `BaseReranker` → `LLMReranker` (OpenAI or OpenAI-compatible chat), `HTTPCrossEncoderReranker` (TEI, Infinity, Cohere, Jina, Voyage shapes), `ONNXCrossEncoderReranker` (optional extra), `NoReranker`; `RerankOutcome(results, applied, error)`; env config + factory; `rerank_with_llm()` kept as a wrapper. |
| `server/vault_api/store_factory.py` | Builds the embedder from env (default unchanged), opens the index with the signature, caches `get_reranker()`; records why vectors are unavailable for the 503 detail. |
| `server/vault_api/router.py` | `EmbeddingError` → 503 `embedding_provider_unavailable`; `IndexSignatureMismatch` → 503 `vec_index_model_mismatch`; rerank via the configured reranker, `RerankError` → 503 `rerank_provider_unavailable` (hard mode) or `meta.rerank_applied=false` (soft); `allow_bm25_fallback` opt-in with `meta.fallback="bm25"`. |
| `server/vault_api/schemas.py` | `SearchMeta` on every search response; `HybridSearchRequest.allow_bm25_fallback`. |
| `server/admin_cli/config.py` | `AdminConfig.env` — merged env-file + process env, so `vec-rebuild` sees the same provider settings as the server. |
| `server/admin_cli/commands/vec_rebuild.py` | Works with any provider; prints the configured and on-disk signatures; refuses to mix spaces without `--force`; `--force` resets and re-stamps. |
| `pyproject.toml` | optional extra `onnx = [onnxruntime, tokenizers, numpy]`. |
| `tests/…` | `test_embedding_providers.py`, `test_rerankers.py`, `test_vec_index_signature.py`, `server/test_search_providers.py`, additions to `server/test_admin_cli.py`, `tests/fakes.py` (deterministic `FakeEmbedder`, sqlite-vec probe). |

Design rules the code follows:

- One interface for everything that embeds: `embed(text)` is a *query*,
  `embed_batch(texts)` is *documents*. Providers with asymmetric input
  types (Voyage `input_type`, Jina `task`, Cohere `input_type`) get the
  right one automatically; e5-style prefixes are configurable.
- The signature is the identity of the vector space: `provider` is the
  kind of provider, not the host — granite-97m served by TEI on another
  box is the same space as granite-97m served by TEI on the droplet. It is
  *not* the same space as granite-97m in-process via ONNX: the two pipelines
  differ (quantisation, truncation), so the provider is part of the key.
  Switching between sidecar and in-process therefore needs a rebuild, which
  is the honest answer.
- Failures are typed and explicit. Nothing in the search path swallows a
  provider error; the only fallback is the one the caller asked for.
- No new dependencies in the base install. httpx was already a dependency.

## 3. Environment-variable reference

All variables are read at process start (server) or at command start
(`amperstand-admin`). The admin CLI reads `/etc/amperstand/env` (or
`--env-file`), so both see the same settings.

### Embeddings

| Variable | Values / default | Meaning |
|---|---|---|
| `AMPERSTAND_EMBED_PROVIDER` | `openai` (default), `openai_compatible`, `voyage`, `jina`, `cohere`, `tei`, `onnx`, `none` | Which provider. `none` disables semantic search explicitly. |
| `AMPERSTAND_EMBED_MODEL` | default `text-embedding-3-small` for `openai`; required for the hosted providers; informational for `tei` (TEI serves one model per process) and `onnx` (defaults to the model directory's name) | Model name sent to the API and recorded in the index signature. |
| `AMPERSTAND_EMBED_DIM` | integer, default `512` | Dimension the index is built with. Must equal what the provider returns (or see `MATRYOSHKA`). |
| `AMPERSTAND_EMBED_BASE_URL` | URL | Required for `openai_compatible` (…`/embeddings` is appended) and `tei` (…`/embed`). Optional override for `openai` (falls back to `OPENAI_BASE_URL`), `voyage`, `jina`, `cohere`. |
| `AMPERSTAND_EMBED_API_KEY` | secret | Falls back to `OPENAI_API_KEY` (openai, openai_compatible), `VOYAGE_API_KEY`, `JINA_API_KEY`, `COHERE_API_KEY`. Not needed for a local sidecar. |
| `AMPERSTAND_EMBED_REQUEST_DIM` | `auto` (default), `1`, `0` | Ask the API for `DIM`-sized vectors (OpenAI `dimensions`, Voyage/Cohere `output_dimension`, Jina `dimensions`, TEI `dimensions`). `auto` = on for `openai` only. |
| `AMPERSTAND_EMBED_MATRYOSHKA` | `0` (default) / `1` | Slice longer vectors to `DIM` client-side and re-normalise. Only valid for MRL-trained models (text-embedding-3, granite-311m-r2, EmbeddingGemma, jina-v3, voyage-3.5). Otherwise a wrong dimension is an error. |
| `AMPERSTAND_EMBED_QUERY_PREFIX` / `AMPERSTAND_EMBED_DOC_PREFIX` | strings, default empty | e.g. `query: ` / `passage: ` for e5 models, EmbeddingGemma prompts. Granite-r2 needs none. |
| `AMPERSTAND_EMBED_MAX_TOKENS` | default `8000` | Truncation budget per text (tiktoken count; a budget, not an exact cap, for non-OpenAI tokenizers — TEI/ONNX also truncate at the model side). |
| `AMPERSTAND_EMBED_BATCH_SIZE` | default `100` | Texts per request (Cohere allows 96). |
| `AMPERSTAND_EMBED_TIMEOUT_S` | default `30` | HTTP timeout. |
| `AMPERSTAND_EMBED_ONNX_DIR` | path | `onnx` only: directory with `tokenizer.json` and the graph. |
| `AMPERSTAND_EMBED_ONNX_FILE` | default `onnx/model.onnx` | Graph inside the directory (`onnx/model_quint8_avx2.onnx` for the droplet). |
| `AMPERSTAND_EMBED_ONNX_POOLING` | `cls` (default) / `mean` | granite-r2 = `cls`. |
| `AMPERSTAND_EMBED_ONNX_THREADS` | default `1` | onnxruntime intra-op threads. Keep `1` on the 1 vCPU droplet. |
| `AMPERSTAND_EMBED_ONNX_MAX_TOKENS` | default `2048` | Model-side truncation. |

### Rerank

| Variable | Values / default | Meaning |
|---|---|---|
| `AMPERSTAND_RERANK_KIND` | `llm` (default), `cross_encoder`, `none` | The kind of reranker. `none` makes `rerank=true` requests answer 503 `rerank_unavailable`. |
| `AMPERSTAND_RERANK_ENABLED` | `1` (default) / `0` | Existing kill switch; `0` behaves like `none`. |
| `AMPERSTAND_RERANK_PROVIDER` | llm: `openai` (default), `openai_compatible`; cross_encoder: `tei` (default), `infinity`, `cohere`, `jina`, `voyage`, `onnx` | Which API shape / host. |
| `AMPERSTAND_RERANK_MODEL` | llm default `gpt-4o-mini`; required for `cohere`/`jina`/`voyage`; optional for `infinity`; ignored by `tei` | Model name. |
| `AMPERSTAND_RERANK_BASE_URL` | URL | Required for `openai_compatible`, `tei`, `infinity` (…`/rerank` is appended). Optional override for the vendors. |
| `AMPERSTAND_RERANK_API_KEY` | secret | Falls back to `OPENAI_API_KEY`, `COHERE_API_KEY`, `JINA_API_KEY`, `VOYAGE_API_KEY`. |
| `AMPERSTAND_RERANK_MAX_CHARS` | default `600` | Passage cap sent per candidate (unchanged). |
| `AMPERSTAND_RERANK_TIMEOUT_S` | default `20` | HTTP timeout. |
| `AMPERSTAND_RERANK_FAIL_MODE` | `soft` (default) / `hard` | `soft`: provider failure → fused order with `meta.rerank_applied=false` and a warning. `hard`: 503 `rerank_provider_unavailable`. |
| `AMPERSTAND_RERANK_ONNX_DIR` / `_ONNX_FILE` / `_ONNX_THREADS` / `_ONNX_MAX_TOKENS` | path / `onnx/model.onnx` / `1` / `512` | `provider=onnx` only. |

Response shape (additive, on every search response):

```json
"meta": {
  "mode": "hybrid",
  "embed_provider": "tei", "embed_model": "ibm-granite/granite-embedding-97m-multilingual-r2",
  "rerank_requested": true, "rerank_applied": true, "rerank_kind": "cross_encoder",
  "fallback": null, "warnings": []
}
```

## 4. Worked configurations

Each block is the complete set of lines to add to `/etc/amperstand/env`.
Keys are placeholders.

### BYOK

**OpenAI — the default, unchanged.** Nothing new to set:

```
OPENAI_API_KEY=sk-…
```

**OpenAI-compatible host (DeepInfra, Together, Fireworks, Mistral).**
Example DeepInfra serving BAAI/bge-m3 (1024-d; no `dimensions` support, so
keep DIM at the model's native size):

```
AMPERSTAND_EMBED_PROVIDER=openai_compatible
AMPERSTAND_EMBED_BASE_URL=https://api.deepinfra.com/v1/openai
AMPERSTAND_EMBED_MODEL=BAAI/bge-m3
AMPERSTAND_EMBED_DIM=1024
AMPERSTAND_EMBED_API_KEY=…
# LLM grader through the same host's chat endpoint
AMPERSTAND_RERANK_KIND=llm
AMPERSTAND_RERANK_PROVIDER=openai_compatible
AMPERSTAND_RERANK_BASE_URL=https://api.deepinfra.com/v1/openai
AMPERSTAND_RERANK_MODEL=meta-llama/Llama-3.3-70B-Instruct
AMPERSTAND_RERANK_API_KEY=…
```

(1024-d brute-force KNN over ~50k vectors is ~2× the 512-d cost, still
sub-second on 1 vCPU.)

**Voyage** (embeddings + native rerank; `output_dimension=512` keeps the
index size as today):

```
AMPERSTAND_EMBED_PROVIDER=voyage
AMPERSTAND_EMBED_MODEL=voyage-3.5-lite
AMPERSTAND_EMBED_DIM=512
AMPERSTAND_EMBED_REQUEST_DIM=1
VOYAGE_API_KEY=…
AMPERSTAND_RERANK_KIND=cross_encoder
AMPERSTAND_RERANK_PROVIDER=voyage
AMPERSTAND_RERANK_MODEL=rerank-2.5-lite
```

**Jina**:

```
AMPERSTAND_EMBED_PROVIDER=jina
AMPERSTAND_EMBED_MODEL=jina-embeddings-v3
AMPERSTAND_EMBED_DIM=512
AMPERSTAND_EMBED_REQUEST_DIM=1
JINA_API_KEY=…
AMPERSTAND_RERANK_KIND=cross_encoder
AMPERSTAND_RERANK_PROVIDER=jina
AMPERSTAND_RERANK_MODEL=jina-reranker-v2-base-multilingual
```

**Cohere**:

```
AMPERSTAND_EMBED_PROVIDER=cohere
AMPERSTAND_EMBED_MODEL=embed-v4.0
AMPERSTAND_EMBED_DIM=512
AMPERSTAND_EMBED_REQUEST_DIM=1
AMPERSTAND_EMBED_BATCH_SIZE=96
COHERE_API_KEY=…
AMPERSTAND_RERANK_KIND=cross_encoder
AMPERSTAND_RERANK_PROVIDER=cohere
AMPERSTAND_RERANK_MODEL=rerank-v3.5
```

Model names above are examples; the vendors rename models often (Cohere's
reference now shows `embed-v5.0-fast` / `rerank-v5.0-pro`, Jina has
`jina-embeddings-v5-text-small`, Voyage `rerank-2.5` / `rerank-2.5-lite`).
Check the vendor's model list before enabling — the server records
whatever string you set and only that string will match the index later.
Verified request shapes: Voyage rerank takes `top_k` and answers in
`data[]`; Jina embeddings take `task=retrieval.query|retrieval.passage`
and `dimensions`; Cohere v2 embed takes `output_dimension`
(256/512/1024/1536 for embed-v4) and at most 96 texts per call.

### Nearby

**(a) Sidecar on the droplet — TEI.** Two TEI processes (TEI serves one
model each), bound to localhost:

```
AMPERSTAND_EMBED_PROVIDER=tei
AMPERSTAND_EMBED_BASE_URL=http://127.0.0.1:8080
AMPERSTAND_EMBED_MODEL=ibm-granite/granite-embedding-97m-multilingual-r2
AMPERSTAND_EMBED_DIM=384
AMPERSTAND_EMBED_MAX_TOKENS=2000
AMPERSTAND_RERANK_KIND=cross_encoder
AMPERSTAND_RERANK_PROVIDER=tei
AMPERSTAND_RERANK_BASE_URL=http://127.0.0.1:8081
```

**(b) Separate small box** — identical settings with the box's private
IP instead of `127.0.0.1` (DigitalOcean VPC address, firewall the ports to
the droplet). Infinity serves both models from one process on one port:

```
AMPERSTAND_EMBED_PROVIDER=openai_compatible
AMPERSTAND_EMBED_BASE_URL=http://10.x.y.z:7997
AMPERSTAND_EMBED_MODEL=ibm-granite/granite-embedding-97m-multilingual-r2
AMPERSTAND_EMBED_DIM=384
AMPERSTAND_RERANK_KIND=cross_encoder
AMPERSTAND_RERANK_PROVIDER=infinity
AMPERSTAND_RERANK_BASE_URL=http://10.x.y.z:7997
AMPERSTAND_RERANK_MODEL=cross-encoder/mmarco-mMiniLMv2-L12-H384-v1
```

Note the signature difference: TEI `/embed` is recorded as provider `tei`,
Infinity's OpenAI-shaped `/embeddings` as `openai_compatible`. Pick one
server and keep it, or rebuild when switching.

**(c) In-process ONNX** (needs `pip install -e '.[onnx]'` in the venv and
the model files on disk, see §8):

```
AMPERSTAND_EMBED_PROVIDER=onnx
AMPERSTAND_EMBED_ONNX_DIR=/var/lib/amperstand/models/granite-embedding-97m-multilingual-r2
AMPERSTAND_EMBED_ONNX_FILE=onnx/model_quint8_avx2.onnx
AMPERSTAND_EMBED_MODEL=granite-embedding-97m-multilingual-r2
AMPERSTAND_EMBED_DIM=384
AMPERSTAND_EMBED_ONNX_THREADS=1
AMPERSTAND_EMBED_ONNX_MAX_TOKENS=1024      # bounds the per-section RAM/CPU spike on 1 vCPU (§5)
AMPERSTAND_RERANK_KIND=cross_encoder
AMPERSTAND_RERANK_PROVIDER=onnx
AMPERSTAND_RERANK_ONNX_DIR=/var/lib/amperstand/models/mmarco-mMiniLMv2-L12-H384-v1
AMPERSTAND_RERANK_ONNX_FILE=onnx/model_quint8_avx2.onnx
AMPERSTAND_RERANK_ONNX_THREADS=1
```

## 5. Comparison of the Nearby modes on this droplet

Droplet: 1 shared vCPU (AVX2, no AVX-512/VNNI), 961 MB RAM, ~305 MB
available, 1 GB swap with ~510 MB used; uvicorn ~150 MB RSS plus
~200 MB of unrelated services. No torch / onnxruntime installed.

| | (a) Sidecar on the droplet | (b) Separate box | (c) In-process ONNX |
|---|---|---|---|
| Extra resident memory on the droplet | TEI CPU build loads the fp32 weights (390 MB granite, 470 MB mmarco) plus the same tokenizers as below → ~0.6 GB + ~0.7 GB in two processes. Infinity (torch) more. | ~0 | measured on one M1 core: granite-97m ≈ **350–400 MB** after load — the 180k-token tokenizer alone is ~200 MB, the quint8 session ~100–130 MB — plus a transient **+200–300 MB** while embedding a 2048-token section (ORT arena; ~+50 MB at 512 tokens). mmarco ≈ **300–380 MB** after load (250k-token XLM-R tokenizer ~230 MB). Both in uvicorn: **~0.7–0.8 GB** on top of today's 150 MB. Model files 98 + 119 MB on disk. |
| CPU contention | The sidecar and uvicorn share one vCPU; a 40-passage rerank or a long-section embed stalls API requests for its duration. | none on the droplet; one network hop (~1 ms on a VPC) | same vCPU, same process: embedding a long captured doc blocks that uvicorn worker (captures already run in a background worker thread, so API latency is affected only during the 0.2–3 s inference) |
| Query latency (single M1 thread; expect 2–4× on the droplet's shared vCPU) | ≈ in-process + HTTP overhead | ≈ in-process + ~1 ms | query embed ~11 ms (32 tokens); document embed 0.23 s @512 tokens, 0.8 s @1024, 2.7 s @2048; cross-encoder pool 20 / 40: not yet measured (§9) |
| Full re-embed of the vault (50k sections, ~30M tokens) | hours on the droplet; build on the Mac instead (§8) | hours on a 1 vCPU box; build on the Mac | same — build on the Mac |
| Ops burden | two systemd units + Docker (or the TEI binary), model cache dir, health checks; TEI's Rust binary has no Python deps | one more box to patch, a VPC + firewall, two units | none beyond `pip install '.[onnx]'` and copying two model directories |
| Failure modes | sidecar OOM-killed → 503 `embedding_provider_unavailable` (explicit); both die together under memory pressure | box down / network partition → 503 (explicit); independent of the droplet's memory | model files missing → server starts with semantic search disabled and logs why; inference error → 503 |
| Resize needed | yes: 4 GB for TEI ×2 (≈1.3 GB resident) next to uvicorn and the bots | droplet: none; new box: 2 GB / 1 vCPU for TEI with both models (1 GB fits only one of them), 2 GB for Infinity | **2 GB required** (961 MB today with ~300 MB free and 500 MB already in swap cannot take +0.7–0.8 GB); on 2 GB the total is ≈ uvicorn 0.9 GB + bots 0.3 GB, leaving ~0.6 GB headroom for the transient. Setting `AMPERSTAND_EMBED_ONNX_MAX_TOKENS=1024` halves the transient at the cost of truncating the ~5 % of sections longer than 1024 tokens. |
| Vendor dependency | none | none | none |
| Quality | identical model → identical vectors; TEI runs fp32 (slightly better than quint8) | same | quint8 (dynamic int8) — comparison against the OpenAI index pending (§9) |

Cost: DigitalOcean 1 vCPU / 2 GB is $12/mo vs $6/mo today; a second
1 vCPU / 1 GB box is +$6/mo. BYOK stays at ~$0.70 per full re-embed and
~$0.001 per reranked query.

## 6. Index safety

`vec_meta_kv` now carries three keys next to `schema_version`:

```
embed_provider  openai | openai_compatible | voyage | jina | cohere | tei | onnx
embed_model     e.g. text-embedding-3-small
embed_dim       e.g. 512
```

Rules:

1. A fresh/empty index adopts the configured signature on first open (and
   recreates the `vec0` table at the configured dim).
2. A populated index with no stamp — production's current `vec.db`,
   `schema_version=2`, 49,544 vectors — is assumed to be
   `openai/text-embedding-3-small@512`. If that is what is configured it is
   stamped on open; otherwise it is a mismatch (never stamped with a guess).
3. On a mismatch the index stays intact. Query embedding is refused with
   503 `vec_index_model_mismatch` (both signatures in the detail), the live
   indexer's writes are refused (logged), `GET /vault/{id}/related` keeps
   working (it only uses stored vectors), `vec-rebuild` refuses without
   `--force`.
4. `vec-rebuild --force` drops the tables, re-stamps, re-embeds.

Migration path for a model switch (any setup):

1. Set the new `AMPERSTAND_EMBED_*` in `/etc/amperstand/env`.
2. Do **not** restart the server yet (it would start answering 503 for
   semantic/hybrid until the rebuild finishes). Hybrid callers that can
   live with BM25 meanwhile should pass `allow_bm25_fallback=true`.
3. Build the new index — on the droplet with
   `sudo -u amperstand /opt/amperstand/venv/bin/amperstand-admin vec-rebuild --force`
   (hours for Nearby on 1 vCPU; ~15 min for a hosted API), or on the Mac
   (§8) and upload.
4. Restart the server. `journalctl` shows `vector search: <signature>
   (index <signature>)` on startup.

### Building `vec.db` on the Mac and uploading it

The index is a single SQLite file whose rows are keyed by `doc_id` and the
section content hash, so an index built from a copy of `docs/` is
byte-for-byte usable on the droplet, provided the vault did not change in
between (and if it did, the incremental `vec-rebuild` on the droplet
embeds only the changed docs).

```bash
# Mac, arm64 venv of the worktree with the onnx extra installed
rsync -a root@<host>:/var/lib/amperstand/vault/docs/ /tmp/vault/docs/      # read-only copy
export AMPERSTAND_DATA_DIR=/tmp/vault
export AMPERSTAND_EMBED_PROVIDER=onnx AMPERSTAND_EMBED_ONNX_DIR=/path/to/granite-embedding-97m-multilingual-r2 \
       AMPERSTAND_EMBED_ONNX_FILE=onnx/model_quint8_avx2.onnx AMPERSTAND_EMBED_DIM=384 \
       AMPERSTAND_EMBED_ONNX_THREADS=4
amperstand-admin vec-rebuild --force --env-file /dev/null                 # ~25 min with 4 threads on an M1
sqlite3 /tmp/vault/.store/vec.db 'select * from vec_meta_kv'              # check the stamp
scp /tmp/vault/.store/vec.db root@<host>:/var/lib/amperstand/vault/.store/vec.db.new
```

On the droplet (service user owns the file; swap atomically while the
server is stopped, since sqlite-vec holds the file open):

```bash
sudo systemctl stop amperstand-server
sudo mv /var/lib/amperstand/vault/.store/vec.db.new /var/lib/amperstand/vault/.store/vec.db
sudo rm -f /var/lib/amperstand/vault/.store/vec.db-wal /var/lib/amperstand/vault/.store/vec.db-shm
sudo chown amperstand:amperstand /var/lib/amperstand/vault/.store/vec.db
sudo systemctl start amperstand-server
sudo -u amperstand /opt/amperstand/venv/bin/amperstand-admin vec-rebuild   # incremental catch-up, no --force
```

Note: the ONNX graph must be the *same file* on both machines
(`model_quint8_avx2.onnx`), and the signature records the provider as
`onnx`; a droplet configured for `tei` would refuse a Mac-built `onnx`
index, by design.

## 7. Error behaviour

| Situation | Before | Now |
|---|---|---|
| Provider quota exhausted / key invalid | 4 retries, ~14 s, bare `500` | immediate `503 {"error":"embedding_provider_unavailable","provider":…,"model":…,"detail":"openai embeddings rejected (429): …insufficient_quota…"}` |
| Provider down (5xx, timeout, connection refused) | same bare 500 after retries | 4 attempts with backoff, then the same 503 with the cause |
| Index built by another model | undefined (dim error → 500, or wrong-space results) | `503 {"error":"vec_index_model_mismatch","index":{…},"configured":{…}}` |
| No provider configured | `503 feature_requires_openai_key` | unchanged when the cause is the missing OpenAI key; otherwise `503 vector_index_unavailable` with `reason` |
| Hybrid when vectors unavailable | 503 (and `/vault/search?mode=hybrid` silently fell back to BM25) | 503 unless `allow_bm25_fallback=true` → 200 with `meta.fallback="bm25"` and the cause in `meta.warnings`; the legacy `/vault/search` keeps its fallback but now says so in `meta` |
| Rerank requested, not available | 503 | 503 `rerank_unavailable` (or `feature_requires_openai_key` when that is the reason) |
| Rerank provider fails mid-request | silent fused order | fused order + `meta.rerank_applied=false` + warning (`soft`), or 503 `rerank_provider_unavailable` (`hard`) |

Clients should treat
`meta.fallback == "bm25"` and `meta.rerank_applied == false` as
degraded-quality signals and label their results accordingly.

## 8. Deploy steps (written, not executed)

### Nearby (c), in-process ONNX — the recommended option

1. Resize the droplet to 1 vCPU / 2 GB (DigitalOcean: power off → Resize →
   RAM-only resize keeps the disk; ~1 min downtime). Skipping this means
   the extra ~0.7–0.8 GB lands in swap and search latency becomes
   unpredictable.
2. Pull the branch and install the extra:
   ```bash
   cd /opt/amperstand && sudo -u amperstand git fetch && sudo -u amperstand git checkout feat/pluggable-embed-rerank
   sudo -u amperstand /opt/amperstand/venv/bin/pip install -e './amperstand-core[onnx]'
   ```
   (`onnxruntime` wheel ~15 MB for x86_64 Linux, `tokenizers` ~3 MB, `numpy` ~17 MB.)
3. Put the model files where the hardened unit can read them
   (`ProtectSystem=strict`, `ReadWritePaths=/var/lib/amperstand`):
   ```bash
   sudo -u amperstand mkdir -p /var/lib/amperstand/models
   # from the Mac, the same files used for the eval:
   scp -r granite-embedding-97m-multilingual-r2 mmarco-mMiniLMv2-L12-H384-v1 root@<host>:/var/lib/amperstand/models/
   sudo chown -R amperstand:amperstand /var/lib/amperstand/models
   ```
   Needed files per model: `tokenizer.json`, `onnx/model_quint8_avx2.onnx`
   (98 MB and 119 MB). Download them once on the Mac from
   `huggingface.co/ibm-granite/granite-embedding-97m-multilingual-r2` and
   `huggingface.co/cross-encoder/mmarco-mMiniLMv2-L12-H384-v1`.
4. Append the Nearby (c) block from §4 to `/etc/amperstand/env`
   (`sudo $EDITOR /etc/amperstand/env`). Keep `OPENAI_API_KEY` for chat and
   audio; it is no longer used by search.
5. Build the index on the Mac and upload it (§6), or run
   `vec-rebuild --force` on the droplet overnight.
6. `sudo systemctl restart amperstand-server`, then check:
   ```bash
   sudo journalctl -u amperstand-server -n 50 | grep -E "vector search|rerank|onnx"
   curl -s -H "Authorization: Bearer $KEY" -X POST localhost:8765/vault/search/hybrid \
        -H 'content-type: application/json' -d '{"q":"test","limit":3,"rerank":true}' | jq .meta
   ```
   Expect `embed_provider: "onnx"`, `rerank_applied: true`.
7. Run `pytest` once on the droplet's venv? Not needed; run the suite on a
   dev machine — the droplet only needs the smoke test above.

### Nearby (a), TEI sidecar on the droplet

Requires the 2 GB resize (4 GB if both TEI processes run). TEI ships a
CPU Docker image and a static binary; the unit-file route avoids Docker:

```bash
# one unit per model; TEI 1.9.x CPU build
sudo mkdir -p /opt/tei /var/lib/tei
# install: download the cpu release binary for x86_64 from the text-embeddings-inference releases page to /opt/tei/text-embeddings-router
sudo tee /etc/systemd/system/tei-embed.service >/dev/null <<'EOF'
[Unit]
Description=TEI embeddings (granite-97m)
After=network-online.target
[Service]
User=amperstand
Environment=HF_HOME=/var/lib/tei
ExecStart=/opt/tei/text-embeddings-router --model-id ibm-granite/granite-embedding-97m-multilingual-r2 --port 8080 --hostname 127.0.0.1 --max-client-batch-size 128 --max-batch-tokens 16384 --auto-truncate
Restart=on-failure
[Install]
WantedBy=multi-user.target
EOF
# tei-rerank.service: same with --model-id cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 --port 8081
sudo systemctl enable --now tei-embed tei-rerank
curl -s 127.0.0.1:8080/info | jq .model_id
```

Then the Nearby (a) env block, `vec-rebuild --force` (or upload), restart.

### Nearby (b), separate box

Create a 1 vCPU / 1 GB droplet in the same VPC, install TEI (as above) or
Infinity (`pip install infinity-emb[all]`, then
`infinity_emb v2 --model-id ibm-granite/granite-embedding-97m-multilingual-r2 --model-id cross-encoder/mmarco-mMiniLMv2-L12-H384-v1 --engine optimum --device cpu --port 7997 --host 10.x.y.z`),
open 7997/8080/8081 only to the API droplet's private IP (`ufw allow from
<api-private-ip> to any port 7997`), use the Nearby (b) env block, rebuild
or upload the index, restart.

### BYOK

Append the vendor block from §4, `vec-rebuild --force` on the droplet
(hosted APIs embed the vault in ~15–30 min), restart. For the default
OpenAI setup nothing changes except that the branch must be deployed for
the new error behaviour:

```bash
cd /opt/amperstand && sudo -u amperstand git fetch && sudo -u amperstand git checkout feat/pluggable-embed-rerank
sudo -u amperstand /opt/amperstand/venv/bin/pip install -e ./amperstand-core
sudo systemctl restart amperstand-server
```

The existing `vec.db` is adopted (stamped `openai/text-embedding-3-small@512`)
on first start; no rebuild.

## 9. Local evaluation (Mac, Apple M1, arm64)

Partial. Memory and single-thread latency of the in-process ONNX path
were measured on one Apple M1 core and are reported in §5.

Not yet run, and worth running before resizing the droplet for Nearby:

- retrieval quality against the current OpenAI index: top-10 doc overlap
  on a set of saved research queries, with and without the cross-encoder,
  plus a few spot-checked queries where the results diverge;
- cross-encoder latency at candidate pools of 20 and 40, single-threaded.

## 10. Recommendation

See §1. The ranking, for this vault and this droplet:

1. **Nearby (c) in-process ONNX + 2 GB resize.** No vendor, no sidecar, no
   second box; the only operational surface is two model directories and
   the `onnx` extra. The measured RAM fits a 2 GB droplet with the existing
   bots. Quality: granite-97m is the strongest open multilingual model under
   100M params (MMTEB retrieval 60.3 vs 50.9 for multilingual-e5-small, IBM's
   own numbers) and its 32k context covers our long sections, which the
   512-token e5 models truncate.
2. **Nearby (b) separate box** if the droplet's memory profile should stay
   untouched: same models, same quality, +$6/mo, one more
   machine.
3. **BYOK** if no resize and no second box: keep OpenAI, or move to Voyage /
   Jina / Cohere with their native rerankers. The 503-with-cause and the
   explicit BM25 fallback make the next balance outage visible instead of a
   mystery, but a prepaid balance still has to be watched.
4. **Nearby (a) sidecar on the droplet** last: it needs the biggest resize
   for the least benefit over (c).

## 11. Not verified / open

- Vendor model names and free tiers change; the examples use names current
  at the time of writing and were not exercised against live APIs (tests
  use mocked HTTP with the documented shapes; TEI and Infinity shapes were
  checked against their published OpenAPI specs, Cohere/Jina/Voyage against
  their reference docs).
- The in-process numbers are measured on an M1 core; the droplet's shared
  x86 vCPU is slower per core (expect 2–4× on the same single-thread
  path) and the `quint8_avx2` graph is the one tuned for it.
- `ONNXEmbedder` and `ONNXCrossEncoderReranker` were exercised with the two
  default models only. The real-model smoke test is skipped unless
  `AMPERSTAND_TEST_ONNX_DIR` points at a downloaded model.
- Retrieval quality of the Nearby defaults versus OpenAI on this vault is
  not measured yet (§9).
- Voyage's `/v1/embeddings` was implemented from its documentation; Mistral
  rejects `dimensions`, use `AMPERSTAND_EMBED_MATRYOSHKA=1` or the model's
  native dim.
