"""Test doubles shared by the provider tests."""

from __future__ import annotations

import hashlib
import math
import re

from amperstand_core.embeddings import BaseEmbedder, EmbeddingError


_SQLITE_VEC_OK: bool | None = None


def sqlite_vec_loadable() -> bool:
    """True when this interpreter can open a VectorIndex (sqlite-vec present
    and the sqlite3 module allows extension loading).

    Probed in a subprocess: on an x86_64 Python running under Rosetta the
    sqlite-vec dylib dies with SIGILL (AVX), which would take the whole
    pytest process down rather than raise."""
    global _SQLITE_VEC_OK
    if _SQLITE_VEC_OK is None:
        import subprocess
        import sys
        probe = (
            "import sqlite3, sqlite_vec; c = sqlite3.connect(':memory:'); "
            "c.enable_load_extension(True); sqlite_vec.load(c); "
            "c.execute('select vec_version()')"
        )
        try:
            r = subprocess.run([sys.executable, "-c", probe], capture_output=True, timeout=60)
            _SQLITE_VEC_OK = r.returncode == 0
        except Exception:  # noqa: BLE001
            _SQLITE_VEC_OK = False
    return _SQLITE_VEC_OK


class FakeEmbedder(BaseEmbedder):
    """Deterministic bag-of-words embedder: each word hashes to one of
    `dim` buckets, so texts sharing words are close. No network."""

    provider = "fake"

    def __init__(self, *, model: str = "fake-model", dim: int = 8, **kwargs) -> None:
        super().__init__(model=model, dim=dim, **kwargs)
        self.calls: list[tuple[str, list[str]]] = []
        self.fail_with: str | None = None

    def _embed_raw(self, texts: list[str], *, kind: str) -> list[list[float]]:
        self.calls.append((kind, list(texts)))
        if self.fail_with:
            raise EmbeddingError(self.fail_with)
        out = []
        for t in texts:
            v = [0.0] * self.dim
            for w in re.findall(r"\w+", t.lower()):
                h = int(hashlib.md5(w.encode()).hexdigest(), 16)
                v[h % self.dim] += 1.0
            n = math.sqrt(sum(x * x for x in v)) or 1.0
            out.append([x / n for x in v])
        return out
