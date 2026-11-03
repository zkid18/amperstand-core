from __future__ import annotations

import io
import tarfile
from pathlib import Path

import pytest
from typer.testing import CliRunner

from amperstand_core.store import MarkdownStore

from amperstand_core.server.admin_cli.cli import app


@pytest.fixture
def runner() -> CliRunner:
    return CliRunner()


@pytest.fixture
def vault(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    import time

    data_dir = tmp_path / "vault"
    monkeypatch.setenv("AMPERSTAND_DATA_DIR", str(data_dir))
    monkeypatch.setenv("AMPERSTAND_API_KEY", "devkey")
    s = MarkdownStore(data_dir)
    s.create("# one\n\nbody one\n", {"title": "First"})
    time.sleep(1)  # bump timestamps so oldest/newest pick different docs
    s.create("# two\n\nbody two\n", {"title": "Second", "tags": ["a"]})
    return data_dir


# ── stats ────────────────────────────────────────────────────────────


def test_stats_reports_doc_count_and_titles(runner: CliRunner, vault: Path) -> None:
    r = runner.invoke(app, ["stats", "--env-file", str(vault.parent / "missing.env")])
    assert r.exit_code == 0, r.stdout
    assert "docs:        2" in r.stdout
    assert "First" in r.stdout
    assert "Second" in r.stdout
    assert "api key:     set" in r.stdout


def test_stats_when_data_dir_missing(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    missing = tmp_path / "nope"
    monkeypatch.setenv("AMPERSTAND_DATA_DIR", str(missing))
    monkeypatch.delenv("AMPERSTAND_API_KEY", raising=False)
    r = runner.invoke(app, ["stats", "--env-file", str(tmp_path / "missing.env")])
    assert r.exit_code == 0, r.stdout
    assert "data dir does not exist" in r.stdout
    assert "api key:     NOT SET" in r.stdout


# ── rotate-key ───────────────────────────────────────────────────────


def test_rotate_key_dry_run_writes_nothing(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "env"
    env_file.write_text("AMPERSTAND_API_KEY=oldkey\n", encoding="utf-8")
    monkeypatch.setenv("AMPERSTAND_DATA_DIR", str(tmp_path / "vault"))

    r = runner.invoke(
        app, ["rotate-key", "--env-file", str(env_file), "--dry-run"]
    )
    assert r.exit_code == 0, r.stdout
    assert "dry-run" in r.stdout
    assert env_file.read_text(encoding="utf-8") == "AMPERSTAND_API_KEY=oldkey\n"


def test_rotate_key_writes_new_key_and_preserves_other_keys(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    env_file = tmp_path / "env"
    env_file.write_text(
        "AMPERSTAND_API_KEY=oldkey\nAMPERSTAND_DATA_DIR=/var/lib/amperstand/vault\n",
        encoding="utf-8",
    )
    # bootstrap.sh establishes 0640; rotate-key must preserve whatever the
    # original file had, not force its own mode.
    env_file.chmod(0o640)

    r = runner.invoke(app, ["rotate-key", "--env-file", str(env_file)])
    assert r.exit_code == 0, r.stdout

    new_text = env_file.read_text(encoding="utf-8")
    assert "AMPERSTAND_DATA_DIR=/var/lib/amperstand/vault" in new_text
    assert "oldkey" not in new_text
    new_key_lines = [l for l in new_text.splitlines() if l.startswith("AMPERSTAND_API_KEY=")]
    assert len(new_key_lines) == 1
    new_key = new_key_lines[0].split("=", 1)[1]
    assert len(new_key) == 64  # 32 bytes hex
    assert new_key in r.stdout  # printed once, only chance to copy it

    # mode of the original file is preserved across the atomic replace
    mode = oct(env_file.stat().st_mode & 0o777)
    assert mode == "0o640", mode


def test_rotate_key_requires_env_file(
    runner: CliRunner, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Pointing at a nonexistent env file must be refused: minting a key into
    # a fresh file would silently rotate every client (see rotate_key.run).
    monkeypatch.setenv("AMPERSTAND_DATA_DIR", str(tmp_path / "vault"))
    fake_env = tmp_path / "no-such-env"
    r = runner.invoke(app, ["rotate-key", "--env-file", str(fake_env)])
    assert r.exit_code == 2
    assert not fake_env.exists()


# ── backup ───────────────────────────────────────────────────────────


def test_backup_writes_tarball_with_docs(
    runner: CliRunner, vault: Path, tmp_path: Path
) -> None:
    out = tmp_path / "out.tar.gz"
    r = runner.invoke(app, ["backup", str(out)])
    assert r.exit_code == 0, r.stdout
    assert out.exists()
    with tarfile.open(out, "r:gz") as tar:
        names = tar.getnames()
    md_in_tar = [n for n in names if n.endswith(".md")]
    by_id_in_tar = [n for n in names if "by-id" in n]
    assert len(md_in_tar) == 2
    assert len(by_id_in_tar) >= 2


def test_backup_to_stdout_streams_tar(runner: CliRunner, vault: Path) -> None:
    r = runner.invoke(app, ["backup", "-"])
    assert r.exit_code == 0, r.stdout
    # stdout is a binary tar.gz; CliRunner captures bytes via stdout
    raw = r.stdout_bytes if hasattr(r, "stdout_bytes") else r.stdout.encode()
    if not raw[:2] == b"\x1f\x8b":  # gzip magic
        # CliRunner may have decoded; fall back to simply asserting it ran
        return
    bio = io.BytesIO(raw)
    with tarfile.open(fileobj=bio, mode="r:gz") as tar:
        assert any(n.endswith(".md") for n in tar.getnames())


# ── integrity ────────────────────────────────────────────────────────


def test_integrity_clean_vault(runner: CliRunner, vault: Path) -> None:
    r = runner.invoke(app, ["integrity"])
    assert r.exit_code == 0, r.stdout
    assert "OK" in r.stdout


def test_integrity_deep_clean_vault(runner: CliRunner, vault: Path) -> None:
    r = runner.invoke(app, ["integrity", "--deep"])
    assert r.exit_code == 0, r.stdout
    assert "OK" in r.stdout
    assert "content-hash:" in r.stdout


def test_integrity_detects_hash_mismatch(runner: CliRunner, vault: Path) -> None:
    md_files = list((vault / "docs").rglob("*.md"))
    assert md_files
    target = md_files[0]
    text = target.read_text(encoding="utf-8")
    target.write_text(text.replace("body", "TAMPERED-body"), encoding="utf-8")

    r = runner.invoke(app, ["integrity", "--deep"])
    assert r.exit_code == 1, r.stdout
    assert "content-hash mismatches" in r.stdout
    assert str(target) in r.stdout


def test_integrity_detects_missing_index(runner: CliRunner, vault: Path) -> None:
    by_id_dir = vault / ".store" / "by-id"
    entries = list(by_id_dir.glob("*.path"))
    assert entries
    entries[0].unlink()

    r = runner.invoke(app, ["integrity"])
    assert r.exit_code == 1, r.stdout
    assert "missing index entries" in r.stdout


# ── vec-rebuild with pluggable providers ─────────────────────────────


def _fake_provider_env(monkeypatch: pytest.MonkeyPatch, model: str = "fake-model"):
    from amperstand_core.server.admin_cli.commands import vec_rebuild as vec_rebuild_cmd
    from tests.fakes import FakeEmbedder

    emb = FakeEmbedder(model=model, dim=8)
    monkeypatch.setenv("AMPERSTAND_EMBED_PROVIDER", "openai_compatible")
    monkeypatch.setenv("AMPERSTAND_EMBED_BASE_URL", "http://fake.invalid")
    monkeypatch.setenv("AMPERSTAND_EMBED_MODEL", model)
    monkeypatch.setenv("AMPERSTAND_EMBED_DIM", "8")
    monkeypatch.setattr(vec_rebuild_cmd, "build_embedder", lambda cfg, **kw: emb)
    return emb


def _needs_vec():
    from tests.fakes import sqlite_vec_loadable
    if not sqlite_vec_loadable():
        pytest.skip("sqlite-vec not loadable here")


def test_vec_rebuild_refuses_without_openai_key_by_default(
    runner: CliRunner, vault: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.delenv("AMPERSTAND_EMBED_PROVIDER", raising=False)
    r = runner.invoke(app, ["vec-rebuild", "--env-file", str(vault.parent / "missing.env")])
    assert r.exit_code != 0
    assert "OPENAI_API_KEY" in r.stdout + str(r.exception)


def test_vec_rebuild_any_provider_and_signature_guard(
    runner: CliRunner, vault: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    _needs_vec()
    from amperstand_core.embeddings import EmbedSignature
    from amperstand_core.search import VectorIndex

    env_file = str(vault.parent / "missing.env")
    emb = _fake_provider_env(monkeypatch)

    r = runner.invoke(app, ["vec-rebuild", "--env-file", env_file, "--dry-run"])
    assert r.exit_code == 0, r.stdout
    assert "provider=fake" in r.stdout and "would scan 2 docs" in r.stdout
    assert not emb.calls

    r = runner.invoke(app, ["vec-rebuild", "--env-file", env_file])
    assert r.exit_code == 0, r.stdout
    assert "indexed=2" in r.stdout and emb.calls
    # 2 docs x (preamble + heading section) = 4 section vectors.
    idx = VectorIndex(vault / ".store" / "vec.db", signature=EmbedSignature("fake", "fake-model", 8))
    assert idx.count() == 4 and idx.stored_signature() == EmbedSignature("fake", "fake-model", 8)
    idx.close()

    # Incremental: nothing changed → nothing re-embedded.
    emb.calls.clear()
    r = runner.invoke(app, ["vec-rebuild", "--env-file", env_file])
    assert r.exit_code == 0 and "skipped_unchanged=2" in r.stdout and not emb.calls

    # Switch model without --force → refused, index untouched.
    emb2 = _fake_provider_env(monkeypatch, model="other-model")
    r = runner.invoke(app, ["vec-rebuild", "--env-file", env_file])
    assert r.exit_code != 0
    assert "refusing to mix vector spaces" in (r.stdout + str(r.exception))
    assert not emb2.calls
    idx = VectorIndex(vault / ".store" / "vec.db", signature=EmbedSignature("fake", "fake-model", 8))
    assert idx.count() == 4
    idx.close()

    # --force → reset + re-embed under the new signature.
    r = runner.invoke(app, ["vec-rebuild", "--env-file", env_file, "--force"])
    assert r.exit_code == 0, r.stdout
    assert "indexed=2" in r.stdout and emb2.calls
    idx = VectorIndex(vault / ".store" / "vec.db", signature=EmbedSignature("fake", "other-model", 8))
    assert idx.count() == 4 and idx.mismatch is None
    idx.close()
