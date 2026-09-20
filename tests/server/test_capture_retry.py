"""Retry queue: failed captures are recorded, requeued with backoff, re-armable.

The behaviour these cover exists because a capture that failed used to leave
no trace at all on the synchronous path — no doc, no job row, no log line — so
a server-side breakage silently ate every URL sent at it until someone noticed.
"""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from amperstand_core.server.app import app, reset_job_store_cache
from amperstand_core.server.capture_jobs import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    JobStore,
    backoff_seconds,
)
from amperstand_core.server.vault_api.store_factory import reset_store_cache

HEAD = {"Authorization": "Bearer devkey"}


@pytest.fixture
def store(tmp_path: Path) -> JobStore:
    return JobStore(tmp_path / "jobs.db")


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("AMPERSTAND_API_KEY", "devkey")
    monkeypatch.setenv("AMPERSTAND_DATA_DIR", str(tmp_path))
    # One attempt, so a failure goes terminal immediately and the tests don't
    # have to wait out a backoff to see the permanent-failure branch.
    monkeypatch.setenv("AMPERSTAND_CAPTURE_MAX_ATTEMPTS", "1")
    reset_store_cache()
    reset_job_store_cache()
    with TestClient(app) as c:
        yield c
    reset_store_cache()
    reset_job_store_cache()


# ── backoff ─────────────────────────────────────────────────────────


class TestBackoff:
    def test_grows_then_caps(self):
        assert backoff_seconds(0) == 0
        assert backoff_seconds(1) == 60
        assert backoff_seconds(2) == 120
        assert backoff_seconds(3) == 240
        # Never waits longer than an hour, however many attempts have piled up.
        assert backoff_seconds(50) == 3600


# ── store-level retry mechanics ─────────────────────────────────────


class TestFailOrRequeue:
    def test_requeues_while_budget_remains(self, store: JobStore):
        job_id = store.enqueue("https://example.com/a")
        store.claim_next()  # attempts -> 1

        status, next_at = store.fail_or_requeue(job_id, "boom", max_attempts=3)

        assert status == STATUS_QUEUED
        assert next_at is not None
        row = store.get(job_id)
        assert row["error"] == "boom"
        assert row["attempts"] == 1
        assert row["completed_at"] is None

    def test_goes_terminal_when_budget_spent(self, store: JobStore):
        job_id = store.enqueue("https://example.com/b")
        store.claim_next()

        status, next_at = store.fail_or_requeue(job_id, "boom", max_attempts=1)

        assert status == STATUS_FAILED
        assert next_at is None
        assert store.get(job_id)["completed_at"] is not None

    def test_backed_off_job_is_not_claimable(self, store: JobStore):
        job_id = store.enqueue("https://example.com/c")
        store.claim_next()
        store.fail_or_requeue(job_id, "boom", max_attempts=3)

        # Requeued, but held behind its backoff — the worker must skip it
        # rather than spin on the same broken URL.
        assert store.get(job_id)["status"] == STATUS_QUEUED
        assert store.claim_next() is None

    def test_missing_row_does_not_raise(self, store: JobStore):
        """The worker loop must survive a job that vanished mid-flight."""
        status, _ = store.fail_or_requeue("NOPE", "boom", max_attempts=3)
        assert status == STATUS_FAILED


class TestManualRetry:
    def test_retry_rearms_failed_job(self, store: JobStore):
        job_id = store.enqueue("https://example.com/d")
        store.claim_next()
        store.fail_or_requeue(job_id, "boom", max_attempts=1)

        assert store.retry(job_id) is True
        row = store.get(job_id)
        assert row["status"] == STATUS_QUEUED
        assert row["attempts"] == 0
        assert row["next_attempt_at"] is None
        # Re-armed jobs are immediately eligible, not stuck behind a backoff.
        assert store.claim_next() is not None

    def test_retry_ignores_non_failed_jobs(self, store: JobStore):
        job_id = store.enqueue("https://example.com/e")
        assert store.retry(job_id) is False

    def test_retry_all_failed(self, store: JobStore):
        for i in range(3):
            jid = store.enqueue(f"https://example.com/bulk{i}")
            store.claim_next()
            store.fail_or_requeue(jid, "boom", max_attempts=1)
        ok = store.enqueue("https://example.com/fine")
        store.claim_next()
        store.mark_done(ok, doc_id="D", doc_path="p", body_hash="h")

        assert store.retry_all_failed() == 3
        counts = store.status_counts()
        assert counts.get(STATUS_QUEUED) == 3
        assert counts.get(STATUS_DONE) == 1
        assert counts.get(STATUS_FAILED) is None


class TestDedup:
    def test_find_active_by_url(self, store: JobStore):
        job_id = store.enqueue("https://example.com/dup")
        assert store.find_active_by_url("https://example.com/dup")["id"] == job_id
        assert store.find_active_by_url("https://example.com/other") is None

    def test_completed_job_is_not_active(self, store: JobStore):
        job_id = store.enqueue("https://example.com/done")
        store.claim_next()
        store.mark_done(job_id, doc_id="D", doc_path="p", body_hash="h")
        assert store.find_active_by_url("https://example.com/done") is None


class TestMigration:
    def test_v1_database_gains_next_attempt_at(self, tmp_path: Path):
        """An existing queue must survive the upgrade with its rows intact."""
        import sqlite3

        db = tmp_path / "old.db"
        conn = sqlite3.connect(db)
        conn.executescript(
            """
            CREATE TABLE jobs (
                id TEXT PRIMARY KEY, url TEXT NOT NULL,
                persist INTEGER NOT NULL DEFAULT 1,
                frontmatter TEXT, html TEXT, fallback_title TEXT,
                status TEXT NOT NULL, created_at TEXT NOT NULL,
                started_at TEXT, completed_at TEXT, doc_id TEXT,
                doc_path TEXT, body_hash TEXT, error TEXT,
                attempts INTEGER NOT NULL DEFAULT 0
            );
            CREATE TABLE meta (key TEXT PRIMARY KEY, value TEXT NOT NULL);
            INSERT INTO meta VALUES ('schema_version', '1');
            INSERT INTO jobs (id, url, status, created_at, attempts)
                VALUES ('OLD1', 'https://example.com/legacy', 'failed', '2026-01-01T00:00:00Z', 2);
            """
        )
        conn.commit()
        conn.close()

        store = JobStore(db)

        row = store.get("OLD1")
        assert row["url"] == "https://example.com/legacy"
        assert row["next_attempt_at"] is None
        # And the migrated row is re-armable like any other.
        assert store.retry("OLD1") is True


# ── HTTP surface ────────────────────────────────────────────────────


def _boom(*_args, **_kwargs):
    raise ValueError("anti-bot wall served")


class TestSyncCaptureEnrollsFailures:
    def test_failed_capture_is_queued_and_named_in_the_error(self, client: TestClient):
        with patch("amperstand_core.server.app.extract_article", _boom):
            r = client.post(
                "/capture", json={"url": "https://example.com/blocked"}, headers=HEAD,
            )
        assert r.status_code == 422
        detail = r.json()["detail"]
        assert "anti-bot wall served" in detail
        assert "queued for retry as job" in detail

        jobs = client.get("/jobs", headers=HEAD).json()
        urls = [j["url"] for j in jobs["items"]]
        assert "https://example.com/blocked" in urls

    def test_repeat_failure_reuses_the_queued_job(self, client: TestClient):
        with patch("amperstand_core.server.app.extract_article", _boom):
            for _ in range(3):
                client.post(
                    "/capture", json={"url": "https://example.com/same"}, headers=HEAD,
                )

        jobs = client.get("/jobs", headers=HEAD).json()
        same = [j for j in jobs["items"] if j["url"] == "https://example.com/same"]
        assert len(same) == 1

    def test_capture_html_failure_keeps_the_html_for_the_retry(self, client: TestClient):
        with patch("amperstand_core.server.app.extract_article_from_html", _boom):
            r = client.post(
                "/capture/html",
                json={"url": "https://example.com/h", "html": "<p>body</p>"},
                headers=HEAD,
            )
        assert r.status_code == 422

        from amperstand_core.server.app import _job_store

        row = _job_store().find_active_by_url("https://example.com/h")
        assert row is not None
        assert row["html"] == "<p>body</p>"


class TestRetryEndpoints:
    def _one_failed_job(self, client: TestClient) -> str:
        from amperstand_core.server.app import _job_store

        store = _job_store()
        job_id = store.enqueue("https://example.com/dead")
        store.claim_next()
        store.fail_or_requeue(job_id, "boom", max_attempts=1)
        return job_id

    def test_retry_one(self, client: TestClient):
        job_id = self._one_failed_job(client)
        r = client.post(f"/jobs/{job_id}/retry", headers=HEAD)
        assert r.status_code == 200
        assert r.json()["status"] == STATUS_QUEUED

    def test_retry_unknown_job_is_404(self, client: TestClient):
        assert client.post("/jobs/NOPE/retry", headers=HEAD).status_code == 404

    def test_retry_non_failed_job_is_409(self, client: TestClient):
        from amperstand_core.server.app import _job_store

        job_id = _job_store().enqueue("https://example.com/queued")
        assert client.post(f"/jobs/{job_id}/retry", headers=HEAD).status_code == 409

    def test_retry_all_failed(self, client: TestClient):
        self._one_failed_job(client)
        r = client.post("/jobs/retry-failed", headers=HEAD)
        assert r.status_code == 200
        assert r.json()["requeued"] >= 1

    def test_retry_requires_auth(self, client: TestClient):
        assert client.post("/jobs/retry-failed").status_code == 401


class TestJobsListing:
    def test_counts_are_reported(self, client: TestClient):
        body = client.get("/jobs", headers=HEAD).json()
        assert "counts" in body
        assert "queue_depth" in body


# ── permanent failures ──────────────────────────────────────────────
#
# An extractor can mark a failure as not worth retrying by raising an error
# whose `retryable` attribute is False (a post that doesn't exist, a rejected
# token). Those must not consume retry attempts on either path.


@pytest.fixture
def client3(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    """Same as `client` but with a real attempt budget, so a job that goes
    terminal on attempt 1 proves it did so because it was told to."""
    monkeypatch.setenv("AMPERSTAND_API_KEY", "devkey")
    monkeypatch.setenv("AMPERSTAND_DATA_DIR", str(tmp_path))
    monkeypatch.setenv("AMPERSTAND_CAPTURE_MAX_ATTEMPTS", "3")
    reset_store_cache()
    reset_job_store_cache()
    with TestClient(app) as c:
        yield c
    reset_store_cache()
    reset_job_store_cache()


class _PermanentError(RuntimeError):
    """Stand-in for an extractor error that declares itself permanent."""
    retryable = False


def _permanent(*_a, **_k):
    raise _PermanentError("Post not found")


class TestPermanentFailures:
    def test_sync_capture_is_not_queued(self, client3: TestClient):
        with patch("amperstand_core.server.app.extract_article", _permanent):
            r = client3.post("/capture", json={"url": "https://example.com/gone"}, headers=HEAD)
        assert r.status_code == 422
        assert "Post not found" in r.json()["detail"]
        assert "queued for retry" not in r.json()["detail"]
        assert client3.get("/jobs", headers=HEAD).json()["items"] == []

    def test_worker_marks_failed_on_first_attempt(self, client3: TestClient):
        import time

        with patch("amperstand_core.server.app.extract_article", _permanent):
            job_id = client3.post(
                "/capture/async", json={"url": "https://example.com/gone"}, headers=HEAD,
            ).json()["job_id"]
            deadline = time.time() + 10
            row = None
            while time.time() < deadline:
                row = client3.get(f"/jobs/{job_id}", headers=HEAD).json()
                if row["status"] in (STATUS_FAILED, STATUS_DONE):
                    break
                time.sleep(0.1)
        assert row is not None
        assert row["status"] == STATUS_FAILED
        assert row["attempts"] == 1
        assert "Post not found" in row["error"]
