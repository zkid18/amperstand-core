"""SQLite-backed capture job queue with retry.

Built for the extension's clip flow: clicking the extension icon enqueues
a job, returns instantly with a job_id, and a background worker drains the
queue. Survives server restarts (queued + running jobs are picked back up
on the next startup; running jobs are reset to queued so they get retried).

This is also the durable record of every capture that FAILED, including the
ones that arrived through the synchronous /capture endpoints. A failed attempt
is requeued with exponential backoff until its attempt budget runs out, so a
capture that broke because the server was temporarily wrong — a missing
browser binary, a wedged network, a full disk — still lands in the vault once
the server is fixed, instead of being lost at the moment it failed.

Single-machine FIFO. No multi-worker coordination beyond a transactional
claim. Drop a row into `jobs`, the worker grabs it, runs it through
the same `_dispatch` + `_persist_capture` path that /capture uses, and writes
the result back. The schema is small enough to inspect with `sqlite3` if
something gets stuck.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Any

from amperstand_core.store.ids import new_id as _new_doc_id

logger = logging.getLogger(__name__)

_SCHEMA_VERSION = 2

# Tables first, then column migrations, then indexes — in that order. An index
# over a column added by a migration can only be built once the migration has
# run, and CREATE TABLE IF NOT EXISTS is a no-op against an older table, so
# putting the two in one script would fail on exactly the databases that need
# migrating.
_TABLES_SQL = """
CREATE TABLE IF NOT EXISTS jobs (
    id              TEXT PRIMARY KEY,
    url             TEXT NOT NULL,
    persist         INTEGER NOT NULL DEFAULT 1,
    frontmatter     TEXT,
    html            TEXT,
    fallback_title  TEXT,
    status          TEXT NOT NULL,
    created_at      TEXT NOT NULL,
    started_at      TEXT,
    completed_at    TEXT,
    doc_id          TEXT,
    doc_path        TEXT,
    body_hash       TEXT,
    error           TEXT,
    attempts        INTEGER NOT NULL DEFAULT 0,
    next_attempt_at TEXT
);

CREATE TABLE IF NOT EXISTS meta (
    key TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
"""

_INDEXES_SQL = """
CREATE INDEX IF NOT EXISTS idx_jobs_status_created
    ON jobs(status, created_at);

CREATE INDEX IF NOT EXISTS idx_jobs_status_next_attempt
    ON jobs(status, next_attempt_at);
"""

STATUS_QUEUED = "queued"
STATUS_RUNNING = "running"
STATUS_DONE = "done"
STATUS_FAILED = "failed"

# Retry pacing. Exponential with a cap: a capture that failed because the
# server itself was wrong (missing browser, bad key, full disk) shouldn't
# hammer the same broken path every second, but it also shouldn't sit idle for
# hours once the operator fixes things. 1min → 2 → 4 → … → 1h.
_BACKOFF_BASE_S = 60
_BACKOFF_CAP_S = 3600


def backoff_seconds(attempts: int) -> int:
    """How long to hold a job back before attempt number `attempts` + 1."""
    if attempts < 1:
        return 0
    return min(_BACKOFF_BASE_S * (2 ** (attempts - 1)), _BACKOFF_CAP_S)


def _ts(dt: datetime) -> str:
    return dt.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _now() -> str:
    return _ts(datetime.now(timezone.utc))


def _later(seconds: float) -> str:
    """A timestamp `seconds` from now, in the same format as _now(). Both are
    fixed-width UTC, so SQLite's lexicographic string comparison orders them
    correctly and we can filter on next_attempt_at <= now in SQL."""
    return _ts(datetime.now(timezone.utc) + timedelta(seconds=seconds))


def _new_id() -> str:
    # Reuse the project's ULID generator so job IDs and doc IDs sort/compare
    # consistently. Job IDs aren't doc IDs — they live in a separate table —
    # but using the same ULID generator keeps the codebase coherent.
    return _new_doc_id()


class JobStore:
    """SQLite-backed capture job store. Thread-safe for our single-worker model
    via a per-instance lock around writes. Reads use a separate connection each
    time, which is fine at the volumes we expect (clip = O(seconds), not O(ms))."""

    def __init__(self, path: Path) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._write_lock = threading.Lock()
        with self._conn() as conn:
            conn.executescript(_TABLES_SQL)
            self._migrate(conn)
            conn.executescript(_INDEXES_SQL)

    def _conn(self) -> sqlite3.Connection:
        c = sqlite3.connect(self.path, isolation_level=None, timeout=30)
        c.row_factory = sqlite3.Row
        c.execute("PRAGMA journal_mode=WAL")
        c.execute("PRAGMA synchronous=NORMAL")
        return c

    def _migrate(self, conn: sqlite3.Connection) -> None:
        """Bring an existing DB up to _SCHEMA_VERSION.

        Additive only. Columns are added, never dropped or retyped, so a DB
        touched by a newer server stays readable by an older one — an operator
        who rolls back a deploy keeps their queue instead of losing it.
        """
        cols = {r["name"] for r in conn.execute("PRAGMA table_info(jobs)")}
        if "next_attempt_at" not in cols:
            conn.execute("ALTER TABLE jobs ADD COLUMN next_attempt_at TEXT")
            logger.info("capture-jobs: migrated schema to v%d", _SCHEMA_VERSION)
        conn.execute(
            "INSERT INTO meta(key, value) VALUES('schema_version', ?)"
            " ON CONFLICT(key) DO UPDATE SET value = excluded.value",
            (str(_SCHEMA_VERSION),),
        )

    # ── writes ──────────────────────────────────────────────────────

    def enqueue(
        self,
        url: str,
        *,
        persist: bool = True,
        frontmatter: dict[str, Any] | None = None,
        html: str | None = None,
        fallback_title: str | None = None,
        error: str | None = None,
        attempts: int = 0,
        delay_s: float = 0,
    ) -> str:
        """Add a new queued job. Returns the job_id.

        `error`, `attempts` and `delay_s` are for enrolling a capture that has
        already failed once: the synchronous /capture endpoints hand their
        failure straight to the queue so the URL isn't lost. Such a row carries
        the first error and is held back until its backoff expires.
        """
        job_id = _new_id()
        fm_json = json.dumps(frontmatter) if frontmatter else None
        next_at = _later(delay_s) if delay_s > 0 else None
        with self._write_lock, self._conn() as conn:
            conn.execute(
                "INSERT INTO jobs"
                "(id, url, persist, frontmatter, html, fallback_title,"
                " status, created_at, attempts, error, next_attempt_at)"
                " VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                (
                    job_id, url, 1 if persist else 0, fm_json,
                    html, fallback_title,
                    STATUS_QUEUED, _now(), attempts, error, next_at,
                ),
            )
        return job_id

    def claim_next(self) -> dict | None:
        """Atomically grab the oldest queued job whose backoff has expired and
        mark it running. Returns None when nothing is eligible. The
        transactional update prevents two workers from grabbing the same job
        (matters if we ever scale beyond the single-worker model)."""
        with self._write_lock, self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE status = ?"
                "   AND (next_attempt_at IS NULL OR next_attempt_at <= ?)"
                " ORDER BY created_at ASC LIMIT 1",
                (STATUS_QUEUED, _now()),
            ).fetchone()
            if row is None:
                return None
            now = _now()
            conn.execute(
                "UPDATE jobs SET status = ?, started_at = ?, attempts = attempts + 1"
                " WHERE id = ? AND status = ?",
                (STATUS_RUNNING, now, row["id"], STATUS_QUEUED),
            )
            return dict(row)

    def mark_done(self, job_id: str, *, doc_id: str | None, doc_path: str | None, body_hash: str | None) -> None:
        with self._write_lock, self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, completed_at = ?, "
                " doc_id = ?, doc_path = ?, body_hash = ?, error = NULL,"
                " next_attempt_at = NULL"
                " WHERE id = ?",
                (STATUS_DONE, _now(), doc_id, doc_path, body_hash, job_id),
            )

    def mark_failed(self, job_id: str, error: str) -> None:
        """Terminal failure — no further retries."""
        with self._write_lock, self._conn() as conn:
            conn.execute(
                "UPDATE jobs SET status = ?, completed_at = ?, error = ?,"
                " next_attempt_at = NULL"
                " WHERE id = ?",
                (STATUS_FAILED, _now(), error[:2000], job_id),
            )

    def fail_or_requeue(
        self, job_id: str, error: str, *, max_attempts: int
    ) -> tuple[str, str | None]:
        """Record a failed attempt and decide what happens next.

        Requeues the job behind an exponential backoff while it still has
        attempts left; marks it terminally failed once the budget is spent.
        Returns (new_status, next_attempt_at) so the caller can log which
        happened.
        """
        with self._write_lock, self._conn() as conn:
            row = conn.execute(
                "SELECT attempts FROM jobs WHERE id = ?", (job_id,)
            ).fetchone()
            # A row that vanished mid-flight (manual DB surgery) gets the
            # terminal branch rather than an exception — the worker loop must
            # survive anything a single job does.
            attempts = int(row["attempts"]) if row else max_attempts
            if attempts < max_attempts:
                next_at = _later(backoff_seconds(attempts))
                conn.execute(
                    "UPDATE jobs SET status = ?, started_at = NULL,"
                    " completed_at = NULL, error = ?, next_attempt_at = ?"
                    " WHERE id = ?",
                    (STATUS_QUEUED, error[:2000], next_at, job_id),
                )
                return STATUS_QUEUED, next_at
            conn.execute(
                "UPDATE jobs SET status = ?, completed_at = ?, error = ?,"
                " next_attempt_at = NULL WHERE id = ?",
                (STATUS_FAILED, _now(), error[:2000], job_id),
            )
            return STATUS_FAILED, None

    def retry(self, job_id: str) -> bool:
        """Re-arm one terminally-failed job for immediate reprocessing with a
        fresh attempt budget. Returns False when there's no such failed job."""
        with self._write_lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = ?, attempts = 0, started_at = NULL,"
                " completed_at = NULL, next_attempt_at = NULL"
                " WHERE id = ? AND status = ?",
                (STATUS_QUEUED, job_id, STATUS_FAILED),
            )
            return cur.rowcount > 0

    def retry_all_failed(self, *, limit: int | None = None) -> int:
        """Re-arm every terminally-failed job. Returns how many were requeued.

        The drain-the-backlog button: after fixing whatever broke captures
        server-side, this replays the ones that died while it was broken.
        """
        with self._write_lock, self._conn() as conn:
            sql = (
                "UPDATE jobs SET status = ?, attempts = 0, started_at = NULL,"
                " completed_at = NULL, next_attempt_at = NULL WHERE status = ?"
            )
            params: tuple = (STATUS_QUEUED, STATUS_FAILED)
            if limit is not None:
                sql += (
                    " AND id IN (SELECT id FROM jobs WHERE status = ?"
                    "            ORDER BY created_at DESC LIMIT ?)"
                )
                params = (STATUS_QUEUED, STATUS_FAILED, STATUS_FAILED, limit)
            cur = conn.execute(sql, params)
            return cur.rowcount

    def reset_running_on_startup(self) -> int:
        """Jobs that were `running` when the server died should be retried.
        Move them back to `queued` and clear any backoff so they go straight
        back into the line. Returns the number reset."""
        with self._write_lock, self._conn() as conn:
            cur = conn.execute(
                "UPDATE jobs SET status = ?, started_at = NULL,"
                " next_attempt_at = NULL WHERE status = ?",
                (STATUS_QUEUED, STATUS_RUNNING),
            )
            return cur.rowcount

    # ── reads ───────────────────────────────────────────────────────

    def get(self, job_id: str) -> dict | None:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE id = ?", (job_id,),
            ).fetchone()
            return dict(row) if row else None

    def find_active_by_url(self, url: str) -> dict | None:
        """The queued/running job for this URL, if there is one. Keeps someone
        who re-sends a failing link by hand from stacking duplicate rows."""
        with self._conn() as conn:
            row = conn.execute(
                "SELECT * FROM jobs WHERE url = ? AND status IN (?, ?)"
                " ORDER BY created_at DESC LIMIT 1",
                (url, STATUS_QUEUED, STATUS_RUNNING),
            ).fetchone()
            return dict(row) if row else None

    def list(self, *, status: str | None = None, limit: int = 50) -> list[dict]:
        with self._conn() as conn:
            if status:
                rows = conn.execute(
                    "SELECT * FROM jobs WHERE status = ? ORDER BY created_at DESC LIMIT ?",
                    (status, limit),
                ).fetchall()
            else:
                rows = conn.execute(
                    "SELECT * FROM jobs ORDER BY created_at DESC LIMIT ?",
                    (limit,),
                ).fetchall()
            return [dict(r) for r in rows]

    def queue_depth(self) -> int:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) FROM jobs WHERE status = ?",
                (STATUS_QUEUED,),
            ).fetchone()
            return row[0] if row else 0

    def status_counts(self) -> dict[str, int]:
        """Row count per status. Cheap enough to include on every /jobs list,
        and it's the fastest way to see a retry backlog building up."""
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT status, COUNT(*) AS n FROM jobs GROUP BY status"
            ).fetchall()
            return {r["status"]: r["n"] for r in rows}
