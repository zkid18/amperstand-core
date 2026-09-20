"""FastAPI application — capture URLs and serve a markdown vault over HTTP."""

from __future__ import annotations

import asyncio
import json
import logging
import os
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any

from fastapi import Depends, FastAPI, HTTPException, Query, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from amperstand_core.converter import to_markdown
from amperstand_core.extractor import (
    extract_article,
    extract_article_from_html,
    is_linkedin_url,
    is_youtube_url,
)
from amperstand_core.linkedin import extract_linkedin
from amperstand_core.youtube import (
    YouTubeTranscriptUnavailable,
    extract_youtube,
    youtube_stub_from_html,
)

from amperstand_core.server.capture_jobs import (
    STATUS_DONE,
    STATUS_FAILED,
    STATUS_QUEUED,
    STATUS_RUNNING,
    JobStore,
    backoff_seconds,
)
from amperstand_core.server.chat_api import router as chat_router
from amperstand_core.server.feed_api import router as feed_router
from amperstand_core.server.vault_api import router as vault_router
from amperstand_core.server.vault_api.auth import require_api_key
from amperstand_core.server.vault_api.store_factory import get_store
from amperstand_core.server.web import mount_static as mount_web_static, router as web_router

# Surface amperstand-core's INFO logs through the same stderr stream uvicorn
# writes to, which systemd's StandardError=journal then captures. Without
# this the worker startup line + per-job done/failed lines were INVISIBLE
# in `journalctl -u amperstand-server` (F3 from the queue e2e report). Only
# touched when no handler is already configured — tests + production both
# fall through cleanly.
if not logging.getLogger("amperstand_core").handlers:
    _h = logging.StreamHandler()
    _h.setFormatter(logging.Formatter(
        "%(asctime)s %(levelname)s %(name)s: %(message)s"
    ))
    _root = logging.getLogger("amperstand_core")
    _root.addHandler(_h)
    _root.setLevel(logging.INFO)

log = logging.getLogger(__name__)

_job_store_singleton: JobStore | None = None


def _job_store() -> JobStore:
    """Lazy singleton — the JobStore lives next to the vault on disk so it
    follows the same data-dir hygiene as everything else (one DB per droplet,
    survives restarts, no separate config to forget)."""
    global _job_store_singleton
    if _job_store_singleton is None:
        data_dir = Path(os.environ.get("AMPERSTAND_DATA_DIR", "/var/lib/amperstand/vault"))
        _job_store_singleton = JobStore(data_dir / ".store" / "capture_jobs.db")
    return _job_store_singleton


def reset_job_store_cache() -> None:
    """Drop the cached JobStore so the next call picks up a new AMPERSTAND_DATA_DIR.
    Used by tests."""
    global _job_store_singleton
    _job_store_singleton = None


# Set by create_app() once the closure-scoped dispatch functions are wired up.
# The worker runs in a separate task and needs to call into the same
# extraction + persistence path that the sync endpoints use, but those are
# defined inside create_app's closure. The cleanest plumbing is to register
# the runner from inside the closure.
_run_job_callback: callable | None = None


def _register_job_runner(fn) -> None:
    global _run_job_callback
    _run_job_callback = fn


def _worker_count() -> int:
    """How many concurrent drain workers to run. 1 by default — fine for the
    typical clip-or-two-at-a-time desktop user. Bump it for boxes that see
    burst traffic (the bot fan-out, a feed-sync that enqueues 50 URLs at once,
    a CLI script that batch-captures a folder of bookmarks).

    The SQLite claim_next is already atomic via JobStore's write lock, so
    multiple workers won't grab the same job. Clamped to a sane range so
    operators can't accidentally fork-bomb their droplet."""
    raw = os.environ.get("AMPERSTAND_CAPTURE_WORKERS", "").strip()
    try:
        v = int(raw) if raw else 1
        return max(1, min(v, 8))
    except ValueError:
        return 1


def _job_timeout_s() -> float:
    """Per-job wall-clock budget for extract+persist. Anything slower gets
    marked failed and the worker moves on. F2 from the queue e2e report:
    a single 62s httpbin failure stalled the line behind it. Configurable
    so you can bump it for slow renders or long Whisper jobs."""
    raw = os.environ.get("AMPERSTAND_CAPTURE_JOB_TIMEOUT_S", "").strip()
    try:
        v = float(raw) if raw else 90.0
        # Clamp to a sane range — silently-tiny timeouts would brick the
        # queue; absurdly large ones defeat the point.
        return max(10.0, min(v, 600.0))
    except ValueError:
        return 90.0


def _max_attempts() -> int:
    """How many times a capture may be tried before the queue gives up on it.

    Three is enough to ride out the failures that actually resolve themselves —
    a rate limit, a flaky origin, a proxy hiccup — without grinding forever on a
    URL that is simply gone. Operators who fixed something server-side don't
    need a bigger budget; they need POST /jobs/retry-failed, which hands every
    dead job a fresh one."""
    raw = os.environ.get("AMPERSTAND_CAPTURE_MAX_ATTEMPTS", "").strip()
    try:
        v = int(raw) if raw else 3
        return max(1, min(v, 10))
    except ValueError:
        return 3


async def _record_attempt_failure(
    store: JobStore, job_id: str, worker_id: int, error: str, *, retryable: bool = True,
) -> None:
    """Book a failed attempt against a job and log which way it went.

    The worker never decides retry policy itself — JobStore.fail_or_requeue
    owns the attempt budget and the backoff, so the sync endpoints and the
    worker can't drift apart on it. The one exception is an extractor that
    has declared the failure permanent (`retryable=False`): that goes
    straight to failed without spending the remaining attempts."""
    if not retryable:
        await asyncio.to_thread(store.mark_failed, job_id, error)
        log.warning(
            "capture-worker[%d]: job %s failed permanently (not retryable): %s",
            worker_id, job_id, error,
        )
        return
    status, next_at = await asyncio.to_thread(
        store.fail_or_requeue, job_id, error, max_attempts=_max_attempts(),
    )
    if status == STATUS_QUEUED:
        log.warning(
            "capture-worker[%d]: job %s failed, requeued for %s: %s",
            worker_id, job_id, next_at, error,
        )
    else:
        log.warning(
            "capture-worker[%d]: job %s failed permanently after %d attempts: %s",
            worker_id, job_id, _max_attempts(), error,
        )


async def _capture_worker(stop: asyncio.Event, *, worker_id: int = 0) -> None:
    """Drain loop. One instance = one concurrent extraction at a time. Multiple
    instances can run safely because JobStore.claim_next is transactional —
    the SQLite UPDATE with the status guard means two workers can't pick up
    the same row. Each worker pulls from the same shared queue and processes
    independently; there's no inter-worker coordination beyond the shared
    `stop` event.

    Runs the actual extract+persist on a thread so we don't block the event
    loop on httpx/playwright/sqlite. Per-job timeout (F2): wraps the thread
    call in asyncio.wait_for so a single hung extraction can't stall the
    line. Note: a timed-out job's underlying thread keeps running (Python
    can't safely kill a thread) until the extraction's own internal timeouts
    fire. We accept short-term thread pile-up — most "slow" jobs are
    network-bound and idle. If this ever becomes a real memory issue we'd
    need subprocess-based workers, not threads.

    Exceptions are caught and recorded on the job, never propagated — the loop
    must survive any single-job failure or one bad URL kills the queue."""
    if _run_job_callback is None:
        log.error("capture-worker[%d]: no runner registered; queue will not drain", worker_id)
        return
    store = _job_store()
    timeout = _job_timeout_s()
    log.info("capture-worker[%d]: started (per-job timeout=%.0fs)", worker_id, timeout)
    while not stop.is_set():
        job = await asyncio.to_thread(store.claim_next)
        if job is None:
            # Empty queue — back off briefly, but stay responsive to shutdown.
            try:
                await asyncio.wait_for(stop.wait(), timeout=1.0)
            except asyncio.TimeoutError:
                pass
            continue
        job_id = job["id"]
        try:
            doc_id, doc_path, body_hash = await asyncio.wait_for(
                asyncio.to_thread(_run_job_callback, job),
                timeout=timeout,
            )
            await asyncio.to_thread(
                store.mark_done, job_id,
                doc_id=doc_id, doc_path=doc_path, body_hash=body_hash,
            )
            log.info("capture-worker[%d]: job %s done (doc_id=%s)", worker_id, job_id, doc_id)
        except asyncio.TimeoutError:
            log.warning(
                "capture-worker[%d]: job %s timed out after %.0fs; "
                "marking failed and moving on (thread continues in background)",
                worker_id, job_id, timeout,
            )
            await _record_attempt_failure(
                store, job_id, worker_id,
                f"Job exceeded {timeout:.0f}s budget — extractor stalled. "
                "Try again later or increase AMPERSTAND_CAPTURE_JOB_TIMEOUT_S.",
            )
        except Exception as e:  # noqa: BLE001
            log.exception("capture-worker[%d]: job %s failed", worker_id, job_id)
            await _record_attempt_failure(
                store, job_id, worker_id, f"{type(e).__name__}: {e}",
                retryable=getattr(e, "retryable", True),
            )
    log.info("capture-worker[%d]: stopped", worker_id)


def _enroll_failed_capture(
    url: str,
    error: Exception,
    *,
    persist: bool,
    frontmatter: dict[str, Any] | None = None,
    html: str | None = None,
    fallback_title: str | None = None,
) -> str | None:
    """Hand a failed synchronous capture to the retry queue.

    The sync endpoints still answer their caller immediately with the failure,
    but the URL itself is worth keeping: most capture failures are the server
    being temporarily wrong, not the link being bad. Enrolling the attempt here
    means fixing the server drains the backlog, instead of the user having to
    remember what they lost. Before this, a failed /capture left no trace
    anywhere — not in the vault, not in the queue, not in the log.

    Returns the job id, or None if the URL could not be queued.
    """
    detail = f"{type(error).__name__}: {error}"
    if getattr(error, "retryable", True) is False:
        # The extractor has said this failure is a property of the URL, not
        # of the moment (a post that doesn't exist, a rejected token). No
        # number of retries changes that, so record it and stop.
        log.warning("capture failed for %s: %s (not retryable, not queued)", url, detail)
        return None
    try:
        store = _job_store()
        existing = store.find_active_by_url(url)
        if existing is not None:
            # Someone re-sending a failing link by hand shouldn't stack up
            # duplicate rows — the one already in flight covers it.
            log.warning(
                "capture failed for %s: %s (already queued as job %s)",
                url, detail, existing["id"],
            )
            return existing["id"]
        job_id = store.enqueue(
            url,
            persist=persist,
            frontmatter=frontmatter,
            html=html,
            fallback_title=fallback_title,
            error=detail,
            attempts=1,
            delay_s=backoff_seconds(1),
        )
    except Exception:  # noqa: BLE001
        # The queue is a safety net, not the contract. If SQLite is unhappy we
        # still owe the caller the real error, so log and carry on.
        log.exception("capture failed for %s and could not be queued: %s", url, detail)
        return None
    log.warning(
        "capture failed for %s: %s (queued for retry as job %s)", url, detail, job_id,
    )
    return job_id


def _with_job_hint(message: str, job_id: str | None) -> str:
    """Append the retry job id to an error, so the response reads as
    'we will try this again' rather than 'your link is gone'."""
    return f"{message} (queued for retry as job {job_id})" if job_id else message


def _env_bool(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


class CaptureRequest(BaseModel):
    url: str
    # When true (default), persist the captured doc to the vault as well as
    # returning the extracted markdown. Was extract-only historically; the
    # README's "POST /capture — for your own scripts" implies persistence
    # and the friction inventory flagged the gap. Backward-compat is
    # preserved by leaving the extract-only fields in CaptureResponse — old
    # callers that ignore the new `id`/`path` fields keep working.
    persist: bool = True
    # Optional frontmatter overrides merged on top of what the extractor
    # produces. Used by the extension/bot to add tags like ["clipper"]
    # without having to do a second POST /vault step.
    frontmatter: dict[str, Any] | None = None


class CaptureHtmlRequest(BaseModel):
    url: str
    html: str
    title: str | None = None
    persist: bool = True
    frontmatter: dict[str, Any] | None = None


class CaptureResponse(BaseModel):
    title: str
    url: str
    content_type: str
    markdown: str
    # Populated when persist=True (default). Mirrors DocResponse identity
    # fields so callers can act on the saved doc without a second round-trip.
    id: str | None = None
    path: str | None = None
    body_hash: str | None = None


def create_app(*, docs_visible: bool | None = None) -> FastAPI:
    """Build the FastAPI app.

    OpenAPI / Swagger / Redoc are EXPOSED by default — integrators need a
    discoverable schema. The auth dependency still protects every endpoint,
    so a passer-by sees the schema but can't actually call anything without
    the API key. To suppress the schema entirely (paranoid deploys), set
    AMPERSTAND_HIDE_DOCS=1.
    """
    if docs_visible is None:
        # New default: docs are ON unless explicitly hidden. Old AMPERSTAND_
        # PUBLIC_DOCS env var is honored for backward compat — if set, treat
        # as "force visible" (matches its old behavior).
        if _env_bool("AMPERSTAND_HIDE_DOCS"):
            docs_visible = False
        else:
            docs_visible = True
    docs_kw = (
        {}
        if docs_visible
        else {"docs_url": None, "redoc_url": None, "openapi_url": None}
    )

    @asynccontextmanager
    async def _lifespan(app: FastAPI):
        # Reset any jobs left in `running` state by a previous crash/restart
        # so the worker picks them up again.
        store = _job_store()
        n = store.reset_running_on_startup()
        if n:
            log.info("capture-jobs: reset %d running jobs to queued", n)
        n_workers = _worker_count()
        log.info("capture-worker: lifespan starting (workers=%d)", n_workers)

        # Spawn N workers, all sharing the same stop event. claim_next on the
        # JobStore is transactional, so they won't grab duplicates.
        stop = asyncio.Event()
        tasks = [
            asyncio.create_task(_capture_worker(stop, worker_id=i))
            for i in range(n_workers)
        ]
        try:
            yield
        finally:
            # Graceful shutdown — F4 from the queue e2e report: the previous
            # 10s wait_for + task.cancel() was too aggressive; mid-fetch
            # jobs run on a thread that ignores asyncio cancellation, and
            # systemd then SIGKILL'd the process ~20s later. Better path:
            # signal the workers (which exit at the top of the next loop
            # iteration), and give in-flight jobs 25s to finish on their
            # own. If they overrun, systemd's StopTimeoutSec eventually wins
            # — but most jobs complete inside the budget. Jobs still
            # `running` at exit are reset to `queued` on next startup.
            stop.set()
            log.info(
                "capture-worker: stop signalled; waiting up to 25s for "
                "%d in-flight workers", n_workers,
            )
            try:
                await asyncio.wait_for(asyncio.gather(*tasks), timeout=25)
                log.info("capture-worker: all workers shut down cleanly")
            except asyncio.TimeoutError:
                log.warning(
                    "capture-worker: %d workers did not shut down within 25s — "
                    "letting systemd terminate the process; in-flight jobs "
                    "will be reset to queued on next startup",
                    sum(1 for t in tasks if not t.done()),
                )
                for t in tasks:
                    if not t.done():
                        t.cancel()
            except asyncio.CancelledError:
                pass

    app = FastAPI(
        title="Amperstand API",
        description="Capture anything from the web as markdown.",
        version="0.1.0",
        lifespan=_lifespan,
        # Trailing-slash normalization OFF so /vault and /vault/ both work
        # the same way (route registered once; both URLs resolve to it).
        # Redirects can drop Authorization headers in strict clients and
        # are surprising for POST bodies — better to just match both shapes.
        redirect_slashes=False,
        **docs_kw,
    )

    # CORS for the browser-extension clipper. Extensions live on opaque
    # chrome-extension:// origins, so we have to allow * here. Auth still
    # gates write endpoints via the Bearer key; CORS only controls who can
    # *attempt* a request, not whether it succeeds without credentials.
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_methods=["GET", "POST", "PUT", "DELETE", "OPTIONS"],
        allow_headers=["Authorization", "Content-Type", "If-Match"],
        expose_headers=["ETag"],
    )

    app.include_router(vault_router)
    app.include_router(feed_router)
    app.include_router(chat_router)
    app.include_router(web_router)
    mount_web_static(app)

    # Root redirect so the bare URL (http://<host>/) lands on the web shell
    # instead of FastAPI's default `{"detail":"Not Found"}`. Anyone who pastes
    # just the host into a browser bar should get *something*, not a JSON 404.
    from fastapi.responses import RedirectResponse

    @app.get("/", include_in_schema=False)
    def _root_redirect() -> RedirectResponse:
        return RedirectResponse(url="/ui/")

    def _dispatch(url: str, *, html: str | None, fallback_title: str | None):
        """Pick the right extractor for a URL.

        YouTube and LinkedIn have URL-only extractors that fetch their own
        canonical sources (oembed + transcript API; LinkedIn embed iframe).
        For everything else, prefer the caller-supplied HTML when we have
        it (cheap, uses the user's authenticated browser session) and fall
        back to a fresh server-side fetch.

        YouTube special case: when no transcript is available AND the caller
        supplied HTML (the clipper), fall back to article-from-HTML
        extraction. The watch page's rendered DOM usually contains the
        description, comments, etc. — beats saving a "no transcript" stub.
        """
        if is_youtube_url(url):
            try:
                return extract_youtube(url)
            except YouTubeTranscriptUnavailable as exc:
                if not html:
                    raise
                # YouTube watch pages render the description into JS-only DOM,
                # so trafilatura would just grab the sidebar/comments chrome.
                # Pull og:title + og:description from the supplied HTML instead.
                return youtube_stub_from_html(url, html, fallback=exc)
        if is_linkedin_url(url):
            return extract_linkedin(url)
        if html:
            return extract_article_from_html(url, html, fallback_title=fallback_title)
        return extract_article(url)

    def _run_queued_job(job: dict) -> tuple[str | None, str | None, str | None]:
        """Worker-side: run a job row through the same extract+persist path
        that POST /capture uses synchronously. Returns (doc_id, path, body_hash)
        — None values when persist=False on the job."""
        url = job["url"]
        html = job.get("html")
        fallback_title = job.get("fallback_title")
        frontmatter = json.loads(job["frontmatter"]) if job.get("frontmatter") else None
        persist = bool(job.get("persist", 1))

        content = _dispatch(url, html=html, fallback_title=fallback_title)
        if not persist:
            return None, None, None
        return _persist_capture(content, frontmatter)

    _register_job_runner(_run_queued_job)

    def _persist_capture(content, frontmatter_override: dict[str, Any] | None):
        """Write captured content to the vault using the same store the vault
        router writes to. Returns (id, path, body_hash). Idempotent on
        `source` URL — re-capturing the same URL updates in place rather
        than creating a duplicate doc."""
        fm: dict[str, Any] = {
            "title": content.title,
            "source": content.url,
            "type": content.content_type.value,
        }
        if getattr(content, "author", None):
            fm["author"] = content.author
        if frontmatter_override:
            fm.update(frontmatter_override)
            # The extractor's title/source/type are authoritative — don't let
            # caller-supplied "" or None silently wipe them.
            for key in ("title", "source", "type"):
                if not fm.get(key):
                    fm.pop(key, None)
        store = get_store()
        doc = store.create(to_markdown(content), fm)
        return doc.meta.id, doc.meta.path, doc.meta.body_hash

    @app.post(
        "/capture",
        response_model=CaptureResponse,
        dependencies=[Depends(require_api_key)],
    )
    def capture(req: CaptureRequest) -> CaptureResponse:
        """Capture a URL and return clean markdown. Requires a Bearer API key.

        By default the doc is also persisted to the vault (idempotent on
        the source URL). Pass `persist=false` to extract without saving."""
        try:
            content = _dispatch(req.url, html=None, fallback_title=None)
        except Exception as e:
            job_id = _enroll_failed_capture(
                req.url, e, persist=req.persist, frontmatter=req.frontmatter,
            )
            raise HTTPException(status_code=422, detail=_with_job_hint(str(e), job_id))

        doc_id = doc_path = body_hash = None
        if req.persist:
            try:
                doc_id, doc_path, body_hash = _persist_capture(content, req.frontmatter)
            except Exception as e:
                job_id = _enroll_failed_capture(
                    req.url, e, persist=req.persist, frontmatter=req.frontmatter,
                )
                raise HTTPException(
                    status_code=500,
                    detail=_with_job_hint(
                        f"capture extracted but persist failed: {e}", job_id,
                    ),
                )

        return CaptureResponse(
            title=content.title,
            url=content.url,
            content_type=content.content_type.value,
            markdown=to_markdown(content),
            id=doc_id,
            path=doc_path,
            body_hash=body_hash,
        )

    # ── async capture (extension queue) ─────────────────────────────
    #
    # Click-and-walk-away flow: the extension/bot POSTs a URL (and optionally
    # HTML), gets a job_id back instantly, and the actual extract+persist
    # happens in a background worker. Lets the popup close in <100ms instead
    # of hanging for 30-120s on YouTube/long articles.

    @app.post(
        "/capture/async",
        status_code=202,
        dependencies=[Depends(require_api_key)],
    )
    def capture_async(req: CaptureRequest) -> dict:
        """Enqueue a URL capture for the background worker. Returns 202 +
        {job_id}. Poll GET /jobs/{id} for status."""
        job_id = _job_store().enqueue(
            url=req.url,
            persist=req.persist,
            frontmatter=req.frontmatter,
        )
        return {"job_id": job_id, "status": STATUS_QUEUED}

    @app.post(
        "/capture/html/async",
        status_code=202,
        dependencies=[Depends(require_api_key)],
    )
    def capture_html_async(req: CaptureHtmlRequest) -> dict:
        """Same as /capture/async but with caller-supplied HTML (extension's
        clip path)."""
        job_id = _job_store().enqueue(
            url=req.url,
            persist=req.persist,
            frontmatter=req.frontmatter,
            html=req.html,
            fallback_title=req.title,
        )
        return {"job_id": job_id, "status": STATUS_QUEUED}

    @app.get(
        "/jobs/{job_id}",
        dependencies=[Depends(require_api_key)],
    )
    def get_job(job_id: str) -> dict:
        """Job status + (when done) the saved doc identity."""
        row = _job_store().get(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="job not found")
        # Drop the bulky html payload from the response — it was useful
        # for resuming work but the caller doesn't need it back.
        row.pop("html", None)
        return row

    @app.get(
        "/jobs",
        dependencies=[Depends(require_api_key)],
    )
    def list_jobs(
        status: str | None = Query(default=None),
        limit: int = Query(default=50, ge=1, le=200),
    ) -> dict:
        """List jobs, newest first. Useful for the extension's 'recent clips'
        view and for operators inspecting queue depth."""
        store = _job_store()
        rows = store.list(status=status, limit=limit)
        for r in rows:
            r.pop("html", None)
        return {
            "items": rows,
            "queue_depth": store.queue_depth(),
            # Per-status totals across the whole table, not just this page —
            # the cheapest way to spot a retry backlog building up.
            "counts": store.status_counts(),
        }

    @app.post(
        "/jobs/{job_id}/retry",
        dependencies=[Depends(require_api_key)],
    )
    def retry_job(job_id: str) -> dict:
        """Re-arm one permanently-failed job with a fresh attempt budget.

        For the case where you know why it failed and you know it's fixed —
        no point waiting for a backoff that already expired."""
        store = _job_store()
        row = store.get(job_id)
        if row is None:
            raise HTTPException(status_code=404, detail="job not found")
        if row["status"] != STATUS_FAILED:
            raise HTTPException(
                status_code=409,
                detail=(
                    f"job is {row['status']}, not {STATUS_FAILED} — "
                    "only permanently-failed jobs can be re-armed"
                ),
            )
        store.retry(job_id)
        log.info("capture-jobs: job %s re-armed", job_id)
        return {"job_id": job_id, "status": STATUS_QUEUED, "url": row["url"]}

    @app.post(
        "/jobs/retry-failed",
        dependencies=[Depends(require_api_key)],
    )
    def retry_failed_jobs(
        limit: int | None = Query(default=None, ge=1, le=1000),
    ) -> dict:
        """Re-arm every permanently-failed job.

        The drain-the-backlog button: run it after fixing whatever broke
        captures server-side and the queue replays everything that died while
        it was broken. `limit` re-arms only the newest N."""
        store = _job_store()
        n = store.retry_all_failed(limit=limit)
        log.info("capture-jobs: re-armed %d failed job(s)", n)
        return {"requeued": n, "queue_depth": store.queue_depth()}

    @app.post(
        "/capture/html",
        response_model=CaptureResponse,
        dependencies=[Depends(require_api_key)],
    )
    def capture_html(req: CaptureHtmlRequest) -> CaptureResponse:
        """Capture a page from caller-supplied HTML (browser-extension clipper).

        YouTube/LinkedIn URLs route through their own URL-based extractors
        (transcript API, embed iframe) — the supplied page HTML is ignored
        for those because the canonical content lives elsewhere. Other
        URLs run trafilatura on the supplied HTML.
        """
        try:
            content = _dispatch(req.url, html=req.html, fallback_title=req.title)
        except Exception as e:
            job_id = _enroll_failed_capture(
                req.url, e, persist=req.persist, frontmatter=req.frontmatter,
                html=req.html, fallback_title=req.title,
            )
            raise HTTPException(status_code=422, detail=_with_job_hint(str(e), job_id))

        doc_id = doc_path = body_hash = None
        if req.persist:
            try:
                doc_id, doc_path, body_hash = _persist_capture(content, req.frontmatter)
            except Exception as e:
                job_id = _enroll_failed_capture(
                    req.url, e, persist=req.persist, frontmatter=req.frontmatter,
                    html=req.html, fallback_title=req.title,
                )
                raise HTTPException(
                    status_code=500,
                    detail=_with_job_hint(
                        f"capture extracted but persist failed: {e}", job_id,
                    ),
                )

        return CaptureResponse(
            title=content.title,
            url=content.url,
            content_type=content.content_type.value,
            markdown=to_markdown(content),
            id=doc_id,
            path=doc_path,
            body_hash=body_hash,
        )

    @app.get("/health")
    def health():
        return {"status": "ok"}

    @app.get(
        "/auth/verify",
        dependencies=[Depends(require_api_key)],
    )
    def auth_verify() -> dict:
        """Side-effect-free check that your API key works.

        Returns 200 with a small payload if the key is valid; returns the
        structured 401 from require_api_key otherwise. Use this from
        clients / CI to confirm credentials before making real calls —
        beats hitting /vault which returns real data.
        """
        return {"ok": True, "auth": "bearer", "scheme": "amperstand-api-key"}

    # Rotation is serialized by a single lock so two concurrent /admin/rotate-key
    # calls can't race the disk vs in-memory write (which would let the caller
    # who won the response disagree with the caller whose key ended up on disk
    # — silent lockout after the next restart). Also enforces a minimum interval
    # between rotations to defang the tight-loop abuse case where a leaked key
    # invalidates every newly-issued key faster than the operator can paste one.
    _rotate_lock = __import__("threading").Lock()
    _rotate_state: dict = {"last_rotation_ts": 0.0}
    _MIN_ROTATE_INTERVAL_S = float(os.environ.get("AMPERSTAND_ROTATE_MIN_INTERVAL_S", "10"))

    # Snapshot the env file path at app startup, NOT per-request. Otherwise an
    # operator who has AMPERSTAND_ENV_FILE in the process env can have it
    # influenced later (defense in depth — and it forecloses an
    # "arbitrary-file-overwrite via env-var" confused-deputy).
    _frozen_env_file = Path(os.environ.get("AMPERSTAND_ENV_FILE", "/etc/amperstand/env"))

    @app.post(
        "/admin/rotate-key",
        dependencies=[Depends(require_api_key)],
    )
    def admin_rotate_key(request: Request) -> dict:
        """Mint a new AMPERSTAND_API_KEY, write it to the env file, update the
        running process's in-memory key so the new value is accepted immediately.

        Gated by the EXISTING API key. Implication: anyone holding the current
        key can rotate it (potentially locking you out if it leaks). Recovery
        is SSH-only — `amperstand-admin rotate-key` on the box. Document that
        trade-off in the CLI's `rotate-key` help text.

        Other services that read /etc/amperstand/env at startup (Telegram bot,
        vault-watcher, feed-sync) still hold the old key in their process
        memory until their own restart — the response payload tells the caller.
        """
        import hashlib as _hashlib
        import secrets as _secrets
        import time as _time
        from amperstand_core.server.admin_cli.commands.rotate_key import (
            _atomic_write_preserve_meta,
            _KEY_LINE_RE,
        )

        # ── 1. Multi-worker guard. The in-memory env write only updates THIS
        #    worker process. With uvicorn --workers >1, sibling workers serve
        #    auth against stale env values → chaos. Refuse rather than silently
        #    create the mismatch.
        try:
            worker_count = int(os.environ.get("AMPERSTAND_UVICORN_WORKERS", "1"))
        except ValueError:
            worker_count = 1
        if worker_count > 1:
            raise HTTPException(
                status_code=503,
                detail=(
                    "rotate-key is not multi-worker safe; this server is configured "
                    f"with {worker_count} uvicorn workers. Use `amperstand-admin "
                    "rotate-key --env-file /etc/amperstand/env` via SSH instead."
                ),
            )

        # ── 2. HTTPS guard. Rotation returns the new key in the response body;
        #    over HTTP a MITM gets old key (request header) AND new key
        #    (response body) in one round-trip. Caddy sets X-Forwarded-Proto.
        forwarded_proto = request.headers.get("X-Forwarded-Proto", "").lower()
        if forwarded_proto != "https":
            if os.environ.get("AMPERSTAND_ALLOW_HTTP_ROTATE") != "1":
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "rotate-key requires HTTPS (X-Forwarded-Proto: https). "
                        "Returning a fresh secret over plaintext lets a MITM "
                        "capture both old and new key in one round-trip. Configure "
                        "Caddy with Caddyfile.tls, or set "
                        "AMPERSTAND_ALLOW_HTTP_ROTATE=1 on the server for dev "
                        "(NOT production)."
                    ),
                )

        # ── 3. Origin allowlist for /admin/*. If the request has an Origin
        #    header (i.e. it's from a browser), require the origin to be in the
        #    server's allowlist. CLI / curl / bot requests have no Origin —
        #    those pass through. Closes the CORS+XSS-driven rotation vector.
        origin = request.headers.get("Origin", "").strip()
        if origin:
            allowlist = [
                o.strip()
                for o in os.environ.get("AMPERSTAND_ADMIN_ORIGINS", "").split(",")
                if o.strip()
            ]
            if origin not in allowlist:
                raise HTTPException(
                    status_code=403,
                    detail=(
                        f"origin {origin!r} not in AMPERSTAND_ADMIN_ORIGINS "
                        "allowlist for /admin endpoints"
                    ),
                )

        # ── 4. Serialize + rate limit. Outside the lock first because the
        #    rate-limit check should be cheap to fail.
        with _rotate_lock:
            now = _time.monotonic()
            since_last = now - _rotate_state["last_rotation_ts"]
            if since_last < _MIN_ROTATE_INTERVAL_S:
                wait = _MIN_ROTATE_INTERVAL_S - since_last
                raise HTTPException(
                    status_code=429,
                    detail=(
                        f"rate limit: rotated {since_last:.1f}s ago; minimum "
                        f"interval is {_MIN_ROTATE_INTERVAL_S}s. Try again in "
                        f"{wait:.1f}s."
                    ),
                )

            env_file = _frozen_env_file
            if not env_file.exists():
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"env file not found on the server. Rotate via SSH: "
                        f"`amperstand-admin rotate-key --env-file <path>`."
                    ),
                )

            try:
                original = env_file.read_text(encoding="utf-8")
            except PermissionError:
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "server process cannot read its own env file. Rotate via "
                        "SSH: `amperstand-admin rotate-key`."
                    ),
                )

            if not _KEY_LINE_RE.search(original):
                raise HTTPException(
                    status_code=500,
                    detail="no `AMPERSTAND_API_KEY=` line in env file (misconfigured)",
                )

            new_key = _secrets.token_hex(32)
            rewritten = _KEY_LINE_RE.sub(
                f"AMPERSTAND_API_KEY={new_key}", original, count=1
            )

            # Broaden exception catch: any OSError (ENOSPC, EROFS, EIO) returns
            # a clean 503 instead of letting FastAPI surface a traceback that
            # may have `data` (the env-file bytes incl. the new key) in locals
            # if debug logging is on. PermissionError is an OSError subclass.
            try:
                _atomic_write_preserve_meta(env_file, rewritten.encode("utf-8"))
            except OSError as e:
                # Type name only, not str(e) — paths in the message leak the
                # frozen env file location to an unauthenticated attacker doing
                # error fingerprinting (well, authenticated here, but defence
                # in depth).
                raise HTTPException(
                    status_code=503,
                    detail=(
                        f"server failed to write env file ({type(e).__name__}). "
                        "Rotate via SSH instead."
                    ),
                )

            # Update in-memory so this worker accepts the new key on its very
            # next request. Sibling workers (if any) wouldn't see this, but
            # we already refused worker_count > 1 above.
            os.environ["AMPERSTAND_API_KEY"] = new_key
            _rotate_state["last_rotation_ts"] = _time.monotonic()

        # ── 5. Audit log. Logged AFTER the rotation succeeds; OUTSIDE the lock
        #    so log I/O doesn't hold up concurrent rate-limit checks. Fingerprint
        #    only — don't log the key itself.
        client_ip = request.client.host if request.client else "unknown"
        xff = request.headers.get("X-Forwarded-For", "")
        new_key_fp = _hashlib.sha256(new_key.encode()).hexdigest()[:12]
        log.warning(
            "admin/rotate-key invoked: client_ip=%s xff=%s new_key_fp=%s",
            client_ip, xff, new_key_fp,
        )

        return {
            "new_api_key": new_key,
            "warning": (
                "Server is using the new key immediately. Other services that "
                "read /etc/amperstand/env at startup (Telegram bot, vault-watcher, "
                "feed-sync) still hold the OLD key in process memory until their "
                "own restart — they will start getting 401 on requests. SSH in and "
                "`sudo systemctl restart amperstand-tg-bot amperstand-vault-watcher "
                "amperstand-feed-sync.service` to bring them current."
            ),
        }

    return app


# Module-level singleton used by `uvicorn amperstand_core.server.app:app`.
app = create_app()
