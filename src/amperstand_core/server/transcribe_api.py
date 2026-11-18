"""Transcription jobs: upload a recording, the server transcribes it and saves the transcript.

    POST /transcribe/jobs?filename=talk.m4a&title=…   raw audio/video bytes → 202 {id, …}
    GET  /transcribe/jobs/{id}                         status, progress, doc_id once saved

Jobs run one at a time, in upload order, each in its own worker process (see
amperstand_core.transcription). A job's files live under
`{data dir}/.store/transcribe/{id}/`; the audio is deleted once the transcript
is saved. A job interrupted by a restart starts over when the server comes back.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re
import secrets
import shutil
import subprocess
import sys
import threading
import time
from pathlib import Path
from typing import Any

from fastapi import APIRouter, Depends, HTTPException, Query, Request

from amperstand_core import transcription
from amperstand_core.server.vault_api.auth import require_api_key
from amperstand_core.server.vault_api.store_factory import _resolve_data_dir, get_store

log = logging.getLogger(__name__)

router = APIRouter(prefix="/transcribe", tags=["transcribe"], dependencies=[Depends(require_api_key)])

# Containers PyAV reads that people actually record or download in.
AUDIO_EXTENSIONS = frozenset(
    {"aac", "flac", "m4a", "mkv", "mov", "mp3", "mp4", "mpeg", "oga", "ogg", "opus", "wav", "webm"}
)
_JOB_ID = re.compile(r"^\d{10}-[0-9a-f]{8}$")
QUEUED, RUNNING, DONE, FAILED = "queued", "running", "done", "failed"

# The command that transcribes one job dir. Tests swap in a fake.
WORKER_CMD: list[str] = [sys.executable, "-m", "amperstand_core.transcription"]


def jobs_root() -> Path:
    return _resolve_data_dir() / ".store" / "transcribe"


def max_upload_bytes() -> int:
    raw = os.environ.get("AMPERSTAND_TRANSCRIBE_MAX_MB", "").strip()
    try:
        mb = int(raw) if raw else 500
    except ValueError:
        mb = 500
    return max(1, mb) * 1024 * 1024


def _read(job_dir: Path) -> dict[str, Any]:
    return json.loads((job_dir / "job.json").read_text())


def _write(job_dir: Path, job: dict[str, Any]) -> None:
    tmp = job_dir / "job.json.tmp"
    tmp.write_text(json.dumps(job, ensure_ascii=False))
    tmp.replace(job_dir / "job.json")


def _update(job_dir: Path, **changes: Any) -> dict[str, Any]:
    job = _read(job_dir) | changes
    _write(job_dir, job)
    return job


def _jobs(status: str) -> list[Path]:
    root = jobs_root()
    if not root.is_dir():
        return []
    out = []
    for d in sorted(root.iterdir()):  # ids start with a timestamp: oldest first
        try:
            if d.is_dir() and _read(d)["status"] == status:
                out.append(d)
        except (OSError, ValueError, KeyError):
            continue
    return out


class Runner:
    """One background thread that works through queued jobs, oldest first."""

    def __init__(self) -> None:
        self._wake = threading.Event()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()

    def start(self) -> None:
        with self._lock:
            if self._thread and self._thread.is_alive():
                self._stop.clear()  # a stop() that hasn't landed yet is cancelled
                return
            for d in _jobs(RUNNING):  # interrupted by a restart: start over
                _update(d, status=QUEUED, started=None)
            self._stop.clear()
            self._thread = threading.Thread(target=self._loop, name="transcribe-runner", daemon=True)
            self._thread.start()

    def wake(self) -> None:
        self._wake.set()

    def stop(self) -> None:
        self._stop.set()
        self._wake.set()

    def _loop(self) -> None:
        while not self._stop.is_set():
            queued = _jobs(QUEUED)
            if not queued:
                self._wake.wait(timeout=60)
                self._wake.clear()
                continue
            try:
                self.run_one(queued[0])
            except Exception:  # noqa: BLE001 — never let one job kill the runner
                log.exception("transcribe: job %s crashed the runner step", queued[0].name)

    def run_one(self, job_dir: Path) -> None:
        job = _update(job_dir, status=RUNNING, started=time.time())
        log.info("transcribe: job %s started (%s, %.0f s)", job["id"], job["file_name"], job["duration"])
        proc = subprocess.run(
            [*WORKER_CMD, str(job_dir)],
            capture_output=True,
            text=True,
            env={**os.environ, **transcription.MALLOC_ENV},
        )
        if proc.returncode != 0:
            tail = " | ".join(proc.stderr.strip().splitlines()[-3:])[:500] or f"exit status {proc.returncode}"
            _update(job_dir, status=FAILED, finished=time.time(), error=tail)
            log.warning("transcribe: job %s failed: %s", job["id"], tail)
            return
        try:
            result = json.loads((job_dir / "result.json").read_text())
            segments = [transcription.Segment(**s) for s in result["segments"]]
            body = transcription.build_body(
                segments, file_name=job["file_name"], duration=job["duration"], language=result.get("language")
            )
            frontmatter: dict[str, Any] = {"title": job["title"], "type": job["type"], "tags": job["tags"]}
            if job.get("source"):
                frontmatter["source"] = job["source"]
            doc = get_store().create(body, frontmatter)
        except Exception as exc:  # noqa: BLE001
            _update(job_dir, status=FAILED, finished=time.time(), error=f"saving the transcript failed: {exc}"[:500])
            log.exception("transcribe: job %s could not be saved", job["id"])
            return
        for name in (job["audio_file"], "result.json", "progress.json"):
            (job_dir / name).unlink(missing_ok=True)
        _update(job_dir, status=DONE, finished=time.time(), doc_id=doc.meta.id, language=result.get("language"))
        log.info("transcribe: job %s saved as %s", job["id"], doc.meta.id)


runner = Runner()


def _status(job_dir: Path) -> dict[str, Any]:
    job = _read(job_dir)
    out = {k: job.get(k) for k in ("id", "status", "title", "duration", "doc_id", "error", "created", "started", "finished")}
    if job["status"] == RUNNING:
        try:
            out["done_seconds"] = json.loads((job_dir / "progress.json").read_text())["done_seconds"]
        except (OSError, ValueError, KeyError):
            out["done_seconds"] = 0.0
    if job["status"] == QUEUED:
        out["queue_position"] = [d.name for d in _jobs(QUEUED)].index(job_dir.name) + 1
    return out


@router.post("/jobs", status_code=202)
async def submit_job(
    request: Request,
    filename: str = Query(min_length=1, max_length=255, description="Name of the uploaded file; its extension tells the format."),
    original_name: str | None = Query(
        default=None, max_length=255, description="Name of the recording the audio came from, shown in the transcript."
    ),
    title: str | None = Query(default=None, max_length=300, description="Document title. Default: the file name."),
    source: str | None = Query(default=None, max_length=2000, description="URL where the recording is published."),
    tags: list[str] = Query(default=[], description="Tags for the document, besides `transcript`."),
    media_type: str = Query(default="audio", alias="type", pattern=r"^(audio|video)$"),
    language: str | None = Query(default=None, pattern=r"^[a-z]{2}$", description="ISO 639-1 code. Default: detected."),
) -> dict[str, Any]:
    """Queue a recording for transcription. The request body is the raw file."""
    if not transcription.available():
        raise HTTPException(
            status_code=503,
            detail={
                "error": "transcription_unavailable",
                "detail": "faster-whisper is not installed on the server. "
                "Install it with: pip install 'amperstand-core[transcribe]', then restart amperstand-server.",
            },
        )
    ext = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    if ext not in AUDIO_EXTENSIONS:
        raise HTTPException(
            status_code=415,
            detail={"error": "unsupported_media_format", "detail": f"use one of: {', '.join(sorted(AUDIO_EXTENSIONS))}"},
        )
    limit = max_upload_bytes()
    declared = request.headers.get("content-length", "")
    if declared.isdigit() and int(declared) > limit:
        raise HTTPException(status_code=413, detail={"error": "file_too_large", "detail": f"the limit is {limit // 2**20} MB"})

    job_id = f"{int(time.time()):010d}-{secrets.token_hex(4)}"
    job_dir = jobs_root() / job_id
    job_dir.mkdir(parents=True)
    audio_file = f"audio.{ext}"
    size = 0
    try:
        with open(job_dir / audio_file, "wb") as f:
            async for chunk in request.stream():
                size += len(chunk)
                if size > limit:
                    raise HTTPException(
                        status_code=413, detail={"error": "file_too_large", "detail": f"the limit is {limit // 2**20} MB"}
                    )
                f.write(chunk)
        if not size:
            raise HTTPException(status_code=400, detail={"error": "empty_body", "detail": "the request body is empty"})
        try:
            duration = await asyncio.to_thread(transcription.probe_duration, job_dir / audio_file)
        except ValueError as exc:
            raise HTTPException(status_code=415, detail={"error": "unreadable_media", "detail": str(exc)})
    except BaseException:
        shutil.rmtree(job_dir, ignore_errors=True)
        raise

    file_name = (original_name or "").strip() or filename
    stem = file_name.rsplit(".", 1)[0].strip()
    job = {
        "id": job_id,
        "status": QUEUED,
        "created": time.time(),
        "file_name": file_name,
        "audio_file": audio_file,
        "bytes": size,
        "duration": duration,
        "title": (title or "").strip() or stem or file_name,
        "source": source or None,
        "tags": list(dict.fromkeys(["transcript", *[t.strip() for t in tags if t.strip()]])),
        "type": media_type,
        "language": language,
    }
    _write(job_dir, job)
    log.info("transcribe: job %s queued (%s, %d bytes, %.0f s)", job_id, filename, size, duration)
    runner.start()
    runner.wake()
    return _status(job_dir)


@router.get("/jobs/{job_id}")
def get_job(job_id: str) -> dict[str, Any]:
    """A job's status. `done_seconds` (while running) counts audio transcribed so far."""
    job_dir = jobs_root() / job_id
    if not _JOB_ID.match(job_id) or not (job_dir / "job.json").is_file():
        raise HTTPException(status_code=404, detail="transcription job not found")
    return _status(job_dir)
