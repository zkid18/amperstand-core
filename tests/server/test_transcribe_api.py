"""Transcription jobs over HTTP: upload, queue, worker process, saved transcript."""

from __future__ import annotations

import io
import json
import math
import struct
import sys
import time
import wave
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from amperstand_core import transcription
from amperstand_core.server import transcribe_api
from amperstand_core.server.app import app
from amperstand_core.server.vault_api.store_factory import reset_store_cache

pytest.importorskip("av")

HEAD = {"Authorization": "Bearer devkey"}

# Stands in for `python -m amperstand_core.transcription JOB_DIR`.
FAKE_WORKER = """
import json, pathlib, sys
d = pathlib.Path(sys.argv[1])
job = json.loads((d / "job.json").read_text())
if job["title"] == "boom":
    print("loading model", file=sys.stderr)
    print("RuntimeError: model exploded", file=sys.stderr)
    sys.exit(1)
(d / "result.json").write_text(json.dumps({"language": "ru", "model": "fake", "segments": [
    {"start": 1.0, "end": 2.0, "text": "Привет."},
    {"start": 301.0, "end": 303.0, "text": "Дальше."},
]}))
"""


def wav_bytes(seconds: float = 2.0, rate: int = 16000) -> bytes:
    buf = io.BytesIO()
    with wave.open(buf, "wb") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(rate)
        w.writeframes(b"".join(struct.pack("<h", int(4000 * math.sin(i / 10))) for i in range(int(seconds * rate))))
    return buf.getvalue()


@pytest.fixture
def client(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> TestClient:
    monkeypatch.setenv("AMPERSTAND_API_KEY", "devkey")
    monkeypatch.setenv("AMPERSTAND_DATA_DIR", str(tmp_path / "vault"))
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    monkeypatch.setattr(transcription, "available", lambda: True)
    worker = tmp_path / "fake_worker.py"
    worker.write_text(FAKE_WORKER)
    monkeypatch.setattr(transcribe_api, "WORKER_CMD", [sys.executable, str(worker)])
    reset_store_cache()
    with TestClient(app) as c:
        yield c
    reset_store_cache()


def submit(client: TestClient, body: bytes | None = None, **params):
    params.setdefault("filename", "Weekly sync.m4a")
    return client.post("/transcribe/jobs", params=params, content=wav_bytes() if body is None else body, headers=HEAD)


def wait_for(client: TestClient, job_id: str, status: str, timeout: float = 15.0) -> dict:
    deadline = time.time() + timeout
    while time.time() < deadline:
        job = client.get(f"/transcribe/jobs/{job_id}", headers=HEAD).json()
        if job["status"] == status:
            return job
        time.sleep(0.1)
    raise AssertionError(f"job {job_id} never reached {status}: {job}")


def test_upload_transcribe_and_save(client: TestClient, tmp_path: Path):
    r = submit(client, filename="talk.wav", title="Team sync", tags=["work", "work"], type="video",
               source="https://youtu.be/xyz", language="ru")
    assert r.status_code == 202, r.text
    job = r.json()
    assert job["status"] in {"queued", "running"} and job["duration"] == pytest.approx(2.0, abs=0.05)

    done = wait_for(client, job["id"], "done")
    doc = client.get(f"/vault/{done['doc_id']}", headers=HEAD).json()
    assert doc["title"] == "Team sync"
    assert doc["source"] == "https://youtu.be/xyz"
    assert doc["tags"] == ["transcript", "work"]
    assert "### 0:00:00\n\nПривет." in doc["body"] and "### 0:05:00\n\nДальше." in doc["body"]
    assert "**Language**: ru" in doc["body"] and "**File**: talk.wav" in doc["body"]
    # Only the job record is kept once the transcript is saved.
    job_dir = tmp_path / "vault" / ".store" / "transcribe" / job["id"]
    assert sorted(p.name for p in job_dir.iterdir()) == ["job.json"]


def test_title_defaults_to_the_file_name(client: TestClient):
    job = submit(client, filename="Weekly sync.wav").json()
    done = wait_for(client, job["id"], "done")
    assert client.get(f"/vault/{done['doc_id']}", headers=HEAD).json()["title"] == "Weekly sync"


def test_original_name_labels_extracted_audio(client: TestClient):
    job = submit(client, filename="audio.wav", original_name="Board meeting.mov").json()
    doc_id = wait_for(client, job["id"], "done")["doc_id"]
    doc = client.get(f"/vault/{doc_id}", headers=HEAD).json()
    assert doc["title"] == "Board meeting" and "**File**: Board meeting.mov" in doc["body"]


def test_worker_failure_is_reported_and_keeps_the_audio(client: TestClient, tmp_path: Path):
    job = submit(client, filename="x.wav", title="boom").json()
    failed = wait_for(client, job["id"], "failed")
    assert "model exploded" in failed["error"] and failed["doc_id"] is None
    assert (tmp_path / "vault" / ".store" / "transcribe" / job["id"] / "audio.wav").exists()


def test_rejects_formats_it_cannot_read(client: TestClient, tmp_path: Path):
    assert submit(client, filename="notes.txt").status_code == 415
    r = submit(client, body=b"definitely not audio", filename="clip.m4a")
    assert r.status_code == 415 and r.json()["detail"]["error"] == "unreadable_media"
    assert list((tmp_path / "vault" / ".store" / "transcribe").iterdir()) == []


def test_rejects_uploads_over_the_limit(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AMPERSTAND_TRANSCRIBE_MAX_MB", "1")
    r = submit(client, body=b"\0" * (1024 * 1024 + 1), filename="big.wav")
    assert r.status_code == 413 and r.json()["detail"]["error"] == "file_too_large"


def test_without_faster_whisper_says_how_to_install(client: TestClient, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setattr(transcription, "available", lambda: False)
    r = submit(client)
    assert r.status_code == 503 and "amperstand-core[transcribe]" in r.json()["detail"]["detail"]


def test_requires_api_key(client: TestClient):
    assert client.post("/transcribe/jobs", params={"filename": "a.wav"}, content=wav_bytes()).status_code == 401
    assert client.get("/transcribe/jobs/1700000000-deadbeef").status_code == 401


def test_unknown_job_is_404(client: TestClient):
    assert client.get("/transcribe/jobs/1700000000-deadbeef", headers=HEAD).status_code == 404
    assert client.get("/transcribe/jobs/..%2F..%2Fsecrets", headers=HEAD).status_code == 404


def test_a_job_interrupted_by_a_restart_runs_again(tmp_path: Path, monkeypatch: pytest.MonkeyPatch):
    monkeypatch.setenv("AMPERSTAND_DATA_DIR", str(tmp_path / "vault"))
    job_dir = tmp_path / "vault" / ".store" / "transcribe" / "1700000000-deadbeef"
    job_dir.mkdir(parents=True)
    (job_dir / "job.json").write_text(json.dumps({"id": job_dir.name, "status": "running", "started": 1.0}))
    runner = transcribe_api.Runner()
    monkeypatch.setattr(runner, "_loop", lambda: None)
    runner.start()
    assert json.loads((job_dir / "job.json").read_text())["status"] == "queued"
