"""Speech-to-text on the server with faster-whisper, for uploaded audio and video.

A transcription runs in its own process, started by the server's job runner:

    python -m amperstand_core.transcription JOB_DIR

so the model's memory is returned when the job ends and a crash can't take
the API down with it. The worker lowers its own CPU priority; searches and
captures stay responsive while it runs.

Audio is decoded and transcribed two minutes at a time. Whisper builds a
spectrogram of everything it is given in one go, so memory grows with the
input; two-minute windows keep that part small. Each window ends at the
quietest moment of its last few seconds, so a cut rarely lands in the middle
of a word.

Needs the `transcribe` extra: `pip install 'amperstand-core[transcribe]'`.

Environment:
    AMPERSTAND_TRANSCRIBE_MODEL    faster-whisper model name or path (default: small)
    AMPERSTAND_TRANSCRIBE_THREADS  CPU threads for the model (default: all cores)
"""

from __future__ import annotations

import importlib.util
import json
import os
import sys
import time
from collections.abc import Callable, Iterator
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

SAMPLE_RATE = 16_000
# Longest stretch of audio handed to the model per call; bounds memory.
WINDOW_SECONDS = 120
# Each window ends at the quietest 100 ms within its last this-many seconds.
CUT_SEARCH_SECONDS = 10
# One transcript section per this many seconds of audio. Search indexes and
# embeds per section, so one heading over a two-hour recording would leave
# most of it out of semantic search.
SECTION_SECONDS = 300
# Within a section, start a new paragraph about once a minute.
PARAGRAPH_SECONDS = 60
DEFAULT_MODEL = "small"
# How often the worker rewrites progress.json.
PROGRESS_EVERY_S = 10.0


@dataclass(frozen=True)
class Segment:
    start: float  # seconds from the start of the recording
    end: float
    text: str


def available() -> bool:
    """True when faster-whisper is installed (the `transcribe` extra)."""
    return importlib.util.find_spec("faster_whisper") is not None


def model_name() -> str:
    return os.environ.get("AMPERSTAND_TRANSCRIBE_MODEL", "").strip() or DEFAULT_MODEL


def cpu_threads() -> int:
    raw = os.environ.get("AMPERSTAND_TRANSCRIBE_THREADS", "").strip()
    try:
        return max(1, int(raw)) if raw else (os.cpu_count() or 1)
    except ValueError:
        return os.cpu_count() or 1


def probe_duration(path: Path) -> float:
    """Length in seconds. Raises ValueError when the file has no readable audio."""
    import av

    try:
        with av.open(str(path)) as container:
            if not container.streams.audio:
                raise ValueError("the file has no audio track")
            if container.duration is not None:
                return container.duration / av.time_base
            stream = container.streams.audio[0]
            if stream.duration is not None and stream.time_base is not None:
                return float(stream.duration * stream.time_base)
    except av.error.FFmpegError as exc:
        raise ValueError(f"can't read the audio: {exc}") from exc
    raise ValueError("can't tell how long the audio is")


def quiet_cut(samples: Any, size: int, search: int) -> int:
    """Index at which to end a window of at most `size` samples: the middle of
    the quietest 100 ms frame within the last `search` samples."""
    import numpy as np

    frame = SAMPLE_RATE // 10
    lo = max(0, size - search)
    tail = samples[lo:size].astype(np.float32)
    n = tail.size // frame
    if n < 1:
        return size
    energy = (tail[: n * frame].reshape(n, frame) ** 2).mean(axis=1)
    return lo + int(energy.argmin()) * frame + frame // 2


def iter_windows(path: Path, seconds: int | None = None) -> Iterator[tuple[float, Any]]:
    """Yield (start_seconds, float32 16 kHz mono samples), up to WINDOW_SECONDS at a time.

    Windows are contiguous: each starts exactly where the previous one ended.
    """
    import av
    import numpy as np

    size = (seconds or WINDOW_SECONDS) * SAMPLE_RATE
    search = min(CUT_SEARCH_SECONDS * SAMPLE_RATE, size // 2)
    pending: list[Any] = []
    held = 0
    consumed = 0  # samples already yielded
    with av.open(str(path)) as container:
        stream = container.streams.audio[0]
        resampler = av.AudioResampler(format="s16", layout="mono", rate=SAMPLE_RATE)

        def frames():
            for frame in container.decode(stream):
                yield from resampler.resample(frame)
            yield from resampler.resample(None)  # flush

        for frame in frames():
            chunk = frame.to_ndarray().reshape(-1)
            pending.append(chunk)
            held += chunk.size
            while held >= size:
                buf = np.concatenate(pending)
                cut = quiet_cut(buf, size, search)
                yield consumed / SAMPLE_RATE, buf[:cut].astype(np.float32) / 32768.0
                consumed += cut
                rest = buf[cut:]
                pending, held = ([rest] if rest.size else []), rest.size
    if held:
        yield consumed / SAMPLE_RATE, np.concatenate(pending).astype(np.float32) / 32768.0


def transcribe(
    path: Path,
    *,
    language: str | None = None,
    on_progress: Callable[[float], None] | None = None,
    model: Any = None,
) -> tuple[list[Segment], str | None]:
    """Transcribe a whole recording. Returns (segments, language code).

    Without `language`, the first window's detected language is kept for the
    rest, so a recording doesn't flip languages on a quiet stretch.
    """
    if model is None:
        from faster_whisper import WhisperModel

        model = WhisperModel(model_name(), device="cpu", compute_type="int8", cpu_threads=cpu_threads())
    segments: list[Segment] = []
    for start, audio in iter_windows(path):
        # best_of=1: when a stretch decodes too repetitively, Whisper retries at
        # higher temperatures, sampling best_of candidates side by side. Five
        # (the default) took `small` from ~600 MB to 1.1 GB on one window.
        parts, info = model.transcribe(audio, language=language, vad_filter=True, beam_size=1, best_of=1)
        for s in parts:
            text = s.text.strip()
            if text:
                segments.append(Segment(start + s.start, start + s.end, text))
            if on_progress:
                on_progress(start + s.end)
        language = language or info.language
        if on_progress:
            on_progress(start + len(audio) / SAMPLE_RATE)
    return segments, language


def hms(seconds: float) -> str:
    s = int(seconds)
    return f"{s // 3600}:{s % 3600 // 60:02d}:{s % 60:02d}"


def build_body(segments: list[Segment], *, file_name: str, duration: float, language: str | None) -> str:
    """Transcript Markdown: a short header, then a timestamp heading every few minutes."""
    lines = [f"**File**: {file_name}", f"**Duration**: {hms(duration)}"]
    if language:
        lines.append(f"**Language**: {language}")
    lines += ["", "## Transcript"]
    section = -1
    para: list[str] = []
    para_start = 0.0

    def flush() -> None:
        if para:
            lines.extend(["", " ".join(para)])
            para.clear()

    for seg in sorted(segments, key=lambda s: s.start):
        if int(seg.start // SECTION_SECONDS) != section:
            flush()
            section = int(seg.start // SECTION_SECONDS)
            lines += ["", f"### {hms(section * SECTION_SECONDS)}"]
            para_start = seg.start
        elif seg.start - para_start >= PARAGRAPH_SECONDS:
            flush()
            para_start = seg.start
        para.append(seg.text)
    flush()
    if not segments:
        lines += ["", "_No speech was recognised._"]
    return "\n".join(lines) + "\n"


# ── worker process ────────────────────────────────────────────────────

# glibc's default malloc keeps freed spectrogram and decoder buffers on the
# heap; on a 1-vCPU / 2 GB server a long recording takes `small` past 900 MB
# that way. Returning allocations of 1 MB and up straight to the OS keeps it
# near 450 MB. glibc reads these only when a process starts, so the worker
# re-runs itself with them set when its launcher didn't (see main()).
MALLOC_ENV = {
    "MALLOC_ARENA_MAX": "2",
    "MALLOC_MMAP_THRESHOLD_": str(1 << 20),
    "MALLOC_TRIM_THRESHOLD_": str(1 << 20),
}


def _write_json(path: Path, data: dict) -> None:
    tmp = path.with_suffix(".tmp")
    tmp.write_text(json.dumps(data, ensure_ascii=False))
    tmp.replace(path)


def run_job(job_dir: Path) -> None:
    """Transcribe job_dir/audio.* into job_dir/result.json, updating progress.json as it goes."""
    job = json.loads((job_dir / "job.json").read_text())
    audio = job_dir / job["audio_file"]
    last = 0.0

    def progress(done: float) -> None:
        nonlocal last
        now = time.monotonic()
        if now - last >= PROGRESS_EVERY_S:
            last = now
            _write_json(job_dir / "progress.json", {"done_seconds": round(done, 1)})

    segments, language = transcribe(audio, language=job.get("language"), on_progress=progress)
    _write_json(
        job_dir / "result.json",
        {"language": language, "model": model_name(), "segments": [asdict(s) for s in segments]},
    )


def main(argv: list[str] | None = None) -> int:
    args = argv if argv is not None else sys.argv[1:]
    if len(args) != 1:
        print("usage: python -m amperstand_core.transcription JOB_DIR", file=sys.stderr)
        return 2
    if sys.platform.startswith("linux") and any(os.environ.get(k) != v for k, v in MALLOC_ENV.items()):
        os.execve(sys.executable, [sys.executable, *sys.orig_argv[1:]], {**os.environ, **MALLOC_ENV})
    try:
        os.nice(10)  # leave the CPU to the API while a long job runs
    except (AttributeError, OSError):
        pass
    try:
        # If memory runs out, the kernel kills this worker before the API server.
        Path("/proc/self/oom_score_adj").write_text("1000")
    except OSError:
        pass
    run_job(Path(args[0]))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
