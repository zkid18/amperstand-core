"""Server-side transcription engine: windowed decoding, timestamp offsets, transcript Markdown."""

from __future__ import annotations

import json
import math
import struct
import wave
from pathlib import Path
from types import SimpleNamespace

import pytest

from amperstand_core import transcription
from amperstand_core.transcription import Segment, build_body

av = pytest.importorskip("av")


def make_wav(path: Path, seconds: float, rate: int = 8000, channels: int = 2) -> Path:
    """A tone at an odd rate and channel count, so decoding has to resample and downmix."""
    with wave.open(str(path), "wb") as w:
        w.setnchannels(channels)
        w.setsampwidth(2)
        w.setframerate(rate)
        frames = bytearray()
        for i in range(int(seconds * rate)):
            v = int(8000 * math.sin(2 * math.pi * 440 * i / rate))
            frames += struct.pack("<h", v) * channels
        w.writeframes(bytes(frames))
    return path


def test_iter_windows_yields_contiguous_bounded_16k_mono_chunks(tmp_path: Path):
    wav = make_wav(tmp_path / "tone.wav", 25)
    windows = list(transcription.iter_windows(wav, seconds=10))
    expected_start = 0.0
    for start, samples in windows[:-1]:
        assert start == pytest.approx(expected_start)
        assert 5 * 16_000 <= len(samples) <= 10 * 16_000  # cut within the last half of the window
        expected_start = start + len(samples) / transcription.SAMPLE_RATE
    start, samples = windows[-1]
    assert start == pytest.approx(expected_start)
    expected_start += len(samples) / transcription.SAMPLE_RATE
    assert expected_start == pytest.approx(25, abs=0.05)
    assert windows[0][1].dtype.name == "float32" and float(abs(windows[0][1]).max()) <= 1.0


def test_windows_end_in_the_quietest_moment_near_the_limit():
    import numpy as np

    rate = transcription.SAMPLE_RATE
    t = np.arange(12 * rate)
    loud = (8000 * np.sin(2 * np.pi * 440 * t / rate)).astype(np.int16)
    loud[int(7.2 * rate) : int(7.5 * rate)] = 0  # a pause between words
    cut = transcription.quiet_cut(loud, 10 * rate, 5 * rate)
    assert 7.2 * rate <= cut <= 7.5 * rate


def test_probe_duration(tmp_path: Path):
    assert transcription.probe_duration(make_wav(tmp_path / "t.wav", 3)) == pytest.approx(3, abs=0.05)
    junk = tmp_path / "junk.m4a"
    junk.write_bytes(b"not audio at all")
    with pytest.raises(ValueError):
        transcription.probe_duration(junk)


class FakeModel:
    """Two segments per window; reports Russian when asked to detect."""

    def __init__(self):
        self.languages: list[str | None] = []

    def transcribe(self, audio, *, language, vad_filter, beam_size, best_of):
        assert beam_size == 1 and best_of == 1  # memory: see transcription.transcribe
        self.languages.append(language)
        segs = [SimpleNamespace(start=1.0, end=2.0, text=" Раз. "), SimpleNamespace(start=3.0, end=4.0, text="  ")]
        return iter(segs), SimpleNamespace(language="ru")


def test_transcribe_offsets_windows_and_keeps_the_first_detected_language(tmp_path: Path, monkeypatch):
    monkeypatch.setattr(transcription, "WINDOW_SECONDS", 10)
    wav = make_wav(tmp_path / "t.wav", 25)
    starts = [start for start, _ in transcription.iter_windows(wav)]
    model = FakeModel()
    seen: list[float] = []
    segments, language = transcription.transcribe(wav, model=model, on_progress=seen.append)
    assert language == "ru"
    assert model.languages == [None] + ["ru"] * (len(starts) - 1)  # detected once, then fixed
    assert segments == [Segment(s + 1.0, s + 2.0, "Раз.") for s in starts]
    assert seen[-1] == pytest.approx(25, abs=0.05)


def test_build_body_sections_and_paragraphs():
    segments = [
        Segment(310.0, 315.0, "Five minutes in."),
        Segment(0.0, 4.0, "Hello."),
        Segment(70.0, 74.0, "A minute later."),
        Segment(605.0, 609.0, "Part two."),
    ]
    assert build_body(segments, file_name="talk.mp4", duration=900, language="en") == (
        "**File**: talk.mp4\n"
        "**Duration**: 0:15:00\n"
        "**Language**: en\n"
        "\n## Transcript\n"
        "\n### 0:00:00\n\nHello.\n\nA minute later.\n"
        "\n### 0:05:00\n\nFive minutes in.\n"
        "\n### 0:10:00\n\nPart two.\n"
    )


def test_build_body_without_speech():
    assert "_No speech was recognised._" in build_body([], file_name="x.m4a", duration=3, language=None)


def test_run_job_writes_result(tmp_path: Path, monkeypatch):
    (tmp_path / "audio.wav").write_bytes(b"")
    (tmp_path / "job.json").write_text(json.dumps({"audio_file": "audio.wav", "language": "ru"}))
    calls = {}

    def fake_transcribe(path, *, language, on_progress):
        calls.update(path=path, language=language)
        on_progress(12.0)
        return [Segment(0.5, 1.5, "Привет.")], "ru"

    monkeypatch.setattr(transcription, "transcribe", fake_transcribe)
    monkeypatch.setattr(transcription, "PROGRESS_EVERY_S", 0)
    transcription.run_job(tmp_path)
    assert calls == {"path": tmp_path / "audio.wav", "language": "ru"}
    assert json.loads((tmp_path / "progress.json").read_text()) == {"done_seconds": 12.0}
    result = json.loads((tmp_path / "result.json").read_text())
    assert result["segments"] == [{"start": 0.5, "end": 1.5, "text": "Привет."}] and result["language"] == "ru"


def test_worker_reruns_itself_with_malloc_tuning(monkeypatch):
    for k in transcription.MALLOC_ENV:
        monkeypatch.delenv(k, raising=False)
    monkeypatch.setattr(transcription.sys, "platform", "linux")
    monkeypatch.setattr(transcription.sys, "orig_argv", ["python", "-m", "amperstand_core.transcription", "/jobs/x"])
    calls = []

    def fake_execve(path, args, env):
        calls.append((args, env))
        raise SystemExit(0)

    monkeypatch.setattr(transcription.os, "execve", fake_execve)
    with pytest.raises(SystemExit):
        transcription.main(["/jobs/x"])
    (args, env), = calls
    assert args[1:] == ["-m", "amperstand_core.transcription", "/jobs/x"]
    assert {k: env[k] for k in transcription.MALLOC_ENV} == transcription.MALLOC_ENV


def test_worker_with_tuning_set_does_not_rerun(monkeypatch, tmp_path):
    for k, v in transcription.MALLOC_ENV.items():
        monkeypatch.setenv(k, v)
    monkeypatch.setattr(transcription.sys, "platform", "linux")
    monkeypatch.setattr(transcription.os, "execve", lambda *a: pytest.fail("re-exec with tuning already set"))
    monkeypatch.setattr(transcription.os, "nice", lambda n: 0)
    ran = []
    monkeypatch.setattr(transcription, "run_job", lambda d: ran.append(d))
    assert transcription.main([str(tmp_path)]) == 0 and ran == [tmp_path]
