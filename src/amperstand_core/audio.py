"""Whisper transcription for captured video.

Used by the LinkedIn extractor when a post carries a video: the media file
is downloaded and sent to OpenAI's Whisper API, which returns plain text
for roughly $0.006 per minute of audio. Gated behind
AMPERSTAND_AUDIO_TRANSCRIPTION so it can't surprise-spend; the older
AMPERSTAND_YOUTUBE_AUDIO_FALLBACK name is honoured for existing env files.
YouTube transcripts are caption tracks via Anysite and never pass through
here.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path

import httpx

logger = logging.getLogger(__name__)

# 25 MB — Whisper API hard limit.
WHISPER_MAX_BYTES = 25 * 1024 * 1024

_ENABLE_ENVS = ("AMPERSTAND_AUDIO_TRANSCRIPTION", "AMPERSTAND_YOUTUBE_AUDIO_FALLBACK")


class AudioError(RuntimeError):
    pass


def audio_transcription_enabled() -> bool:
    """Off unless one of the enabling env vars is explicitly truthy."""
    for env in _ENABLE_ENVS:
        if os.environ.get(env, "").strip().lower() in {"1", "true", "yes", "on"}:
            return True
    return False


def download_media(url: str, dest: Path, *, max_bytes: int = WHISPER_MAX_BYTES) -> int:
    """Stream a media file to `dest`, giving up as soon as it passes `max_bytes`.

    Returns the size written. Stops early rather than pulling a whole
    long video only to find Whisper won't accept it.
    """
    written = 0
    try:
        with httpx.stream("GET", url, follow_redirects=True, timeout=120) as resp:
            resp.raise_for_status()
            with open(dest, "wb") as f:
                for chunk in resp.iter_bytes(chunk_size=65_536):
                    written += len(chunk)
                    if written > max_bytes:
                        raise AudioError(
                            f"media exceeds {max_bytes // 1024 // 1024}MB Whisper limit"
                        )
                    f.write(chunk)
    except httpx.HTTPError as exc:
        raise AudioError(f"media download failed: {exc}") from exc
    return written


def transcribe_audio_file(path: Path, *, api_key: str | None = None, model: str = "whisper-1") -> str:
    """Transcribe an audio/video file via OpenAI Whisper. Returns plain text.

    Whisper auto-detects language and accepts mp4/m4a/webm/mp3 directly,
    so no local conversion is needed.
    """
    if not path.exists():
        raise AudioError(f"audio file not found: {path}")
    size = path.stat().st_size
    if size > WHISPER_MAX_BYTES:
        raise AudioError(
            f"audio file is {size} bytes; Whisper max is {WHISPER_MAX_BYTES}"
        )
    api_key = api_key or os.environ.get("OPENAI_API_KEY", "").strip()
    if not api_key:
        raise AudioError("OPENAI_API_KEY not set; cannot transcribe via Whisper")

    try:
        from openai import OpenAI
    except ImportError as exc:
        raise AudioError("openai package not installed") from exc

    client = OpenAI(api_key=api_key)
    with open(path, "rb") as f:
        resp = client.audio.transcriptions.create(model=model, file=f)
    return resp.text
