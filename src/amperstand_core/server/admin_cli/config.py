"""Resolve admin CLI config from env vars and an optional env-file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path

DEFAULT_ENV_FILE = Path("/etc/amperstand/env")
DEFAULT_DATA_DIR = Path.home() / ".amperstand" / "vault"


@dataclass(frozen=True)
class AdminConfig:
    data_dir: Path
    api_key_present: bool
    env_file: Path | None
    env_file_exists: bool
    openai_api_key: str | None = None
    # Merged view of the env-file and the process environment (process
    # wins), so provider settings (AMPERSTAND_EMBED_*, AMPERSTAND_RERANK_*,
    # vendor keys) reach the embedding factory the same way they reach the
    # server through systemd's EnvironmentFile=.
    env: dict[str, str] = field(default_factory=dict)


def load_env_file(path: Path) -> dict[str, str]:
    """Parse a KEY=VALUE env file. Comments and blanks ignored. Quotes stripped."""
    if not path.exists():
        return {}
    out: dict[str, str] = {}
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#"):
            continue
        if "=" not in line:
            continue
        key, value = line.split("=", 1)
        value = value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in {'"', "'"}:
            value = value[1:-1]
        out[key.strip()] = value
    return out


def resolve_config(env_file: Path | None = None) -> AdminConfig:
    """Resolve config preferring process env, then env-file, then defaults."""
    env_path = env_file
    file_env = load_env_file(env_path) if env_path else {}
    merged = {**file_env, **{k: v for k, v in os.environ.items()}}

    raw_dir = os.environ.get("AMPERSTAND_DATA_DIR") or file_env.get("AMPERSTAND_DATA_DIR")
    data_dir = Path(raw_dir).expanduser() if raw_dir else DEFAULT_DATA_DIR

    api_key = os.environ.get("AMPERSTAND_API_KEY") or file_env.get("AMPERSTAND_API_KEY")
    openai_key = (
        os.environ.get("OPENAI_API_KEY") or file_env.get("OPENAI_API_KEY") or None
    )

    return AdminConfig(
        data_dir=data_dir,
        api_key_present=bool(api_key),
        env_file=env_path,
        env_file_exists=bool(env_path and env_path.exists()),
        openai_api_key=openai_key,
        env=merged,
    )
