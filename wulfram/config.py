"""Environment/config helpers for the Wulfram server facade."""

from __future__ import annotations

import os
from pathlib import Path
from typing import MutableMapping


FALSE_VALUES = {"0", "false", "off", "no"}
TRUE_VALUES = {"1", "true", "on", "yes"}


def load_env_file(path: Path | None = None, *, overwrite: bool = False) -> None:
    """Load the server .env file written by helper scripts."""
    env_path = path or Path(__file__).resolve().parent.parent / ".env"
    if not env_path.exists():
        return
    for line in env_path.read_text().splitlines():
        line = line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, val = line.partition("=")
        key = key.strip()
        if overwrite:
            os.environ[key] = val.strip()
        else:
            os.environ.setdefault(key, val.strip())


def env_flag(name: str, default: bool = False, env: MutableMapping[str, str] | None = None) -> bool:
    env = env or os.environ
    raw = env.get(name)
    if raw is None:
        return bool(default)
    value = raw.strip().lower()
    if value in TRUE_VALUES:
        return True
    if value in FALSE_VALUES:
        return False
    return bool(default)


def env_int(name: str, default: int, env: MutableMapping[str, str] | None = None, *, base: int = 10) -> int:
    env = env or os.environ
    try:
        return int(env.get(name, str(default)), base)
    except (TypeError, ValueError):
        return int(default)


def env_float(name: str, default: float, env: MutableMapping[str, str] | None = None) -> float:
    env = env or os.environ
    try:
        return float(env.get(name, str(default)))
    except (TypeError, ValueError):
        return float(default)


def configure_core_server(server: object, host: str | None, port: int) -> None:
    """Set core bind/entity/axis fields from environment without side effects."""
    load_env_file()
    server.host = host or os.environ.get("WULFRAM_BIND_ADDR", "0.0.0.0")
    server.public_addr = os.environ.get("WULFRAM_PUBLIC_ADDR", server.host)
    server.port = env_int("WULFRAM_PORT", int(port))
    server.next_entity_id = env_int("WULFRAM_START_ENTITY_ID", 1337)
    if server.next_entity_id <= 0:
        server.next_entity_id = 1337
    server.up_axis = os.environ.get("WULFRAM_UP_AXIS", "z").lower()
    if server.up_axis not in ("y", "z"):
        server.up_axis = "z"

