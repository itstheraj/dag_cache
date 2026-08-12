"""Global configuration and promotion/demotion policy."""

from __future__ import annotations

import os
from dataclasses import dataclass


@dataclass
class Config:
    db_path: str = ".dagcache/store.db"
    enabled: bool = True
    force_record: bool = False  # always run live + record (still stores DAGs)
    replay_mode: str = "verified"  # "verified" | "frozen"
    auto_replay: bool = True  # replay staging DAGs, not only approved ones
    fallback_demote_threshold: int = 3  # staging DAGs die after this many fallbacks
    default_ttl_seconds: int | None = None


_CFG: Config | None = None

_VALID_REPLAY_MODES = ("verified", "frozen")


def get_config() -> Config:
    global _CFG
    if _CFG is None:
        mode = os.environ.get("DAGCACHE_MODE", "").lower()
        _CFG = Config(
            db_path=os.environ.get("DAGCACHE_DB", Config.db_path),
            enabled=mode != "off",
            force_record=mode == "record",
            replay_mode=os.environ.get("DAGCACHE_REPLAY", "verified"),
            auto_replay=os.environ.get("DAGCACHE_AUTO_REPLAY", "1") not in ("0", "false"),
        )
    return _CFG


def configure(**kwargs) -> Config:
    """Update global config, e.g. ``configure(db_path=":memory:")``."""
    cfg = get_config()
    for key, value in kwargs.items():
        if not hasattr(cfg, key):
            raise TypeError(f"unknown dagcache setting {key!r}")
        if key == "replay_mode" and value not in _VALID_REPLAY_MODES:
            raise ValueError(f"replay_mode must be one of {_VALID_REPLAY_MODES}")
        setattr(cfg, key, value)
    return cfg


def reset_config() -> None:
    """Restore defaults (mainly for tests)."""
    global _CFG
    _CFG = None
