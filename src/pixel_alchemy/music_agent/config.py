"""Config for MusicAgent — replaces App/config.py with tau-friendly settings."""

from __future__ import annotations

import json
import os
from pathlib import Path

PKG_DIR = Path(__file__).parent
PROJECT_ROOT = PKG_DIR.parent.parent.parent  # repo root

DEFAULT_MODEL = "qwen3.6-35b-a3b-mtp"
DEFAULT_BASE_URL = "http://localhost:1234/v1"
DEFAULT_TAU_API_KEY = "lm-studio"  # LMStudio default


def get_tau_config() -> dict:
    """Resolve model/base_url/api_key from env or defaults."""
    return {
        "model": os.environ.get("MUSICAGENT_MODEL", os.environ.get("TAU_MODEL", DEFAULT_MODEL)),
        "base_url": os.environ.get("MUSICAGENT_BASE_URL", os.environ.get("OPENAI_BASE_URL", os.environ.get("TAU_BASE_URL", DEFAULT_BASE_URL))),
        "api_key": os.environ.get("MUSICAGENT_API_KEY", os.environ.get("OPENAI_API_KEY", os.environ.get("TAU_API_KEY", DEFAULT_TAU_API_KEY))),
        "timeout_seconds": float(os.environ.get("MUSICAGENT_TIMEOUT", "120")),
    }


def load_json_config(name: str) -> dict:
    p = PKG_DIR / "config" / name
    if p.exists():
        return json.loads(p.read_text(encoding="utf-8"))
    return {}
