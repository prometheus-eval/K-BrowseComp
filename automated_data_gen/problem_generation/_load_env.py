"""Load API credentials from repo-root `api_keys/` files into os.environ.

Reads (if present, in repo_root/api_keys/):
- api_key_perplexity.txt → PERPLEXITY_API_KEY
- litellm.txt            → LITELLM_PROXY_API_KEY  + OPENAI_API_KEY (mirror for OpenAI SDK)
- base_url.txt           → LITELLM_PROXY_BASE_URL + OPENAI_BASE_URL (mirror)
- gemini.txt             → GEMINI_API_KEY + GOOGLE_API_KEY

The mirror to OPENAI_* lets the existing non-litellm code path (which uses
AsyncOpenAI) route through the same LiteLLM proxy via the OpenAI-compatible
chat-completions endpoint. The dedicated LITELLM_PROXY_* vars are what the
new --litellm code path uses explicitly.

If an env var is already set, it is NOT overwritten — explicit shell exports
always win. Backward-compat: also falls back to repo_root/<file>.txt if
api_keys/ is empty (so legacy layouts still work).
"""
from __future__ import annotations

import os
from pathlib import Path

_REPO_ROOT = Path(__file__).resolve().parents[2]
_API_KEYS = _REPO_ROOT / "api_keys"


def _first_existing(*paths: Path) -> Path | None:
    for p in paths:
        if p.exists():
            return p
    return None


# (filename, [env var(s) to set with that value])
_FILE_TO_VARS: list[tuple[str, list[str]]] = [
    ("api_key_perplexity.txt", ["PERPLEXITY_API_KEY"]),
    ("litellm.txt", ["LITELLM_PROXY_API_KEY", "OPENAI_API_KEY"]),
    ("base_url.txt", ["LITELLM_PROXY_BASE_URL", "OPENAI_BASE_URL"]),
    ("gemini.txt", ["GEMINI_API_KEY", "GOOGLE_API_KEY"]),
]


def load_env() -> dict[str, str]:
    """Populate os.environ from credential files. Returns {var: source_path}
    for what was actually applied."""
    applied: dict[str, str] = {}
    for filename, env_vars in _FILE_TO_VARS:
        path = _first_existing(_API_KEYS / filename, _REPO_ROOT / filename)
        if path is None:
            continue
        value = path.read_text(encoding="utf-8").strip()
        if not value:
            continue
        for var in env_vars:
            if os.environ.get(var):
                continue  # explicit env wins
            os.environ[var] = value
            applied[var] = str(path)
    return applied


# Apply on import.
load_env()
