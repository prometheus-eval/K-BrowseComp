#!/usr/bin/env bash
#
# launch_codex.sh — Codex-proposer problem generation, REVERSE seed order.
#
# Counterpart to launch.sh. Uses run_codex.py (OpenAI Codex CLI as the proposer)
# and walks the interleaved seed pool BACK-TO-FRONT (--reverse). Paired with a
# forward launch.sh (Claude) run, the two cover opposite ends of the same 1298-
# seed sequence and only overlap once both pass the midpoint (~seed 649 each).
#
# Outputs are Codex-specific and never collide with the Claude run:
#   work_codex/ , accepted_problems_codex.jsonl , rejected_problems_codex.jsonl
#
# Portable: paths derive from this script's own location, so a coauthor can run
# it from their own checkout. They need, on their machine:
#   - the Codex CLI installed + authenticated (`codex login` -> ~/.codex/auth.json)
#   - `uv` on PATH
#   - repo-root api_keys/ files (api_key_perplexity.txt, litellm.txt, base_url.txt, gemini.txt)
#     so _load_env can route the solver/grader through the LiteLLM proxy
#
# Usage:
#   ./launch_codex.sh                 # resume the reverse run
#   ./launch_codex.sh --fresh         # archive work_codex/ + *_codex.jsonl, then run
#   ./launch_codex.sh --max-seeds 2   # smoke test (extra args pass through to run_codex.py)
#
# Multi-day job — run under tmux, and keep the machine awake:
#   tmux new -s datagen_codex
#   caffeinate -is ./launch_codex.sh        # macOS; on Linux just ./launch_codex.sh
#
set -euo pipefail

PG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"   # = .../problem_generation
REPO="$(cd "$PG/../.." && pwd)"

# --- preflight: Codex CLI + uv must be on PATH ---
command -v codex >/dev/null || { echo "ERROR: 'codex' CLI not on PATH — install it and run 'codex login'." >&2; exit 1; }
command -v uv    >/dev/null || { echo "ERROR: 'uv' not on PATH." >&2; exit 1; }

cd "$REPO"

# --- optional fresh start: archive Codex work + aggregates (reversible mv) ---
FRESH=0
if [[ "${1:-}" == "--fresh" ]]; then FRESH=1; shift; fi
if [[ "$FRESH" == "1" ]]; then
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  if [[ -d "$PG/work_codex" ]]; then
    mv "$PG/work_codex" "$PG/work_codex.bak.$ts"
    echo "[launch_codex] archived work_codex/ -> work_codex.bak.$ts"
  fi
  for f in accepted_problems_codex.jsonl rejected_problems_codex.jsonl; do
    if [[ -f "$PG/$f" ]]; then
      mv "$PG/$f" "$PG/$f.bak.$ts"
      echo "[launch_codex] archived $f -> $f.bak.$ts"
    fi
  done
fi

# --- launch: Codex proposer, REVERSE interleaved order, sequential, resume-safe ---
mkdir -p "$PG/logs"
log="$PG/logs/run_codex_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "[launch_codex] logging to $log"
echo "[launch_codex] starting run_codex.py --litellm --resume --reverse $* (Codex proposer, reverse order)"

uv run python "$PG/run_codex.py" --litellm --resume --reverse "$@" 2>&1 | tee -a "$log"
