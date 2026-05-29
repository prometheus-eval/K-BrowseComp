#!/usr/bin/env bash
#
# launch.sh — sequential adversarial K-BrowseComp problem generation.
#
# Runs problem_generation/run.py over ALL seeds in
# seed_expansion/{intra,inter}_domain_candidates.json, ONE Claude Code
# proposer session at a time (no --parallel), routed through the LiteLLM
# proxy (--litellm) so it uses the configured azure_ai + gemini target models.
#
# Usage:
#   ./launch.sh                 # resume: skip seeds with a valid final.json
#   ./launch.sh --fresh         # archive work/ + aggregates, then run all seeds
#   ./launch.sh --max-seeds 2   # smoke test (pass-through args go to run.py)
#   ./launch.sh --fresh --max-seeds 2
#
# Long job (up to ~1h/seed). Run it inside tmux/nohup so it survives logout:
#   tmux new -s datagen
#   ./launch.sh --fresh
#
set -euo pipefail

PG="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(cd "$PG/../.." && pwd)"

# --- preflight: tools the pipeline shells out to must be on PATH ---
command -v claude >/dev/null || { echo "ERROR: 'claude' not on PATH" >&2; exit 1; }
command -v uv     >/dev/null || { echo "ERROR: 'uv' not on PATH"     >&2; exit 1; }

cd "$REPO"

# --- optional fresh start: move work/ + aggregates aside (reversible) ---
FRESH=0
if [[ "${1:-}" == "--fresh" ]]; then FRESH=1; shift; fi
if [[ "$FRESH" == "1" ]]; then
  ts="$(date -u +%Y%m%dT%H%M%SZ)"
  if [[ -d "$PG/work" ]]; then
    mv "$PG/work" "$PG/work.bak.$ts"
    echo "[launch] archived work/ -> work.bak.$ts"
  fi
  for f in accepted_problems.jsonl rejected_problems.jsonl; do
    if [[ -f "$PG/$f" ]]; then
      mv "$PG/$f" "$PG/$f.bak.$ts"
      echo "[launch] archived $f -> $f.bak.$ts"
    fi
  done
fi

# --- launch (sequential: NO --parallel; --resume makes re-runs safe) ---
mkdir -p "$PG/logs"
log="$PG/logs/run_$(date -u +%Y%m%dT%H%M%SZ).log"
echo "[launch] logging to $log"
echo "[launch] starting run.py --litellm --resume $* (one session at a time)"

uv run python "$PG/run.py" --litellm --resume "$@" 2>&1 | tee -a "$log"
