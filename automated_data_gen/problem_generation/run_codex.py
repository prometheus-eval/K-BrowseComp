"""
run_codex.py — same orchestration as run.py, but uses OpenAI's Codex CLI
(`codex exec --json`) as the proposer instead of Anthropic's Claude Code.

Differences vs run.py:

- Binary:        `codex` (set with CODEX_BIN env var)
- Headless mode: `codex exec --json --dangerously-bypass-approvals-and-sandbox`
- Model:         CODEX_MODEL env var (default `gpt-5-codex`)
- No effort arg: Codex exec does not expose a reasoning-effort flag
- No session-id: Codex assigns its own thread IDs; no UUID arg needed
- Auth:          OPENAI_API_KEY / CODEX_API_KEY env var, OR ~/.codex/auth.json
- Workspaces:    `work_codex/` (sibling to run.py's `work/`)
- Outputs:       `accepted_problems_codex.jsonl` / `rejected_problems_codex.jsonl`
- Event schema:  `thread.started` / `turn.started` / `turn.completed` /
                 `item.started` / `item.completed` / `error`
                 (vs Claude Code's `system` / `assistant` / `user` / `result`)

Everything else — seed loading, lineage, the 3-gate brief, workspace
restoration, the FM-quota logic, per-seed final.json enrichment — is
inherited from run.py via monkey-patching the Claude-specific entry points.

Usage (mirrors run.py):
  python3 run_codex.py --litellm --section 1 --parallel 4
  python3 run_codex.py --max-seeds 1 --source-classes species_inventory --litellm
  python3 run_codex.py --aggregate-only-ish: no equivalent — outputs accumulate
    on the per-seed final.json files, then merged at the end as usual.

References:
  https://developers.openai.com/codex/noninteractive
  https://developers.openai.com/codex/cli/reference
"""
from __future__ import annotations

import json
import os
import subprocess
import sys
import threading
from pathlib import Path
from typing import Any

# Import the shared orchestrator + ensure credential env vars are loaded.
# `run` triggers `_load_env` on import, which populates LITELLM/OPENAI/etc.
import run

# ---------------------------------------------------------------------------
# Codex-specific constants.
# ---------------------------------------------------------------------------

CODEX_BIN = os.environ.get("CODEX_BIN", "codex")
CODEX_TIMEOUT = int(os.environ.get("CODEX_TIMEOUT", 3600))  # 1 hour per seed
CODEX_MODEL = os.environ.get("CODEX_MODEL", "gpt-5-codex")

# Per-CLI workspace + aggregate output files so Codex and Claude runs don't
# collide on the same seed_id.
run.WORK_DIR = run.ROOT / "work_codex"
run.OUT_ACCEPTED = run.ROOT / "accepted_problems_codex.jsonl"
run.OUT_REJECTED = run.ROOT / "rejected_problems_codex.jsonl"

# Banner overrides so `[run]` lines name Codex correctly.
run.PROPOSER_CLI_NAME = "Codex"
run.CLAUDE_BIN = CODEX_BIN
run.REQUIRED_CLAUDE_VERSION = "(unpinned)"
run.MODEL = CODEX_MODEL
run.EFFORT = "n/a"
run.CLAUDE_TIMEOUT = CODEX_TIMEOUT

# Extend the run-wide rate-limit patterns with OpenAI / Codex-specific
# phrases so a quota event in either CLI aborts the entire orchestration.
for _pat in (
    "insufficient_quota",
    "You exceeded your current quota",
    "Rate limit reached",
    "rate_limit_exceeded",
):
    if _pat not in run.RATE_LIMIT_PATTERNS:
        run.RATE_LIMIT_PATTERNS.append(_pat)


# ---------------------------------------------------------------------------
# Codex auth + version check (replacing the Claude-specific versions).
# ---------------------------------------------------------------------------


def check_auth() -> None:
    """Codex CLI auth: env var (OPENAI_API_KEY or CODEX_API_KEY) OR
    ~/.codex/auth.json (OAuth via `codex login`)."""
    if os.environ.get("OPENAI_API_KEY") or os.environ.get("CODEX_API_KEY"):
        return
    if (Path.home() / ".codex" / "auth.json").exists():
        return
    sys.exit(
        "[run] Codex CLI is not authenticated. Either:\n"
        "  - set OPENAI_API_KEY (or CODEX_API_KEY) in env, or\n"
        "  - run `codex login` once to authenticate via OAuth\n"
        "    (creates ~/.codex/auth.json)."
    )


def check_codex_version() -> None:
    """Just verify the codex binary is reachable. Codex has no fixed version
    requirement — we print whatever is installed."""
    result = subprocess.run(
        [CODEX_BIN, "--version"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        sys.exit(f"[run] `{CODEX_BIN} --version` failed: {result.stderr.strip()}")
    print(f"[run] Codex version: {result.stdout.strip()}", flush=True)


# ---------------------------------------------------------------------------
# Stream-JSON parser for `codex exec --json`. Event shapes documented at
# https://developers.openai.com/codex/noninteractive.
# ---------------------------------------------------------------------------


def _codex_format_event(evt: dict[str, Any], prefix: str = "") -> str | None:
    """Render one `codex exec --json` event into a single human-readable line.

    Codex stream events (per docs):
      {type: "thread.started", thread_id: "..."}
      {type: "turn.started"}
      {type: "turn.completed", usage: {input_tokens, output_tokens, ...}}
      {type: "item.started",   item: {id, type, ...payload...}}
      {type: "item.completed", item: {id, type, ...payload...}}
      {type: "error",          message: "..."}

    Item `type` values include (per docs): agent_message, reasoning,
    command_execution, file_change, mcp_tool_call, web_search, plan_update.
    Field names inside `item` are not fully documented, so we look up the
    most likely keys (text/command/query/path/name/arguments/output) and
    fall back to a compact JSON dump.
    """
    t = evt.get("type", "")

    if t == "thread.started":
        tid = (evt.get("thread_id") or "?")[:8]
        return f"{prefix}[thread] id={tid}"

    if t == "turn.started":
        return None  # too noisy; ignore

    if t == "turn.completed":
        usage = evt.get("usage", {}) or {}
        return (
            f"{prefix}[turn done] "
            f"in={usage.get('input_tokens', '?')}/"
            f"out={usage.get('output_tokens', '?')}/"
            f"reasoning={usage.get('reasoning_output_tokens', '?')}"
        )

    if t == "error":
        return f"{prefix}[ERROR] {evt.get('message', '')}"

    if t in ("item.started", "item.completed"):
        item = evt.get("item", {}) or {}
        it = item.get("type", "?")
        completed = (t == "item.completed")

        # Agent text — show only on completion.
        if it == "agent_message":
            if not completed:
                return None
            txt = (item.get("text") or "").strip()
            return f"{prefix}{run._short(txt, 400)}" if txt else None

        # Reasoning — too long, skip for the human log (but it's still
        # in proposer_stream.jsonl).
        if it == "reasoning":
            return None

        # Shell command execution — show command on start, result on complete.
        if it in ("command_execution", "shell_command", "exec_command"):
            if not completed:
                cmd = item.get("command") or item.get("script") or "?"
                if isinstance(cmd, list):
                    cmd = " ".join(str(c) for c in cmd)
                return f"{prefix}→ shell: {run._short(str(cmd))}"
            status = item.get("status", "?")
            marker = "✓" if status in ("completed", "success", "ok") else "✗"
            output = item.get("output") or item.get("stdout") or item.get("result") or ""
            return f"{prefix}  {marker} {run._short(str(output).strip())}"

        # Web search.
        if it == "web_search":
            if not completed:
                q = item.get("query") or item.get("q") or "?"
                return f"{prefix}→ web_search: {run._short(str(q))}"
            results = item.get("results") or []
            n = len(results) if isinstance(results, list) else "?"
            return f"{prefix}  ✓ {n} results"

        # File edits.
        if it in ("file_change", "file_edit", "patch"):
            if not completed:
                path = item.get("path") or item.get("file") or "?"
                kind = item.get("change_type") or item.get("kind") or "edit"
                return f"{prefix}→ {kind}: {path}"
            return f"{prefix}  ✓ applied"

        # MCP tool calls.
        if it == "mcp_tool_call":
            if not completed:
                name = item.get("name") or "?"
                args_blob = run._short(json.dumps(item.get("arguments", {})), 160)
                return f"{prefix}→ mcp:{name} {args_blob}"
            result = item.get("result", "done")
            return f"{prefix}  ✓ {run._short(str(result))}"

        # Plan updates from the planner.
        if it == "plan_update":
            if not completed:
                return None
            text = item.get("text") or item.get("plan") or ""
            return f"{prefix}[plan] {run._short(str(text), 240)}"

        # Fallback for any item types we don't recognize.
        verb = "→" if not completed else "  ✓"
        return f"{prefix}{verb} {it}: {run._short(json.dumps(item), 200)}"

    return None  # Unknown top-level event — silently skip


# ---------------------------------------------------------------------------
# Codex subprocess runner — mirrors run_claude's workspace-restore logic.
# ---------------------------------------------------------------------------


def run_codex(args: list[str], cwd: Path, label: str, timeout: int = CODEX_TIMEOUT) -> int:
    """Stream Codex events to stdout AND persist them to two files in cwd.

    Mirrors run.run_claude: same defensive workspace restoration on the rare
    Bash-sandbox-wipe event, same proposer_stream.jsonl + proposer_log.txt
    artifact layout (so monitor.py can inspect Codex runs the same way it
    inspects Claude runs).
    """
    prefix = f"[{label}] " if label else ""
    seed_path = cwd / "seed.json"
    spec_path = cwd / "task_spec.md"
    stream_path = cwd / "proposer_stream.jsonl"
    log_path = cwd / "proposer_log.txt"
    seed_content = seed_path.read_text(encoding="utf-8") if seed_path.exists() else ""
    spec_content = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""

    def _restore_workspace_if_wiped() -> bool:
        if cwd.exists() and stream_path.exists() and log_path.exists():
            return False
        cwd.mkdir(parents=True, exist_ok=True)
        if seed_content and not seed_path.exists():
            seed_path.write_text(seed_content, encoding="utf-8")
        if spec_content and not spec_path.exists():
            spec_path.write_text(spec_content, encoding="utf-8")
        return True

    proc = subprocess.Popen(
        [CODEX_BIN] + args,
        cwd=str(cwd),
        stdin=subprocess.DEVNULL,
        stdout=subprocess.PIPE,
        start_new_session=True,
        text=True,
        bufsize=1,
    )
    timed_out = threading.Event()

    def _on_timeout() -> None:
        timed_out.set()
        run._kill_process_group(proc.pid)

    timer = threading.Timer(timeout, _on_timeout)
    timer.start()

    fs = stream_path.open("a", encoding="utf-8")
    fl = log_path.open("a", encoding="utf-8")
    events_since_check = 0
    try:
        for line in proc.stdout:  # type: ignore[union-attr]
            line = line.rstrip()
            if not line:
                continue
            events_since_check += 1
            if events_since_check >= 20:
                events_since_check = 0
                if _restore_workspace_if_wiped():
                    try:
                        fs.close()
                        fl.close()
                    except Exception:
                        pass
                    fs = stream_path.open("a", encoding="utf-8")
                    fl = log_path.open("a", encoding="utf-8")
                    marker = (
                        f"{prefix}[run] WARNING: workspace dir was wiped; "
                        f"restored seed.json + task_spec.md + reopened logs."
                    )
                    print(marker, flush=True)
                    fl.write(marker + "\n")
                    fl.flush()
            try:
                fs.write(line + "\n")
                fs.flush()
            except Exception:
                pass
            try:
                pretty = _codex_format_event(json.loads(line), prefix=prefix)
            except json.JSONDecodeError:
                pretty = prefix + line
            if pretty:
                print(pretty, flush=True)
                try:
                    fl.write(pretty + "\n")
                    fl.flush()
                except Exception:
                    pass
            if run._check_rate_limit(line, prefix, fl):
                run._kill_process_group(proc.pid)
                break
        proc.wait()
        if _restore_workspace_if_wiped():
            try:
                fs.close()
                fl.close()
            except Exception:
                pass
            fs = stream_path.open("a", encoding="utf-8")
            fl = log_path.open("a", encoding="utf-8")
            fl.write(f"{prefix}[run] WARNING: final workspace restore after proposer exit.\n")
            fl.flush()
        if timed_out.is_set():
            msg = f"{prefix}[run] Timeout after {timeout}s; killed process group."
            print(msg, flush=True)
            try:
                fl.write(msg + "\n")
                fl.flush()
            except Exception:
                pass
        return proc.returncode
    finally:
        try:
            fs.close()
            fl.close()
        except Exception:
            pass
        timer.cancel()
        run._kill_process_group(proc.pid)


def run_proposer_codex(ws: Path, label: str) -> int:
    """Codex equivalent of run.run_proposer.

    `codex exec` takes the prompt as a positional argument. There's no
    session-id or effort flag in non-interactive mode; the model is the
    only knob, plus the sandbox bypass."""
    prompt = (
        "Read task_spec.md and drive the full adversarial loop yourself "
        "(draft → run solver_runner.py via Bash → analyze trajectory → "
        "evolve → up to 4 attempts). Write ./final.json before exiting."
    )
    codex_args = [
        "exec",
        "--json",
        "--dangerously-bypass-approvals-and-sandbox",
        "--model", CODEX_MODEL,
        prompt,
    ]
    print(f"[{label}] proposer start run_id={ws.name} model={CODEX_MODEL}", flush=True)
    rc = run_codex(codex_args, cwd=ws, label=label)
    print(f"[{label}] proposer done rc={rc}", flush=True)
    return rc


# ---------------------------------------------------------------------------
# Wire the overrides into the `run` module so its `process_seed` and
# orchestrator pick up Codex behavior via Python's late-binding name lookup.
# ---------------------------------------------------------------------------

run.check_auth = check_auth                # OAuth + env key check
run.check_claude_version = check_codex_version
run._format_event = _codex_format_event    # used by run_claude internally; we
                                           # also override run_claude below so
                                           # this is belt-and-suspenders.
run.run_claude = run_codex                 # in case anything still calls it
run.run_proposer = run_proposer_codex      # the actual call site in process_seed


if __name__ == "__main__":
    run.main()
