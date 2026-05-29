"""
problem_generation/run.py — Per-seed adversarial K-BrowseComp problem generation.

For each seed in seed_expansion's intra/inter aggregated JSONs, spawn ONE
headless Claude Code subprocess. The proposer (inside that subprocess) reads
its task_spec.md, drafts a problem, invokes solver_runner.py via Bash to
test it against the K-BrowseComp deep-research solver, analyzes the
trajectory, and iterates up to 4 attempts. The orchestrator just spawns and
collects the resulting final.json.

Per-seed workspace:
  work/<source_class>/<sha1(url)>/<run_id>/
    seed.json                cached seed entry
    task_spec.md             rendered brief
    attempt_1/
      problem.json           proposer's draft
      solver_convo.json      solver_runner output
      solver_response.txt
      grader_result.json
      summary.json
    attempt_2/...
    attempt_3/...
    attempt_4/...
    final.json               written by the proposer at the end

Top-level aggregates (rewritten after each invocation):
  accepted_problems.jsonl    one JSON object per accepted seed
  rejected_problems.jsonl    one JSON object per rejected seed

Usage:
  python3 run.py --max-seeds 1
  python3 run.py --seed-json /tmp/local_seed.json --dry-run
  python3 run.py --classes species_inventory --max-seeds 3
  python3 run.py --seed-id 1a2b3c4d --dry-run
"""
from __future__ import annotations

import argparse
import concurrent.futures as futures
import datetime as _dt
import hashlib
import json
import os
import re
import signal
import subprocess
import sys
import threading
import time
import uuid
from pathlib import Path
from typing import Any

CLAUDE_BIN = os.environ.get("CLAUDE_BIN", "claude")
CLAUDE_TIMEOUT = int(os.environ.get("CLAUDE_TIMEOUT", 3600))  # 1 hour per seed
REQUIRED_CLAUDE_VERSION = os.environ.get("CLAUDE_VERSION", "2.1.138")
MODEL = os.environ.get("CLAUDE_MODEL", "claude-opus-4-7")
EFFORT = os.environ.get("CLAUDE_EFFORT", "max")
PROPOSER_CLI_NAME = "Claude"  # banner label; run_codex.py overrides to "Codex"
MAX_ATTEMPTS = 4

# Substring patterns indicating the proposer CLI has hit a usage cap / rate
# limit. When any pattern matches a line of proposer output, we set
# _RATE_LIMIT_HIT and abort the entire orchestration so we don't burn through
# remaining seeds with empty rejects. run_codex.py extends this list with
# OpenAI-specific phrases.
RATE_LIMIT_PATTERNS: list[str] = [
    # Anthropic Claude Code usage caps. Kept apostrophe-agnostic (no leading
    # "You're"/"You've") so they match straight quotes, curly quotes, and the
    # JSON ’ escape form alike, in both raw stream-json and rendered lines.
    "out of extra usage",     # "You're out of extra usage" — extra-usage cap
    "hit your weekly limit",  # "You've hit your weekly limit · resets ..." — weekly cap
]
_RATE_LIMIT_HIT = threading.Event()

# Outcomes are appended to the JSONLs incrementally (one line per finished
# seed) instead of batched at the end, so a Ctrl-C / crash never loses
# completed work. _WRITE_LOCK guards the append when --parallel > 1.
SKIP_REASONS = {"skipped_resume", "skipped_rate_limit", "dry_run"}
_WRITE_LOCK = threading.Lock()

ROOT = Path(__file__).resolve().parent
PARENT = ROOT.parent
REPO_ROOT = PARENT.parent
SEED_EXPANSION = PARENT / "seed_expansion"
SEED_MATERIAL = PARENT / "seed_material"
WORK_DIR = ROOT / "work"
TEMPLATE = ROOT / "task_spec_template.md"
SOLVER_RUNNER = ROOT / "solver_runner.py"
PERPLEXITY_CHECK = ROOT / "perplexity_check.py"
OUT_ACCEPTED = ROOT / "accepted_problems.jsonl"
OUT_REJECTED = ROOT / "rejected_problems.jsonl"

# Populate API credentials from repo-root txt files.
sys.path.insert(0, str(ROOT))
import _load_env  # noqa: F401  side-effect: populates os.environ

os.environ.setdefault("IS_SANDBOX", "1")


FAILURE_MODE_NAMES = {
    "F1": "첫 검색어를 좁힐 수 없음 / 사후 검증형 문제 — starting-entity ambiguity",
    "F3": "비인접 도메인 hopping — cross-domain chain retrieval",
    "F4": "semi-structured parsing 실패 — extracting from tables / notice fields / sub-sections",
    "F6": "희소 엔티티 정규화 실패 — rare-entity normalization (Korean variant names, transliteration)",
    "F7": "조건 누적 / constraint tracking 실패 — accumulating multiple constraints",
    "F8": "중간 계산 / 절차형 reasoning 실패 — intermediate computation / multi-step arithmetic",
    "F9": "검색 결과 선택 실패 — choosing the right SERP result among many",
    "F10": "iframe / 동적 페이지 / 특정 페이지 진입 실패 — dynamic / login-walled / iframe content",
}


# ---------------------------------------------------------------------------
# Preflight.
# ---------------------------------------------------------------------------


def check_auth() -> None:
    """Same soft check as seed_expansion/run.py — tolerant of macOS Keychain auth."""
    if os.environ.get("ANTHROPIC_API_KEY"):
        return
    if (Path.home() / ".claude" / ".credentials.json").exists():
        return
    if sys.platform == "darwin" and (Path.home() / ".claude").exists():
        print(
            "[run] Warning: no ANTHROPIC_API_KEY and no ~/.claude/.credentials.json, "
            "but ~/.claude/ exists. Assuming macOS Keychain OAuth.",
            flush=True,
        )
        return
    sys.exit(
        "[run] Claude Code is not authenticated. Either:\n"
        "  - set ANTHROPIC_API_KEY=<your-api-key>, or\n"
        "  - run `claude login` once to authenticate via OAuth."
    )


def check_claude_version() -> None:
    result = subprocess.run(
        [CLAUDE_BIN, "--version"], capture_output=True, text=True, check=False
    )
    if result.returncode != 0:
        sys.exit(f"[run] `{CLAUDE_BIN} --version` failed: {result.stderr.strip()}")
    m = re.search(r"\d+\.\d+\.\d+", result.stdout)
    if not m:
        sys.exit(f"[run] Could not parse claude version: {result.stdout.strip()!r}")
    version = m.group(0)
    if version != REQUIRED_CLAUDE_VERSION:
        print(
            f"[run] Warning: claude {version} found, expected {REQUIRED_CLAUDE_VERSION}.",
            flush=True,
        )


def check_solver_prereqs(args: argparse.Namespace) -> None:
    """Sanity-check that solver_runner.py is reachable and the env has what it needs."""
    if not SOLVER_RUNNER.exists():
        sys.exit(f"[run] solver_runner.py not found at {SOLVER_RUNNER}")
    needed = []
    if args.litellm:
        if not os.environ.get("LITELLM_PROXY_BASE_URL") and not os.environ.get("OPENAI_BASE_URL"):
            needed.append("LITELLM_PROXY_BASE_URL")
        if not os.environ.get("LITELLM_PROXY_API_KEY") and not os.environ.get("OPENAI_API_KEY"):
            needed.append("LITELLM_PROXY_API_KEY")
    elif (args.solver_model.startswith("gpt") or args.grader_model.startswith("gpt")
          or any(m.startswith("gpt") for m in args.target_models_list)):
        if not os.environ.get("OPENAI_API_KEY"):
            needed.append("OPENAI_API_KEY")
    if (
        args.solver_model.startswith("gemini")
        or args.grader_model.startswith("gemini")
        or any(m.startswith("gemini") for m in args.target_models_list)
    ) and not (os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")):
        needed.append("GEMINI_API_KEY")
    if args.search_engine == "perplexity" and not os.environ.get("PERPLEXITY_API_KEY"):
        needed.append("PERPLEXITY_API_KEY")
    if args.search_engine == "brave" and not os.environ.get("BRAVE_API_KEY"):
        needed.append("BRAVE_API_KEY")
    if args.search_engine == "exa" and not os.environ.get("EXA_API_KEY"):
        needed.append("EXA_API_KEY")
    if args.search_engine == "tavily" and not os.environ.get("TAVILY_API_KEY"):
        needed.append("TAVILY_API_KEY")
    if needed:
        print(
            f"[run] Warning: missing env vars: {', '.join(needed)}. "
            "The solver subprocess will fail. Add to shell env or to the "
            "repo-root credential files (api_key_perplexity.txt, litellm.txt, base_url.txt, gemini.txt).",
            flush=True,
        )


# ---------------------------------------------------------------------------
# Subprocess plumbing (mirrors seed_expansion/run.py).
# ---------------------------------------------------------------------------


def _kill_process_group(pid: int) -> None:
    try:
        pgid = os.getpgid(pid)
        os.killpg(pgid, signal.SIGTERM)
        time.sleep(2)
        os.killpg(pgid, signal.SIGKILL)
    except (ProcessLookupError, PermissionError):
        pass


def _short(s: str, n: int = 200) -> str:
    s = s.replace("\n", " | ")
    return s if len(s) <= n else s[:n] + "..."


def _tool_summary(name: str, inp: dict[str, Any]) -> str:
    if name == "Bash":
        return _short(inp.get("command", ""))
    if name in ("Read", "Edit", "Write", "NotebookEdit"):
        return inp.get("file_path", "?")
    if name in ("Grep", "Glob"):
        return f"pattern={inp.get('pattern', '?')}"
    if name in ("Task", "Agent"):
        return inp.get("description", "?")
    if name == "WebFetch":
        return inp.get("url", "?")
    if name == "WebSearch":
        return inp.get("query", "?")
    return _short(json.dumps(inp))


def _format_event(evt: dict[str, Any], prefix: str = "") -> str | None:
    t = evt.get("type")
    if t == "system":
        sub = evt.get("subtype", "system")
        if sub == "init":
            return f"{prefix}[init] session={evt.get('session_id', '?')[:8]}"
        return f"{prefix}[{sub}]"
    if t == "assistant":
        parts = []
        for block in evt.get("message", {}).get("content", []):
            bt = block.get("type")
            if bt == "text":
                txt = block.get("text", "").strip()
                if txt:
                    parts.append(f"{prefix}{txt}")
            elif bt == "tool_use":
                name = block.get("name", "?")
                parts.append(f"{prefix}→ {name}: {_tool_summary(name, block.get('input', {}))}")
        return "\n".join(parts) if parts else None
    if t == "user":
        parts = []
        for block in evt.get("message", {}).get("content", []):
            if block.get("type") == "tool_result":
                marker = "✗" if block.get("is_error") else "✓"
                content = block.get("content", "")
                if isinstance(content, list):
                    content = "".join(c.get("text", "") for c in content if isinstance(c, dict))
                parts.append(f"{prefix}  {marker} {_short(str(content).strip())}")
        return "\n".join(parts) if parts else None
    if t == "result":
        success = evt.get("subtype") == "success"
        cost = evt.get("total_cost_usd")
        cost_str = f" cost=${cost:.4f}" if cost else ""
        return f"{prefix}[result] success={success}{cost_str}"
    return None


def _check_rate_limit(line: str, prefix: str, fl) -> bool:
    """Scan one line of raw proposer output for a usage-cap indicator.

    On first match, set the module-level event so the orchestrator can
    bail. Returns True iff the caller should kill its proposer and break
    out of the stream loop.
    """
    for pat in RATE_LIMIT_PATTERNS:
        if pat not in line:
            continue
        first = not _RATE_LIMIT_HIT.is_set()
        _RATE_LIMIT_HIT.set()
        msg = (
            f"{prefix}[run] FATAL: proposer hit usage cap ({pat!r}). "
            f"Aborting; pending seeds will be skipped."
            if first else f"{prefix}[run] (rate-limit; aborting)"
        )
        print(msg, flush=True)
        try:
            fl.write(msg + "\n")
            fl.flush()
        except Exception:
            pass
        return True
    return False


def run_claude(args: list[str], cwd: Path, label: str, timeout: int = CLAUDE_TIMEOUT) -> int:
    """Stream proposer events to stdout AND persist them to two files in cwd.

    - `proposer_stream.jsonl`: raw stream-json, one event per line. Canonical
      record — replayable, machine-readable, includes everything claude emits
      (init, assistant text, tool_use, tool_result, result, etc.).
    - `proposer_log.txt`: human-readable rendering (same as stdout). Easy
      to `tail -f` while a run is live.

    Defensive workspace restoration: if the Claude Code sandbox wipes the
    workspace mid-run (observed on Bash subcommand failure), the four
    orchestrator-owned files (seed.json, task_spec.md, proposer_stream.jsonl,
    proposer_log.txt) silently lose their inodes. We snapshot seed.json and
    task_spec.md before launching, periodically check whether the workspace
    still exists, and re-create everything if it doesn't.
    """
    prefix = f"[{label}] " if label else ""
    seed_path = cwd / "seed.json"
    spec_path = cwd / "task_spec.md"
    stream_path = cwd / "proposer_stream.jsonl"
    log_path = cwd / "proposer_log.txt"
    seed_content = seed_path.read_text(encoding="utf-8") if seed_path.exists() else ""
    spec_content = spec_path.read_text(encoding="utf-8") if spec_path.exists() else ""

    def _restore_workspace_if_wiped() -> bool:
        """Return True if a restoration actually happened."""
        if cwd.exists() and stream_path.exists() and log_path.exists():
            return False
        cwd.mkdir(parents=True, exist_ok=True)
        if seed_content and not seed_path.exists():
            seed_path.write_text(seed_content, encoding="utf-8")
        if spec_content and not spec_path.exists():
            spec_path.write_text(spec_content, encoding="utf-8")
        return True

    proc = subprocess.Popen(
        [CLAUDE_BIN] + args,
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
        _kill_process_group(proc.pid)

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
            # Every ~20 events, check if the workspace got wiped under us.
            # POSIX: writes to a deleted inode succeed but produce nothing on
            # disk, so we'd silently lose logs without this check.
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
                    marker = f"{prefix}[run] WARNING: workspace dir was wiped; restored seed.json + task_spec.md + reopened logs."
                    print(marker, flush=True)
                    fl.write(marker + "\n")
                    fl.flush()
            # Always persist the raw line first — even if it's not valid JSON,
            # so we can debug malformed events.
            try:
                fs.write(line + "\n")
                fs.flush()
            except Exception:
                pass
            try:
                pretty = _format_event(json.loads(line), prefix=prefix)
            except json.JSONDecodeError:
                pretty = prefix + line
            if pretty:
                print(pretty, flush=True)
                try:
                    fl.write(pretty + "\n")
                    fl.flush()
                except Exception:
                    pass
            if _check_rate_limit(line, prefix, fl):
                _kill_process_group(proc.pid)
                break
        proc.wait()
        # Final restore pass so post-run inspection finds intact files.
        if _restore_workspace_if_wiped():
            try:
                fs.close(); fl.close()
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
        _kill_process_group(proc.pid)


# ---------------------------------------------------------------------------
# Seed loading + workspace setup.
# ---------------------------------------------------------------------------


def seed_id(seed: dict[str, Any]) -> str:
    return hashlib.sha1(seed["website_url"].encode("utf-8")).hexdigest()[:10]


def _load_original_index() -> dict[str, dict[str, Any]]:
    """Build seed.source URL → original-question metadata mapping.

    Joins seed_material/seed_url_index.json (URL ⇄ question_id) with
    seed_material/seed_questions.json (the 300 human-crafted problems)
    so each seed can carry its original-problem provenance into the final
    result JSON.

    Returns {url: {question_id, original_question, original_answer,
    all_question_ids}}. A URL may be referenced by multiple original
    questions; we store the first as the canonical scalar fields and the
    full list under all_question_ids.
    """
    if not SEED_MATERIAL.exists():
        return {}
    try:
        url_idx = json.loads((SEED_MATERIAL / "seed_url_index.json").read_text(encoding="utf-8"))
        questions = json.loads((SEED_MATERIAL / "seed_questions.json").read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[run] Warning: could not load original problem lineage: {e}", flush=True)
        return {}

    # Group url_idx entries by URL, preserving order so the first match wins.
    by_url: dict[str, list[dict[str, Any]]] = {}
    for entry in url_idx:
        url = entry.get("url")
        if not url:
            continue
        by_url.setdefault(url, []).append(entry)

    out: dict[str, dict[str, Any]] = {}
    for url, entries in by_url.items():
        all_qids: list[str] = []
        primary: dict[str, Any] | None = None
        for e in entries:
            qid = e.get("question_id", "")
            if not qid:
                continue
            try:
                idx = int(qid.split("_", 1)[1])
            except Exception:
                continue
            if idx < 0 or idx >= len(questions):
                continue
            if qid not in all_qids:
                all_qids.append(qid)
            if primary is None:
                q = questions[idx]
                primary = {
                    "original_question_id": qid,
                    "original_question": q.get("problem", ""),
                    "original_answer": q.get("answer", ""),
                }
        if primary is not None:
            out[url] = {**primary, "original_url": url, "all_question_ids": all_qids}
    return out


def _interleave_by_source_class(seeds: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Round-robin seeds across their source_class so processing cycles
    one-per-class instead of running a whole class back-to-back.

    Within-class order is preserved (so intra still precedes inter for a given
    class); classes appear in first-seen order. This avoids a long front-loaded
    run of a single class — e.g. the 20 consecutive academic_korean seeds at the
    head of the pool — so early results sample the full variety of the dataset.
    """
    groups: dict[str, list[dict[str, Any]]] = {}
    for s in seeds:
        groups.setdefault(s.get("source_class", "unknown"), []).append(s)
    queues = list(groups.values())
    out: list[dict[str, Any]] = []
    while any(queues):
        for q in queues:
            if q:
                out.append(q.pop(0))
    return out


def _read_seed_entries(path: Path) -> list[dict[str, Any]]:
    """Read local seed candidates from JSON list, {"seeds": [...]}, or JSONL."""
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    if path.suffix == ".jsonl":
        entries = [json.loads(line) for line in text.splitlines() if line.strip()]
    else:
        parsed = json.loads(text)
        entries = parsed.get("seeds", []) if isinstance(parsed, dict) else parsed
    if not isinstance(entries, list):
        raise ValueError("seed file must contain a JSON list, a {'seeds': [...]} object, or JSONL rows")
    for entry in entries:
        if not isinstance(entry, dict):
            raise ValueError("each seed entry must be a JSON object")
        if "website_url" not in entry:
            raise ValueError("each seed entry must include website_url")
    return entries


def load_seeds(args: argparse.Namespace) -> list[dict[str, Any]]:
    seeds: list[dict[str, Any]] = []
    original_index = _load_original_index()
    seed_sources: list[tuple[Path, str]] = []
    if args.seed_json:
        seed_sources.extend((Path(p).expanduser(), "custom") for p in args.seed_json.split(",") if p.strip())
    else:
        seed_sources.extend(
            (
                (SEED_EXPANSION / "intra_domain_candidates.json", "intra"),
                (SEED_EXPANSION / "inter_domain_candidates.json", "inter"),
            )
        )

    for p, axis in seed_sources:
        if not p.exists():
            print(f"[run] Warning: seed file not found: {p}", flush=True)
            continue
        try:
            entries = _read_seed_entries(p)
        except Exception as e:
            print(f"[run] Warning: cannot parse {p}: {e}", flush=True)
            continue
        for entry in entries:
            entry.setdefault("axis", axis)
            # Join in original-problem lineage if the seed's `source` URL
            # matches one of the 300 human-crafted problems' source URLs.
            src = entry.get("source")
            if src and src in original_index:
                entry.update(original_index[src])
            seeds.append(entry)

    # --section is for splitting work across people. It slices the canonical
    # seed pool (intra concatenated with inter, JSON-order within each) into
    # fixed 100-seed chunks. Applied BEFORE other filters so the section→seeds
    # mapping is deterministic across machines/runs.
    if args.section is not None:
        section_size = 100
        total = len(seeds)
        last_section = (total + section_size - 1) // section_size
        if args.section < 1 or args.section > last_section:
            sys.exit(
                f"[run] --section must be in [1, {last_section}] "
                f"(canonical pool size: {total} seeds); got {args.section}"
            )
        start = (args.section - 1) * section_size
        end = min(args.section * section_size, total)
        seeds = seeds[start:end]
        print(
            f"[run] Section {args.section}/{last_section}: "
            f"seeds [{start}, {end}) of {total}",
            flush=True,
        )

    if args.source_classes:
        allowed = {c.strip() for c in args.source_classes.split(",") if c.strip()}
        seeds = [s for s in seeds if s.get("source_class") in allowed]
    if args.axes:
        allowed_axes = {a.strip() for a in args.axes.split(",") if a.strip()}
        seeds = [s for s in seeds if s.get("axis") in allowed_axes]
    if args.target_failure_modes:
        wanted_fms = {f.strip() for f in args.target_failure_modes.split(",") if f.strip()}
        seeds = [
            s for s in seeds
            if wanted_fms & set(s.get("target_failure_modes", []) or [])
        ]
    if args.seed_id:
        seeds = [s for s in seeds if seed_id(s) == args.seed_id]
    # Reorder for variety BEFORE --max-seeds so a capped/smoke run also samples
    # across classes rather than taking the first N of one class.
    if getattr(args, "order", "interleave") == "interleave":
        seeds = _interleave_by_source_class(seeds)
    # --reverse walks the SAME ordered list back-to-front, so a forward run and
    # a --reverse run start from opposite ends of the identical sequence and
    # only overlap once both pass the midpoint.
    if getattr(args, "reverse", False):
        seeds = seeds[::-1]
    if args.max_seeds:
        seeds = seeds[: args.max_seeds]
    return seeds


def _failure_mode_glossary(target_fms: list[str]) -> str:
    # Top of the glossary lists the targeted FMs; full glossary follows.
    lines = []
    if target_fms:
        lines.append("**Targeted failure modes for this seed:**")
        for fm in target_fms:
            lines.append(f"- **{fm}** — {FAILURE_MODE_NAMES.get(fm, '')}")
        lines.append("")
    lines.append("**Full failure mode glossary:**")
    for fm, desc in FAILURE_MODE_NAMES.items():
        lines.append(f"- **{fm}** — {desc}")
    return "\n".join(lines)


def _model_slug(model: str) -> str:
    """Strip provider prefixes and any litellm_proxy/ wrapper so the output dir
    name stays human-readable (e.g., `gpt-5.4-mini`, `gemini-3-flash-preview`)."""
    m = model
    if m.startswith("litellm_proxy/"):
        m = m[len("litellm_proxy/"):]
    return m.rsplit("/", 1)[-1]


def _render_target_models_block(
    models: list[str],
    *,
    repo_root: str,
    solver_runner_path: str,
    litellm_flag: str,
    search_engine: str,
    grader_model: str,
) -> str:
    """Generate the per-model adversarial bash invocations for Gate 3.

    The proposer will substitute `attempt_N` with the actual attempt number
    (e.g. attempt_1, attempt_2). Each model gets its own output subdir
    `attempt_N/adversarial/<slug>/` so summary.json / solver_convo.json /
    grader_result.json don't collide across models.
    """
    blocks: list[str] = []
    for i, m in enumerate(models, start=1):
        slug = _model_slug(m)
        blocks.append(
            f"# === Target {i}/{len(models)}: {slug} ===\n"
            f"cd {repo_root} && uv run python {solver_runner_path}{litellm_flag} \\\n"
            f"  --mode adversarial \\\n"
            f"  --problem $(pwd)/attempt_N/problem.json \\\n"
            f"  --output-dir $(pwd)/attempt_N/adversarial/{slug} \\\n"
            f"  --search-engine {search_engine} \\\n"
            f"  --solver-model {m} \\\n"
            f"  --grader-model {grader_model}"
        )
    return "\n\n".join(blocks)


def render_brief(seed: dict[str, Any], args: argparse.Namespace) -> str:
    template = TEMPLATE.read_text(encoding="utf-8")
    target_fms = seed.get("target_failure_modes", []) or []
    # When --litellm is on, expand to " --litellm" (with leading space) so the
    # placeholder slots into the rendered bash command without an extra blank arg.
    litellm_flag = " --litellm" if args.litellm else ""
    target_models_block = _render_target_models_block(
        args.target_models_list,
        repo_root=str(REPO_ROOT),
        solver_runner_path=str(SOLVER_RUNNER),
        litellm_flag=litellm_flag,
        search_engine=args.search_engine,
        grader_model=args.grader_model,
    )
    target_models_summary = ", ".join(_model_slug(m) for m in args.target_models_list)
    return template.format(
        seed_url=seed["website_url"],
        source_class=seed.get("source_class", "unknown"),
        axis=seed.get("axis", ""),
        seed_summary=seed.get("website_summary", ""),
        seed_evidence=(seed.get("verification") or {}).get("evidence", ""),
        target_failure_modes=", ".join(target_fms) or "(none specified)",
        why_this_is_promising=seed.get("why_this_is_promising", ""),
        solver_runner_path=str(SOLVER_RUNNER),
        perplexity_check_path=str(PERPLEXITY_CHECK),
        repo_root=str(REPO_ROOT),
        search_engine=args.search_engine,
        solver_model=args.solver_model,
        grader_model=args.grader_model,
        max_attempts=MAX_ATTEMPTS,
        failure_mode_glossary=_failure_mode_glossary(target_fms),
        target_models_bash_block=target_models_block,
        target_models_summary=target_models_summary,
        litellm_flag=litellm_flag,
        litellm_mode_note=(
            "\n**LiteLLM proxy mode is ENABLED.** Pass `--litellm` to `solver_runner.py` "
            "as shown below; the runner will auto-prefix model names with `litellm_proxy/` "
            "and route through the LITELLM_PROXY_BASE_URL / LITELLM_PROXY_API_KEY env vars.\n"
            if args.litellm else ""
        ),
    )


def setup_seed_workspace(seed: dict[str, Any], run_id: str, args: argparse.Namespace) -> Path:
    cls = seed.get("source_class", "unknown")
    sid = seed_id(seed)
    ws = WORK_DIR / cls / sid / run_id
    ws.mkdir(parents=True, exist_ok=True)
    (ws / "seed.json").write_text(json.dumps(seed, indent=2, ensure_ascii=False), encoding="utf-8")
    (ws / "task_spec.md").write_text(render_brief(seed, args), encoding="utf-8")
    return ws


# ---------------------------------------------------------------------------
# Per-seed runner.
# ---------------------------------------------------------------------------


def seed_has_prior_final(seed: dict[str, Any]) -> bool:
    cls = seed.get("source_class", "unknown")
    sid = seed_id(seed)
    seed_dir = WORK_DIR / cls / sid
    if not seed_dir.is_dir():
        return False
    for run_dir in seed_dir.iterdir():
        if not run_dir.is_dir():
            continue
        p = run_dir / "final.json"
        if not p.exists():
            continue
        try:
            json.loads(p.read_text(encoding="utf-8"))
            return True
        except Exception:
            continue
    return False


def run_proposer(ws: Path, label: str) -> int:
    session_id = str(uuid.uuid4())
    claude_args = [
        "-p",
        (
            "Read task_spec.md and drive the full adversarial loop yourself "
            "(draft → run solver_runner.py via Bash → analyze trajectory → "
            "evolve → up to 4 attempts). Write ./final.json before exiting."
        ),
        "--output-format", "stream-json",
        "--verbose",
        "--model", MODEL,
        "--effort", EFFORT,
        "--session-id", session_id,
        "--dangerously-skip-permissions",
    ]
    print(f"[{label}] proposer start run_id={ws.name} session={session_id[:8]}", flush=True)
    rc = run_claude(claude_args, cwd=ws, label=label)
    print(f"[{label}] proposer done rc={rc}", flush=True)
    return rc


def _collect_proposed_iterations(ws: Path) -> tuple[dict[str, str], dict[str, str], str | None]:
    """Walk attempt_N/problem.json files and return per-iter question/answer + the
    most recent source_url the proposer landed on.

    attempt_1 → iter0, attempt_2 → iter1, etc. — matches the user-facing
    iter naming and lets the result JSON show how the proposer's draft
    evolved across re-attempts.
    """
    questions_by_iter: dict[str, str] = {}
    answers_by_iter: dict[str, str] = {}
    last_url: str | None = None
    for attempt_dir in sorted(ws.iterdir() if ws.is_dir() else []):
        if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt_"):
            continue
        try:
            n = int(attempt_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        prob_path = attempt_dir / "problem.json"
        if not prob_path.exists():
            continue
        try:
            p = json.loads(prob_path.read_text(encoding="utf-8"))
        except Exception:
            continue
        key = f"iter{n - 1}"
        questions_by_iter[key] = p.get("problem", "")
        answers_by_iter[key] = str(p.get("answer", ""))
        if p.get("source_url"):
            last_url = p["source_url"]
    return questions_by_iter, answers_by_iter, last_url


def _collect_per_model_adversarial(ws: Path) -> dict[str, dict[str, Any]]:
    """For each attempt_N, list the per-model adversarial summaries.

    Returns {"iterK": {"<model_slug>": <summary.json>, ...}}. Also tolerates
    the legacy flat layout where summary.json lived directly under
    attempt_N/adversarial/ (one model only)."""
    out: dict[str, dict[str, Any]] = {}
    if not ws.is_dir():
        return out
    for attempt_dir in sorted(ws.iterdir()):
        if not attempt_dir.is_dir() or not attempt_dir.name.startswith("attempt_"):
            continue
        try:
            n = int(attempt_dir.name.split("_", 1)[1])
        except ValueError:
            continue
        adv = attempt_dir / "adversarial"
        if not adv.is_dir():
            continue
        per_model: dict[str, Any] = {}
        # Legacy flat case: summary.json directly under adversarial/
        flat = adv / "summary.json"
        if flat.exists():
            try:
                per_model["_legacy"] = json.loads(flat.read_text(encoding="utf-8"))
            except Exception:
                pass
        for sub in sorted(adv.iterdir()):
            if not sub.is_dir():
                continue
            s = sub / "summary.json"
            if not s.exists():
                continue
            try:
                per_model[sub.name] = json.loads(s.read_text(encoding="utf-8"))
            except Exception:
                continue
        if per_model:
            out[f"iter{n - 1}"] = per_model
    return out


def _lineage_fields(seed: dict[str, Any], ws: Path) -> dict[str, Any]:
    """Provenance to attach to every result row (accepted or rejected).

    Layers, oldest to newest:
      original_*    - the human-crafted problem (1 of 300) whose source URL
                      seeded this workspace
      seed_*        - the URL/axis/class proposed by seed_expansion
      proposed_*    - what the proposer drafted during problem_generation,
                      with proposed_question keyed by iter0/iter1/iter2/iter3
      adversarial_results - per-iter, per-model {solved, grade, response} so
                      downstream consumers can verify ALL target models failed
    """
    q_by_iter, a_by_iter, last_url = _collect_proposed_iterations(ws)
    per_model = _collect_per_model_adversarial(ws)
    # Flatten to a tested_models list across the final iter (most recent).
    tested_models: list[str] = []
    last_iter_results: dict[str, Any] = {}
    if per_model:
        last_iter_key = sorted(per_model.keys())[-1]
        last_iter_results = per_model[last_iter_key]
        tested_models = [m for m in last_iter_results.keys() if m != "_legacy"]
    return {
        "original_url": seed.get("original_url"),
        "original_question": seed.get("original_question"),
        "original_answer": seed.get("original_answer"),
        "original_question_id": seed.get("original_question_id"),
        "all_question_ids": seed.get("all_question_ids") or [],
        "seed_url": seed.get("website_url"),
        "seed_axis": seed.get("axis"),
        "seed_source_class": seed.get("source_class"),
        "proposed_url": last_url or seed.get("website_url"),
        "proposed_question": q_by_iter,
        "proposed_answer": a_by_iter,
        "attempts_made": len(q_by_iter),
        "tested_models": tested_models,
        "adversarial_results_by_iter": per_model,
    }


def collect_outcome(ws: Path, seed: dict[str, Any]) -> dict[str, Any]:
    lineage = _lineage_fields(seed, ws)
    base: dict[str, Any] = {
        "seed_id": seed_id(seed),
        "workspace": str(ws),
        "source_class": seed.get("source_class"),
        **lineage,
    }
    final_path = ws / "final.json"
    if not final_path.exists():
        return {**base, "accepted": False, "reason": "no_final_json_written"}
    try:
        final = json.loads(final_path.read_text(encoding="utf-8"))
    except Exception as e:
        return {**base, "accepted": False, "reason": f"malformed_final_json: {e}"}
    # Merge proposer's final.json on top of base — proposer fields win for
    # accepted/rationale/gates/etc., but lineage from base stays authoritative.
    merged: dict[str, Any] = {**base, **final}
    for k, v in base.items():
        merged.setdefault(k, v)
    # Re-overwrite lineage fields so the proposer can't accidentally
    # clobber them by writing same-named keys in final.json.
    merged.update(lineage)
    merged["seed_id"] = base["seed_id"]
    merged["workspace"] = base["workspace"]
    # Persist the enriched record back into the workspace so each per-seed
    # final.json is self-contained (proposer fields + provenance + iter history).
    try:
        final_path.write_text(
            json.dumps(merged, indent=2, ensure_ascii=False), encoding="utf-8"
        )
    except Exception as e:
        print(f"[run] Warning: could not rewrite enriched final.json: {e}", flush=True)
    return merged


def process_seed(seed: dict[str, Any], run_id: str, args: argparse.Namespace) -> dict[str, Any]:
    cls = seed.get("source_class", "unknown")
    sid = seed_id(seed)
    label = f"{cls}:{sid}"

    if _RATE_LIMIT_HIT.is_set():
        return {"accepted": False, "seed_id": sid, "reason": "skipped_rate_limit"}

    if args.resume and seed_has_prior_final(seed):
        print(f"[{label}] skip (prior final.json exists)", flush=True)
        return {"accepted": False, "seed_id": sid, "reason": "skipped_resume"}

    ws = setup_seed_workspace(seed, run_id, args)
    if args.dry_run:
        print(f"[{label}] dry-run: brief written to {ws/'task_spec.md'}", flush=True)
        return {"accepted": False, "seed_id": sid, "reason": "dry_run", "workspace": str(ws)}

    run_proposer(ws, label)
    return collect_outcome(ws, seed)


def _append_result(result: dict[str, Any], args: argparse.Namespace) -> None:
    """Append one seed's outcome to accepted/rejected jsonl immediately, so a
    Ctrl-C or crash mid-run never loses already-finished seeds. Thread-safe for
    --parallel via _WRITE_LOCK. Short-circuit/skip rows are not persisted."""
    if args.dry_run or result.get("reason") in SKIP_REASONS:
        return
    path = OUT_ACCEPTED if result.get("accepted") else OUT_REJECTED
    line = json.dumps(result, ensure_ascii=False)
    with _WRITE_LOCK:
        with path.open("a", encoding="utf-8") as f:
            f.write(line + "\n")
            f.flush()


# ---------------------------------------------------------------------------
# Entry point.
# ---------------------------------------------------------------------------


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Per-seed adversarial K-BrowseComp problem generation."
    )
    ap.add_argument("--source-classes", default=None,
                    help="Comma-separated source class names to include.")
    ap.add_argument("--axes", default=None,
                    help="Comma-separated axes to include: intra,inter (default: both).")
    ap.add_argument("--target-failure-modes", default=None,
                    help="Comma-separated FM codes (e.g. F4,F6); keep only seeds whose target_failure_modes overlap.")
    ap.add_argument("--seed-id", default=None,
                    help="Run only this seed (sha1(url)[:10]).")
    ap.add_argument("--seed-json", default=None,
                    help=("Optional local seed candidate file(s), comma-separated. "
                          "Accepts JSON list, {'seeds': [...]}, or JSONL rows. "
                          "When set, the generated seed_expansion aggregate files "
                          "are not required."))
    ap.add_argument("--max-seeds", type=int, default=None,
                    help="Cap the number of seeds processed.")
    ap.add_argument("--section", type=int, default=None,
                    help=("Process the i-th 100-seed slice of the canonical "
                          "pool. Section 1 = seeds [0, 100); section 13 = seeds "
                          "[1200, 1298). For dividing work across people: each "
                          "person runs `python3 run.py --section i --litellm`. "
                          "Sliced BEFORE other filters so the section→seeds "
                          "mapping is identical across runs."))
    ap.add_argument("--solver-model", default=None,
                    help=("Reference LLM used by the ORACLE gate (single model — "
                          "verifies the problem is well-formed with page in context). "
                          "Default: gpt-5.4-mini, or azure_ai/gpt-5.4-mini with --litellm."))
    ap.add_argument("--target-models", default=None,
                    help=("Comma-list of LLMs to test in the ADVERSARIAL gate. "
                          "Acceptance requires ALL models to fail for a target FM. "
                          "Default: gpt-5.4-mini,gemini/gemini-3-flash-preview; "
                          "with --litellm: azure_ai/gpt-5.4-mini,gemini/gemini-3-flash-preview."))
    ap.add_argument("--grader-model", default=None,
                    help=("LLM used by the grader. Default depends on --litellm: "
                          "without it, gpt-5.4-mini; with it, azure_ai/gpt-5.4-mini."))
    ap.add_argument("--search-engine", default="perplexity",
                    help="perplexity / brave / exa / tavily.")
    ap.add_argument("--litellm", action="store_true",
                    help=("Route solver + grader through a LiteLLM proxy. The rendered "
                          "brief tells the proposer to pass --litellm to solver_runner.py, "
                          "which auto-prefixes models with `litellm_proxy/` and uses "
                          "LITELLM_PROXY_BASE_URL / LITELLM_PROXY_API_KEY."))
    ap.add_argument("--parallel", type=int, default=1,
                    help="Max concurrent per-seed subprocesses (default 1).")
    ap.add_argument("--order", choices=["interleave", "file"], default="interleave",
                    help=("Processing order. interleave (default): round-robin across "
                          "source_class groups so no single class is front-loaded "
                          "(e.g. avoids the 20 academic_korean seeds at the head of the "
                          "pool all running first). file: original intra-then-inter JSON "
                          "order. Note: --section still slices the canonical file order "
                          "first, so section membership is unchanged; only the order "
                          "within the run changes."))
    ap.add_argument("--reverse", action="store_true",
                    help=("Walk the ordered seed list back-to-front. Lets two machines "
                          "split the pool without overlap: one runs forward (run.py / "
                          "launch.sh), the other runs --reverse (run_codex.py / "
                          "launch_codex.sh); with the same default --order interleave they "
                          "start from opposite ends and meet in the middle. Applied after "
                          "ordering, before --max-seeds."))
    ap.add_argument("--resume", action="store_true",
                    help="Skip seeds that already have a prior final.json.")
    ap.add_argument("--dry-run", action="store_true",
                    help="Render briefs but do not launch claude.")
    ap.add_argument("--run-id", default=None,
                    help="Reuse a given run_id across multiple invocations.")
    args = ap.parse_args()

    # Context-aware defaults: when --litellm is on, the proxy expects
    # `azure_ai/gpt-5.4-mini` (which solver_runner.py then auto-prefixes with
    # `litellm_proxy/`); the OpenAI-Responses direct path uses bare `gpt-5.4-mini`.
    if args.solver_model is None:
        args.solver_model = "azure_ai/gpt-5.4-mini" if args.litellm else "gpt-5.4-mini"
    if args.grader_model is None:
        args.grader_model = "azure_ai/gpt-5.4-mini" if args.litellm else "gpt-5.4-mini"
    if args.target_models is None:
        args.target_models = (
            "azure_ai/gpt-5.4-mini,gemini/gemini-3-flash-preview"
            if args.litellm else "gpt-5.4-mini,gemini/gemini-3-flash-preview"
        )
    args.target_models_list = [
        m.strip() for m in args.target_models.split(",") if m.strip()
    ]

    if not args.dry_run:
        check_claude_version()
        check_auth()
        check_solver_prereqs(args)

    if not TEMPLATE.exists():
        sys.exit(f"[run] task_spec_template.md not found at {TEMPLATE}")
    if not SOLVER_RUNNER.exists():
        sys.exit(f"[run] solver_runner.py not found at {SOLVER_RUNNER}")
    if not PERPLEXITY_CHECK.exists():
        sys.exit(f"[run] perplexity_check.py not found at {PERPLEXITY_CHECK}")

    seeds = load_seeds(args)
    if not seeds:
        sys.exit("[run] No seeds matched filters.")

    run_id = args.run_id or _dt.datetime.now(_dt.timezone.utc).strftime("%Y%m%dT%H%M%SZ")

    print(f"[run] {PROPOSER_CLI_NAME}:        {CLAUDE_BIN} (expected {REQUIRED_CLAUDE_VERSION})", flush=True)
    print(f"[run] Proposer:      {MODEL} (effort={EFFORT})", flush=True)
    print(f"[run] Oracle model:  {args.solver_model} on {args.search_engine}", flush=True)
    print(f"[run] Target models: {', '.join(args.target_models_list)}", flush=True)
    print(f"[run] Grader:        {args.grader_model}", flush=True)
    print(f"[run] LiteLLM proxy: {'ON' if args.litellm else 'off'}", flush=True)
    print(f"[run] Order:         {args.order}{' + REVERSE' if args.reverse else ''}", flush=True)
    print(f"[run] Run ID:        {run_id}", flush=True)
    if args.section is not None:
        print(f"[run] Section:       {args.section}", flush=True)
    print(f"[run] Seeds:         {len(seeds)} (after filters)", flush=True)
    print(f"[run] Parallel:      {args.parallel}", flush=True)
    print(f"[run] Workspace:     {WORK_DIR}", flush=True)

    WORK_DIR.mkdir(parents=True, exist_ok=True)

    results: list[dict[str, Any]] = []
    aborted = False
    interrupted = False
    # Each seed's outcome is appended to the JSONLs immediately (via
    # _append_result) rather than batched here at the end — so Ctrl-C or a
    # crash keeps every seed finished so far.
    if args.parallel <= 1:
        try:
            for i, seed in enumerate(seeds):
                if _RATE_LIMIT_HIT.is_set():
                    print(
                        f"[run] Aborting after {i}/{len(seeds)} seeds: usage cap hit.",
                        flush=True,
                    )
                    aborted = True
                    break
                print(
                    f"\n=== seed {i+1}/{len(seeds)}: "
                    f"{seed.get('website_url', '?')[:90]} ===",
                    flush=True,
                )
                r = process_seed(seed, run_id, args)
                results.append(r)
                _append_result(r, args)
        except KeyboardInterrupt:
            interrupted = True
            print(
                f"\n[run] Interrupted (Ctrl-C) after {len(results)}/{len(seeds)} "
                f"seeds. Finished outcomes already saved to {OUT_ACCEPTED.name} / "
                f"{OUT_REJECTED.name}; re-run with --resume to continue.",
                flush=True,
            )
    else:
        with futures.ThreadPoolExecutor(max_workers=args.parallel) as ex:
            futs = [ex.submit(process_seed, s, run_id, args) for s in seeds]
            for f in futures.as_completed(futs):
                try:
                    r = f.result()
                except futures.CancelledError:
                    continue
                results.append(r)
                _append_result(r, args)
                if _RATE_LIMIT_HIT.is_set() and not aborted:
                    aborted = True
                    print(
                        f"[run] Usage cap hit; cancelling pending seeds "
                        f"(in-flight will short-circuit on their next event).",
                        flush=True,
                    )
                    for pending in futs:
                        pending.cancel()

    accepted = sum(1 for r in results if r.get("accepted"))
    skipped = sum(1 for r in results if r.get("reason") in SKIP_REASONS)
    rejected = len(results) - accepted - skipped
    print(
        f"\n[run] Done. Accepted: {accepted}  Rejected: {rejected}  Skipped: {skipped}",
        flush=True,
    )
    if aborted:
        print(
            "[run] Aborted by usage cap. After the cap resets, re-run the "
            "same command with --resume to retry only the unfinished seeds.",
            flush=True,
        )
        sys.exit(1)
    if interrupted:
        sys.exit(130)


if __name__ == "__main__":
    main()
