"""monitor.py — Inspect K-BrowseComp problem-generation workspaces.

The run.py orchestrator and solver_runner.py write a per-seed trajectory to:

  work/<source_class>/<seed_id>/<run_id>/
    seed.json                  # seed entry the proposer was given
    task_spec.md               # rendered brief
    proposer_stream.jsonl      # raw stream-json from claude (one event per line)
    proposer_log.txt           # human-readable mirror of proposer stdout
    attempt_<N>/
      problem.json             # proposer's draft
      solver_convo.json        # full solver trajectory (queries + results)
      solver_response.txt      # solver's final answer
      grader_result.json       # grader judgement
      summary.json             # {solved, grade, solver_response, ...}
    final.json                 # proposer's accept/reject verdict

Commands:
  python monitor.py list                              # table of all workspaces + status
  python monitor.py list --run RUN_ID                 # filter to one run
  python monitor.py show <workspace>                  # full detail of one workspace
  python monitor.py show --latest --trace             # latest workspace + solver trace
  python monitor.py tail <workspace>                  # follow proposer_log.txt live
  python monitor.py tail --latest                     # follow newest workspace
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parent
WORK_DIR = ROOT / "work"


class C:
    """ANSI colors — disabled when not writing to a TTY."""
    if sys.stdout.isatty():
        RESET = "\033[0m"
        DIM = "\033[2m"
        BOLD = "\033[1m"
        GREEN = "\033[32m"
        RED = "\033[31m"
        YELLOW = "\033[33m"
        CYAN = "\033[36m"
        BLUE = "\033[34m"
    else:
        RESET = DIM = BOLD = GREEN = RED = YELLOW = CYAN = BLUE = ""


def _short(s: str, n: int) -> str:
    s = s.replace("\n", " | ")
    return s if len(s) <= n else s[: n - 3] + "..."


def _load_json(p: Path) -> Any:
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return None


def _attempt_dirs(ws: Path) -> list[Path]:
    out = []
    for d in ws.iterdir():
        if d.is_dir() and d.name.startswith("attempt_"):
            try:
                idx = int(d.name.split("_", 1)[1])
            except ValueError:
                continue
            out.append((idx, d))
    return [d for _, d in sorted(out)]


def _all_workspaces() -> list[Path]:
    out: list[Path] = []
    if not WORK_DIR.exists():
        return out
    for cls_dir in sorted(WORK_DIR.iterdir()):
        if not cls_dir.is_dir():
            continue
        for seed_dir in sorted(cls_dir.iterdir()):
            if not seed_dir.is_dir():
                continue
            for run_dir in sorted(seed_dir.iterdir()):
                if run_dir.is_dir():
                    out.append(run_dir)
    return out


def _latest_workspace() -> Path | None:
    wss = _all_workspaces()
    if not wss:
        return None
    return max(wss, key=lambda p: p.stat().st_mtime)


def _gate_status(attempt_dir: Path) -> dict[str, Any]:
    """Return per-gate outcomes for one attempt_N directory.

    Each attempt has: sanity/, oracle/, and adversarial/<model_slug>/ subdirs.
    Older layouts (flat adversarial/summary.json, or a single attempt-level
    summary.json) are also recognized.
    """
    sanity = _load_json(attempt_dir / "sanity" / "sanity_check.json")
    oracle = _load_json(attempt_dir / "oracle" / "summary.json")

    adversarial_by_model: dict[str, Any] = {}
    adv_dir = attempt_dir / "adversarial"
    if adv_dir.is_dir():
        flat = _load_json(adv_dir / "summary.json")
        if flat:
            adversarial_by_model["_legacy"] = flat
        for sub in sorted(adv_dir.iterdir()):
            if sub.is_dir():
                s = _load_json(sub / "summary.json")
                if s:
                    adversarial_by_model[sub.name] = s

    flat_attempt = _load_json(attempt_dir / "summary.json")  # legacy
    if not adversarial_by_model and flat_attempt:
        adversarial_by_model["_legacy"] = flat_attempt

    return {
        "sanity": sanity,
        "oracle": oracle,
        "adversarial": adversarial_by_model,
        "flat_legacy": flat_attempt,
    }


def _ws_status(ws: Path) -> dict[str, Any]:
    seed = _load_json(ws / "seed.json") or {}
    final = _load_json(ws / "final.json")
    attempts = _attempt_dirs(ws)

    accepted: bool | None = None
    grade_summary: str = "—"
    if final is not None:
        accepted = bool(final.get("accepted"))
        grade_summary = "ACCEPTED" if accepted else "rejected"
    elif attempts:
        last = attempts[-1]
        gates = _gate_status(last)
        adv = gates["adversarial"]
        if adv:
            # Aggregate across models: solved_count / tested_count
            solved = sum(1 for s in adv.values() if s and s.get("solved"))
            grade_summary = f"{last.name}/adv: {solved}/{len(adv)} solved"
        elif gates["oracle"]:
            grade_summary = f"{last.name}/oracle: {gates['oracle'].get('grade', '?')}"
        elif gates["sanity"]:
            grade_summary = f"{last.name}/sanity: {gates['sanity'].get('verdict', '?')}"
        else:
            grade_summary = f"{last.name}: running…"
    else:
        grade_summary = "no attempts yet"

    return {
        "workspace": ws,
        "run_id": ws.name,
        "seed_id": ws.parent.name,
        "source_class": ws.parent.parent.name,
        "axis": seed.get("axis", "?"),
        "url": seed.get("website_url", "?"),
        "target_fms": seed.get("target_failure_modes") or [],
        "attempts": len(attempts),
        "accepted": accepted,
        "grade_summary": grade_summary,
        "has_final": final is not None,
    }


def cmd_list(args: argparse.Namespace) -> int:
    rows = [_ws_status(ws) for ws in _all_workspaces()]
    if args.run:
        rows = [r for r in rows if r["run_id"] == args.run]
    if args.source_class:
        rows = [r for r in rows if r["source_class"] == args.source_class]
    if not rows:
        print("No matching workspaces.")
        return 0

    print(
        f"{C.BOLD}{'RUN_ID':<22} {'SEED':<10} {'CLASS':<20} "
        f"{'AX':<5} {'TRY':<4} {'STATUS':<28} URL{C.RESET}"
    )
    for r in rows:
        color = (
            C.GREEN if r["accepted"] is True
            else C.RED if r["accepted"] is False
            else C.YELLOW
        )
        url = r["url"]
        if len(url) > 60:
            url = url[:57] + "..."
        print(
            f"{r['run_id']:<22} {r['seed_id']:<10} {r['source_class']:<20} "
            f"{r['axis']:<5} {r['attempts']:<4} "
            f"{color}{r['grade_summary']:<28}{C.RESET} {C.DIM}{url}{C.RESET}"
        )
    return 0


def _proposer_log_preview(ws: Path, head: int, tail: int) -> str:
    p = ws / "proposer_log.txt"
    if not p.exists():
        return f"{C.DIM}(no proposer_log.txt yet){C.RESET}"
    lines = p.read_text(encoding="utf-8").splitlines()
    if len(lines) <= head + tail:
        return "\n".join(lines)
    omitted = len(lines) - head - tail
    return "\n".join(
        lines[:head]
        + [f"{C.DIM}…[{omitted} lines omitted]…{C.RESET}"]
        + lines[-tail:]
    )


def _solver_trace_lines(convo_path: Path, max_lines: int) -> list[str]:
    """Decode solver_convo.json into '→ tool: input' / '  ✓ result' lines."""
    raw = _load_json(convo_path)
    if not raw:
        return [f"{C.DIM}(no solver_convo.json){C.RESET}"]
    messages = raw.get("messages") if isinstance(raw, dict) else raw
    if not isinstance(messages, list):
        return [f"{C.DIM}(unrecognized solver_convo shape){C.RESET}"]

    out: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            continue
        role = msg.get("role", "?")
        content = msg.get("content", [])
        if isinstance(content, str):
            txt = content.strip()
            if txt:
                out.append(f"{C.CYAN}[{role}]{C.RESET} {_short(txt, 280)}")
            continue
        if not isinstance(content, list):
            continue
        for block in content:
            if not isinstance(block, dict):
                continue
            bt = block.get("type")
            if bt == "text":
                txt = block.get("text", "").strip()
                if txt:
                    out.append(f"{C.CYAN}[{role}]{C.RESET} {_short(txt, 280)}")
            elif bt == "tool_use":
                name = block.get("name", "?")
                inp = block.get("input", {}) or {}
                summary = (
                    inp.get("query")
                    or inp.get("url")
                    or inp.get("command")
                    or _short(json.dumps(inp), 200)
                )
                out.append(f"{C.YELLOW}→ {name}{C.RESET}: {_short(str(summary), 280)}")
            elif bt == "tool_result":
                rc = block.get("content", "")
                if isinstance(rc, list):
                    rc = "".join(
                        c.get("text", "") if isinstance(c, dict) else str(c)
                        for c in rc
                    )
                marker = f"{C.RED}✗{C.RESET}" if block.get("is_error") else f"{C.GREEN}✓{C.RESET}"
                out.append(f"  {marker} {_short(str(rc).strip(), 280)}")

    if len(out) > max_lines:
        head_n = max_lines // 3
        tail_n = max_lines - head_n
        out = (
            out[:head_n]
            + [f"{C.DIM}…[{len(out) - max_lines} entries omitted]…{C.RESET}"]
            + out[-tail_n:]
        )
    return out


def _resolve_ws(value: str | None, use_latest: bool) -> Path:
    if use_latest:
        latest = _latest_workspace()
        if latest is None:
            sys.exit("No workspaces in work/ yet.")
        return latest
    if not value:
        sys.exit("Need a workspace path or --latest.")
    p = Path(value).resolve()
    if not p.is_dir():
        sys.exit(f"Not a directory: {p}")
    return p


def cmd_show(args: argparse.Namespace) -> int:
    ws = _resolve_ws(args.workspace, args.latest)
    status = _ws_status(ws)
    seed = _load_json(ws / "seed.json") or {}
    final = _load_json(ws / "final.json")

    print(f"{C.BOLD}Workspace:{C.RESET} {ws}")
    print(f"  run_id:        {status['run_id']}")
    print(f"  seed_id:       {status['seed_id']}")
    print(f"  source_class:  {status['source_class']}")
    print(f"  axis:          {status['axis']}")
    print(f"  url:           {seed.get('website_url')}")
    print(f"  target FMs:    {', '.join(status['target_fms']) or '(none)'}")
    print(f"  status:        {status['grade_summary']}")
    print()

    print(f"{C.BOLD}Proposer log (preview):{C.RESET}")
    print(_proposer_log_preview(ws, head=args.head, tail=args.tail))
    print()

    for ad in _attempt_dirs(ws):
        problem = _load_json(ad / "problem.json")
        gates = _gate_status(ad)
        print(f"{C.BOLD}── {ad.name} ──{C.RESET}")
        if problem:
            print(f"  problem:   {_short(problem.get('problem', ''), 320)}")
            print(f"  answer:    {_short(str(problem.get('answer', '')), 200)}")
            if problem.get("target_failure_modes"):
                print(f"  target FM: {', '.join(problem['target_failure_modes'])}")

        sanity = gates["sanity"]
        if sanity:
            verdict = sanity.get("verdict", "?")
            col = C.RED if verdict == "TOO_EASY" else C.GREEN if verdict == "PROCEED" else C.YELLOW
            print(
                f"  {C.BOLD}Gate 1 sanity:{C.RESET} {col}{verdict}{C.RESET} "
                f"answer_in_serp={sanity.get('answer_in_serp')} "
                f"queries={sanity.get('num_queries')}"
            )

        oracle = gates["oracle"]
        if oracle:
            solved = oracle.get("solved")
            col = C.GREEN if solved else C.RED
            print(
                f"  {C.BOLD}Gate 2 oracle:{C.RESET} grade={col}{oracle.get('grade')}{C.RESET}  "
                f"solved={solved}"
            )
            print(f"    response:  {_short(oracle.get('solver_response', ''), 320)}")
            g = _load_json(ad / "oracle" / "grader_result.json")
            if g:
                print(f"    reasoning: {_short(g.get('grade_text', ''), 320)}")
            if args.trace:
                print(f"    {C.DIM}-- solver trajectory (oracle) --{C.RESET}")
                for ln in _solver_trace_lines(ad / "oracle" / "solver_convo.json",
                                              max_lines=args.trace_lines):
                    print(f"      {ln}")

        adv = gates["adversarial"]
        if adv:
            solved_count = sum(1 for s in adv.values() if s and s.get("solved"))
            agg_col = C.GREEN if solved_count == 0 else C.RED
            print(
                f"  {C.BOLD}Gate 3 adversarial (multi-model):{C.RESET} "
                f"{agg_col}{len(adv) - solved_count}/{len(adv)} failed (target = ALL fail){C.RESET}"
            )
            for slug in sorted(adv.keys()):
                s = adv[slug] or {}
                solved = s.get("solved")
                col = C.RED if solved else C.GREEN
                print(
                    f"    {C.BOLD}[{slug}]{C.RESET} grade={col}{s.get('grade')}{C.RESET}  "
                    f"solved={solved}"
                )
                print(f"      response:  {_short(s.get('solver_response', ''), 280)}")
                model_dir = ad / "adversarial" / (slug if slug != "_legacy" else "")
                g = _load_json(model_dir / "grader_result.json") if slug != "_legacy" \
                    else _load_json(ad / "adversarial" / "grader_result.json")
                if g:
                    print(f"      reasoning: {_short(g.get('grade_text', ''), 280)}")
                if args.trace:
                    convo_path = (
                        model_dir / "solver_convo.json"
                        if slug != "_legacy"
                        else ad / "adversarial" / "solver_convo.json"
                    )
                    print(f"      {C.DIM}-- trajectory ({slug}) --{C.RESET}")
                    for ln in _solver_trace_lines(convo_path, max_lines=args.trace_lines):
                        print(f"        {ln}")
        print()

    if final:
        print(f"{C.BOLD}final.json:{C.RESET}")
        print(f"  accepted:      {final.get('accepted')}")
        print(f"  rationale:     {_short(final.get('rationale', ''), 400)}")
        if not final.get("accepted"):
            print(f"  reject_reason: {_short(final.get('reason_if_rejected', ''), 400)}")
        # Lineage (only present after orchestrator enrichment).
        if final.get("original_url") or final.get("proposed_question"):
            print(f"{C.BOLD}Lineage:{C.RESET}")
            print(f"  original_qid:  {final.get('original_question_id')}")
            print(f"  original_url:  {_short(str(final.get('original_url') or ''), 120)}")
            print(f"  original_Q:    {_short(str(final.get('original_question') or ''), 200)}")
            print(f"  original_A:    {_short(str(final.get('original_answer') or ''), 120)}")
            print(f"  seed_url:      {_short(str(final.get('seed_url') or ''), 120)}")
            print(f"  proposed_url:  {_short(str(final.get('proposed_url') or ''), 120)}")
            pq = final.get("proposed_question") or {}
            if pq:
                print(f"  proposed_question ({len(pq)} iters):")
                for k in sorted(pq.keys()):
                    print(f"    {k}: {_short(str(pq[k]), 200)}")

    return 0


def cmd_tail(args: argparse.Namespace) -> int:
    ws = _resolve_ws(args.workspace, args.latest)
    log = ws / "proposer_log.txt"
    if not log.exists():
        print(f"Waiting for {log} to appear…", flush=True)
        while not log.exists():
            if (ws / "final.json").exists():
                print("final.json appeared before proposer_log.txt — nothing to tail.")
                return 0
            time.sleep(0.5)
    print(f"{C.DIM}--- tailing {log} ---{C.RESET}", flush=True)
    with log.open("r", encoding="utf-8") as f:
        if not args.from_start:
            f.seek(0, 2)
        try:
            while True:
                line = f.readline()
                if not line:
                    if (ws / "final.json").exists() and not args.persist:
                        for tail_line in f:
                            print(tail_line.rstrip(), flush=True)
                        print(
                            f"{C.DIM}--- final.json present; exiting "
                            f"(use --persist to keep tailing) ---{C.RESET}"
                        )
                        return 0
                    time.sleep(0.5)
                    continue
                print(line.rstrip(), flush=True)
        except KeyboardInterrupt:
            return 0


def main() -> None:
    ap = argparse.ArgumentParser(
        description="Inspect K-BrowseComp problem-generation workspaces."
    )
    sub = ap.add_subparsers(dest="cmd", required=True)

    p_list = sub.add_parser("list", help="Table of all workspaces and statuses.")
    p_list.add_argument("--run", default=None, help="Filter to one run_id.")
    p_list.add_argument("--source-class", default=None,
                        help="Filter to one source_class.")

    p_show = sub.add_parser("show", help="Detailed view of a single workspace.")
    p_show.add_argument("workspace", nargs="?",
                        help="Path to the run_id workspace directory.")
    p_show.add_argument("--latest", action="store_true",
                        help="Use the most recently modified workspace.")
    p_show.add_argument("--head", type=int, default=40,
                        help="Proposer log head lines (default 40).")
    p_show.add_argument("--tail", type=int, default=60,
                        help="Proposer log tail lines (default 60).")
    p_show.add_argument("--trace", action="store_true",
                        help="Include decoded solver trajectory for each attempt.")
    p_show.add_argument("--trace-lines", type=int, default=80,
                        help="Max trace entries per attempt (default 80).")

    p_tail = sub.add_parser("tail", help="Follow proposer_log.txt of a workspace.")
    p_tail.add_argument("workspace", nargs="?",
                        help="Path to the run_id workspace directory.")
    p_tail.add_argument("--latest", action="store_true",
                        help="Use the most recently modified workspace.")
    p_tail.add_argument("--persist", action="store_true",
                        help="Keep tailing even after final.json appears.")
    p_tail.add_argument("--from-start", action="store_true",
                        help="Start from the beginning of the log, not the end.")

    args = ap.parse_args()
    if args.cmd == "list":
        sys.exit(cmd_list(args))
    elif args.cmd == "show":
        sys.exit(cmd_show(args))
    elif args.cmd == "tail":
        sys.exit(cmd_tail(args))


if __name__ == "__main__":
    main()
