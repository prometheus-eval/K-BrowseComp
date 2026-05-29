"""Single-problem solver runner.

Wraps search_evals.agents.DeepResearchAgent + search_evals.suites.graders.DeepResearchGrader
to run one adversarial problem end-to-end and persist its trajectory + grade.
Intended to be invoked by the problem_generation proposer (via Bash) inside
its per-seed workspace.

Two modes for differential testing:
  --mode adversarial (default)
      Solver gets just the question. This is the real test — a clean
      acceptance requires this to FAIL for a target failure-mode reason.
  --mode oracle --oracle-page-path PATH
      Solver gets the page contents (HTML stripped to text) PLUS the question.
      Confirms the problem is well-formed: if oracle FAILS, the answer
      isn't actually extractable from the given page → problem is malformed.

Outputs (in --output-dir):
- solver_convo.json     full search trajectory (all queries + tool results)
- solver_response.txt   the solver's final text response
- grader_result.json    grader judgement: CORRECT / INCORRECT / NOT_ATTEMPTED + reasoning
- summary.json          one-shot summary (solved, grade, paths, reasoning, mode)

Exit code 0 on success (regardless of solved/not solved); non-zero on
infrastructure failure (e.g., missing API key, problem file malformed).

Usage:
  # Adversarial (default)
  python solver_runner.py \\
    --problem path/to/problem.json \\
    --output-dir path/to/out/adversarial \\
    --search-engine perplexity \\
    --solver-model gpt-5.4-mini \\
    --grader-model gpt-5.4-mini

  # Oracle (page in context)
  python solver_runner.py \\
    --mode oracle --oracle-page-path /tmp/page.html \\
    --problem path/to/problem.json \\
    --output-dir path/to/out/oracle \\
    ...
"""
from __future__ import annotations

import argparse
import asyncio
import json
import re
import sys
import traceback
from pathlib import Path
from typing import Any

# Make search_evals importable + load credentials from repo-root txt files.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HERE))

import _load_env  # noqa: F401  side-effect: populates os.environ

from search_evals.agents import DeepResearchAgent
from search_evals.datasets import Datum
from search_evals.suites.graders import DeepResearchGrader


def _ensure_litellm_prefix(model: str) -> str:
    """Add `litellm_proxy/` prefix if not already present."""
    return model if model.startswith("litellm_proxy/") else f"litellm_proxy/{model}"


def _strip_html_to_text(html: str) -> str:
    """Convert HTML to plain text. Strips <script>/<style>/<noscript> chrome,
    collapses whitespace, preserves block structure with newlines."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "html.parser")
        for tag in soup(["script", "style", "noscript", "template"]):
            tag.decompose()
        text = soup.get_text(separator="\n")
    except ImportError:
        # Crude fallback if bs4 isn't installed.
        text = re.sub(r"<script[^>]*>.*?</script>", "", html, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<style[^>]*>.*?</style>", "", text, flags=re.DOTALL | re.IGNORECASE)
        text = re.sub(r"<[^>]+>", "\n", text)
    text = re.sub(r"\n[ \t]+", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return text.strip()


def _build_oracle_prompt(problem_text: str, page_path: Path, max_chars: int, source_url: str | None) -> str:
    raw = page_path.read_text(encoding="utf-8", errors="replace")
    text = _strip_html_to_text(raw) if raw.lstrip().lower().startswith(("<!doctype", "<html", "<")) else raw
    text = text[:max_chars]
    src = source_url or page_path.name
    return (
        f"You are given the full text contents of the source page that contains the answer "
        f"to the following question. Use this page text to find and extract the answer. "
        f"You may also issue search queries if helpful, but the answer is in the page below.\n\n"
        f"=== BEGIN PAGE CONTENTS (from {src}) ===\n"
        f"{text}\n"
        f"=== END PAGE CONTENTS ===\n\n"
        f"=== QUESTION ===\n"
        f"{problem_text}"
    )


async def main_async(args: argparse.Namespace) -> int:
    problem_path: Path = args.problem
    output_dir: Path = args.output_dir

    if not problem_path.exists():
        print(f"[solver] ERROR: problem file not found: {problem_path}", file=sys.stderr)
        return 2

    try:
        problem_obj = json.loads(problem_path.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[solver] ERROR: cannot parse {problem_path}: {e}", file=sys.stderr)
        return 2

    problem_text = problem_obj.get("problem")
    answer = problem_obj.get("answer")
    if not problem_text or not answer:
        print(f"[solver] ERROR: {problem_path} must contain 'problem' and 'answer'", file=sys.stderr)
        return 2

    output_dir.mkdir(parents=True, exist_ok=True)

    solver_model = args.solver_model
    grader_model = args.grader_model
    if args.litellm:
        solver_model = _ensure_litellm_prefix(solver_model)
        grader_model = _ensure_litellm_prefix(grader_model)

    if args.mode == "oracle":
        if not args.oracle_page_path:
            print("[solver] ERROR: --mode oracle requires --oracle-page-path", file=sys.stderr)
            return 2
        if not args.oracle_page_path.exists():
            print(f"[solver] ERROR: oracle page not found: {args.oracle_page_path}", file=sys.stderr)
            return 2
        agent_input = _build_oracle_prompt(
            problem_text=problem_text,
            page_path=args.oracle_page_path,
            max_chars=args.oracle_max_chars,
            source_url=problem_obj.get("source_url"),
        )
        print(
            f"[solver] oracle mode: page={args.oracle_page_path.name} "
            f"chars={len(agent_input)}",
            flush=True,
        )
    else:
        agent_input = problem_text

    print(
        f"[solver] running mode={args.mode} solver_model={solver_model} "
        f"search_engine={args.search_engine} litellm={args.litellm}",
        flush=True,
    )
    # Construction can fail (unknown model name, missing creds, missing search
    # engine wiring, etc.). Catch separately so the proposer sees a clean
    # exit code instead of an unhandled traceback that may cascade into the
    # surrounding Claude Code session's workspace state.
    try:
        agent = DeepResearchAgent(search_engine=args.search_engine, model=solver_model)
    except Exception as e:
        print(f"[solver] ERROR: agent construction failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 5
    try:
        convo = await agent(agent_input)
    except Exception as e:
        print(f"[solver] ERROR: solver crashed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 3

    convo_path = output_dir / "solver_convo.json"
    response_path = output_dir / "solver_response.txt"
    grader_path = output_dir / "grader_result.json"
    summary_path = output_dir / "summary.json"

    try:
        convo_path.write_text(convo.model_dump_json(indent=2), encoding="utf-8")
        response_text = convo.last_text() or ""
        response_path.write_text(response_text, encoding="utf-8")
    except Exception as e:
        print(f"[solver] ERROR: writing solver outputs failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 6

    print(f"[solver] grading with grader_model={grader_model}", flush=True)
    try:
        if args.litellm:
            from litellm_grader import LiteLLMProxyGrader
            grader: Any = LiteLLMProxyGrader(model=grader_model)
        else:
            grader = DeepResearchGrader(model=grader_model)
    except Exception as e:
        print(f"[solver] ERROR: grader construction failed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 5
    datum = Datum(id="problem-gen-test", problem=problem_text, answer=answer)
    try:
        grader_result = await grader(datum, convo)
    except Exception as e:
        print(f"[solver] ERROR: grader crashed: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        return 4

    grader_path.write_text(grader_result.model_dump_json(indent=2), encoding="utf-8")

    solved = grader_result.grade_type.name == "CORRECT"
    summary = {
        "mode": args.mode,
        "solved": solved,
        "grade": grader_result.grade_type.name,
        "solver_response": response_text,
        "grader_reasoning": grader_result.grade_text,
        "files": {
            "convo": str(convo_path),
            "response": str(response_path),
            "grader": str(grader_path),
        },
        "solver_model": solver_model,
        "grader_model": grader_model,
        "search_engine": args.search_engine,
        "litellm": args.litellm,
        "oracle_page_path": str(args.oracle_page_path) if args.oracle_page_path else None,
    }
    summary_path.write_text(
        json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8"
    )

    print(f"[solver] solved={solved} grade={summary['grade']}", flush=True)
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Single-problem K-BrowseComp solver runner.")
    ap.add_argument("--problem", type=Path, required=True,
                    help="Path to a problem.json with {problem, answer} fields.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory to write solver_convo.json + grader_result.json + summary.json.")
    ap.add_argument("--search-engine", default="perplexity",
                    help="Search engine for the solver (perplexity / brave / exa / tavily).")
    ap.add_argument("--solver-model", default=None,
                    help=("LLM driving the solver agent. Default depends on --litellm: "
                          "without it, gpt-5.4-mini (OpenAI Responses); with it, "
                          "azure_ai/gpt-5.4-mini (auto-prefixed to litellm_proxy/...)."))
    ap.add_argument("--grader-model", default=None,
                    help=("LLM used by the grader. Default depends on --litellm: "
                          "without it, gpt-5.4-mini; with it, azure_ai/gpt-5.4-mini."))
    ap.add_argument("--litellm", action="store_true",
                    help=("Route solver + grader through a LiteLLM proxy via "
                          "litellm.acompletion(). Auto-prefixes solver/grader "
                          "model names with `litellm_proxy/` if missing, uses "
                          "LITELLM_PROXY_BASE_URL / LITELLM_PROXY_API_KEY from env."))
    ap.add_argument("--mode", choices=["adversarial", "oracle"], default="adversarial",
                    help=("adversarial (default): solver gets question only — the real "
                          "K-BrowseComp test. oracle: solver gets the page contents + "
                          "question, to verify the answer is extractable from the page."))
    ap.add_argument("--oracle-page-path", type=Path, default=None,
                    help="Path to the source HTML page (used when --mode oracle).")
    ap.add_argument("--oracle-max-chars", type=int, default=80_000,
                    help="Truncate stripped page text to this many chars (default 80K).")
    args = ap.parse_args()
    # Apply context-dependent defaults.
    if args.solver_model is None:
        args.solver_model = "azure_ai/gpt-5.4-mini" if args.litellm else "gpt-5.4-mini"
    if args.grader_model is None:
        args.grader_model = "azure_ai/gpt-5.4-mini" if args.litellm else "gpt-5.4-mini"
    # Outer guard: any uncaught exception in main_async would otherwise bubble
    # out of asyncio.run and exit with the implicit Python crash code (1) +
    # an unhandled traceback. In a Claude Code Bash session this can cascade
    # into a workspace-state reset. Always exit with a deterministic code.
    try:
        sys.exit(asyncio.run(main_async(args)))
    except SystemExit:
        raise
    except Exception as e:
        print(f"[solver] ERROR: uncaught exception: {e}", file=sys.stderr)
        traceback.print_exc(file=sys.stderr)
        sys.exit(7)


if __name__ == "__main__":
    main()
