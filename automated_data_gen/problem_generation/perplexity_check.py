"""perplexity_check.py — Pre-flight Perplexity sanity check.

Tests whether a drafted problem is trivially solvable by Perplexity SERP
*before* the proposer spends the cost of a full solver run. If the known
answer string appears in any SERP snippet returned by Perplexity, the
problem is too easy and should be revised.

Flow:
  1. Generate N candidate Perplexity queries from the problem text (via LLM).
  2. Run each query against the Perplexity Search API.
  3. Concatenate all titles + snippets.
  4. Substring-match the known answer against the concat.
  5. Write sanity_check.json with verdict TOO_EASY / PROCEED.

Exit code:
  0 on success (whatever the verdict)
  2 on input/config error
  3 on Perplexity / LLM call failure

Usage:
  python perplexity_check.py \\
    --problem path/to/problem.json \\
    --output-dir path/to/out/sanity \\
    --litellm
"""
from __future__ import annotations

import argparse
import asyncio
import json
import os
import sys
from pathlib import Path
from typing import Any

# Make search_evals importable + load credentials from repo-root txt files.
_HERE = Path(__file__).resolve().parent
_REPO_ROOT = _HERE.parents[1]
sys.path.insert(0, str(_REPO_ROOT))
sys.path.insert(0, str(_HERE))

import _load_env  # noqa: F401  side-effect: populates os.environ

import orjson
from perplexity import AsyncPerplexity


async def _generate_queries(problem_text: str, n: int, model: str, litellm: bool) -> list[str]:
    """Use an LLM to draft N search queries for this problem.

    We deliberately ask the LLM to write queries a person would naturally
    type, not adversarial/uniquely-identifying queries — the goal is to see
    whether Perplexity gives away the answer for ordinary queries.
    """
    prompt = (
        f"You are a search query generator. Given the following question, "
        f"generate {n} distinct, well-formed search queries that someone "
        f"would type into a search engine to find the answer.\n\n"
        f"Requirements:\n"
        f"- Same language as the question (Korean if the question is Korean)\n"
        f"- Cover different angles: direct keyword, indirect descriptor, structured\n"
        f"- 8–20 words each\n"
        f"- Do NOT include the answer\n\n"
        f"Question:\n{problem_text}\n\n"
        f"Return ONLY a JSON object: {{\"queries\": [...]}}"
    )

    if litellm:
        import litellm  # noqa: WPS433
        full_model = model if model.startswith("litellm_proxy/") else f"litellm_proxy/{model}"
        resp = await litellm.acompletion(
            model=full_model,
            messages=[{"role": "user", "content": prompt}],
            base_url=os.getenv("LITELLM_PROXY_BASE_URL") or os.getenv("OPENAI_BASE_URL"),
            api_key=os.getenv("LITELLM_PROXY_API_KEY") or os.getenv("OPENAI_API_KEY"),
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content
    else:
        from openai import AsyncOpenAI  # noqa: WPS433
        client = AsyncOpenAI()
        resp = await client.chat.completions.create(
            model=model,
            messages=[{"role": "user", "content": prompt}],
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content

    try:
        data = orjson.loads(text or "{}")
        queries = data.get("queries", []) or []
        return [str(q) for q in queries[:n] if isinstance(q, (str, int))]
    except Exception:
        return []


async def main_async(args: argparse.Namespace) -> int:
    if not args.problem.exists():
        print(f"[sanity] ERROR: problem file not found: {args.problem}", file=sys.stderr)
        return 2

    try:
        problem_obj = json.loads(args.problem.read_text(encoding="utf-8"))
    except Exception as e:
        print(f"[sanity] ERROR: cannot parse {args.problem}: {e}", file=sys.stderr)
        return 2

    problem_text = problem_obj.get("problem", "").strip()
    answer = str(problem_obj.get("answer", "")).strip()
    if not problem_text or not answer:
        print(f"[sanity] ERROR: problem.json missing 'problem' or 'answer'", file=sys.stderr)
        return 2

    args.output_dir.mkdir(parents=True, exist_ok=True)

    print(f"[sanity] generating {args.num_queries} Perplexity queries…", flush=True)
    try:
        queries = await _generate_queries(
            problem_text, args.num_queries, args.query_model, args.litellm
        )
    except Exception as e:
        print(f"[sanity] WARNING: query generation failed: {e}", file=sys.stderr)
        queries = []
    if not queries:
        queries = [problem_text[:200]]  # fallback
    print(f"[sanity] queries: {queries}", flush=True)

    client = AsyncPerplexity()
    serp_entries: list[dict[str, Any]] = []
    for q in queries:
        try:
            resp = await client.search.create(
                query=q,
                max_results=10,
                max_tokens=3_000,
                max_tokens_per_page=3_000,
            )
            results = [
                {"url": r.url, "title": r.title or "", "snippet": r.snippet or ""}
                for r in resp.results
            ]
        except Exception as e:
            print(f"[sanity] ERROR: query failed: {e}", file=sys.stderr)
            results = []
        print(f"[sanity] q={q[:80]!r} → {len(results)} results", flush=True)
        serp_entries.append({"query": q, "results": results})

    all_text = "\n".join(
        f"{r['title']}\n{r['snippet']}"
        for entry in serp_entries
        for r in entry["results"]
    )
    answer_in_serp = answer in all_text
    verdict = "TOO_EASY" if answer_in_serp else "PROCEED"

    sanity = {
        "verdict": verdict,
        "answer_in_serp": answer_in_serp,
        "answer": answer,
        "problem": problem_text,
        "num_queries": len(queries),
        "queries": queries,
        "serp_total_chars": len(all_text),
        "serp_entries": serp_entries,
    }
    out_path = args.output_dir / "sanity_check.json"
    out_path.write_text(json.dumps(sanity, indent=2, ensure_ascii=False), encoding="utf-8")

    print(
        f"[sanity] verdict={verdict} answer_in_serp={answer_in_serp} "
        f"serp_chars={len(all_text)}",
        flush=True,
    )
    return 0


def main() -> None:
    ap = argparse.ArgumentParser(description="Pre-flight Perplexity sanity check.")
    ap.add_argument("--problem", type=Path, required=True,
                    help="Path to problem.json with {problem, answer}.")
    ap.add_argument("--output-dir", type=Path, required=True,
                    help="Directory to write sanity_check.json.")
    ap.add_argument("--num-queries", type=int, default=4,
                    help="How many candidate queries to run (default 4).")
    ap.add_argument("--query-model", default=None,
                    help=("LLM used to generate queries. Default depends on --litellm: "
                          "without it, gpt-5.4-mini; with it, azure_ai/gpt-5.4-mini."))
    ap.add_argument("--litellm", action="store_true",
                    help="Route the query-generation LLM call through LiteLLM proxy.")
    args = ap.parse_args()
    if args.query_model is None:
        args.query_model = "azure_ai/gpt-5.4-mini" if args.litellm else "gpt-5.4-mini"
    sys.exit(asyncio.run(main_async(args)))


if __name__ == "__main__":
    main()
