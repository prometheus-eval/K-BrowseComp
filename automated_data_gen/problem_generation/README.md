# problem_generation

Stage 3 of `automated_data_gen`: per-seed adversarial generation of K-BrowseComp
problems. For each seed produced by Stage 2 (`../seed_expansion/`), one
proposer subprocess — **Claude Code** (`run.py`) or **OpenAI Codex** (`run_codex.py`) —
drafts a Korean question and puts it through a **3-gate test**, iterating
up to 4 attempts (`iter0`–`iter3`). The orchestrator just spawns one CLI
per seed and collects `final.json`.

## Quick start

### Pick a proposer: Claude Code vs Codex

Two interchangeable proposer drivers live in this directory. They share
everything (3-gate logic, solver/grader plumbing, FM quotas, seed lineage)
— `run_codex.py` imports `run.py` and only swaps the proposer subprocess.
CLI flags are identical between them.

|                | `run.py`                                                                                       | `run_codex.py`                                                                       |
|----------------|------------------------------------------------------------------------------------------------|--------------------------------------------------------------------------------------|
| Proposer CLI   | Anthropic Claude Code (`claude`)                                                               | OpenAI Codex (`codex`)                                                               |
| Default model  | `claude-opus-4-7` (effort=max)                                                                 | `gpt-5-codex`                                                                        |
| Auth           | `ANTHROPIC_API_KEY` env, or `claude login` → `~/.claude/.credentials.json`, or macOS Keychain   | `OPENAI_API_KEY` / `CODEX_API_KEY` env, or `codex login` → `~/.codex/auth.json`      |
| Workspaces     | `work/`                                                                                        | `work_codex/`                                                                        |
| Outputs        | `accepted_problems.jsonl` / `rejected_problems.jsonl`                                          | `accepted_problems_codex.jsonl` / `rejected_problems_codex.jsonl`                    |
| Override env   | `CLAUDE_BIN`, `CLAUDE_MODEL`, `CLAUDE_TIMEOUT`, `CLAUDE_EFFORT`                                | `CODEX_BIN`, `CODEX_MODEL`, `CODEX_TIMEOUT`                                          |

### Per-section launch (work split into 13 chunks of 100)

The seed pool has 1298 candidates. `--section i` slices it deterministically
into 13 chunks so multiple people can run in parallel without collisions:
section 1 = seeds `[0, 100)`, section 2 = `[100, 200)`, …, section 13 = `[1200, 1298)`.
Slicing happens BEFORE other filters, so the section→seeds mapping is identical
across runs.

```bash
cd automated_data_gen/problem_generation

# Claude Code — section i, in-parallel 4 seeds at a time
python3 run.py       --litellm --section <i> --parallel 4

# Codex CLI — same shape, different proposer
python3 run_codex.py --litellm --section <i> --parallel 4
```

`accepted_problems[_codex].jsonl` accumulates at the directory root; concatenating
across people/sections at the end is safe. Use `--resume` to skip seeds whose
`final.json` already exists if a section gets interrupted.

### Where API keys go

Shared credentials (both proposers, both code paths) — gitignored files
under `<repo_root>/api_keys/`, auto-loaded by `_load_env.py` at startup:

| File                              | Purpose                                                |
|-----------------------------------|--------------------------------------------------------|
| `api_keys/api_key_perplexity.txt` | Perplexity search (default search engine)              |
| `api_keys/litellm.txt`            | LiteLLM proxy key (solver + grader with `--litellm`)   |
| `api_keys/base_url.txt`           | LiteLLM proxy base URL                                 |
| `api_keys/gemini.txt`             | Gemini solver key                                      |

Proposer auth (only the one you're using):

- **Claude Code** — `export ANTHROPIC_API_KEY=…` OR run `claude login` once (writes
  `~/.claude/.credentials.json`) OR rely on the macOS Keychain entry.
- **Codex** — `export OPENAI_API_KEY=…` (or `CODEX_API_KEY=…`) OR run `codex login` once
  (writes `~/.codex/auth.json`).

If you swap to a non-default search engine (`brave` / `exa` / `tavily`), also
export the matching key (`BRAVE_API_KEY`, `EXA_API_KEY`, `TAVILY_API_KEY`).

## Files

| File | Role |
|---|---|
| `run.py` | Orchestrator: loads seeds (with original-problem provenance), renders each proposer brief, spawns one Claude Code per seed, collects + enriches `final.json`, appends to `accepted_problems.jsonl` / `rejected_problems.jsonl`. |
| `task_spec_template.md` | Proposer brief. Drives the 4-attempt loop with the 3-gate test, FM-classification honesty guardrail, and per-iter evolution rules. |
| `solver_runner.py` | CLI wrapper around `search_evals.agents.DeepResearchAgent` + grader. Supports `--mode {adversarial,oracle}` and `--oracle-page-path` (strips HTML to text via BeautifulSoup). |
| `perplexity_check.py` | Pre-flight sanity gate. LLM-drafts 4 Perplexity queries, runs them, substring-matches the known answer against SERP text, writes `sanity_check.json` with `verdict: TOO_EASY|PROCEED`. |
| `litellm_grader.py` | Grader for the `--litellm` code path. Mirrors `DeepResearchGrader` but routes through `litellm.acompletion()`. |
| `monitor.py` | Inspector for per-seed workspaces: `list`, `show [<ws>\|--latest] [--trace]`, `tail [<ws>\|--latest]`. |
| `_load_env.py` | Reads API credentials from `<repo_root>/api_keys/{api_key_perplexity,litellm,base_url,gemini}.txt` and exports to env. |

## The 3-gate test (per attempt)

```
Attempt N (iter{N-1}):
  ├─ proposer drafts attempt_N/problem.json (Korean question + answer + target FMs)
  │
  ├─ Gate 1: sanity (perplexity_check.py)
  │     LLM-generated SERP queries → run on Perplexity → substring-match the answer
  │     │
  │     ├─ verdict = TOO_EASY  → revise (skip Gates 2 & 3)
  │     └─ verdict = PROCEED   → continue
  │
  ├─ Gate 2: oracle (solver_runner.py --mode oracle --oracle-page-path …)
  │     Solver gets the page (HTML stripped to text, ~80K chars) + question
  │     │
  │     ├─ NOT solved → problem is malformed, answer not extractable → revise
  │     └─ SOLVED     → continue
  │
  └─ Gate 3: adversarial, MULTI-MODEL (solver_runner.py --mode adversarial)
        Run once per target model (default: gpt-5.4-mini + gemini/gemini-3-flash-preview)
        │
        ├─ ANY model SOLVED → too easy for that model → revise
        ├─ ALL models failed for non-target FM → wrong reason → revise honestly
        └─ ALL models failed for a target FM → ACCEPT (write final.json, exit)
```

The brief enforces an **honesty guardrail**: if Gate 3 fails because the
target page never appeared in SERP (F10/F9), the proposer must NOT
fake-classify it as F4. It either re-targets the FM in `final.json` or
revises the problem so the failure manifests as the originally-targeted
FM (typically by choosing a detail whose page IS reachable in SERP).

## Per-seed workspace layout

```
work/<source_class>/<sha1(url)[:10]>/<run_id>/
  seed.json                  cached seed entry, augmented with original-problem lineage
                             (original_url, original_question, original_question_id, …)
  task_spec.md               rendered proposer brief
  proposer_stream.jsonl      raw stream-json from Claude Code (one event per line)
  proposer_log.txt           human-readable mirror of proposer stdout
  attempt_1/                 iter0
    problem.json             proposer's draft (problem, answer, target_failure_modes, …)
    sanity/sanity_check.json Gate 1 output
    oracle/                  Gate 2 output (single model — the Oracle reference model)
      summary.json
      solver_convo.json
      solver_response.txt
      grader_result.json
    adversarial/             Gate 3 output (one subdir per target model)
      gpt-5.4-mini/
        summary.json, solver_convo.json, solver_response.txt, grader_result.json
      gemini-3-flash-preview/
        summary.json, solver_convo.json, solver_response.txt, grader_result.json
  attempt_2/...              iter1
  attempt_3/...               iter2
  attempt_4/...               iter3
  final.json                 written by the proposer, then enriched by the orchestrator
                             with full lineage (see schema below)
```

After each invocation, the orchestrator appends each seed's outcome to:
- `accepted_problems.jsonl` — accepted problems (one JSON per line)
- `rejected_problems.jsonl` — seeds that couldn't yield a defeating problem

## Result schema

Each row in `accepted_problems.jsonl` (and the per-workspace `final.json`)
has these lineage + result fields:

```jsonc
{
  // Proposer's decision
  "accepted": true,
  "rationale": "Clean F4. Gates: Gate 1 PROCEED, Gate 2 SOLVED, Gate 3 all failed for F4 reason.",
  "gates": { "sanity": {...}, "oracle": {...}, "adversarial": { "<model>": {...}, ... } },

  // Original (1 of 300 hand-curated K-BrowseComp problems whose source URL seeded this)
  "original_url":         "https://species.nibr.go.kr/...",
  "original_question":    "...",
  "original_answer":      "...",
  "original_question_id": "seed_0239",
  "all_question_ids":     ["seed_0239"],   // if multiple originals share this URL

  // Seed (proposed by seed_expansion)
  "seed_url":          "https://www.mbris.kr/...?spcTxnId=270000005487",
  "seed_axis":         "intra",
  "seed_source_class": "species_inventory",

  // Proposer's drafts across iterations
  "proposed_url":      "https://www.mbris.kr/...?spcTxnId=270000005487",
  "proposed_question": { "iter0": "...", "iter1": "...", ... },
  "proposed_answer":   { "iter0": "...", "iter1": "...", ... },
  "attempts_made":     2,

  // Multi-model verification
  "tested_models":            ["gpt-5.4-mini", "gemini-3-flash-preview"],
  "adversarial_results_by_iter": {
    "iter0": { "gpt-5.4-mini": <summary>, "gemini-3-flash-preview": <summary> },
    "iter1": { "gpt-5.4-mini": <summary>, "gemini-3-flash-preview": <summary> }
  },

  // Bookkeeping
  "workspace": "/.../work/species_inventory/e63cbf0481/20260511T...",
  "seed_id":   "e63cbf0481",
  "source_class": "species_inventory"
}
```

## Credentials

`_load_env.py` reads gitignored files from `<repo_root>/api_keys/` and
populates env. The mirror to `OPENAI_*` lets the non-litellm code path
(which uses `AsyncOpenAI`) route through the same LiteLLM proxy via the
OpenAI-compatible chat-completions endpoint.

| File | Env var(s) | Used for |
|---|---|---|
| `api_keys/api_key_perplexity.txt` | `PERPLEXITY_API_KEY` | Search engine (default: perplexity) |
| `api_keys/litellm.txt`            | `LITELLM_PROXY_API_KEY` + `OPENAI_API_KEY` | Solver + grader, both code paths |
| `api_keys/base_url.txt`           | `LITELLM_PROXY_BASE_URL` + `OPENAI_BASE_URL` | LiteLLM proxy URL |
| `api_keys/gemini.txt`             | `GEMINI_API_KEY` + `GOOGLE_API_KEY` | Gemini target model |

Explicit shell exports always win over the txt files. The orchestrator also
verifies `ANTHROPIC_API_KEY` or `~/.claude/.credentials.json` (or macOS
Keychain) for the proposer's Claude Code subprocess.

If you swap to other search engines (`brave` / `exa` / `tavily`) you'll need
the corresponding key (`BRAVE_API_KEY`, `EXA_API_KEY`, `TAVILY_API_KEY`).

## Usage

```bash
cd automated_data_gen/problem_generation

# Sanity check: 1 seed, multi-model adversarial (gpt-5.4-mini + gemini-3-flash-preview)
python3 run.py --litellm --max-seeds 1 \
  --source-classes species_inventory --target-failure-modes F4

# Smoke test with a local/private seed file instead of generated aggregate data
python3 run.py --dry-run --seed-json /tmp/local_seed.json --max-seeds 1

# Single specific seed by id (sha1(url)[:10])
python3 run.py --litellm --seed-id 1a2b3c4d

# Restrict by axis (intra / inter) and failure modes
python3 run.py --litellm --axes intra --target-failure-modes F4,F6 --max-seeds 5

# Concurrent (interleaved logs)
python3 run.py --litellm --parallel 4

# Resume — skip seeds that already have a prior final.json
python3 run.py --litellm --resume

# Custom target models for Gate 3 (comma-list — must ALL fail for acceptance)
python3 run.py --litellm \
  --target-models "azure_ai/gpt-5.4-mini,gemini/gemini-3-flash-preview,anthropic/claude-sonnet-4-6"

# Custom Oracle (well-formedness reference) model
python3 run.py --litellm --solver-model azure_ai/gpt-5.4-mini

# Dry-run — render task_spec.md per seed without launching claude
python3 run.py --litellm --dry-run --max-seeds 2

# Non-litellm code path (direct OpenAI + Gemini; needs OPENAI_API_KEY and GEMINI_API_KEY)
python3 run.py --max-seeds 1
```

Configurable env vars:

| Var | Default | Purpose |
|---|---|---|
| `CLAUDE_BIN` | `claude` | Path to claude binary |
| `CLAUDE_TIMEOUT` | `3600` | Per-seed timeout (seconds) |
| `CLAUDE_VERSION` | `2.1.138` | Expected claude version (warning only) |
| `CLAUDE_MODEL` | `claude-opus-4-7` | Proposer model |
| `CLAUDE_EFFORT` | `max` | Proposer effort |

## Monitoring a run

`monitor.py` reads workspaces from `work/` without re-running anything:

```bash
# Tabular status of all workspaces
python3 monitor.py list

# Filter to one run_id
python3 monitor.py list --run 20260511T052406Z

# Full detail of one workspace (gates + iter history)
python3 monitor.py show <workspace>
python3 monitor.py show --latest                  # most recently modified
python3 monitor.py show --latest --trace          # + decoded solver trajectory per model

# Follow proposer events live while a run is in progress
python3 monitor.py tail --latest
```

## Cost note

Per seed (worst case 4 attempts × 3 gates × 2 target models for Gate 3):
- 1 proposer Claude Code session (`claude-opus-4-7` at max effort, plus
  WebFetch/WebSearch/curl calls and ~12 `solver_runner.py` invocations)
- 4 sanity checks (`perplexity_check.py`: 4 Perplexity SERP queries + 1 LLM
  query-generation call each)
- 4 oracle runs (single model, page in context)
- 4 × 2 = 8 adversarial runs (10 Perplexity search calls each)
- 1 grader call per solver run

Single-seed sanity check observed: ~$3–5 with 2 attempts. For 700 problems
across ~1000 seeds (assuming ~70% acceptance), budget a few thousand dollars
in API spend. Run a single seed first.

## How this differs from a vanilla "generate then test" pipeline

Three structural choices that took several iterations to land on:

1. **The proposer is a Claude Code subprocess, not a plain LLM call.**
   It has `WebFetch`, `Bash` (curl), `Read`, `Grep`, `Write`, and the
   ability to invoke `solver_runner.py` and `perplexity_check.py` via
   Bash. That gives it the same "open the page, inspect the table, write
   a question targeting a specific row cell" workflow a human author
   would use.

2. **The adversarial gate is multi-model.** A problem that defeats one
   solver but not another isn't measuring a fundamental retrieval/parsing
   pathology — it's measuring a model-specific weakness. Default targets
   are `gpt-5.4-mini` and `gemini-3-flash-preview`; configurable via
   `--target-models`. Acceptance requires ALL of them to fail.

3. **The Oracle gate catches malformed problems.** If you only run the
   adversarial test, a malformed problem (answer not actually on the page,
   ambiguous wording) reads as a clean adversarial success — false win.
   Oracle pre-loads the page text and confirms the answer is at least
   extractable when handed to the solver on a plate.

## Sanity check command (one-liner)

```bash
python3 run.py --litellm --max-seeds 1 \
  --source-classes species_inventory --target-failure-modes F4
```

When it finishes, inspect with:

```bash
python3 monitor.py show --latest --trace
```
