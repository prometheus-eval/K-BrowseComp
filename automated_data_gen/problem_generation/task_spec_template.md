# Adversarial K-BrowseComp problem generation

You are designing **ONE** adversarial K-BrowseComp problem from a single
Korean website. Your goal: produce a question whose answer lives on that
page and which a frontier search-augmented LLM (the SOLVER) cannot answer
correctly via web search alone.

You drive the entire 4-attempt loop yourself in this single Claude Code
session. The orchestrator just spawned you and collects your `final.json`
at the end. You have unrestricted access to `WebFetch`, `WebSearch`, `Bash`
(curl, python), `Read`, `Write`, `Edit`, etc.

## Seed

- **URL**: {seed_url}
- **Source class**: `{source_class}` ({axis})
- **Target failure modes**: {target_failure_modes}
- **Page summary** (from seed expansion): {seed_summary}
- **Why this seed is promising**: {why_this_is_promising}

### Load-bearing content already observed on the page
```
{seed_evidence}
```

## Failure mode glossary (which solver behaviors count as "failed for the right reason")

{failure_mode_glossary}

## Pipeline you must drive

You will run **at most {max_attempts} attempts**. Each attempt has the same
three phases: draft → test → analyze. Stop the moment the solver fails for a
target failure mode (no need to "improve" past acceptance). After the final
attempt, write `./final.json` and exit.

### Step 0 — Open the page

Call `WebFetch` on the seed URL. If it returns only page chrome or is
blocked, fall back to `Bash`:

```bash
curl -sSL -m 30 \\
  -A 'Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 Chrome/130.0.0.0 Safari/537.36' \\
  -o /tmp/page.html '{seed_url}'
```

then `Read` `/tmp/page.html`. Identify the structured content (table rows,
notice fields, sub-sections, label-value pairs) you might use as the answer.

### Step 1 — Attempt N: draft a problem

Pick a small, hidden, uniquely-identifying textual detail from the page
that:
- Aligns with the target failure modes ({target_failure_modes})
- Is NOT visible in standard SERP snippets
- Is a single, objectively verifiable fact
- Is a row cell / label-value / sub-section value (not a page title or
  obvious metadata)

Design the question **backwards** from that target detail:
- The answer is the target detail
- The question requires multi-step retrieval — broad indirect constraints
  that intersect at this page
- The question is in **Korean** (this is K-BrowseComp)
- The question must NOT include:
  - The target detail or any near-paraphrase
  - The source URL or the domain name
  - The most identifying entity name from the page (that's the giveaway
    path the solver will use)

Write to `./attempt_N/problem.json`:

```json
{{
  "problem": "<the question, in Korean>",
  "answer": "<the exact target detail>",
  "target_detail_on_page": "<verbatim quote from the page>",
  "source_url": "{seed_url}",
  "target_failure_modes": [...],
  "rationale": "<one sentence: why this should defeat the solver via FM X>"
}}
```

### Step 2 — Attempt N: run the 3-gate test

Each attempt goes through **three gates in order**. Stop at the first
revision signal — do NOT continue past a failed gate.

The solver gets 10 search calls and is the standard K-BrowseComp deep
research agent (same one this benchmark is evaluating). It has only a
`search_web` tool (Perplexity SERP — no page-fetch), so a "failure" can
be either (a) the answer isn't reachable via SERP at all (F9/F10) or
(b) the answer is reachable but the solver couldn't extract it
(F4/F6/F7/F8). The gates below disambiguate.

**You must use `uv run python` (not plain `python`)** so the
search_evals project venv is activated.
{litellm_mode_note}

#### Gate 1 — Pre-flight Perplexity sanity check (cheap, ~5s)

Confirm Perplexity does not already serve the answer directly via SERP.
If it does, the problem is trivially solvable and you should revise
immediately — skip Gates 2/3.

```bash
cd {repo_root} && uv run python {perplexity_check_path}{litellm_flag} \\
  --problem $(pwd)/attempt_N/problem.json \\
  --output-dir $(pwd)/attempt_N/sanity
```

Read `./attempt_N/sanity/sanity_check.json`:
- `verdict: "TOO_EASY"` (i.e., the answer string appears in some
  Perplexity SERP snippet) → **revise.** Pick a deeper detail from
  the page that snippets don't expose. Go to Step 4 (do not run
  Gate 2 or Gate 3 — you'd just confirm what we already know).
- `verdict: "PROCEED"` → continue to Gate 2.

#### Gate 2 — Oracle solver test (medium, ~30s)

Pass the page contents directly to the solver alongside the question.
If the solver fails *even with the page in context*, the problem is
malformed: the answer is not actually extractable from the page you
chose, or the question's wording doesn't uniquely identify the row.

Make sure the page HTML you read in Step 0 is still on disk at the
path you pass below (e.g. `/tmp/page.html`).

```bash
cd {repo_root} && uv run python {solver_runner_path}{litellm_flag} \\
  --mode oracle \\
  --oracle-page-path /tmp/page.html \\
  --problem $(pwd)/attempt_N/problem.json \\
  --output-dir $(pwd)/attempt_N/oracle \\
  --search-engine {search_engine} \\
  --solver-model {solver_model} \\
  --grader-model {grader_model}
```

Read `./attempt_N/oracle/summary.json`:
- `solved: false` → **revise.** The page doesn't actually contain
  the answer in extractable form, or the question is ambiguous. Pick a
  different detail, or rewrite the question to uniquely target the row.
  Go to Step 4.
- `solved: true` → great, problem is well-formed. Continue to Gate 3.

#### Gate 3 — Adversarial solver test, MULTI-MODEL (~30s × N models)

The real test. Run the adversarial problem against **every** target
solver model below. Each model gets its own output subdir under
`attempt_N/adversarial/<model_slug>/`. Acceptance requires **ALL**
target models to fail.

Target models for this run: **{target_models_summary}**

Run each block in order:

```bash
{target_models_bash_block}
```

After all blocks complete, read each
`./attempt_N/adversarial/<slug>/summary.json`:
- If **ANY** model has `solved: true` → problem is too easy for at
  least one model. **Revise.** Go to Step 4.
- If **EVERY** model has `solved: false` → continue to Step 3 to
  analyze each model's failure mode (each must fail for a target FM
  reason — otherwise revise honestly).

Each `solver_runner.py` invocation produces, in its `--output-dir`:
- `solver_convo.json` — full trajectory (every query + every result)
- `solver_response.txt` — solver's final text answer
- `grader_result.json` — CORRECT / INCORRECT / NOT_ATTEMPTED + reasoning
- `summary.json` — `{{mode, solved, grade, solver_response, ...}}`

(Use absolute paths for `--problem` / `--output-dir` since the `cd
{repo_root}` switches working directory. The orchestrator put you in
the seed's workspace at `$(pwd)`.)

### Step 3 — Attempt N: analyze each model's adversarial trajectory

Only reached when **Gate 1 = PROCEED, Gate 2 = solved, Gate 3 = ALL models NOT solved**.
For **each** target model, read its
`./attempt_N/adversarial/<slug>/solver_convo.json` and answer:

**Did this model fail for the targeted failure mode ({target_failure_modes})?**

Walk through every `tool_use` query in the convo. Match the failure
pattern to one of the target FMs:
- **F4** = solver reached the right page (saw it in snippets) but
  couldn't extract the specific row cell — answered with a sibling
  cell, "I don't know", or guessed. *Note: with this solver toolkit
  (Perplexity SERP only, no page-fetch), pure F4 is hard to distinguish
  from F10 unless the page IS present in the snippets.*
- **F6** = solver got tripped on Korean variant naming / transliteration
  (e.g., searched for one transliteration of an entity name and missed
  the canonical Korean form on the source page).
- **F3** = solver kept hopping between unrelated domains, never
  converging on the target source.
- **F7** = solver had partial info, couldn't combine constraints to
  uniquely identify the row.
- **F9** = solver saw the right page in SERP but couldn't pick it out
  from competing results.
- **F10** = the target page never appeared in SERP at all (or only
  appeared as chrome-only snippets) — solver had no way to reach it.

If **every** target model's failure aligns with a target FM → **ACCEPT**.
Go to Step 5.

If **any** model failed for a non-target reason (e.g. F10 when you
targeted F4), the problem is not yet clean across all models. Either
re-target the FM in `final.json` honestly, or **revise** the problem so
the next attempt fails *for the right reason on every model*. Don't
fake-classify F10 as F4.

### Step 4 — Attempts 2, 3, 4: evolve

Use your analysis to revise. The signal that fired tells you what to change:

| Failing gate / behavior | Evolution |
|---|---|
| Gate 1 TOO_EASY (answer in Perplexity SERP) | Swap the answer for a deeper detail in the same source page (a different row/cell that SERP snippets don't expose) |
| Gate 2 oracle FAILED (answer not in page) | The detail you picked isn't actually load-bearing on the page, or the question doesn't uniquely identify the row. Pick a different cell or rewrite the question to disambiguate |
| Any target model solved Gate 3 via giveaway entity | Remove that name; replace with a broader indirect descriptor requiring multi-page correlation |
| Any target model solved Gate 3 via direct semantic search | Tighten the F4/F6 angle: ask about a row value requiring table parsing |
| Only one model solved Gate 3 (other(s) failed cleanly) | Problem is too easy for that one model. Strengthen retrieval ambiguity or pick a deeper row cell — must defeat ALL target models |
| Any model failed Gate 3 for non-target FM (e.g. F10 when targeting F4) | Either re-target FM honestly OR pick a detail whose page IS in SERP, so failure manifests as F4 not F10 |
| Adversarial response hesitated between alternatives | Tighten constraints to force a unique answer |

**Do NOT** just stack more constraints. More constraints make problems
more guessable through triangulation, not less.

Write the revised problem to `./attempt_{{N+1}}/problem.json`, re-run
all three gates (Gate 3 fans out across every target model with
`--output-dir ./attempt_{{N+1}}/adversarial/<model_slug>`), re-analyze.
Repeat until acceptance or attempt 4 exhausts.

### Step 5 — Write final.json

When you stop (acceptance or 4 attempts done), write `./final.json`:

```json
{{
  "accepted": true,
  "seed_url": "{seed_url}",
  "source_class": "{source_class}",
  "target_failure_modes": [...],
  "final_attempt_num": 1,
  "problem": <full problem object from the accepted attempt>,
  "gates": {{
    "sanity": <sanity_check.json from the accepted attempt>,
    "oracle": <oracle/summary.json from the accepted attempt>,
    "adversarial": {{
      "<model_slug_1>": <adversarial/<model_slug_1>/summary.json>,
      "<model_slug_2>": <adversarial/<model_slug_2>/summary.json>
    }}
  }},
  "rationale": "<one or two sentences: why this defeats every target model for the right reason, citing all three gate outcomes>",
  "attempts_taken": 1
}}
```

If you exhausted 4 attempts without a clean acceptance, or if the page
turned out to have no extractable load-bearing detail:

```json
{{
  "accepted": false,
  "seed_url": "{seed_url}",
  "source_class": "{source_class}",
  "target_failure_modes": [...],
  "final_attempt_num": 4,
  "attempts_taken": 4,
  "problem": <best attempt's problem, or null>,
  "reason_if_rejected": "<one sentence: which gate kept failing and why>"
}}
```

## Guardrails — read carefully

- **Stop early on success.** If attempt 1 defeats the solver for a target
  FM, write `final.json` and exit. Do NOT keep going to "improve" the
  problem.
- **Stop if irreparable.** If the page has no extractable detail, or every
  question you can write is too easy, write `final.json` with
  `accepted: false`. Don't burn 4 attempts on a hopeless seed.
- **Korean only.** Both `problem` and `answer` must be in Korean.
- **One problem per seed.** Even if multiple good details exist on the
  page, pick one and develop it across attempts. Other details can be
  used in future runs.
- **Always run all three gates via the exact commands above.** Do not
  invent a different solver setup, skip a gate, or try to predict
  outcomes without actually running them.
- **Honest FM classification.** If Gate 3 fails for a non-target FM
  (e.g. F10 when you targeted F4), DO NOT relabel the failure to match
  your target — revise the problem or re-target honestly.
- **The brief is your contract.** When in doubt, re-read Steps 3–4 and
  the failure mode glossary.

When `./final.json` exists with valid JSON, your task is done.
