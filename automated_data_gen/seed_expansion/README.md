# seed_expansion

Agent-driven seed-bank expansion. For each source class in
`../seed_material/seed_bank.json`, a headless Claude Code subprocess is
launched in a class-specific workspace. The agent reads a generated brief
(`task_spec.md`), uses `WebSearch` + `WebFetch` + `Bash` (`curl`) to discover
candidate URLs on two axes, validates each URL against structural-similarity
criteria, and writes JSON. The orchestrator then aggregates per-class
outputs into the top-level files.

## The two axes

Both axes look for Korean websites on hosts **not in the seed class's seed
bank**. They differ in whether content topic is held constant:

- **Intra-archetype** (`intra.json`): same kind of content as the seeds, just
  on different hosts. Same retrieval pathology, same failure modes. For a
  class like `species_inventory` (other Korean biodiversity portals such as
  mbris.kr, kbr.go.kr, naturing.net, fbp.nnibr.re.kr).
- **Inter-archetype** (`inter.json`): same failure modes as the seed class
  but content can be entirely different — "weird corners of the Korean web"
  that solvers struggle to reach (regional cultural encyclopedias, cultural
  property archives, statistical-table portals, regulatory annexes, etc.).

The intent is twofold: intra makes the seed pool host-diverse without
changing the difficulty profile; inter expands into adjacent retrieval
pathologies so downstream problem generation isn't confined to one content
domain.

## Inputs (read from `../seed_material/`)

- `seed_bank.json` — labeled source-class taxonomy with hosts, sample URLs,
  empirical failure-mode distribution per class.
- `failure_mode_exemplars.json` — used in `task_spec_template.md` indirectly
  (failure-mode names are baked into the rendered brief).
- `seed_questions.json` / `failure_summary_*.md` — available to the agent but
  not required.

## Outputs

Generated outputs are local/private artifacts and are gitignored in the public
repo.

- **Per-run workspaces** at `work/<class_name>/<run_id>/`:
  - `task_spec.md` — the rendered agent brief
  - `class_context.json` — machine-readable class metadata
  - `intra.json`, `inter.json` — agent's per-class outputs for this run

  `run_id` defaults to the current UTC timestamp (e.g. `20251210T143052Z`), so
  reruns never clobber prior results. Override with `--run-id <string>` to
  share a directory across multiple `run.py` invocations within one logical
  run.
- **Aggregated top-level files** (rewritten each invocation):
  - `intra_domain_candidates.json` — every entry across every `(class, run_id)`,
    deduplicated by `website_url` (later run_ids win on conflict). Each entry
    is annotated with `source_class` and `run_id`.
  - `inter_domain_candidates.json` — same, inter-domain.

Each entry carries the schema enforced by `task_spec_template.md`:
`{source, website_url, website_summary, why_this_is_promising,
target_failure_modes, verification: {method, http_status, evidence}}`
plus the `source_class` and `run_id` fields the aggregator attaches.

## What Stage 3 (problem_generation) needs from here

For each seed entry, Stage 3's proposer/3-gate pipeline reads:
- `website_url` — starting page (the URL the proposer will draft problems from)
- `website_summary` + `verification.evidence` — page context (real load-bearing
  content the agent already extracted, so the proposer can draft a question
  before re-fetching)
- `target_failure_modes` + `why_this_is_promising` — which failure mode the
  proposer should engineer toward
- `source_class` — class-level metadata via `seed_material/seed_bank.json`
- `run_id` — traceability
- `source` — the URL of the original human-crafted K-BrowseComp problem (one of
  the 300) that this seed was generated from. Stage 3's orchestrator joins
  this URL back against `seed_material/seed_url_index.json` +
  `seed_material/seed_questions.json` to attach `original_url`,
  `original_question`, `original_answer`, `original_question_id` to every
  result row.

Per-class metadata and failure-mode exemplars come from
`../seed_material/seed_bank.json` and `../seed_material/failure_mode_exemplars.json`.
The seed pool for Stage 3 is the union of `intra_domain_candidates.json` and
`inter_domain_candidates.json`.

## Usage

```bash
# Run all classes sequentially (default targets: 15 intra + 15 inter per class)
python3 automated_data_gen/seed_expansion/run.py

# Only a couple of classes
python3 automated_data_gen/seed_expansion/run.py \
  --classes species_inventory,encyclopedia_korean_culture

# Higher per-class targets
python3 automated_data_gen/seed_expansion/run.py \
  --target-intra 25 --target-inter 25

# Concurrent (up to 4 agents at a time; output lines will interleave)
python3 automated_data_gen/seed_expansion/run.py --parallel 4

# Resume — skip classes that already have at least one valid prior run
python3 automated_data_gen/seed_expansion/run.py --resume

# Reuse a run_id across multiple invocations (e.g. across tiers)
python3 automated_data_gen/seed_expansion/run.py --run-id batch-v2 --classes ...

# Aggregate only — useful after manually editing work/<class>/<run_id>/{intra,inter}.json
python3 automated_data_gen/seed_expansion/run.py --aggregate-only

# Dry-run — render briefs into work/<class>/<run_id>/task_spec.md without launching claude
python3 automated_data_gen/seed_expansion/run.py --dry-run
```

Configurable via env vars:

| Var | Default | Purpose |
|---|---|---|
| `CLAUDE_BIN` | `claude` | Path to claude binary |
| `CLAUDE_TIMEOUT` | `7200` | Per-class timeout in seconds |
| `CLAUDE_VERSION` | `2.1.138` | Expected claude version (warning only) |
| `CLAUDE_MODEL` | `claude-opus-4-7` | Model for the agent |
| `CLAUDE_EFFORT` | `max` | Effort level |

## Agent behavior

Each agent gets a class-specific brief that includes:
- The class's retrieval hypothesis and prior failure-mode hypotheses
- The empirical failure-mode histogram (computed from joining the seed bank
  with both solver-model classifications)
- The top hosts and 8 reference URLs
- Concrete discovery strategies for intra and inter axes
- A **mandatory verification protocol**: every entry must include a
  `verification` object recording the HTTP status the agent observed and a
  100–300 character distinguishing snippet from the page. Sitemaps and
  indexes are preferred over ID-pattern guessing. URLs with non-2xx/3xx
  status, DNS errors, or boilerplate-only content must be dropped.
- An exact output schema enforced by both the agent and the validator

The agent has `--dangerously-skip-permissions` so it can use `WebSearch`,
`WebFetch`, `Bash`, etc. without confirmation prompts. Run in a controlled
environment.

## Aggregator behavior

After all per-class runs, the orchestrator walks `work/<class>/<run_id>/intra.json`
and `work/<class>/<run_id>/inter.json` across **all** run_ids per class,
deduplicates by `website_url` (later run_ids win on conflict), attaches
`source_class` and `run_id`, and writes the two top-level aggregates. This
step is rerunnable via `--aggregate-only`.

## Cost note

With 36 classes × `claude-opus-4-7` at max effort, the full run is expensive
(each class likely uses dozens of `WebSearch` / `WebFetch` calls). Recommended
warm-up:

1. Run on 2–3 classes first (`--classes species_inventory,encyclopedia_korean_culture,sports_records`)
   to sanity-check the brief and output quality.
2. Inspect the per-class outputs in `work/`.
3. Iterate on `task_spec_template.md` if needed.
4. Then run the full set, optionally `--parallel 4` for throughput.

## Out of scope here

- No problem generation. That happens in `../problem_generation/`.
- No structural validator beyond what the agent does in-line. A separate
  post-hoc validator can be added if needed.
