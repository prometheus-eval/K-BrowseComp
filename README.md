# Ko-BrowseComp Search Evals

Evaluation and automated problem-generation code for Ko-BrowseComp.

This repository is adapted from
[perplexityai/search_evals](https://github.com/perplexityai/search_evals).

This public repo includes only the Ko-BrowseComp datasets:

- `search_evals/datasets/ko_browsecomp.jsonl`
- `search_evals/datasets/ko_browsecomp_synthetic.jsonl`

Generated run outputs, model trajectories, seed-bank artifacts, local API keys,
virtual environments, and logs are ignored by git.

## Setup

```bash
uv sync
```

Python dependencies are managed with `uv`. Most commands below should be run
from the repository root.

## Credentials

`search_evals/run_eval.py` reads credentials from environment variables.

```bash
export OPENAI_API_KEY="..."
export PERPLEXITY_API_KEY="..."
```

Use the matching search key for the search engine you select:

| Search engine | Required env var |
|---|---|
| `perplexity` / `perplexity-long` | `PERPLEXITY_API_KEY` |
| `brave` | `BRAVE_API_KEY` |
| `exa` | `EXA_API_KEY` |
| `tavily` | `TAVILY_API_KEY` |

Model keys depend on the model prefix:

| Model prefix | Required env var |
|---|---|
| `gpt...` | `OPENAI_API_KEY` |
| `gemini...` | `GEMINI_API_KEY` or `GOOGLE_API_KEY` |
| `claude...` | `ANTHROPIC_API_KEY` |
| `openrouter:...` | `OPENROUTER_API_KEY` |
| `litellm_proxy/...` | `LITELLM_PROXY_BASE_URL` and `LITELLM_PROXY_API_KEY` |

For `automated_data_gen/problem_generation/`, gitignored files under
`api_keys/` are also auto-loaded by `_load_env.py`; see the autogen section
below.

## Run Evaluations

Run a small live sample:

```bash
uv run python search_evals/run_eval.py \
  search_engine=perplexity \
  model=gpt-5.4-mini \
  suite=ko_browsecomp \
  dry_run=true \
  run_suffix=smoke
```

For KO-BrowseComp, `dry_run=true` runs the first 10 samples and still calls the
model, search engine, and grader APIs. Use `run_suffix` to make the output
folder easier to distinguish.

Run the full default dataset:

```bash
uv run python search_evals/run_eval.py \
  search_engine=perplexity \
  model=gpt-5.4-mini \
  suite=ko_browsecomp
```

Run the synthetic dataset:

```bash
uv run python search_evals/run_eval.py \
  search_engine=perplexity \
  model=gpt-5.4-mini \
  suite=ko_browsecomp \
  dataset_path=search_evals/datasets/ko_browsecomp_synthetic.jsonl \
  dry_run=true \
  run_suffix=synthetic-smoke
```

Outputs are written under:

- per-question task files: `runs/<search>-<model>_<suite>[-suffix]/`
- aggregate score: `runs/results/<search>-<model>_<suite>[-suffix].json`

By default, existing per-question task files are reused. To delete cached files
for a run and start over:

```bash
uv run python search_evals/run_eval.py \
  search_engine=perplexity \
  model=gpt-5.4-mini \
  suite=ko_browsecomp \
  dry_run=true \
  run_suffix=smoke \
  rerun=true
```

## Automated Problem Generation

Autogen code lives under `automated_data_gen/`.

The main Stage 3 problem-generation entry point is:

```bash
cd automated_data_gen/problem_generation
python3 run.py --litellm --max-seeds 1
```

Default adversarial target models are:

- `azure_ai/gpt-5.4-mini` when `--litellm` is enabled; otherwise `gpt-5.4-mini`
- `gemini/gemini-3-flash-preview`

Acceptance requires every target model to fail for a target failure mode. You
can override the list:

```bash
python3 run.py --litellm \
  --target-models "azure_ai/gpt-5.4-mini,gemini/gemini-3-flash-preview"
```

Autogen credentials can be exported as environment variables or placed in
gitignored files:

| File | Env var(s) populated |
|---|---|
| `api_keys/api_key_perplexity.txt` | `PERPLEXITY_API_KEY` |
| `api_keys/litellm.txt` | `LITELLM_PROXY_API_KEY`, `OPENAI_API_KEY` |
| `api_keys/base_url.txt` | `LITELLM_PROXY_BASE_URL`, `OPENAI_BASE_URL` |
| `api_keys/gemini.txt` | `GEMINI_API_KEY`, `GOOGLE_API_KEY` |

Autogen `--dry-run` has a different meaning from eval `dry_run=true`: it only
renders per-seed `task_spec.md` files and does not launch the proposer loop.

```bash
cd automated_data_gen/problem_generation
python3 run.py --dry-run --seed-json /path/to/local_seed.json --max-seeds 1
```

Local/generated seed material is not part of the public repo. If you have a
private seed file, pass it with `--seed-json`; otherwise Stage 3 looks for the
gitignored Stage 2 aggregate files under `automated_data_gen/seed_expansion/`.

## Tests

```bash
uv run --extra dev pytest
```

