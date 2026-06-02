# K-BrowseComp

## Introduction
K-BrowseComp is a Korean web-browsing agent benchmark inspired by
[BrowseComp](https://arxiv.org/abs/2504.12516). 
It evaluates whether agents can retrieve hard-to-find public information from Korean websites, keep track of multi-hop or parallel evidence, and return a single stable short answer grounded in Korean contexts.

📄 Paper: [K-BrowseComp](https://arxiv.org/abs/2606.02404)

🤗 Dataset: [K-BrowseComp](https://huggingface.co/datasets/prometheus-eval/k-browsecomp)

K-BrowseComp contains **400 problems**:

- **K-BrowseComp-Verified**: 300 problems manually written and validated by native Korean speakers.
- **K-BrowseComp-Synthetic**: 100 machine-generated diagnostic problems created with hard few-shot exemplars and failure-mode-targeted generation. (See `automated_data_gen/` for the generation pipeline.)

Each verified item is designed to require either **multi-hop reasoning** or **parallel-branching** constraint satisfaction over public Korean web evidence. 
The dataset release includes the *problem, gold answer, expected trajectory, source URLs, and checklist values for key intermediate evidence*, enabling trajectory diagnostics in addition to final-answer scoring.

In our experiments, even the strongest evaluated model remains below 50% on K-BrowseComp-Verified and reaches only 26% on the Synthetic split.

<p align="center">
  <img src="misc/error_example.png" alt="Representative trajectory-level failures in K-BrowseComp" width="100%">
</p>

The benchmark therefore tests more than whether a model can search in Korean: it tests whether the model can preserve the exact evidence chain needed to answer Korean users' browsing questions reliably.


## K-BrowseComp Search Evals

Evaluation and automated problem-generation code for K-BrowseComp.

This repository is adapted from
[perplexityai/search_evals](https://github.com/perplexityai/search_evals).

Runtime evals download K-BrowseComp from
[prometheus-eval/k-browsecomp](https://huggingface.co/datasets/prometheus-eval/k-browsecomp)
by default. The repo also keeps fallback local JSONL copies:

- `search_evals/datasets/ko_browsecomp.jsonl`
- `search_evals/datasets/ko_browsecomp_synthetic.jsonl`

Generated run outputs, model trajectories, seed-bank artifacts, local API keys,
virtual environments, and logs are ignored by git.

## License and Attribution

Unless otherwise noted, this repository is distributed under the MIT License;
see [LICENSE](LICENSE).

The evaluation framework is adapted from
[perplexityai/search_evals](https://github.com/perplexityai/search_evals),
which is also MIT-licensed. Files adapted from that repository include
file-level source comments at the top of the file.

## Setup

```bash
uv sync
```

Python dependencies are managed with `uv`. Most commands below should be run
from the repository root.

## Colab Quickstart

Run the Colab notebook to evaluate `gpt-5-nano` on a dry-run K-BrowseComp
sample with only notebook cells. The notebook downloads the dataset from
[prometheus-eval/k-browsecomp](https://huggingface.co/datasets/prometheus-eval/k-browsecomp):

[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/prometheus-eval/K-BrowseComp/blob/main/examples/k_browsecomp_colab.ipynb)

The first cell is the only cell you need to edit. It asks for `OPENAI_API_KEY`
and one search API key for the selected search engine.

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

By default, `suite=ko_browsecomp` loads the `verified` config and `test` split
from [prometheus-eval/k-browsecomp](https://huggingface.co/datasets/prometheus-eval/k-browsecomp).

Run a small live sample:

```bash
uv run python search_evals/run_eval.py \
  search_engine=perplexity \
  model=gpt-5.4-mini \
  suite=ko_browsecomp \
  dry_run=true \
  run_suffix=smoke
```

For K-BrowseComp, `dry_run=true` runs the first 10 samples and still calls the
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
  hf_dataset_config=synthetic \
  dry_run=true \
  run_suffix=synthetic-smoke
```

You can still evaluate a local JSONL file by passing `dataset_path=...`; when
`dataset_path` is set, the Hugging Face dataset settings are ignored.

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
