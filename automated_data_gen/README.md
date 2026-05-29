# automated_data_gen

Three-stage pipeline for synthesizing additional KO BrowseComp-style
problems. The code is public; generated seed material, trajectories, run
workspaces, and aggregate outputs are intentionally gitignored.

## Stages

1. `mine_seed_bank.py` builds source-class metadata from private/local seed
   material and writes generated artifacts under `seed_material/`.
2. `seed_expansion/run.py` expands source classes into candidate Korean web
   pages and writes candidate JSON.
3. `problem_generation/run.py` turns candidates into adversarial problems via
   sanity, oracle, and multi-model adversarial gates.

## Problem Generation Defaults

The adversarial gate tests every candidate against both default target models:

- `gpt-5.4-mini`
- `gemini/gemini-3-flash-preview`

With `--litellm`, the GPT target is `azure_ai/gpt-5.4-mini`; the Gemini target
remains `gemini/gemini-3-flash-preview`.

## Local Credentials

Credentials may be exported in the shell or placed under the gitignored
repo-root `api_keys/` directory:

- `api_keys/api_key_perplexity.txt` -> `PERPLEXITY_API_KEY`
- `api_keys/litellm.txt` -> `LITELLM_PROXY_API_KEY` and `OPENAI_API_KEY`
- `api_keys/base_url.txt` -> `LITELLM_PROXY_BASE_URL` and `OPENAI_BASE_URL`
- `api_keys/gemini.txt` -> `GEMINI_API_KEY` and `GOOGLE_API_KEY`

## Smoke Checks

Render a Stage 3 brief without launching a proposer:

```bash
cd automated_data_gen/problem_generation
python3 run.py --dry-run --max-seeds 1
```

Run one Codex-proposer dry run:

```bash
cd automated_data_gen/problem_generation
python3 run_codex.py --dry-run --max-seeds 1
```
