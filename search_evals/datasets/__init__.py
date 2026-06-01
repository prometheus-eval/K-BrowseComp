# Portions adapted from https://github.com/perplexityai/search_evals/blob/main/search_evals/datasets/__init__.py
# Original license: MIT.

from pathlib import Path

from search_evals.datasets.dataset import Dataset, Datum

DATASETS_DIR = Path(__file__).parent

KO_BROWSECOMP = DATASETS_DIR / "ko_browsecomp.jsonl"
KO_BROWSECOMP_SYNTHETIC = DATASETS_DIR / "ko_browsecomp_synthetic.jsonl"
KO_BROWSECOMP_HF_REPO = "prometheus-eval/k-browsecomp"
KO_BROWSECOMP_HF_CONFIG = "verified"
KO_BROWSECOMP_HF_SYNTHETIC_CONFIG = "synthetic"
KO_BROWSECOMP_HF_SPLIT = "test"

__all__ = [
    "KO_BROWSECOMP",
    "KO_BROWSECOMP_HF_CONFIG",
    "KO_BROWSECOMP_HF_REPO",
    "KO_BROWSECOMP_HF_SPLIT",
    "KO_BROWSECOMP_HF_SYNTHETIC_CONFIG",
    "KO_BROWSECOMP_SYNTHETIC",
    "Dataset",
    "Datum",
]
