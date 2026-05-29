from pathlib import Path

from search_evals.datasets.dataset import Dataset, Datum

DATASETS_DIR = Path(__file__).parent

KO_BROWSECOMP = DATASETS_DIR / "ko_browsecomp.jsonl"
KO_BROWSECOMP_SYNTHETIC = DATASETS_DIR / "ko_browsecomp_synthetic.jsonl"

__all__ = [
    "KO_BROWSECOMP",
    "KO_BROWSECOMP_SYNTHETIC",
    "Dataset",
    "Datum",
]
