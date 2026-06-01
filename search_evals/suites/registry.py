# Portions adapted from https://github.com/perplexityai/search_evals/blob/main/search_evals/suites/registry.py
# Original license: MIT.

from pathlib import Path

from search_evals.suites.graders import DEFAULT_GRADER_MODEL
from search_evals.suites.ko_browsecomp import KoBrowseCompSuite
from search_evals.suites.types import AsyncBaseSuite

SUITES: dict[str, type[AsyncBaseSuite]] = {
    "ko_browsecomp": KoBrowseCompSuite,
}


def make_suite(
    suite: str,
    search_engine: str,
    model: str,
    output_dir: Path,
    dry_run: bool = False,
    max_workers: int = 3,
    grader_model: str = DEFAULT_GRADER_MODEL,
    dataset_path: Path | None = None,
) -> AsyncBaseSuite:
    """Create a suite instance by name."""
    suite_class = SUITES.get(suite)
    if suite_class is None:
        available = ", ".join(sorted(SUITES))
        raise ValueError(f"Suite '{suite}' not found. Available suites: {available}")
    kwargs = {
        "search_engine": search_engine,
        "model": model,
        "output_dir": output_dir,
        "dry_run": dry_run,
        "max_workers": max_workers,
        "grader_model": grader_model,
    }
    if dataset_path is not None:
        kwargs["dataset_path"] = dataset_path
    return suite_class(**kwargs)
