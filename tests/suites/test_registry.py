# Portions adapted from https://github.com/perplexityai/search_evals/blob/main/tests/suites/test_registry.py
# Original license: MIT.

from pathlib import Path
from typing import Any

import pytest

from search_evals.datasets import Dataset
from search_evals.suites import make_suite
from search_evals.suites.ko_browsecomp import KoBrowseCompSuite


class TestMakeSuite:
    @pytest.fixture
    def mock_hf_dataset(self, monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
        captured: dict[str, Any] = {}

        def fake_from_huggingface(repo_id: str, config: str, split: str, limit: int | None) -> Dataset:
            captured.update({"repo_id": repo_id, "config": config, "split": split, "limit": limit})
            return Dataset(Path("search_evals/datasets/ko_browsecomp.jsonl"), limit=limit)

        monkeypatch.setattr(Dataset, "from_huggingface", staticmethod(fake_from_huggingface))
        return captured

    def test_invalid_name_raises_error(self, tmp_path: Path) -> None:
        with pytest.raises(ValueError, match="Available suites: ko_browsecomp"):
            make_suite("invalid", "perplexity", "gpt-4.1", output_dir=tmp_path)

    def test_creates_ko_browsecomp_suite(self, tmp_path: Path, mock_hf_dataset: dict[str, Any]) -> None:
        suite = make_suite("ko_browsecomp", "perplexity", "gpt-4.1", output_dir=tmp_path, dry_run=True, max_workers=1)

        assert isinstance(suite, KoBrowseCompSuite)
        assert suite.search_engine == "perplexity"
        assert suite.model == "gpt-4.1"
        assert suite.dry_run is True
        assert suite.max_workers == 1
        assert suite.output_dir == tmp_path
        assert mock_hf_dataset == {
            "repo_id": "prometheus-eval/k-browsecomp",
            "config": "verified",
            "split": "test",
            "limit": 10,
        }

    def test_can_use_hf_dataset_config(self, tmp_path: Path, mock_hf_dataset: dict[str, Any]) -> None:
        suite = make_suite(
            "ko_browsecomp",
            "perplexity",
            "gpt-4.1",
            output_dir=tmp_path,
            dry_run=True,
            max_workers=1,
            hf_dataset_config="synthetic",
        )

        assert isinstance(suite, KoBrowseCompSuite)
        assert mock_hf_dataset["config"] == "synthetic"

    def test_can_use_synthetic_dataset_path(self, tmp_path: Path) -> None:
        dataset_path = Path("search_evals/datasets/ko_browsecomp_synthetic.jsonl")
        suite = make_suite(
            "ko_browsecomp",
            "perplexity",
            "gpt-4.1",
            output_dir=tmp_path,
            dry_run=True,
            max_workers=1,
            dataset_path=dataset_path,
        )

        assert isinstance(suite, KoBrowseCompSuite)
        assert suite.dataset.path == dataset_path

    def test_different_search_engines(self, tmp_path: Path, mock_hf_dataset: dict[str, Any]) -> None:
        suite_perplexity = make_suite(
            "ko_browsecomp", "perplexity", "gpt-4.1", output_dir=tmp_path, dry_run=True, max_workers=1
        )
        suite_tavily = make_suite(
            "ko_browsecomp", "tavily", "gpt-4.1", output_dir=tmp_path, dry_run=True, max_workers=1
        )

        assert suite_perplexity.search_engine == "perplexity"
        assert suite_tavily.search_engine == "tavily"
