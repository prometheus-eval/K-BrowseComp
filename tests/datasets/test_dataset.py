from pathlib import Path

import pytest

from search_evals.datasets import Dataset, Datum


class TestDatum:
    def test_create_from_dict_with_all_fields(self) -> None:
        data: dict[str, object] = {
            "id": "test-123",
            "problem": "What is 2+2?",
            "answer": "4",
        }
        datum = Datum.create(data)

        assert datum.id == "test-123"
        assert datum.problem == "What is 2+2?"
        assert datum.answer == "4"
        assert datum.metadata is None

    def test_create_from_dict_with_extra_fields_becomes_metadata(self) -> None:
        data: dict[str, object] = {
            "id": "test-123",
            "problem": "What is 2+2?",
            "answer": "4",
            "category": "math",
            "difficulty": "easy",
        }
        datum = Datum.create(data)

        assert datum.metadata == {"category": "math", "difficulty": "easy"}

    def test_create_coerces_non_string_values_to_strings(self) -> None:
        data: dict[str, object] = {
            "id": 123,
            "problem": "What is 2+2?",
            "answer": 4,
            "score": 0.95,
        }
        datum = Datum.create(data)

        assert datum.id == "123"
        assert datum.answer == "4"
        assert datum.metadata == {"score": "0.95"}

    def test_direct_construction(self) -> None:
        datum = Datum(id="test", problem="Question?", answer="Answer")

        assert datum.id == "test"
        assert datum.problem == "Question?"
        assert datum.answer == "Answer"
        assert datum.metadata is None

    def test_direct_construction_with_metadata(self) -> None:
        datum = Datum(id="test", problem="Q?", answer="A", metadata={"key": "value"})

        assert datum.metadata == {"key": "value"}

    def test_model_dump_roundtrip(self) -> None:
        original = Datum(id="test", problem="Q?", answer="A", metadata={"k": "v"})
        dumped = original.model_dump()
        restored = Datum.model_validate(dumped)

        assert restored == original


class TestDataset:
    @pytest.fixture
    def sample_jsonl_file(self, tmp_path: Path) -> Path:
        jsonl_content = """\
{"id": "1", "problem": "What is 1+1?", "answer": "2"}
{"id": "2", "problem": "What is 2+2?", "answer": "4", "category": "math"}
{"id": "3", "problem": "What is 3+3?", "answer": "6"}
"""
        file_path = tmp_path / "test_dataset.jsonl"
        file_path.write_text(jsonl_content)
        return file_path

    def test_load_dataset_from_file(self, sample_jsonl_file: Path) -> None:
        dataset = Dataset(sample_jsonl_file)

        assert len(dataset) == 3
        assert dataset[0].id == "1"
        assert dataset[1].id == "2"
        assert dataset[2].id == "3"

    def test_dataset_with_limit(self, sample_jsonl_file: Path) -> None:
        dataset = Dataset(sample_jsonl_file, limit=2)

        assert len(dataset) == 2
        assert dataset[0].id == "1"
        assert dataset[1].id == "2"

    def test_dataset_iteration(self, sample_jsonl_file: Path) -> None:
        dataset = Dataset(sample_jsonl_file)
        ids = [datum.id for datum in dataset]

        assert ids == ["1", "2", "3"]

    def test_dataset_indexing(self, sample_jsonl_file: Path) -> None:
        dataset = Dataset(sample_jsonl_file)

        assert dataset[0].answer == "2"
        assert dataset[-1].answer == "6"

    def test_dataset_slicing(self, sample_jsonl_file: Path) -> None:
        dataset = Dataset(sample_jsonl_file)
        subset = dataset[1:3]

        assert len(subset) == 2
        assert subset[0].id == "2"
        assert subset[1].id == "3"

    def test_dataset_preserves_metadata(self, sample_jsonl_file: Path) -> None:
        dataset = Dataset(sample_jsonl_file)

        assert dataset[0].metadata is None
        assert dataset[1].metadata == {"category": "math"}

    def test_dataset_stores_path(self, sample_jsonl_file: Path) -> None:
        dataset = Dataset(sample_jsonl_file)

        assert dataset.path == sample_jsonl_file

    def test_empty_dataset(self, tmp_path: Path) -> None:
        empty_file = tmp_path / "empty.jsonl"
        empty_file.write_text("")
        dataset = Dataset(empty_file)

        assert len(dataset) == 0
        assert list(dataset) == []
