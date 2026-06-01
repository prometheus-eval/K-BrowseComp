# Portions adapted from https://github.com/perplexityai/search_evals/blob/main/search_evals/datasets/dataset.py
# Original license: MIT.

from collections.abc import Iterator
from pathlib import Path
from typing import Any, Self, overload

import orjson
from pydantic import BaseModel

from search_evals.io_utils import decrypt_dataset, hash_key, load_jsonl_file


class Datum(BaseModel):
    id: str
    problem: str
    answer: str
    metadata: dict[str, str] | None = None

    @classmethod
    def create(cls, data: dict[str, object]) -> Self:
        if "id" not in data:
            data = {
                **data,
                "id": hash_key(orjson.dumps(data, option=orjson.OPT_SORT_KEYS).decode()),
            }
        metadata = {k: str(v) for k, v in data.items() if k not in {"id", "problem", "answer"}}
        return cls(
            id=str(data["id"]),
            problem=str(data["problem"]),
            answer=str(data["answer"]),
            metadata=metadata if metadata else None,
        )


class Dataset:
    def __init__(
        self,
        path: Path,
        encrypted: bool = False,
        limit: int | None = None,
    ) -> None:
        self.path: Path | None = path
        self.source = str(path)
        self.encrypted = encrypted
        self.data: list[Datum] = self._load_data(limit)

    @classmethod
    def from_huggingface(
        cls,
        repo_id: str,
        config: str = "verified",
        split: str = "test",
        limit: int | None = None,
    ) -> Self:
        try:
            from datasets import load_dataset
        except ImportError as exc:
            raise ImportError("Install the `datasets` package to load Ko-BrowseComp from Hugging Face.") from exc

        hf_dataset = load_dataset(repo_id, name=config, split=split)
        raw_data: list[dict[str, object]] = []
        for i, row in enumerate(hf_dataset):
            if limit is not None and i >= limit:
                break
            raw_data.append(cls._coerce_record(row))

        dataset = cls.__new__(cls)
        dataset.path = None
        dataset.source = f"hf://datasets/{repo_id}/{config}/{split}"
        dataset.encrypted = False
        dataset.data = [Datum.create(item) for item in raw_data]
        return dataset

    @staticmethod
    def _coerce_record(row: Any) -> dict[str, object]:
        if isinstance(row, dict):
            return {str(key): value for key, value in row.items()}
        return dict(row)

    def _load_data(self, limit: int | None) -> list[Datum]:
        if self.path is None:
            raise ValueError("A local dataset path is required for JSONL loading.")
        raw_data = load_jsonl_file(str(self.path), limit=limit)
        if self.encrypted:
            raw_data = decrypt_dataset(raw_data)
        return [Datum.create(item) for item in raw_data]

    def __iter__(self) -> Iterator[Datum]:
        return iter(self.data)

    @overload
    def __getitem__(self, index: int) -> Datum: ...

    @overload
    def __getitem__(self, index: slice) -> list[Datum]: ...

    def __getitem__(self, index: int | slice) -> Datum | list[Datum]:
        return self.data[index]

    def __len__(self) -> int:
        return len(self.data)
