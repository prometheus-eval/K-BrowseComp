from collections.abc import Iterator
from pathlib import Path
from typing import Self, overload

from pydantic import BaseModel

from search_evals.io_utils import decrypt_dataset, load_jsonl_file


class Datum(BaseModel):
    id: str
    problem: str
    answer: str
    metadata: dict[str, str] | None = None

    @classmethod
    def create(cls, data: dict[str, object]) -> Self:
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
        self.path = path
        self.encrypted = encrypted
        self.data: list[Datum] = self._load_data(limit)

    def _load_data(self, limit: int | None) -> list[Datum]:
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
