from typing import Any, ClassVar

from pydantic import BaseModel, Field

from search_evals.agents.tools.base import Tool, ToolDef
from search_evals.search_engines import AsyncSearchEngine


class SearchWebInput(BaseModel):
    query: str = Field(description="A concise search query.")


class SearchWebResult(BaseModel):
    title: str
    url: str
    snippet: str


class SearchWebOutput(BaseModel):
    results: list[SearchWebResult]


class SearchWebToolDef(ToolDef[SearchWebInput, SearchWebOutput]):
    name: ClassVar[str] = "search_web"
    description: ClassVar[str] = (
        "Searches the web for current and factual information, "
        "returning relevant results with titles, URLs, and content snippets. "
        "Use for questions about up-to-date or externally verified information. "
    )
    input_schema: ClassVar[type[BaseModel]] = SearchWebInput
    output_schema: ClassVar[type[BaseModel]] = SearchWebOutput


class SearchWebTool(Tool[SearchWebInput, SearchWebOutput]):
    tool_def: ClassVar[type[ToolDef[Any, Any]]] = SearchWebToolDef

    def __init__(
        self,
        search_engine: AsyncSearchEngine,
        max_results: int = 10,
    ) -> None:
        self.search_engine = search_engine
        self.max_results = max_results

    async def __call__(self, input: SearchWebInput) -> SearchWebOutput:
        results = await self.search_engine(input.query, num_results=self.max_results)
        return SearchWebOutput(
            results=[
                SearchWebResult(
                    url=result.url,
                    title=result.title,
                    snippet=result.snippet,
                )
                for result in results
            ]
        )
