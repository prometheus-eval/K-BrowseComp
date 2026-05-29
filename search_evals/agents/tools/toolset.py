import asyncio
from typing import Any, Literal, cast, overload

import orjson
from anthropic.types import ToolParam
from openai.types.responses import FunctionToolParam

from search_evals.agents.tools.base import LLMProvider, Tool, ToolSchema
from search_evals.agents.types import ToolCallBlock, ToolChoice, ToolResult


class ToolSet:
    def __init__(
        self,
        tools: list[Tool[Any, Any]],
        tool_choice: ToolChoice = ToolChoice.AUTO,
    ) -> None:
        self.tools: dict[str, Tool[Any, Any]] = {tool.tool_def.name: tool for tool in tools}
        self.tool_choice = tool_choice

    def get_schemas(self, provider: LLMProvider) -> list[ToolSchema]:
        return [tool.tool_def.get_schema(provider) for tool in self.tools.values()]

    def with_tool_choice(self, tool_choice: ToolChoice) -> "ToolSet":
        return ToolSet(list(self.tools.values()), tool_choice=tool_choice)

    @overload
    def get_defs(self, provider: Literal[LLMProvider.OPENAI]) -> list[FunctionToolParam]: ...

    @overload
    def get_defs(self, provider: Literal[LLMProvider.ANTHROPIC]) -> list[ToolParam]: ...

    def get_defs(self, provider: LLMProvider) -> list[FunctionToolParam] | list[ToolParam]:
        return cast(
            list[FunctionToolParam] | list[ToolParam],
            [schema.model_dump() for schema in self.get_schemas(provider)],
        )

    async def _execute(self, tool_call: ToolCallBlock) -> ToolResult:
        try:
            tool = self.tools[tool_call.name]
            input = tool.tool_def.parse_input(tool_call.input)
            output = await tool(input)
            return ToolResult(
                tool_call_id=tool_call.id,
                output=orjson.dumps(output.model_dump()).decode("utf-8"),
            )
        except Exception as e:
            return ToolResult(
                tool_call_id=tool_call.id,
                output=f"Error: {e}",
            )

    async def __call__(self, tool_calls: list[ToolCallBlock]) -> list[ToolResult]:
        return list(await asyncio.gather(*[self._execute(tc) for tc in tool_calls]))
