import builtins
from abc import ABC, abstractmethod
from typing import Any, ClassVar, Literal, Self

from pydantic import BaseModel

from search_evals.agents.types import LLMProvider as LLMProvider


class ToolSchema(BaseModel, ABC):
    name: str
    description: str


class ToolDef[InputT: BaseModel, OutputT: BaseModel]:
    name: ClassVar[str]
    description: ClassVar[str]
    input_schema: ClassVar[builtins.type[BaseModel]]
    output_schema: ClassVar[builtins.type[BaseModel]]

    @classmethod
    def get_schema(cls, provider: LLMProvider) -> ToolSchema:
        match provider:
            case LLMProvider.OPENAI:
                return OpenAIToolSchema.from_tool_def(cls)
            case LLMProvider.ANTHROPIC:
                return AnthropicToolSchema.from_tool_def(cls)

    @classmethod
    def parse_input(cls, arguments: dict[str, Any]) -> InputT:
        return cls.input_schema.model_validate(arguments)  # type: ignore[return-value]


class OpenAIToolSchema(ToolSchema):
    type: Literal["function"] = "function"
    parameters: dict[str, Any]

    @classmethod
    def from_tool_def(cls, tool_def: builtins.type[ToolDef[Any, Any]]) -> Self:
        return cls(
            name=tool_def.name,
            description=tool_def.description,
            parameters=tool_def.input_schema.model_json_schema(),
        )


class AnthropicToolSchema(ToolSchema):
    input_schema: dict[str, Any]

    @classmethod
    def from_tool_def(cls, tool_def: builtins.type[ToolDef[Any, Any]]) -> Self:
        schema = tool_def.input_schema.model_json_schema()
        schema["additionalProperties"] = False
        return cls(
            name=tool_def.name,
            description=tool_def.description,
            input_schema=schema,
        )


class Tool[InputT: BaseModel, OutputT: BaseModel](ABC):
    tool_def: ClassVar[type[ToolDef[Any, Any]]]

    @abstractmethod
    async def __call__(self, input_: InputT) -> OutputT:
        pass
