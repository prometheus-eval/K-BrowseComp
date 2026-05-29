from __future__ import annotations

import os
import re
from dataclasses import dataclass
from typing import Any

import openai
import orjson
from openai import AsyncOpenAI
from tenacity import retry, retry_if_exception_type, wait_exponential

from search_evals.agents.llms.base import DEFAULT_MAX_CONTEXT_TOKENS, BaseConversation, BaseLLM, LLMOutput
from search_evals.agents.llms.tool_helper import (
    normalize_content,
    recover_search_web_tool_call,
)
from search_evals.agents.tools import LLMProvider, ToolSet
from search_evals.agents.types import ResponseBlock, TextBlock, ToolCallBlock, ToolChoice, ToolResult

OPENROUTER_REASONING_EFFORTS = {"none", "minimal", "low", "medium", "high", "xhigh"}
FINAL_ANSWER_INSTRUCTION = (
    "You have no search calls remaining. Do not call search_web, do not emit tool-call markup, "
    "and do not ask to search again. Provide your final answer now using only the information "
    "already available in the conversation. You must always follow the required response format, "
    "even if the research is incomplete. Empty responses are not allowed. If you cannot determine "
    "the answer, write exactly \"Exact Answer: I don't know\" in the exact-answer field. When "
    "the exact answer is \"I don't know\", still fill the Explanation field with the useful "
    "facts or candidates found so far and the specific gap, conflict, or uncertainty that "
    "prevents a precise answer."
)
EMPTY_FINAL_ANSWER_RETRY_INSTRUCTION = (
    "Your previous final response was empty. You must now return the required response format. "
    "If you cannot determine the answer, explain what you found so far and why it is insufficient, "
    "then write:\nExact Answer: I don't know"
)
EMPTY_REQUIRED_TOOL_RETRY_INSTRUCTION = (
    "Your previous response was empty, but this turn requires a search_web tool call. "
    "You must call search_web now with one focused query. Do not answer directly."
)


@dataclass(frozen=True)
class OpenRouterModelInfo:
    context_length: int | None = None
    max_completion_tokens: int | None = None
    default_reasoning_effort: str | None = None
    default_max_tokens: int | None = None
    reasoning_efforts: frozenset[str] | None = None


OPENROUTER_MODEL_INFO: dict[str, OpenRouterModelInfo] = {
    "z-ai/glm-5.1": OpenRouterModelInfo(
        context_length=202_752,
        max_completion_tokens=65_535,
        default_reasoning_effort="medium",
    ),
    "deepseek/deepseek-v4-pro": OpenRouterModelInfo(
        context_length=1_048_576,
        max_completion_tokens=384_000,
        default_reasoning_effort="high",
        reasoning_efforts=frozenset({"high", "xhigh"}),
    ),
}


def _responses_tools_to_chat_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    if not tools:
        return None

    out: list[dict[str, Any]] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": tool["name"],
                    "description": tool.get("description"),
                    "parameters": tool.get("parameters"),
                },
            }
        )
    return out or None


def _looks_like_tool_call_text(text: str) -> bool:
    lowered = text.lower()
    if (
        "search_web" in lowered
        or "<tool_call" in lowered
        or "tool_calls" in lowered
        or "dsml" in lowered
    ):
        return True
    if re.search(r"\{\s*[\"']query[\"']\s*:", text, flags=re.IGNORECASE | re.DOTALL):
        return True
    if re.search(r"(?:^|\n)\s*action\s*:\s*search", text, flags=re.IGNORECASE):
        return True
    return False


def _parse_positive_int_env(name: str) -> int | None:
    value = os.getenv(name)
    if value is None or value == "":
        return None
    parsed = int(value)
    if parsed <= 0:
        return None
    return parsed


def _parse_bool_env(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None or value == "":
        return default
    lowered = value.lower()
    if lowered in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if lowered in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"{name} must be a boolean value")


def _get_reasoning_effort(model: str) -> str | None:
    model_info = OPENROUTER_MODEL_INFO.get(model, OpenRouterModelInfo())
    effort = os.getenv("OPENROUTER_REASONING_EFFORT") or model_info.default_reasoning_effort
    if not effort:
        return None
    allowed_efforts = model_info.reasoning_efforts or OPENROUTER_REASONING_EFFORTS
    if effort not in allowed_efforts:
        raise ValueError(
            f"OPENROUTER_REASONING_EFFORT for {model} must be one of: "
            + ", ".join(sorted(allowed_efforts))
        )
    return effort


def _get_max_tokens(model: str) -> int | None:
    env_max_tokens = _parse_positive_int_env("OPENROUTER_MAX_TOKENS")
    if env_max_tokens is not None:
        return env_max_tokens
    return OPENROUTER_MODEL_INFO.get(model, OpenRouterModelInfo()).default_max_tokens


def _get_max_context_tokens(model: str) -> int:
    env_max_context_tokens = _parse_positive_int_env("OPENROUTER_MAX_CONTEXT_TOKENS")
    if env_max_context_tokens is not None:
        return env_max_context_tokens

    model_info = OPENROUTER_MODEL_INFO.get(model)
    if model_info and model_info.context_length:
        return min(model_info.context_length, DEFAULT_MAX_CONTEXT_TOKENS)
    return DEFAULT_MAX_CONTEXT_TOKENS


def _get_usage_value(obj: Any, name: str) -> Any:
    if obj is None:
        return None
    if isinstance(obj, dict):
        return obj.get(name)
    return getattr(obj, name, None)


def _get_int_usage_value(obj: Any, name: str) -> int:
    value = _get_usage_value(obj, name)
    return value if isinstance(value, int) else 0


def _get_reasoning_tokens(usage: Any) -> int:
    """Extract hidden reasoning-token counts from OpenRouter/OpenAI-style usage metadata."""
    direct = _get_int_usage_value(usage, "reasoning_tokens")
    if direct:
        return direct

    completion_details = _get_usage_value(usage, "completion_tokens_details")
    from_completion = _get_int_usage_value(completion_details, "reasoning_tokens")
    if from_completion:
        return from_completion

    output_details = _get_usage_value(usage, "output_tokens_details")
    return _get_int_usage_value(output_details, "reasoning_tokens")


class OpenRouterConversation(BaseConversation[dict[str, Any]]):
    def _make_role_message(self, role: str, content: str) -> dict[str, Any]:
        assert role in ("user", "assistant")
        return {"role": role, "content": content}

    def _get_assistant_text(self, message: dict[str, Any]) -> str | None:
        if message.get("role") == "assistant" and message.get("content"):
            return message["content"]
        return None

    def _add_response_messages(self, blocks: list[ResponseBlock]) -> None:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for block in blocks:
            if isinstance(block, ToolCallBlock):
                tool_calls.append(
                    {
                        "id": block.id,
                        "type": "function",
                        "function": {
                            "name": block.name,
                            "arguments": orjson.dumps(block.input).decode(),
                        },
                    }
                )
            elif isinstance(block, TextBlock):
                text_parts.append(block.text)

        message: dict[str, Any] = {"role": "assistant"}
        if text_parts:
            message["content"] = "\n".join(text_parts)
        elif not tool_calls:
            message["content"] = ""
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)

    def add_tool_results(self, results: list[ToolResult]) -> OpenRouterConversation:
        for result in results:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": result.tool_call_id,
                    "content": result.output,
                }
            )
        return self

    def to_api_format(self) -> list[dict[str, Any]]:
        messages = [{k: v for k, v in message.items() if v is not None} for message in self.messages]
        if self.system:
            messages.insert(0, {"role": "system", "content": self.system})
        return messages


class OpenRouterLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str,
        default_headers: dict[str, str] | None = None,
        reasoning_effort: str | None = None,
        reasoning_exclude: bool = True,
        max_tokens: int | None = None,
        max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS,
    ) -> None:
        self.model = model
        self.reasoning_effort = reasoning_effort
        self.reasoning_exclude = reasoning_exclude
        self.max_tokens = max_tokens
        self.max_context_tokens = max_context_tokens
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
            default_headers=default_headers,
        )

    def create_conversation(self, max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS) -> OpenRouterConversation:
        if max_context_tokens == DEFAULT_MAX_CONTEXT_TOKENS:
            max_context_tokens = self.max_context_tokens
        return OpenRouterConversation(max_context_tokens=max_context_tokens)

    @retry(
        retry=retry_if_exception_type(openai.APITimeoutError),
        wait=wait_exponential(multiplier=1, max=300),
    )
    async def __call__(self, convo: OpenRouterConversation, toolset: ToolSet) -> LLMOutput:
        requested_tool_choice = toolset.tool_choice
        tools = _responses_tools_to_chat_tools(toolset.get_defs(LLMProvider.OPENAI))
        messages = convo.to_api_format()

        if requested_tool_choice == ToolChoice.NONE:
            tool_choice: Any = "none"
            tools = None
            messages.append({"role": "user", "content": FINAL_ANSWER_INSTRUCTION})
        elif requested_tool_choice == ToolChoice.REQUIRED:
            tool_choice = "required"
        else:
            tool_choice = "auto"

        request_kwargs: dict[str, Any] = {}
        if self.max_tokens is not None:
            request_kwargs["max_tokens"] = self.max_tokens
        if self.reasoning_effort:
            request_kwargs["extra_body"] = {
                "reasoning": {
                    "effort": self.reasoning_effort,
                    "exclude": self.reasoning_exclude,
                }
            }

        request: dict[str, Any] = {
            "model": self.model,
            "messages": messages,
            **request_kwargs,
        }
        if tools is not None:
            request["tools"] = tools
            request["tool_choice"] = tool_choice

        response = await self.client.chat.completions.create(**request)
        output = self._parse_response(
            response,
            recover_text_tool_call=requested_tool_choice != ToolChoice.NONE,
        )
        if requested_tool_choice == ToolChoice.NONE and output.text and _looks_like_tool_call_text(output.text):
            messages.append({"role": "assistant", "content": output.text})
            messages.append(
                {
                    "role": "user",
                    "content": (
                        "That response attempted to call a tool, but no search calls remain. "
                        "Provide the final answer now without any tool-call markup."
                    ),
                }
            )
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=messages,
                **request_kwargs,
            )
            output = self._parse_response(response, recover_text_tool_call=False)
        if requested_tool_choice == ToolChoice.NONE and not (output.text and output.text.strip()):
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[*messages, {"role": "user", "content": EMPTY_FINAL_ANSWER_RETRY_INSTRUCTION}],
                **request_kwargs,
            )
            output = self._parse_response(response, recover_text_tool_call=False)
        if requested_tool_choice == ToolChoice.REQUIRED and not output.blocks:
            response = await self.client.chat.completions.create(
                model=self.model,
                messages=[*messages, {"role": "user", "content": EMPTY_REQUIRED_TOOL_RETRY_INSTRUCTION}],
                **request_kwargs,
            )
            output = self._parse_response(response, recover_text_tool_call=True)
        return output

    def _parse_response(self, response: Any, *, recover_text_tool_call: bool = True) -> LLMOutput:
        message = response.choices[0].message
        blocks: list[ResponseBlock] = []

        content = normalize_content(getattr(message, "content", None))
        if content:
            blocks.append(TextBlock(text=content))

        tool_calls: list[ToolCallBlock] = []
        for tool_call in getattr(message, "tool_calls", None) or []:
            try:
                args = orjson.loads(tool_call.function.arguments)
            except Exception:
                args = {}
            tool_calls.append(
                ToolCallBlock(
                    id=tool_call.id,
                    name=tool_call.function.name,
                    input=args,
                )
            )

        if recover_text_tool_call and not tool_calls and content:
            recovered = recover_search_web_tool_call(content)
            if recovered is not None:
                tool_calls.append(recovered)
                blocks = []

        blocks.extend(tool_calls)

        usage = getattr(response, "usage", None)
        input_tokens = _get_int_usage_value(usage, "prompt_tokens")
        output_tokens = _get_int_usage_value(usage, "completion_tokens")
        total_tokens = _get_int_usage_value(usage, "total_tokens") or input_tokens + output_tokens
        reasoning_tokens = _get_reasoning_tokens(usage)
        return LLMOutput(
            blocks=blocks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
            reasoning_tokens=reasoning_tokens,
        )


def make_openrouter_llm(model: str) -> OpenRouterLLM:
    api_key = os.getenv("OPENROUTER_API_KEY")
    if not api_key:
        raise ValueError("Missing OPENROUTER_API_KEY for openrouter: models")

    default_headers = {
        "HTTP-Referer": os.getenv("OPENROUTER_HTTP_REFERER", ""),
        "X-Title": os.getenv("OPENROUTER_X_TITLE", "search-evals"),
    }
    default_headers = {key: value for key, value in default_headers.items() if value}

    return OpenRouterLLM(
        model=model,
        base_url=os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1"),
        api_key=api_key,
        default_headers=default_headers or None,
        reasoning_effort=_get_reasoning_effort(model),
        reasoning_exclude=_parse_bool_env("OPENROUTER_REASONING_EXCLUDE", True),
        max_tokens=_get_max_tokens(model),
        max_context_tokens=_get_max_context_tokens(model),
    )
