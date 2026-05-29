from __future__ import annotations

from typing import Any

import orjson
from openai import AsyncOpenAI

from search_evals.agents.llms.base import DEFAULT_MAX_CONTEXT_TOKENS, BaseConversation, BaseLLM, LLMOutput
from search_evals.agents.llms.tool_helper import (
    normalize_content,
    recover_search_web_tool_call,
)
from search_evals.agents.tools import LLMProvider, ToolSet
from search_evals.agents.types import ResponseBlock, TextBlock, ToolCallBlock, ToolChoice, ToolResult


def _responses_tools_to_chat_tools(tools: list[dict[str, Any]] | None) -> list[dict[str, Any]] | None:
    """ToolSet(OpenAI provider)이 뱉는 responses-style tool schema를 chat-completions 스타일로 변환."""
    if not tools:
        return None

    out: list[dict[str, Any]] = []
    for t in tools:
        if t.get("type") != "function":
            continue
        out.append(
            {
                "type": "function",
                "function": {
                    "name": t["name"],
                    "description": t.get("description"),
                    "parameters": t.get("parameters"),
                },
            }
        )
    return out or None


class OpenAIChatConversation(BaseConversation[dict[str, Any]]):
    def _make_role_message(self, role: str, content: str) -> dict[str, Any]:
        # system은 BaseConversation.system으로 관리
        assert role in ("user", "assistant")
        return {"role": role, "content": content}

    def _get_assistant_text(self, message: dict[str, Any]) -> str | None:
        if message.get("role") == "assistant" and message.get("content"):
            return message["content"]
        return None

    def _add_response_messages(self, blocks: list[ResponseBlock]) -> None:
        text_parts: list[str] = []
        tool_calls: list[dict[str, Any]] = []

        for b in blocks:
            if isinstance(b, ToolCallBlock):
                tool_calls.append(
                    {
                        "id": b.id,
                        "type": "function",
                        "function": {
                            "name": b.name,
                            "arguments": orjson.dumps(b.input).decode(),
                        },
                    }
                )
            elif isinstance(b, TextBlock):
                text_parts.append(b.text)

        message: dict[str, Any] = {"role": "assistant"}
        if text_parts:
            message["content"] = "\n".join(text_parts)
        elif not tool_calls:
            message["content"] = ""
        if tool_calls:
            message["tool_calls"] = tool_calls
        self.messages.append(message)

    def add_tool_results(self, results: list[ToolResult]) -> OpenAIChatConversation:
        for r in results:
            self.messages.append(
                {
                    "role": "tool",
                    "tool_call_id": r.tool_call_id,
                    "content": r.output,
                }
            )
        return self

    def to_api_format(self) -> list[dict[str, Any]]:
        msgs = [{k: v for k, v in msg.items() if v is not None} for msg in self.messages]
        if self.system:
            msgs.insert(0, {"role": "system", "content": self.system})
        return msgs


class OpenAIChatLLM(BaseLLM):
    """OpenAI Chat Completions 호환 엔드포인트(vLLM / Ollama / TGI 등)용."""

    def __init__(
        self,
        model: str,
        *,
        base_url: str,
        api_key: str,
    ) -> None:
        self.model = model
        self.client = AsyncOpenAI(
            api_key=api_key,
            base_url=base_url,
        )

    def create_conversation(self, max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS) -> OpenAIChatConversation:
        return OpenAIChatConversation(max_context_tokens=max_context_tokens)

    async def __call__(self, convo: OpenAIChatConversation, toolset: ToolSet) -> LLMOutput:
        tools = _responses_tools_to_chat_tools(toolset.get_defs(LLMProvider.OPENAI))

        if toolset.tool_choice == ToolChoice.NONE:
            tool_choice: Any = "none"
        elif toolset.tool_choice == ToolChoice.REQUIRED:
            tool_choice = "required"
        else:
            tool_choice = "auto"

        resp = await self.client.chat.completions.create(
            model=self.model,
            messages=convo.to_api_format(),
            tools=tools,
            tool_choice=tool_choice,
        )
        return self._parse(resp)

    def _parse(self, resp: Any) -> LLMOutput:
        msg = resp.choices[0].message
        blocks: list[ResponseBlock] = []

        content = normalize_content(getattr(msg, "content", None))
        if content:
            blocks.append(TextBlock(text=content))

        tool_calls: list[ToolCallBlock] = []

        # 1) 정상적인 tool_calls 우선 처리
        for tc in (getattr(msg, "tool_calls", None) or []):
            try:
                args = orjson.loads(tc.function.arguments)
            except Exception:
                args = {}
            tool_calls.append(
                ToolCallBlock(
                    id=tc.id,
                    name=tc.function.name,
                    input=args,
                )
            )

        # 2) 정식 tool_calls가 없으면 content에서 recovery
        if not tool_calls and content:
            recovered = recover_search_web_tool_call(content)
            if recovered is not None:
                tool_calls.append(recovered)
                blocks = []

        blocks.extend(tool_calls)

        input_tokens = resp.usage.prompt_tokens if getattr(resp, "usage", None) else 0
        output_tokens = resp.usage.completion_tokens if getattr(resp, "usage", None) else 0
        total_tokens = resp.usage.total_tokens if getattr(resp, "usage", None) else input_tokens + output_tokens
        return LLMOutput(
            blocks=blocks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
