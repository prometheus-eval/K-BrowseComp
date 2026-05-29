from __future__ import annotations

import base64
import os
import uuid
from typing import Any, Self

import orjson
from google import genai
from google.genai import types
from pydantic import BaseModel, ConfigDict, PrivateAttr, field_serializer

from search_evals.agents.llms.base import DEFAULT_MAX_CONTEXT_TOKENS, BaseConversation, BaseLLM, LLMOutput
from search_evals.agents.tools import LLMProvider, ToolSet
from search_evals.agents.types import ResponseBlock, TextBlock, ToolCallBlock, ToolResult, ToolChoice


def _clean_schema_for_gemini(schema: dict[str, Any]) -> dict[str, Any]:
    s = dict(schema)
    s.pop("title", None)
    s.pop("$defs", None)
    s.pop("definitions", None)
    return s


def _to_jsonable(x: Any) -> Any:
    """Make nested objects JSON-safe for orjson (handles bytes + genai/pydantic models)."""
    if isinstance(x, bytes):
        # stable + reversible
        return {"__bytes__": base64.b64encode(x).decode("ascii")}
    if isinstance(x, (str, int, float, bool)) or x is None:
        return x
    if isinstance(x, dict):
        return {str(k): _to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [_to_jsonable(v) for v in x]
    # google-genai types are pydantic-ish; try model_dump if available
    if hasattr(x, "model_dump"):
        try:
            return _to_jsonable(x.model_dump())
        except Exception:
            pass
    # fallback: repr
    return repr(x)


class GeminiMessage(BaseModel):
    """
    We keep raw google-genai Content objects in `content` for API calls,
    but we serialize them safely when saving results (bytes -> base64 dict).
    """
    content: Any
    model_config = ConfigDict(arbitrary_types_allowed=True)

    @field_serializer("content", when_used="always")
    def _ser_content(self, v: Any) -> Any:
        return _to_jsonable(v)


class GeminiConversation(BaseConversation[GeminiMessage]):
    model_config = ConfigDict(arbitrary_types_allowed=True)

    _pending_model_content: Any | None = PrivateAttr(default=None)
    _tool_id_to_name: dict[str, str] = PrivateAttr(default_factory=dict)

    def set_pending_model_content(self, content: Any) -> None:
        self._pending_model_content = content

    def _make_role_message(self, role: str, content: str) -> GeminiMessage:
        gem_role = "user" if role == "user" else "model"
        return GeminiMessage(
            content=types.Content(
                role=gem_role,
                parts=[types.Part.from_text(text=content)],  # ✅ keyword-only + [] correctly closed
            )
        )

    def _get_assistant_text(self, message: GeminiMessage) -> str | None:
        c = message.content
        if getattr(c, "role", None) != "model":
            return None
        texts: list[str] = []
        for p in (getattr(c, "parts", None) or []):
            t = getattr(p, "text", None)
            if t:
                texts.append(t)
        return "\n".join(texts) if texts else None

    def _add_response_messages(self, blocks: list[ResponseBlock]) -> None:
        # Prefer storing raw model content (keeps thought/signature fields intact)
        if self._pending_model_content is not None:
            self.messages.append(GeminiMessage(content=self._pending_model_content))
            self._pending_model_content = None
        else:
            # Fallback reconstruction (still valid)
            for block in blocks:
                match block:
                    case TextBlock():
                        self.messages.append(
                            GeminiMessage(
                                content=types.Content(
                                    role="model",
                                    parts=[types.Part.from_text(text=block.text)],  # ✅ correct
                                )
                            )
                        )
                    case ToolCallBlock():
                        self.messages.append(
                            GeminiMessage(
                                content=types.Content(
                                    role="model",
                                    parts=[
                                        types.Part.from_function_call(
                                            name=block.name,
                                            args=block.input,
                                        )
                                    ],
                                )
                            )
                        )

        for b in blocks:
            if isinstance(b, ToolCallBlock):
                self._tool_id_to_name[b.id] = b.name

    def add_tool_results(self, results: list[ToolResult]) -> Self:
        for r in results:
            name = self._tool_id_to_name.get(r.tool_call_id, "unknown_tool")
            try:
                payload = orjson.loads(r.output)
            except Exception:
                payload = {"output": r.output}

            self.messages.append(
                GeminiMessage(
                    content=types.Content(
                        role="tool",
                        parts=[
                            types.Part.from_function_response(
                                name=name,
                                response=payload,
                            )
                        ],
                    )
                )
            )
        return self

    def to_api_format(self) -> Any:
        # google-genai expects list[types.Content]
        return [m.content for m in self.messages]


class GeminiLLM(BaseLLM):
    def __init__(
        self,
        model: str,
        api_key: str | None = None,
    ) -> None:
        self.model = model
        if not api_key:
            # your envs: GEMINI_API_KEY or GOOGLE_API_KEY
            api_key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
        if not api_key:
            raise ValueError("Missing GEMINI_API_KEY (or GOOGLE_API_KEY).")
        self.client = genai.Client(api_key=api_key)

    def create_conversation(self, max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS) -> GeminiConversation:
        return GeminiConversation(max_context_tokens=max_context_tokens)

    async def __call__(self, convo: GeminiConversation, toolset: ToolSet) -> LLMOutput:
        tools_resp = toolset.get_defs(LLMProvider.OPENAI)
        function_decls: list[types.FunctionDeclaration] = []

        for t in tools_resp:
            if t.get("type") != "function":
                continue
            schema = _clean_schema_for_gemini(t.get("parameters") or {})
            function_decls.append(
                types.FunctionDeclaration(
                    name=t["name"],
                    description=t.get("description"),
                    parameters_json_schema=schema,
                )
            )

        tools = [types.Tool(function_declarations=function_decls)] if function_decls else None

        if toolset.tool_choice == ToolChoice.NONE:
            mode = "NONE"
        elif toolset.tool_choice == ToolChoice.REQUIRED:
            mode = "ANY"
        else:
            mode = "AUTO"

        tool_config = (
            types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode=mode))
            if tools
            else None
        )

        config_kwargs: dict[str, Any] = {
            "system_instruction": convo.system or None,
            "tools": tools,
            "tool_config": tool_config,
        }

        config = types.GenerateContentConfig(**config_kwargs)

        resp = await self.client.aio.models.generate_content(
            model=self.model,
            contents=convo.to_api_format(),
            config=config,
        )

        if resp.candidates and resp.candidates[0].content is not None:
            convo.set_pending_model_content(resp.candidates[0].content)

        blocks: list[ResponseBlock] = []
        content = resp.candidates[0].content if resp.candidates else None

        for p in (getattr(content, "parts", None) or []):
            if getattr(p, "text", None):
                blocks.append(TextBlock(text=p.text))
            if getattr(p, "function_call", None):
                fc = p.function_call
                call_id = getattr(fc, "id", None) or str(uuid.uuid4())
                blocks.append(ToolCallBlock(id=call_id, name=fc.name, input=fc.args or {}))

        usage = getattr(resp, "usage_metadata", None)
        input_tokens = getattr(usage, "prompt_token_count", 0) or 0
        output_tokens = getattr(usage, "candidates_token_count", 0) or 0
        total_tokens = getattr(usage, "total_token_count", 0) or input_tokens + output_tokens
        return LLMOutput(
            blocks=blocks,
            input_tokens=input_tokens,
            output_tokens=output_tokens,
            total_tokens=total_tokens,
        )
