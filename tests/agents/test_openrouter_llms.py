from types import SimpleNamespace
from typing import Any

import pytest

import search_evals.agents.llms as llms
from search_evals.agents.llms.openrouter import (
    EMPTY_FINAL_ANSWER_RETRY_INSTRUCTION,
    FINAL_ANSWER_INSTRUCTION,
    OpenRouterConversation,
    OpenRouterLLM,
)
from search_evals.agents.tools import ToolSet
from search_evals.agents.types import TextBlock, ToolCallBlock, ToolChoice, ToolResult


class FakeCompletions:
    def __init__(self, responses: list[Any], on_create: Any | None = None) -> None:
        self.responses = responses
        self.on_create = on_create
        self.calls: list[dict[str, Any]] = []

    async def create(self, **kwargs: Any) -> Any:
        self.calls.append(kwargs)
        if self.on_create is not None:
            self.on_create()
        return self.responses.pop(0)


class FakeOpenAIClient:
    def __init__(self, responses: list[Any], on_create: Any | None = None) -> None:
        self.completions = FakeCompletions(responses, on_create=on_create)
        self.chat = SimpleNamespace(completions=self.completions)


def make_chat_response(content: str, *, reasoning_tokens: int = 0) -> Any:
    return SimpleNamespace(
        choices=[SimpleNamespace(message=SimpleNamespace(content=content, tool_calls=[]))],
        usage=SimpleNamespace(
            prompt_tokens=11,
            completion_tokens=13,
            total_tokens=24,
            completion_tokens_details=SimpleNamespace(reasoning_tokens=reasoning_tokens),
        ),
    )


def test_glm_51_openrouter_defaults_are_token_conservative(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("OPENROUTER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("OPENROUTER_MAX_CONTEXT_TOKENS", raising=False)

    llm = llms.make_llm("openrouter:z-ai/glm-5.1")

    assert isinstance(llm, OpenRouterLLM)
    assert llm.model == "z-ai/glm-5.1"
    assert llm.reasoning_effort == "medium"
    assert llm.max_tokens is None
    assert llm.max_context_tokens == llms.DEFAULT_MAX_CONTEXT_TOKENS


def test_glm_51_openrouter_env_overrides(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "low")
    monkeypatch.setenv("OPENROUTER_MAX_TOKENS", "2048")
    monkeypatch.setenv("OPENROUTER_MAX_CONTEXT_TOKENS", "8000")

    llm = llms.make_llm("openrouter:z-ai/glm-5.1")

    assert isinstance(llm, OpenRouterLLM)
    assert llm.reasoning_effort == "low"
    assert llm.max_tokens == 2_048
    assert llm.max_context_tokens == 8_000


def test_deepseek_v4_pro_openrouter_defaults(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.delenv("OPENROUTER_REASONING_EFFORT", raising=False)
    monkeypatch.delenv("OPENROUTER_MAX_TOKENS", raising=False)
    monkeypatch.delenv("OPENROUTER_MAX_CONTEXT_TOKENS", raising=False)

    llm = llms.make_llm("openrouter:deepseek/deepseek-v4-pro")

    assert isinstance(llm, OpenRouterLLM)
    assert llm.reasoning_effort == "high"
    assert llm.max_tokens is None
    assert llm.max_context_tokens == llms.DEFAULT_MAX_CONTEXT_TOKENS


def test_deepseek_v4_pro_rejects_unsupported_reasoning_effort(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "medium")

    with pytest.raises(ValueError, match="deepseek/deepseek-v4-pro"):
        llms.make_llm("openrouter:deepseek/deepseek-v4-pro")


def test_openrouter_reasoning_effort_validation(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENROUTER_API_KEY", "sk-test")
    monkeypatch.setenv("OPENROUTER_REASONING_EFFORT", "very-high")

    with pytest.raises(ValueError, match="OPENROUTER_REASONING_EFFORT"):
        llms.make_llm("openrouter:z-ai/glm-5.1")


def test_openrouter_conversation_uses_native_tool_messages() -> None:
    convo = OpenRouterConversation()
    convo.add_response(
        [
            TextBlock(text="Searching."),
            ToolCallBlock(id="call_1", name="search_web", input={"query": "glm 5.1"}),
        ]
    )
    convo.add_tool_results([ToolResult(tool_call_id="call_1", output='{"results": []}')])

    api_messages = convo.to_api_format()

    assert api_messages[0] == {
        "role": "assistant",
        "content": "Searching.",
        "tool_calls": [
            {
                "id": "call_1",
                "type": "function",
                "function": {
                    "name": "search_web",
                    "arguments": '{"query":"glm 5.1"}',
                },
            }
        ],
    }
    assert api_messages[1] == {
        "role": "tool",
        "tool_call_id": "call_1",
        "content": '{"results": []}',
    }


@pytest.mark.asyncio
async def test_openrouter_retries_empty_final_response() -> None:
    final_text = "Explanation: I could not determine it.\nExact Answer: I don't know\nConfidence: 0%"
    client = FakeOpenAIClient([make_chat_response(""), make_chat_response(final_text)])
    llm = OpenRouterLLM(model="test-model", base_url="https://example.test", api_key="sk-test")
    llm.client = client

    convo = OpenRouterConversation()
    convo.add_user("Question?")

    output = await llm(convo, ToolSet([], tool_choice=ToolChoice.NONE))

    assert output.text == final_text
    assert len(client.completions.calls) == 2
    assert client.completions.calls[0]["messages"][-1] == {
        "role": "user",
        "content": FINAL_ANSWER_INSTRUCTION,
    }
    assert client.completions.calls[1]["messages"][-1] == {
        "role": "user",
        "content": EMPTY_FINAL_ANSWER_RETRY_INSTRUCTION,
    }


@pytest.mark.asyncio
async def test_openrouter_records_reasoning_tokens() -> None:
    client = FakeOpenAIClient([make_chat_response("Done", reasoning_tokens=123)])
    llm = OpenRouterLLM(model="test-model", base_url="https://example.test", api_key="sk-test")
    llm.client = client

    convo = OpenRouterConversation()
    convo.add_user("Question?")

    output = await llm(convo, ToolSet([], tool_choice=ToolChoice.NONE))

    assert output.input_tokens == 11
    assert output.output_tokens == 13
    assert output.total_tokens == 24
    assert output.reasoning_tokens == 123


@pytest.mark.asyncio
async def test_openrouter_snapshots_tool_choice_before_await() -> None:
    toolset = ToolSet([], tool_choice=ToolChoice.AUTO)
    pseudo_tool_call = (
        "<tool_call>search_web<arg_key>query</arg_key>"
        "<arg_value>버즈 BUZZ 의성어 노래 제목</arg_value></tool_call>"
    )
    client = FakeOpenAIClient(
        [make_chat_response(pseudo_tool_call)],
        on_create=lambda: setattr(toolset, "tool_choice", ToolChoice.NONE),
    )
    llm = OpenRouterLLM(model="test-model", base_url="https://example.test", api_key="sk-test")
    llm.client = client

    convo = OpenRouterConversation()
    convo.add_user("Question?")

    output = await llm(convo, toolset)

    assert output.text is None
    assert len(output.tool_calls) == 1
    assert output.tool_calls[0].name == "search_web"
    assert output.tool_calls[0].input == {"query": "버즈 BUZZ 의성어 노래 제목"}
