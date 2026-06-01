# Portions adapted from https://github.com/perplexityai/search_evals/blob/main/tests/agents/test_types.py
# Original license: MIT.

from search_evals.agents.llms import AnthropicConversation, OpenAIConversation
from search_evals.agents.tools import OpenAIToolSchema
from search_evals.agents.types import (
    TextBlock,
    ToolCallBlock,
    ToolChoice,
    ToolResult,
)


class TestOpenAIConversation:
    def test_empty_conversation(self) -> None:
        convo = OpenAIConversation()
        assert len(convo.messages) == 0

    def test_add_user_message(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Hello")

        assert len(convo.messages) == 1
        assert convo.messages[0].role == "user"
        assert convo.messages[0].content == "Hello"

    def test_add_multiple_messages(self) -> None:
        convo = OpenAIConversation()
        convo.add_system("You are helpful")
        convo.add_user("Hi")
        convo.add_assistant("Hello!")

        assert convo.system == "You are helpful"
        assert len(convo.messages) == 2
        assert convo.messages[0].role == "user"
        assert convo.messages[1].role == "assistant"

    def test_fluent_api(self) -> None:
        convo = OpenAIConversation()
        result = convo.add_system("System").add_user("User").add_assistant("Assistant")

        assert result is convo
        assert convo.system == "System"
        assert len(convo.messages) == 2

    def test_add_response_with_tool_use(self) -> None:
        convo = OpenAIConversation()
        convo.add_response(
            [
                ToolCallBlock(id="call_123", name="search", input={"query": "test"}),
            ]
        )

        assert len(convo.messages) == 1
        assert convo.messages[0].type == "function_call"
        assert convo.messages[0].name == "search"

    def test_add_response_with_text_and_tool_use(self) -> None:
        convo = OpenAIConversation()
        convo.add_response(
            [
                TextBlock(text="Let me search for that"),
                ToolCallBlock(id="call_123", name="search", input={"query": "test"}),
            ]
        )

        # OpenAI stores as separate messages
        assert len(convo.messages) == 2
        assert convo.messages[0].role == "assistant"
        assert convo.messages[0].content == "Let me search for that"
        assert convo.messages[1].type == "function_call"

    def test_add_tool_results(self) -> None:
        convo = OpenAIConversation()
        convo.add_tool_results(
            [
                ToolResult(tool_call_id="call_123", output="Search results here"),
            ]
        )

        assert len(convo.messages) == 1
        assert convo.messages[0].type == "function_call_output"
        assert convo.messages[0].output == "Search results here"

    def test_last_text_returns_assistant_message(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Question")
        convo.add_assistant("Answer")

        assert convo.last_text() == "Answer"

    def test_last_text_returns_empty_when_no_assistant(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Question")

        assert convo.last_text() == ""

    def test_to_api_format(self) -> None:
        convo = OpenAIConversation()
        convo.add_system("System prompt")
        convo.add_user("User message")

        api_format = convo.to_api_format()
        assert isinstance(api_format, list)
        assert len(api_format) == 2
        # Use dict access to avoid TypedDict key issues
        assert dict(api_format[0])["role"] == "system"
        assert dict(api_format[1])["role"] == "user"

    def test_to_api_format_excludes_none_values(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Hello")

        api_format = convo.to_api_format()
        assert "type" not in api_format[0]
        assert "call_id" not in api_format[0]

    def test_model_dump(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Hello")

        dumped = convo.model_dump()
        assert "messages" in dumped
        assert len(dumped["messages"]) == 1


class TestToolSchema:
    def test_openai_tool_schema_creation(self) -> None:
        schema = OpenAIToolSchema(
            name="search",
            description="Search the web",
            parameters={"type": "object", "properties": {"query": {"type": "string"}}},
        )

        assert schema.type == "function"
        assert schema.name == "search"
        assert schema.description == "Search the web"
        assert "query" in schema.parameters["properties"]


class TestTextBlock:
    def test_text_block_creation(self) -> None:
        block = TextBlock(text="Hello world")

        assert block.type == "text"
        assert block.text == "Hello world"


class TestToolCallBlock:
    def test_tool_call_block_creation(self) -> None:
        block = ToolCallBlock(
            id="toolu_abc123",
            name="get_weather",
            input={"city": "London"},
        )

        assert block.type == "tool_call"
        assert block.id == "toolu_abc123"
        assert block.name == "get_weather"
        assert block.input == {"city": "London"}


class TestToolResult:
    def test_tool_result_creation(self) -> None:
        result = ToolResult(
            tool_call_id="toolu_abc123",
            output="The weather in London is sunny, 22°C",
        )

        assert result.tool_call_id == "toolu_abc123"
        assert result.output == "The weather in London is sunny, 22°C"


class TestToolChoice:
    def test_enum_values(self) -> None:
        assert ToolChoice.AUTO.value == "auto"
        assert ToolChoice.NONE.value == "none"
        assert ToolChoice.REQUIRED.value == "required"


class TestAnthropicConversation:
    def test_add_response_with_text_and_tool_use(self) -> None:
        convo = AnthropicConversation()
        convo.add_response(
            [
                TextBlock(text="Let me search for that"),
                ToolCallBlock(id="toolu_123", name="search", input={"query": "test"}),
            ]
        )

        # Anthropic stores text + tool_use in single message
        assert len(convo.messages) == 1
        assert convo.messages[0].role == "assistant"
        assert isinstance(convo.messages[0].content, list)
        assert len(convo.messages[0].content) == 2
        assert convo.messages[0].content[0]["type"] == "text"
        assert convo.messages[0].content[0]["text"] == "Let me search for that"
        assert convo.messages[0].content[1]["type"] == "tool_use"
        assert convo.messages[0].content[1]["name"] == "search"

    def test_add_response_tool_use_only(self) -> None:
        convo = AnthropicConversation()
        convo.add_response(
            [
                ToolCallBlock(id="toolu_123", name="search", input={"query": "test"}),
            ]
        )

        assert len(convo.messages) == 1
        assert convo.messages[0].role == "assistant"
        assert isinstance(convo.messages[0].content, list)
        assert len(convo.messages[0].content) == 1
        assert convo.messages[0].content[0]["type"] == "tool_use"

    def test_add_multiple_tool_uses(self) -> None:
        convo = AnthropicConversation()
        convo.add_response(
            [
                TextBlock(text="Let me search for both"),
                ToolCallBlock(id="toolu_1", name="search", input={"query": "weather"}),
                ToolCallBlock(id="toolu_2", name="search", input={"query": "news"}),
            ]
        )

        assert len(convo.messages) == 1
        content = convo.messages[0].content
        assert isinstance(content, list)
        assert len(content) == 3  # 1 text + 2 tool_use
        assert content[0]["type"] == "text"
        assert content[1]["type"] == "tool_use"
        assert content[2]["type"] == "tool_use"

    def test_add_tool_results(self) -> None:
        convo = AnthropicConversation()
        convo.add_tool_results(
            [
                ToolResult(tool_call_id="toolu_1", output="Result 1"),
                ToolResult(tool_call_id="toolu_2", output="Result 2"),
            ]
        )

        assert len(convo.messages) == 1
        assert convo.messages[0].role == "user"
        content = convo.messages[0].content
        assert isinstance(content, list)
        assert len(content) == 2
        assert content[0]["type"] == "tool_result"
        assert content[0]["tool_use_id"] == "toolu_1"
        assert content[1]["type"] == "tool_result"
        assert content[1]["tool_use_id"] == "toolu_2"

    def test_last_text_from_text_block(self) -> None:
        convo = AnthropicConversation()
        convo.add_user("Question")
        convo.add_response([TextBlock(text="Answer")])

        assert convo.last_text() == "Answer"

    def test_last_text_from_simple_assistant(self) -> None:
        convo = AnthropicConversation()
        convo.add_user("Question")
        convo.add_assistant("Simple answer")

        assert convo.last_text() == "Simple answer"


class TestContextSnapshots:
    def test_record_snapshot_creates_snapshot(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Question")
        convo.add_response([TextBlock(text="Answer")]).record_snapshot(100)

        assert len(convo.context_snapshots) == 1
        assert convo.context_snapshots[0].message_index == 2
        assert convo.context_snapshots[0].input_tokens == 100
        assert convo.context_snapshots[0].output_tokens == 0
        assert convo.context_snapshots[0].total_tokens == 0
        assert convo.context_snapshots[0].reasoning_tokens == 0

    def test_record_snapshot_stores_usage_metadata(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Question")
        convo.add_response([TextBlock(text="Answer")]).record_snapshot(
            input_tokens=100,
            output_tokens=30,
            total_tokens=130,
            reasoning_tokens=20,
        )

        snapshot = convo.context_snapshots[0]
        assert snapshot.input_tokens == 100
        assert snapshot.output_tokens == 30
        assert snapshot.total_tokens == 130
        assert snapshot.reasoning_tokens == 20

    def test_multiple_responses_create_snapshots(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Q1")
        convo.add_response([TextBlock(text="A1")]).record_snapshot(100)
        convo.add_user("Q2")
        convo.add_response([TextBlock(text="A2")]).record_snapshot(250)
        convo.add_user("Q3")
        convo.add_response([TextBlock(text="A3")]).record_snapshot(400)

        assert len(convo.context_snapshots) == 3
        assert convo.context_snapshots[0].input_tokens == 100
        assert convo.context_snapshots[1].input_tokens == 250
        assert convo.context_snapshots[2].input_tokens == 400


class TestTruncatePrefix:
    def test_truncate_removes_early_messages(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Q1")
        convo.add_response([TextBlock(text="A1")]).record_snapshot(100)  # snapshot 0
        convo.add_user("Q2")
        convo.add_response([TextBlock(text="A2")]).record_snapshot(250)  # snapshot 1
        convo.add_user("Q3")
        convo.add_response([TextBlock(text="A3")]).record_snapshot(400)  # snapshot 2

        # Total=400, limit=300 means keep at most 300 tokens
        # Need to remove >= 100 tokens, first snapshot with >= 100 is snapshot 0
        convo.truncate_prefix(limit=300)

        # Should have 4 messages: Q2, A2, Q3, A3 (remaining = 400-100 = 300)
        assert len(convo.messages) == 4
        assert convo.messages[0].content == "Q2"

    def test_truncate_rebuilds_snapshot_indices(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Q1")
        convo.add_response([TextBlock(text="A1")]).record_snapshot(100)
        convo.add_user("Q2")
        convo.add_response([TextBlock(text="A2")]).record_snapshot(250)

        # Total=250, limit=150 means keep at most 150 tokens
        # Need to remove >= 100, truncate at snapshot 0
        convo.truncate_prefix(limit=150)

        # First snapshot should now have message_index=0, input_tokens=0
        assert convo.context_snapshots[0].message_index == 0
        assert convo.context_snapshots[0].input_tokens == 0
        # Second snapshot: 250-100=150 remaining tokens
        assert convo.context_snapshots[1].message_index == 2
        assert convo.context_snapshots[1].input_tokens == 150

    def test_truncate_already_within_limit(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Q1")
        convo.add_response([TextBlock(text="A1")]).record_snapshot(100)

        original_len = len(convo.messages)
        convo.truncate_prefix(limit=200)  # Already within limit

        assert len(convo.messages) == original_len

    def test_truncate_empty_snapshots_no_op(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Question")

        convo.truncate_prefix(limit=100)

        assert len(convo.messages) == 1

    def test_truncate_to_minimum_remaining(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Q1")
        convo.add_response([TextBlock(text="A1")]).record_snapshot(100)
        convo.add_user("Q2")
        convo.add_response([TextBlock(text="A2")]).record_snapshot(200)
        convo.add_user("Q3")
        convo.add_response([TextBlock(text="A3")]).record_snapshot(300)

        # Total=300, limit=100 means keep at most 100 tokens
        # Need to remove >= 200, first snapshot with >= 200 is snapshot 1
        convo.truncate_prefix(limit=100)

        # Should have Q3, A3 remaining (300-200=100 tokens)
        assert len(convo.messages) == 2
        assert convo.messages[0].content == "Q3"
        # Remaining tokens should be <= limit
        assert convo.context_snapshots[-1].input_tokens <= 100

    def test_truncate_returns_self(self) -> None:
        convo = OpenAIConversation()
        convo.add_user("Q1")
        convo.add_response([TextBlock(text="A1")]).record_snapshot(100)

        result = convo.truncate_prefix(limit=200)

        assert result is convo
