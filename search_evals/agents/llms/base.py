# Portions adapted from https://github.com/perplexityai/search_evals/blob/main/search_evals/agents/llms/base.py
# Original license: MIT.

from abc import ABC, abstractmethod
from typing import Any, Self

from pydantic import BaseModel

from search_evals.agents.tools import ToolSet
from search_evals.agents.types import ResponseBlock, TextBlock, ToolCallBlock, ToolResult

DEFAULT_MAX_CONTEXT_TOKENS = 120_000


class LLMOutput(BaseModel):
    blocks: list[ResponseBlock]
    input_tokens: int = 0
    output_tokens: int = 0
    total_tokens: int = 0
    reasoning_tokens: int = 0

    @property
    def text(self) -> str | None:
        """Get concatenated text from all text blocks."""
        texts = [b.text for b in self.blocks if isinstance(b, TextBlock)]
        return "\n".join(texts) if texts else None

    @property
    def tool_calls(self) -> list[ToolCallBlock]:
        """Get all tool call blocks."""
        return [b for b in self.blocks if isinstance(b, ToolCallBlock)]


class ContextSnapshot(BaseModel):
    """Snapshot of context size at a point in conversation, enabling prefix truncation."""

    message_index: int  # Index into messages list after this response was added
    input_tokens: int  # Input tokens reported by API at this point
    output_tokens: int = 0  # Output tokens reported by API for this response
    total_tokens: int = 0  # Total tokens reported by API for this response
    reasoning_tokens: int = 0  # Hidden reasoning tokens reported by API, when available


class BaseConversation[MessageT: BaseModel](BaseModel):
    """Base conversation with shared logic using template method pattern."""

    messages: list[MessageT] = []
    system: str = ""
    context_snapshots: list[ContextSnapshot] = []
    max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS

    @abstractmethod
    def _make_role_message(self, role: str, content: str) -> MessageT:
        """Create a simple role-based text message."""
        ...

    @abstractmethod
    def _get_assistant_text(self, message: MessageT) -> str | None:
        """Extract text from an assistant message, or None if not applicable."""
        ...

    def add_user(self, content: str) -> Self:
        self.messages.append(self._make_role_message("user", content))
        return self

    def add_assistant(self, content: str) -> Self:
        self.messages.append(self._make_role_message("assistant", content))
        return self

    def add_system(self, content: str) -> Self:
        """Store system prompt separately, added in to_api_format by subclasses."""
        self.system = content
        return self

    def last_text(self) -> str:
        """Get the last assistant text."""
        for message in reversed(self.messages):
            if (text := self._get_assistant_text(message)) is not None:
                return text
        return ""

    @abstractmethod
    def _add_response_messages(self, blocks: list[ResponseBlock]) -> None:
        """Add assistant response messages (provider-specific)."""
        ...

    def add_response(self, blocks: list[ResponseBlock]) -> Self:
        """Add assistant response messages."""
        self._add_response_messages(blocks)
        return self

    def record_snapshot(
        self,
        input_tokens: int,
        output_tokens: int = 0,
        total_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> Self:
        """Record context snapshot for truncation. Call after a complete turn (including tool results)."""
        self.context_snapshots.append(
            ContextSnapshot(
                message_index=len(self.messages),
                input_tokens=input_tokens,
                output_tokens=output_tokens,
                total_tokens=total_tokens,
                reasoning_tokens=reasoning_tokens,
            )
        )
        return self

    def truncate_prefix(self, limit: int) -> Self:
        """Truncate conversation prefix to keep at most `limit` tokens remaining."""
        total = 0 if not self.context_snapshots else self.context_snapshots[-1].input_tokens
        if total <= limit:
            return self
        threshold = total - limit
        for idx, base in enumerate(self.context_snapshots):
            if base.input_tokens >= threshold:
                self.messages = self.messages[base.message_index :]
                self.context_snapshots = [
                    ContextSnapshot(
                        message_index=s.message_index - base.message_index,
                        input_tokens=s.input_tokens - base.input_tokens,
                        output_tokens=s.output_tokens,
                        total_tokens=s.total_tokens,
                        reasoning_tokens=s.reasoning_tokens,
                    )
                    for s in self.context_snapshots[idx:]
                ]
                break
        return self

    def maybe_truncate(self) -> Self:
        """Truncate if context exceeds max_context_tokens, keeping half the budget."""
        if (
            self.max_context_tokens
            and self.context_snapshots
            and self.context_snapshots[-1].input_tokens > self.max_context_tokens
        ):
            self.truncate_prefix(self.max_context_tokens // 2)
        return self

    @abstractmethod
    def add_tool_results(self, results: list[ToolResult]) -> Self:
        """Add tool results."""
        ...

    @abstractmethod
    def to_api_format(self) -> Any:
        """Return messages in provider's native API format."""
        ...


class BaseLLM(ABC):
    @abstractmethod
    def create_conversation(self, max_context_tokens: int = DEFAULT_MAX_CONTEXT_TOKENS) -> Any:
        """Create a new conversation in this provider's native format."""
        ...

    @abstractmethod
    async def __call__(self, convo: Any, toolset: ToolSet) -> LLMOutput: ...
