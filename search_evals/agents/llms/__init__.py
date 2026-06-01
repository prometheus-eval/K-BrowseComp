# Portions adapted from https://github.com/perplexityai/search_evals/blob/main/search_evals/agents/llms/__init__.py
# Original license: MIT.

# search_evals/agents/llms/__init__.py
from __future__ import annotations

import os

from search_evals.agents.llms.anthropic import AnthropicConversation, AnthropicLLM
from search_evals.agents.llms.base import DEFAULT_MAX_CONTEXT_TOKENS, BaseLLM, ContextSnapshot, LLMOutput
from search_evals.agents.llms.gemini import GeminiConversation, GeminiLLM
from search_evals.agents.llms.openai import OpenAIConversation, OpenAILLM
from search_evals.agents.llms.openai_chat import OpenAIChatConversation, OpenAIChatLLM
from search_evals.agents.llms.openrouter import OpenRouterConversation, OpenRouterLLM, make_openrouter_llm

Conversation = (
    OpenAIConversation
    | AnthropicConversation
    | GeminiConversation
    | OpenAIChatConversation
    | OpenRouterConversation
)


def make_llm(model: str) -> BaseLLM:
    match model:
        # OpenAI (Responses API-based engine in this repo)
        case m if m.startswith("gpt"):
            return OpenAILLM(model=m)

        # Anthropic
        case m if m.startswith("claude"):
            return AnthropicLLM(model=m)

        # Gemini
        case m if m.startswith("gemini"):
            return GeminiLLM(model=m)

        # OpenRouter (OpenAI Chat Completions compatible)
        case m if m.startswith("openrouter:"):
            real_model = m.split("openrouter:", 1)[1]
            return make_openrouter_llm(real_model)

        # Open-source LLMs via OpenAI-compatible Chat Completions endpoint (vLLM/Ollama/TGI/etc.)
        # Example: "local:Qwen/Qwen3-8B"
        case m if m.startswith("local:"):
            real_model = m.split("local:", 1)[1]
            base_url = os.getenv("OPENAI_COMPAT_BASE_URL", "http://localhost:8000/v1")
            api_key = os.getenv("OPENAI_COMPAT_API_KEY", "local")

            return OpenAIChatLLM(
                model=real_model,
                base_url=base_url,
                api_key=api_key,
            )

        # LiteLLM proxy: route through the LiteLLM gateway via its OpenAI-compatible
        # Chat Completions endpoint. Model name format: "litellm_proxy/<provider>/<model>"
        # e.g. "litellm_proxy/azure_ai/gpt-5.4-mini", "litellm_proxy/gemini/gemini-3-flash-preview".
        case m if m.startswith("litellm_proxy/"):
            real_model = m.split("litellm_proxy/", 1)[1]
            base_url = (
                os.getenv("LITELLM_PROXY_BASE_URL")
                or os.getenv("OPENAI_BASE_URL")
                or "http://localhost:4000"
            )
            api_key = (
                os.getenv("LITELLM_PROXY_API_KEY")
                or os.getenv("OPENAI_API_KEY")
                or "litellm"
            )
            return OpenAIChatLLM(
                model=real_model,
                base_url=base_url,
                api_key=api_key,
            )

        case _:
            raise ValueError(f"Unknown model: {model}")


__all__ = [
    "AnthropicConversation",
    "AnthropicLLM",
    "BaseLLM",
    "ContextSnapshot",
    "Conversation",
    "DEFAULT_MAX_CONTEXT_TOKENS",
    "GeminiConversation",
    "GeminiLLM",
    "LLMOutput",
    "OpenAIConversation",
    "OpenAILLM",
    "OpenAIChatConversation",
    "OpenAIChatLLM",
    "OpenRouterConversation",
    "OpenRouterLLM",
    "make_llm",
    "make_openrouter_llm",
]
