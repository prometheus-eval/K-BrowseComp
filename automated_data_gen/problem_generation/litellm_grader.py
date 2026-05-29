"""Grader for the --litellm code path.

Mirrors `search_evals.suites.graders.DeepResearchGrader` (same prompt, same
output schema, same retry policy) but routes through a LiteLLM proxy using
`litellm.acompletion(...)` with JSON-mode `response_format`. Used by
solver_runner.py when `--litellm` is set.
"""
from __future__ import annotations

import os
from typing import Literal

import litellm
import orjson
from pydantic import BaseModel
from tenacity import retry, stop_after_attempt, wait_exponential

from search_evals.agents.llms import Conversation
from search_evals.datasets import Datum
from search_evals.suites.graders import DEFAULT_DEEP_RESEARCH_GRADER_PROMPT
from search_evals.suites.types import AsyncBaseGrader, GraderResult, GradeType


class LiteLLMProxyGrader(AsyncBaseGrader):
    class ExtractedAnswer(BaseModel):
        extracted_final_answer: str
        reasoning: str
        correct: Literal["yes", "no"]
        confidence: int

    def __init__(
        self,
        prompt: str = DEFAULT_DEEP_RESEARCH_GRADER_PROMPT,
        model: str = "litellm_proxy/azure_ai/gpt-5.4-mini",
        base_url: str | None = None,
        api_key: str | None = None,
    ) -> None:
        self.prompt = prompt
        self.model = model
        self.base_url = (
            base_url
            or os.getenv("LITELLM_PROXY_BASE_URL")
            or os.getenv("OPENAI_BASE_URL")
        )
        self.api_key = (
            api_key
            or os.getenv("LITELLM_PROXY_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )

    @retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=10))
    async def __call__(self, datum: Datum, convo: Conversation) -> GraderResult:
        agent_response = convo.last_text()
        prompt = self.prompt.format(
            question=datum.problem,
            response=agent_response,
            correct_answer=datum.answer,
        )
        prompt_json = (
            prompt
            + "\n\nReturn your judgement as a single JSON object with these fields:\n"
            "  - extracted_final_answer (string): the final exact answer extracted from "
            "the response, or \"None\" if no extractable answer.\n"
            "  - reasoning (string): explain why the extracted answer is correct or "
            "incorrect based on the [correct_answer].\n"
            "  - correct (\"yes\" or \"no\"): whether the extracted answer matches the "
            "[correct_answer].\n"
            "  - confidence (integer 0-100): the confidence score extracted from the "
            "response (or 100 if none stated).\n"
            "Return ONLY the JSON object, nothing else."
        )

        resp = await litellm.acompletion(
            model=self.model,
            messages=[{"role": "user", "content": prompt_json}],
            base_url=self.base_url,
            api_key=self.api_key,
            response_format={"type": "json_object"},
        )
        text = resp.choices[0].message.content or "{}"

        try:
            parsed = self.ExtractedAnswer.model_validate(orjson.loads(text))
        except Exception:
            parsed = None

        grade_text = (
            parsed.reasoning if parsed
            else f"grader output parse failure: {text[:500]}"
        )
        return GraderResult(
            grade_type=GradeType.CORRECT if (parsed and parsed.correct == "yes") else GradeType.INCORRECT,
            problem=datum.problem,
            answer=datum.answer,
            response=agent_response,
            grade_text=grade_text,
        )
