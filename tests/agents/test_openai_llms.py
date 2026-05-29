from types import SimpleNamespace

from search_evals.agents.llms.openai import OpenAILLM


def test_openai_records_reasoning_tokens_from_output_details() -> None:
    response = SimpleNamespace(
        output=[
            SimpleNamespace(
                type="message",
                content=[
                    SimpleNamespace(type="output_text", text="Done"),
                ],
            )
        ],
        usage=SimpleNamespace(
            input_tokens=11,
            output_tokens=13,
            total_tokens=24,
            output_tokens_details=SimpleNamespace(reasoning_tokens=7),
        ),
    )

    output = OpenAILLM("gpt-test")._parse_response(response)

    assert output.text == "Done"
    assert output.input_tokens == 11
    assert output.output_tokens == 13
    assert output.total_tokens == 24
    assert output.reasoning_tokens == 7


def test_openai_records_reasoning_tokens_from_completion_details() -> None:
    response = SimpleNamespace(
        output=[],
        usage={
            "input_tokens": 11,
            "output_tokens": 13,
            "total_tokens": 24,
            "completion_tokens_details": {"reasoning_tokens": 5},
        },
    )

    output = OpenAILLM("gpt-test")._parse_response(response)

    assert output.reasoning_tokens == 5
