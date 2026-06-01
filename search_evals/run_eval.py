# Portions adapted from https://github.com/perplexityai/search_evals/blob/main/search_evals/run_eval.py
# Original license: MIT.

import asyncio
import inspect
import logging
import os
import shutil
import sys
from pathlib import Path

try:
    import chz
except ModuleNotFoundError:
    chz = None

from search_evals.logging_utils import setup_logging
from search_evals.suites import DEFAULT_GRADER_MODEL, make_suite

logger = logging.getLogger(__name__)


def _coerce_cli_value(value: str, annotation: object) -> object:
    if annotation is bool:
        lowered = value.lower()
        if lowered in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if lowered in {"0", "false", "f", "no", "n", "off"}:
            return False
        raise ValueError(f"Cannot parse boolean value: {value}")
    if annotation is int:
        return int(value)
    return value


def _parse_fallback_cli(argv: list[str]) -> dict[str, object]:
    signature = inspect.signature(main)
    kwargs: dict[str, object] = {}
    for arg in argv:
        if "=" not in arg:
            raise ValueError(f"Expected key=value argument, got: {arg}")
        key, value = arg.split("=", 1)
        if key not in signature.parameters:
            raise ValueError(f"Unknown argument: {key}")
        param = signature.parameters[key]
        kwargs[key] = _coerce_cli_value(value, param.annotation)
    return kwargs


async def run_suite(
    search_engine: str,
    model: str,
    suite_name: str,
    output_dir: Path,
    dry_run: bool,
    max_workers: int,
    grader_model: str,
    dataset_path: str = "",
) -> None:
    """Run evaluation with specified search engine and suite."""
    suite = make_suite(
        suite_name,
        search_engine,
        model,
        output_dir=output_dir,
        dry_run=dry_run,
        max_workers=max_workers,
        grader_model=grader_model,
        dataset_path=Path(dataset_path) if dataset_path else None,
    )
    result = await suite()
    logger.info(f"Evaluation complete. Score: {result.score}")


def main(
    search_engine: str,
    suite: str,
    model: str,
    run_suffix: str = "",
    rerun: bool = False,
    dry_run: bool = False,
    max_workers: int = 1,
    grader_model: str = DEFAULT_GRADER_MODEL,
    dataset_path: str = "",
    openrouter_reasoning_effort: str = "",
    openrouter_max_tokens: int = 0,
    openrouter_max_context_tokens: int = 0,
) -> None:
    setup_logging()

    if openrouter_reasoning_effort:
        os.environ["OPENROUTER_REASONING_EFFORT"] = openrouter_reasoning_effort
    if openrouter_max_tokens:
        os.environ["OPENROUTER_MAX_TOKENS"] = str(openrouter_max_tokens)
    if openrouter_max_context_tokens:
        os.environ["OPENROUTER_MAX_CONTEXT_TOKENS"] = str(openrouter_max_context_tokens)

    run_name = f"{search_engine}-{model}_{suite}"
    if run_suffix:
        run_name = f"{run_name}-{run_suffix}"
    should_rerun = rerun
    output_dir = Path("runs") / run_name
    result_file = Path("runs") / "results" / f"{run_name}.json"

    # Existing task files are reused, so an already-written summary file should not block resuming.
    if not should_rerun and result_file.exists():
        logger.info(
            f"Results for {run_name} already exist. Resuming from cached task files instead of starting over."
        )

    if dry_run and not should_rerun:
        logger.info(f"Dry run enabled for {run_name}. Reusing any existing cached task files in {output_dir}.")

    if should_rerun:
        if output_dir.exists():
            shutil.rmtree(output_dir)
        if result_file.exists():
            result_file.unlink()

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Output directory: {output_dir}")

    asyncio.run(run_suite(search_engine, model, suite, output_dir, dry_run, max_workers, grader_model, dataset_path))


if __name__ == "__main__":
    if chz is not None:
        chz.entrypoint(main)
    else:
        main(**_parse_fallback_cli(sys.argv[1:]))
