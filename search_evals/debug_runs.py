# Portions adapted from https://github.com/perplexityai/search_evals/blob/main/search_evals/debug_runs.py
# Original license: MIT.

import re
from pathlib import Path
from typing import Any

import chz
import orjson
from rich.console import Console
from rich.markup import escape
from rich.panel import Panel
from rich.table import Table

from search_evals.suites.types import GradeType, TaskResult

MAX_SNIPPET_LEN = 80
MAX_RESULTS = 10


def load_run_results(run_name: str, runs_dir: Path) -> dict[str, TaskResult]:
    run_path = runs_dir / run_name
    if not run_path.exists():
        raise FileNotFoundError(f"Run directory not found: {run_path}")

    results: dict[str, TaskResult] = {}
    for json_file in run_path.glob("*.json"):
        task_result = TaskResult.load(json_file)
        if task_result is not None:
            results[task_result.datum.id] = task_result

    return results


def find_common_ids(run_results: dict[str, dict[str, TaskResult]]) -> list[str]:
    if not run_results:
        return []

    id_sets = [set(results.keys()) for results in run_results.values()]
    common = set.intersection(*id_sets)
    return sorted(common)


def parse_results_json(data: str | dict[str, Any] | list[Any]) -> list[dict[str, Any]]:
    if isinstance(data, list):
        return data
    if isinstance(data, dict):
        if "results" in data:
            return list(data["results"])
        return [data]
    # isinstance(data, str)
    try:
        parsed = orjson.loads(data)
        return parse_results_json(parsed)
    except Exception:
        return []


class ContextSnapshotInfo:
    """Context snapshot info for display."""

    def __init__(
        self,
        message_index: int,
        input_tokens: int,
        output_tokens: int = 0,
        total_tokens: int = 0,
        reasoning_tokens: int = 0,
    ) -> None:
        self.message_index = message_index
        self.input_tokens = input_tokens
        self.output_tokens = output_tokens
        self.total_tokens = total_tokens
        self.reasoning_tokens = reasoning_tokens


class SearchStep:
    """A single search step with query and results."""

    def __init__(self, query: str, results: list[dict[str, Any]] | None = None) -> None:
        self.query = query
        self.results: list[dict[str, Any]] = results or []
        self.snapshot: ContextSnapshotInfo | None = None


def extract_search_info(convo: Any) -> tuple[list[SearchStep], list[ContextSnapshotInfo], str]:
    """Extract search info, returning queries and results grouped by step."""
    steps: list[SearchStep] = []
    current_step: SearchStep | None = None

    if hasattr(convo, "model_dump"):
        convo = convo.model_dump()

    system = convo.get("system", "")

    # Extract context snapshots
    snapshots = [
        ContextSnapshotInfo(
            s.get("message_index", 0),
            s.get("input_tokens", 0),
            s.get("output_tokens", 0),
            s.get("total_tokens", 0),
            s.get("reasoning_tokens", 0),
        )
        for s in convo.get("context_snapshots", [])
    ]

    for msg in convo.get("messages", []):
        msg_type = msg.get("type")

        # OpenAI format: function_call with arguments
        if msg_type == "function_call" and msg.get("name") == "search_web":
            args = msg.get("arguments", "")
            if args:
                try:
                    parsed = orjson.loads(args)
                    query = parsed.get("query", "")
                    if query:
                        current_step = SearchStep(query)
                        steps.append(current_step)
                except Exception:
                    pass

        # OpenAI format: function_call_output with output
        if msg_type == "function_call_output":
            output = msg.get("output", "")
            if output and current_step is not None:
                current_step.results = parse_results_json(output)

        content = msg.get("content")
        if not content:
            continue

        if isinstance(content, str):
            continue

        for block in content:
            if not isinstance(block, dict):
                continue

            # Anthropic format: tool_use
            if block.get("type") == "tool_use" and block.get("name") == "search_web":
                inp = block.get("input", {})
                if isinstance(inp, dict):
                    query = inp.get("query", "")
                    if query:
                        current_step = SearchStep(query)
                        steps.append(current_step)

            # Anthropic format: tool_result
            if block.get("type") == "tool_result":
                content_str = block.get("content", "")
                if current_step is not None:
                    current_step.results = parse_results_json(content_str)

    # Associate snapshots with steps (snapshot i corresponds to step i)
    for i, step in enumerate(steps):
        if i < len(snapshots):
            step.snapshot = snapshots[i]

    return steps, snapshots, system


def truncate(text: str, max_len: int) -> str:
    text = re.sub(r"<[^>]+>", "", text)
    text = text.replace("\n", " ").replace("\\n", " ").strip()
    return text[:max_len] + "..." if len(text) > max_len else text


def display_comparison(
    console: Console,
    datum_id: str,
    run_results: dict[str, dict[str, TaskResult]],
    run_names: list[str],
    current_idx: int,
    total_count: int,
) -> None:
    console.clear()

    first_result = run_results[run_names[0]][datum_id]
    datum = first_result.datum

    console.print(
        Panel(
            f"[bold]Question {current_idx + 1}/{total_count}[/bold]  |  ID: {datum.id}",
            style="blue",
        )
    )

    console.print(
        Panel(
            escape(datum.problem),
            title="[bold white]Question[/bold white]",
            border_style="white",
        )
    )

    console.print(
        Panel(
            f"[cyan]{escape(datum.answer)}[/cyan]",
            title="[bold cyan]Correct Answer[/bold cyan]",
            border_style="cyan",
        )
    )

    console.print()

    for run_name in run_names:
        task_result = run_results[run_name][datum_id]
        grader = task_result.grader_result
        convo = task_result.convo

        grade_text: str
        border_style: str
        match grader.grade_type:
            case GradeType.CORRECT:
                grade_text = "[bold green]CORRECT[/bold green]"
                border_style = "green"
            case GradeType.INCORRECT:
                grade_text = "[bold red]INCORRECT[/bold red]"
                border_style = "red"
            case GradeType.NOT_ATTEMPTED:
                grade_text = "[bold yellow]NOT ATTEMPTED[/bold yellow]"
                border_style = "yellow"

        display_name = run_name.split("_")[0] if "_" in run_name else run_name

        search_steps, snapshots, _system = extract_search_info(convo)

        lines = []

        if search_steps:
            final_snapshot = snapshots[-1] if snapshots else None
            final_tokens = final_snapshot.input_tokens if final_snapshot else 0
            # Only show token info if we have meaningful data (> 100 tokens)
            has_token_data = final_tokens > 100
            header_tokens = f", {final_tokens:,} input tokens" if has_token_data else ""
            if final_snapshot and final_snapshot.reasoning_tokens:
                header_tokens += f", {final_snapshot.reasoning_tokens:,} reasoning tokens"
            lines.append(f"[bold yellow]Search Steps ({len(search_steps)} searches{header_tokens}):[/bold yellow]")
            for i, step in enumerate(search_steps, 1):
                token_info = ""
                if has_token_data and step.snapshot and step.snapshot.input_tokens > 100:
                    usage_parts = [f"{step.snapshot.input_tokens:,} in"]
                    if step.snapshot.output_tokens:
                        usage_parts.append(f"{step.snapshot.output_tokens:,} out")
                    if step.snapshot.reasoning_tokens:
                        usage_parts.append(f"{step.snapshot.reasoning_tokens:,} reasoning")
                    token_info = f" [cyan]({', '.join(usage_parts)})[/cyan]"
                lines.append(f"  [bold]{i}.[/bold]{token_info} {escape(step.query)}")
                # Show results for this step (limited)
                if step.results:
                    for r in step.results[:MAX_RESULTS]:
                        url = truncate(r.get("url", ""), 45)
                        title = truncate(r.get("title", ""), 30)
                        snippet_len = len(r.get("snippet", ""))
                        lines.append(f"       [dim]{url}[/dim] | {escape(title)} | {snippet_len} chars")
            lines.append("")

        lines.append("[bold yellow]Response:[/bold yellow]")
        response_text = escape(grader.response) if grader.response else "[dim]No response[/dim]"
        lines.append(response_text)

        console.print(
            Panel(
                "\n".join(lines),
                title=f"[bold blue]{display_name}[/bold blue]  {grade_text}",
                border_style=border_style,
                padding=(1, 2),
            )
        )


def display_summary(
    console: Console,
    run_results: dict[str, dict[str, TaskResult]],
    run_names: list[str],
    common_ids: list[str],
) -> None:
    table = Table(title="Run Comparison Summary")
    table.add_column("Run Name", style="cyan")
    table.add_column("Correct", style="green")
    table.add_column("Incorrect", style="red")
    table.add_column("Not Attempted", style="yellow")
    table.add_column("Score", style="bold")

    for run_name in run_names:
        results = run_results[run_name]
        correct = sum(1 for id in common_ids if results[id].grader_result.grade_type == GradeType.CORRECT)
        incorrect = sum(1 for id in common_ids if results[id].grader_result.grade_type == GradeType.INCORRECT)
        not_attempted = sum(1 for id in common_ids if results[id].grader_result.grade_type == GradeType.NOT_ATTEMPTED)
        score = correct / len(common_ids) if common_ids else 0

        table.add_row(
            run_name,
            str(correct),
            str(incorrect),
            str(not_attempted),
            f"{score:.1%}",
        )

    console.print(table)


def interactive_loop(
    console: Console,
    datum_ids: list[str],
    run_results: dict[str, dict[str, TaskResult]],
    run_names: list[str],
) -> None:
    total = len(datum_ids)

    for idx in range(total):
        display_comparison(
            console,
            datum_ids[idx],
            run_results,
            run_names,
            idx,
            total,
        )

        console.print()
        console.print("[dim][Enter] next | [q] quit[/dim]")

        user_input = console.input("[bold]> [/bold]").strip().lower()

        if user_input in ("q", "quit", "exit"):
            console.print("[yellow]Exiting...[/yellow]")
            break


def main(
    runs: list[str],
    runs_dir: str = "runs",
) -> None:
    console = Console()
    runs_path = Path(runs_dir)

    if len(runs) < 2:
        console.print("[red]Error: Need at least 2 runs to compare[/red]")
        return

    console.print("[bold]Loading run results...[/bold]")
    run_results: dict[str, dict[str, TaskResult]] = {}

    for run_name in runs:
        try:
            console.print(f"  Loading {run_name}...")
            run_results[run_name] = load_run_results(run_name, runs_path)
            console.print(f"    Loaded {len(run_results[run_name])} results")
        except FileNotFoundError as e:
            console.print(f"[red]Error: {e}[/red]")
            return

    common_ids = find_common_ids(run_results)
    console.print(f"\n[bold]Found {len(common_ids)} common questions across all runs[/bold]")

    if not common_ids:
        console.print("[red]No common questions found across runs[/red]")
        return

    console.print()
    display_summary(console, run_results, runs, common_ids)

    console.print("\n[dim]Press Enter to start browsing...[/dim]")
    console.input()

    try:
        interactive_loop(console, common_ids, run_results, runs)
    except KeyboardInterrupt:
        console.print("\n[yellow]Interrupted. Exiting...[/yellow]")


if __name__ == "__main__":
    chz.entrypoint(main)
