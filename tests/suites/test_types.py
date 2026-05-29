from pathlib import Path

import pytest

from search_evals.agents.llms import OpenAIConversation
from search_evals.datasets import Datum
from search_evals.suites.types import GraderResult, GradeType, SuiteResult, TaskResult


class TestGradeType:
    def test_grade_type_values(self) -> None:
        assert GradeType.CORRECT.value == 1
        assert GradeType.INCORRECT.value == 2
        assert GradeType.NOT_ATTEMPTED.value == 3

    def test_grade_type_comparison(self) -> None:
        assert GradeType.CORRECT.value < GradeType.INCORRECT.value
        assert GradeType.INCORRECT.value < GradeType.NOT_ATTEMPTED.value


class TestGraderResult:
    def test_correct_score_is_one(self) -> None:
        result = GraderResult(
            grade_type=GradeType.CORRECT,
            problem="What is 2+2?",
            answer="4",
            response="The answer is 4",
            grade_text="correct: yes",
        )
        assert result.score == 1.0

    def test_incorrect_score_is_zero(self) -> None:
        result = GraderResult(
            grade_type=GradeType.INCORRECT,
            problem="What is 2+2?",
            answer="4",
            response="The answer is 5",
            grade_text="correct: no",
        )
        assert result.score == 0.0

    def test_not_attempted_score_is_zero(self) -> None:
        result = GraderResult(
            grade_type=GradeType.NOT_ATTEMPTED,
            problem="What is 2+2?",
            answer="4",
            response="I don't know",
            grade_text="not attempted",
        )
        assert result.score == 0.0

    def test_model_dump_includes_all_fields(self) -> None:
        result = GraderResult(
            grade_type=GradeType.CORRECT,
            problem="Q",
            answer="A",
            response="R",
            grade_text="G",
        )
        dumped = result.model_dump()

        assert dumped["grade_type"] == GradeType.CORRECT
        assert dumped["problem"] == "Q"
        assert dumped["answer"] == "A"
        assert dumped["response"] == "R"
        assert dumped["grade_text"] == "G"


class TestSuiteResult:
    def test_suite_result_creation(self) -> None:
        result = SuiteResult(score=0.75, total_samples=100, total_correct=75)

        assert result.score == 0.75
        assert result.total_samples == 100
        assert result.total_correct == 75

    def test_suite_result_save(self, tmp_path: Path) -> None:
        result = SuiteResult(score=0.8, total_samples=50, total_correct=40)
        output_dir = tmp_path / "test_suite"
        output_dir.mkdir()

        result.save(output_dir)

        results_file = tmp_path / "results" / "test_suite.json"
        assert results_file.exists()

    def test_suite_result_save_creates_results_dir(self, tmp_path: Path) -> None:
        result = SuiteResult(score=0.5, total_samples=10, total_correct=5)
        output_dir = tmp_path / "my_suite"
        output_dir.mkdir()

        result.save(output_dir)

        results_dir = tmp_path / "results"
        assert results_dir.exists()
        assert results_dir.is_dir()


class TestTaskResult:
    @pytest.fixture
    def sample_convo(self) -> OpenAIConversation:
        convo = OpenAIConversation()
        convo.add_user("What is 2+2?")
        convo.add_assistant("4")
        return convo

    @pytest.fixture
    def sample_task_result(self, sample_convo: OpenAIConversation) -> TaskResult:
        datum = Datum(id="test-1", problem="What is 2+2?", answer="4")
        grader_result = GraderResult(
            grade_type=GradeType.CORRECT,
            problem="What is 2+2?",
            answer="4",
            response="4",
            grade_text="A: CORRECT",
        )
        return TaskResult(datum=datum, convo=sample_convo, grader_result=grader_result)

    def test_score_delegates_to_grader_result(self, sample_task_result: TaskResult) -> None:
        assert sample_task_result.score == sample_task_result.grader_result.score

    def test_save_creates_json_file(self, sample_task_result: TaskResult, tmp_path: Path) -> None:
        task_file = tmp_path / "task.json"
        sample_task_result.save(task_file)

        assert task_file.exists()
        assert task_file.stat().st_size > 0

    def test_load_returns_equivalent_result(self, sample_task_result: TaskResult, tmp_path: Path) -> None:
        task_file = tmp_path / "task.json"
        sample_task_result.save(task_file)

        loaded = TaskResult.load(task_file)

        assert loaded is not None
        assert loaded.datum.id == sample_task_result.datum.id
        assert loaded.datum.problem == sample_task_result.datum.problem
        assert len(loaded.convo.messages) == len(sample_task_result.convo.messages)
        assert loaded.grader_result.grade_type == sample_task_result.grader_result.grade_type
        assert loaded.score == sample_task_result.score

    def test_load_nonexistent_file_returns_none(self, tmp_path: Path) -> None:
        nonexistent = tmp_path / "does_not_exist.json"
        result = TaskResult.load(nonexistent)

        assert result is None

    def test_load_invalid_json_returns_none(self, tmp_path: Path) -> None:
        invalid_file = tmp_path / "invalid.json"
        invalid_file.write_text("not valid json")

        result = TaskResult.load(invalid_file)

        assert result is None

    def test_load_wrong_schema_returns_none(self, tmp_path: Path) -> None:
        wrong_schema = tmp_path / "wrong.json"
        wrong_schema.write_text('{"foo": "bar"}')

        result = TaskResult.load(wrong_schema)

        assert result is None

    def test_save_load_preserves_metadata(self, tmp_path: Path) -> None:
        datum = Datum(id="test", problem="Q", answer="A", metadata={"key": "value"})
        convo = OpenAIConversation()
        convo.add_user("Q")
        grader_result = GraderResult(
            grade_type=GradeType.INCORRECT,
            problem="Q",
            answer="A",
            response="B",
            grade_text="wrong",
        )
        original = TaskResult(datum=datum, convo=convo, grader_result=grader_result)

        task_file = tmp_path / "task.json"
        original.save(task_file)
        loaded = TaskResult.load(task_file)

        assert loaded is not None
        assert loaded.datum.metadata == {"key": "value"}

    def test_save_load_preserves_conversation(self, tmp_path: Path) -> None:
        datum = Datum(id="test", problem="Q", answer="A")
        convo = OpenAIConversation()
        convo.add_user("Question 1")
        convo.add_assistant("Answer 1")
        convo.add_user("Question 2")
        convo.add_assistant("Answer 2")
        grader_result = GraderResult(
            grade_type=GradeType.CORRECT,
            problem="Q",
            answer="A",
            response="A",
            grade_text="correct",
        )
        original = TaskResult(datum=datum, convo=convo, grader_result=grader_result)

        task_file = tmp_path / "task.json"
        original.save(task_file)
        loaded = TaskResult.load(task_file)

        assert loaded is not None
        assert len(loaded.convo.messages) == 4
