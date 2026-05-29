import pytest

from search_evals.search_engines import SearchResult
from search_evals.search_engines.types import ContaminationFilter


class TestSearchResult:
    def test_creation_with_all_fields(self) -> None:
        result = SearchResult(
            url="https://example.com/page",
            title="Example Page",
            snippet="This is a test snippet.",
        )

        assert result.url == "https://example.com/page"
        assert result.title == "Example Page"
        assert result.snippet == "This is a test snippet."

    def test_model_dump_returns_dict(self) -> None:
        result = SearchResult(url="https://a.com", title="A", snippet="S")
        dumped = result.model_dump()

        assert isinstance(dumped, dict)
        assert dumped == {"url": "https://a.com", "title": "A", "snippet": "S"}

    def test_model_validate_from_dict(self) -> None:
        data = {"url": "https://b.com", "title": "B", "snippet": "Snippet B"}
        result = SearchResult.model_validate(data)

        assert result.url == "https://b.com"
        assert result.title == "B"


class TestContaminationFilter:
    @pytest.fixture
    def simple_filter(self) -> ContaminationFilter:
        return ContaminationFilter(
            title_ban_re=r"banned|forbidden",
            url_ban_re=r"evil\.com|spam\.org",
            doc_ban_re=r"contaminated|leaked",
        )

    def test_allows_clean_result(self, simple_filter: ContaminationFilter) -> None:
        result = SearchResult(
            url="https://example.com/article",
            title="A Normal Title",
            snippet="Just some normal content here.",
        )
        assert simple_filter(result) is True

    def test_blocks_banned_title(self, simple_filter: ContaminationFilter) -> None:
        result = SearchResult(
            url="https://example.com/article",
            title="This is a BANNED title",
            snippet="Normal content.",
        )
        assert simple_filter(result) is False

    def test_blocks_banned_url(self, simple_filter: ContaminationFilter) -> None:
        result = SearchResult(
            url="https://evil.com/article",
            title="Normal Title",
            snippet="Normal content.",
        )
        assert simple_filter(result) is False

    def test_blocks_banned_snippet(self, simple_filter: ContaminationFilter) -> None:
        result = SearchResult(
            url="https://example.com/article",
            title="Normal Title",
            snippet="This content is contaminated with bad data.",
        )
        assert simple_filter(result) is False

    def test_case_insensitive_matching(self, simple_filter: ContaminationFilter) -> None:
        result_upper = SearchResult(url="https://a.com", title="FORBIDDEN", snippet="s")
        result_lower = SearchResult(url="https://a.com", title="forbidden", snippet="s")
        result_mixed = SearchResult(url="https://a.com", title="FoRbIdDeN", snippet="s")

        assert simple_filter(result_upper) is False
        assert simple_filter(result_lower) is False
        assert simple_filter(result_mixed) is False

    def test_regex_patterns_work(self) -> None:
        filter_with_regex = ContaminationFilter(
            title_ban_re=r"bench.?mark",
            url_ban_re=r"hugging\s*face",
            doc_ban_re=r"simple\s*qa",
        )

        matches_title = SearchResult(url="https://a.com", title="benchmark test", snippet="s")
        matches_url = SearchResult(url="https://huggingface.co", title="t", snippet="s")
        matches_snippet = SearchResult(url="https://a.com", title="t", snippet="simple qa data")

        assert filter_with_regex(matches_title) is False
        assert filter_with_regex(matches_url) is False
        assert filter_with_regex(matches_snippet) is False

    def test_multiple_ban_patterns(self) -> None:
        filter_multi = ContaminationFilter(
            title_ban_re=r"test1|test2|test3",
            url_ban_re=r"bad1|bad2",
            doc_ban_re=r"leak",
        )

        result1 = SearchResult(url="https://a.com", title="test2 page", snippet="s")
        result2 = SearchResult(url="https://bad2.com", title="t", snippet="s")

        assert filter_multi(result1) is False
        assert filter_multi(result2) is False

    def test_partial_match_in_url(self) -> None:
        filter_url = ContaminationFilter(
            title_ban_re=r"x",
            url_ban_re=r"linkedin",
            doc_ban_re=r"x",
        )

        result = SearchResult(
            url="https://www.linkedin.com/in/someone",
            title="Normal",
            snippet="Normal",
        )
        assert filter_url(result) is False

    def test_empty_patterns_allow_all(self) -> None:
        # Empty regex patterns that match nothing
        permissive_filter = ContaminationFilter(
            title_ban_re=r"^$",  # Only matches empty string
            url_ban_re=r"^$",
            doc_ban_re=r"^$",
        )

        result = SearchResult(
            url="https://anything.com",
            title="Any Title",
            snippet="Any content",
        )
        assert permissive_filter(result) is True
