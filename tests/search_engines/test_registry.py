# Portions adapted from https://github.com/perplexityai/search_evals/blob/main/tests/search_engines/test_registry.py
# Original license: MIT.

import pytest

from search_evals.search_engines.registry import make_search_engine


def test_make_search_engine_invalid_name() -> None:
    with pytest.raises(ValueError, match="Search engine 'invalid' not found"):
        make_search_engine("invalid")
