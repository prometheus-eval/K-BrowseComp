# Portions adapted from https://github.com/perplexityai/search_evals/blob/main/search_evals/logging_utils.py
# Original license: MIT.

import logging

_NOISY_LOGGERS = [
    "aiohttp",
    "asyncio",
    "httpx",
    "openai",
]


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="[%(asctime)s] [%(filename)s:%(lineno)d] %(message)s",
    )
    for logger_name in _NOISY_LOGGERS:
        logging.getLogger(logger_name).setLevel(logging.WARNING)
