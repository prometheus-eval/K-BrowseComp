from __future__ import annotations

import re
from html import unescape
from html.parser import HTMLParser
from typing import Any, ClassVar
from urllib.parse import urlparse

import aiohttp
from pydantic import BaseModel, Field

from search_evals.agents.tools.base import Tool, ToolDef


class OpenURLInput(BaseModel):
    url: str = Field(description="The exact http(s) URL to open and inspect.")


class OpenURLOutput(BaseModel):
    url: str
    final_url: str = ""
    status: int = 0
    content_type: str = ""
    title: str = ""
    text: str = ""
    error: str = ""


class OpenURLToolDef(ToolDef[OpenURLInput, OpenURLOutput]):
    name: ClassVar[str] = "open_url"
    description: ClassVar[str] = (
        "Opens an exact http(s) URL and returns extracted readable text. "
        "Use this to verify cited source URLs directly instead of relying only on search snippets."
    )
    input_schema: ClassVar[type[BaseModel]] = OpenURLInput
    output_schema: ClassVar[type[BaseModel]] = OpenURLOutput


class _HTMLTextExtractor(HTMLParser):
    def __init__(self) -> None:
        super().__init__()
        self.in_ignored = False
        self.in_title = False
        self.title_parts: list[str] = []
        self.text_parts: list[str] = []

    def handle_starttag(self, tag: str, attrs: list[tuple[str, str | None]]) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.in_ignored = True
        if tag == "title":
            self.in_title = True

    def handle_endtag(self, tag: str) -> None:
        if tag in {"script", "style", "noscript", "svg"}:
            self.in_ignored = False
        if tag == "title":
            self.in_title = False

    def handle_data(self, data: str) -> None:
        if self.in_ignored:
            return
        text = data.strip()
        if not text:
            return
        if self.in_title:
            self.title_parts.append(text)
        self.text_parts.append(text)


def _is_allowed_url(url: str) -> tuple[bool, str]:
    parsed = urlparse(url)
    if parsed.scheme not in {"http", "https"}:
        return False, "Only http(s) URLs are allowed."
    host = (parsed.hostname or "").lower()
    if not host:
        return False, "URL has no hostname."
    if host in {"localhost", "127.0.0.1", "0.0.0.0", "::1"} or host.endswith(".local"):
        return False, "Localhost/private URLs are not allowed."
    return True, ""


def _html_to_text(raw: str) -> tuple[str, str]:
    parser = _HTMLTextExtractor()
    parser.feed(raw)
    title = " ".join(parser.title_parts)
    text = "\n".join(parser.text_parts)
    text = unescape(text)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    return title.strip(), text.strip()


class OpenURLTool(Tool[OpenURLInput, OpenURLOutput]):
    tool_def: ClassVar[type[ToolDef[Any, Any]]] = OpenURLToolDef

    def __init__(self, max_chars: int = 20000, timeout_s: float = 20.0) -> None:
        self.max_chars = max_chars
        self.timeout_s = timeout_s

    async def __call__(self, input_: OpenURLInput) -> OpenURLOutput:
        allowed, reason = _is_allowed_url(input_.url)
        if not allowed:
            return OpenURLOutput(url=input_.url, error=reason)

        timeout = aiohttp.ClientTimeout(total=self.timeout_s)
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; KBrowseCompVerifier/1.0; "
                "+https://github.com/openai/search-evals)"
            )
        }
        try:
            async with aiohttp.ClientSession(timeout=timeout, headers=headers) as session:
                async with session.get(input_.url, allow_redirects=True) as response:
                    content_type = response.headers.get("content-type", "")
                    raw = await response.text(errors="replace")
                    raw = raw[: self.max_chars * 2]
                    if "html" in content_type.lower():
                        title, text = _html_to_text(raw)
                    else:
                        title, text = "", raw.strip()
                    return OpenURLOutput(
                        url=input_.url,
                        final_url=str(response.url),
                        status=response.status,
                        content_type=content_type,
                        title=title[:500],
                        text=text[: self.max_chars],
                    )
        except Exception as exc:
            return OpenURLOutput(url=input_.url, error=repr(exc))
