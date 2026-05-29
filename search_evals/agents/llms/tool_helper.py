from __future__ import annotations

import re
import uuid
from typing import Any

import orjson

from search_evals.agents.types import ToolCallBlock


def safe_json_loads(s: str) -> dict[str, Any] | None:
    try:
        obj = orjson.loads(s)
        if isinstance(obj, dict):
            return obj
    except Exception:
        pass
    return None


def normalize_content(content: Any) -> str:
    if content is None:
        return ""

    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict):
                if item.get("type") == "text":
                    parts.append(str(item.get("text", "")))
                else:
                    parts.append(str(item))
            else:
                parts.append(str(item))
        return "\n".join(parts)

    return str(content)


def _clean_query(query: str) -> str:
    query = query.strip()
    if len(query) >= 2 and query[0] == query[-1] and query[0] in {"'", '"'}:
        query = query[1:-1]
    return query.strip()


def _make_tool_call(name: str, query: str) -> ToolCallBlock:
    return ToolCallBlock(
        id=f"recovered_{uuid.uuid4().hex[:8]}",
        name=name,
        input={"query": _clean_query(query)},
    )


def recover_search_web_tool_call(text: str) -> ToolCallBlock | None:
    text = text.strip()

    # 1) <tool_call name="search_web">{"query":"..."}</tool_call>
    m = re.search(
        r"<tool_call\s+name\s*=\s*[\"']search_web[\"']\s*>\s*(\{.*?\})\s*</tool_call>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        obj = safe_json_loads(m.group(1))
        if obj and isinstance(obj.get("query"), str):
            return _make_tool_call("search_web", obj["query"])

    # 2) <tool_call>search_web<arg_key>query</arg_key><arg_value>...</arg_value></tool_call>
    m = re.search(
        r"<tool_call>\s*search_web\s*<arg_key>\s*query\s*</arg_key>\s*"
        r"<arg_value>\s*(.*?)\s*</arg_value>\s*</tool_call>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return _make_tool_call("search_web", m.group(1))

    # 3) <｜｜DSML｜｜invoke name="search_web">...<｜｜DSML｜｜parameter name="query">...</...>
    m = re.search(
        r"<[^<>]*DSML[^<>]*invoke\s+name\s*=\s*[\"']search_web[\"'][^>]*>\s*"
        r".*?<[^<>]*DSML[^<>]*parameter\s+name\s*=\s*[\"']query[\"'][^>]*>\s*"
        r"(.*?)\s*</[^<>]*DSML[^<>]*parameter>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return _make_tool_call("search_web", m.group(1))

    # 4) <function=search_web>{"query":"..."}</function>
    m = re.search(
        r"<function\s*=\s*search_web>\s*(\{.*?\})\s*</function>",
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        obj = safe_json_loads(m.group(1))
        if obj and isinstance(obj.get("query"), str):
            return _make_tool_call("search_web", obj["query"])

    # 5) {"name":"search_web","parameters":{"query":"..."}}
    m = re.search(
        r'(\{\s*"name"\s*:\s*"search_web"\s*,\s*"parameters"\s*:\s*\{.*?\}\s*\})',
        text,
        flags=re.DOTALL,
    )
    if m:
        obj = safe_json_loads(m.group(1))
        if (
            obj
            and obj.get("name") == "search_web"
            and isinstance(obj.get("parameters"), dict)
            and isinstance(obj["parameters"].get("query"), str)
        ):
            return _make_tool_call("search_web", obj["parameters"]["query"])

    # 6) {"tool":"search_web","query":"..."} 또는 {"name":"search_web","query":"..."}
    m = re.search(
        r'(\{\s*"(?:tool|name)"\s*:\s*"search_web".*?\})',
        text,
        flags=re.DOTALL,
    )
    if m:
        obj = safe_json_loads(m.group(1))
        if obj:
            query = obj.get("query")
            if query is None and isinstance(obj.get("input"), dict):
                query = obj["input"].get("query")
            if isinstance(query, str):
                return _make_tool_call("search_web", query)

    # 7) search_web("...")
    m = re.search(
        r'search_web\s*\(\s*["\'](.+?)["\']\s*\)',
        text,
        flags=re.IGNORECASE | re.DOTALL,
    )
    if m:
        return _make_tool_call("search_web", m.group(1))

    return None
