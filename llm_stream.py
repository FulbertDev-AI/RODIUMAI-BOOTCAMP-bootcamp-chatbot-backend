import asyncio
import json
from collections.abc import AsyncIterator, Awaitable, Callable, Iterator
from typing import Any

import httpx

SSE_DONE = "[DONE]"

CancelledFn = Callable[[], Awaitable[bool]]


class LLMStreamError(Exception):
    """RodiumAI HTTP/protocol error while streaming."""


class LLMStreamInterrupted(Exception):
    """Upstream stream ended without the [DONE] sentinel."""


class LLMStreamCancelled(Exception):
    """The HTTP client disconnected before the upstream stream finished."""


def format_sse(data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


def _as_token_count(value: object) -> int | None:
    if isinstance(value, bool) or not isinstance(value, int):
        return None
    return value


def normalize_usage(raw: object) -> dict | None:
    """Normalize RodiumAI/OpenAI usage. Official stream chunks may omit total_tokens."""
    if not isinstance(raw, dict):
        return None
    prompt = _as_token_count(raw.get("prompt_tokens", raw.get("input_tokens")))
    completion = _as_token_count(raw.get("completion_tokens", raw.get("output_tokens")))
    if prompt is None or completion is None:
        return None
    total = _as_token_count(raw.get("total_tokens"))
    if total is None:
        total = prompt + completion
    return {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "total_tokens": total,
    }


def usage_from_sse_line(line: str) -> dict | None:
    if not line or not line.startswith("data:"):
        return None
    payload = line[5:].strip()
    if payload == SSE_DONE:
        return None
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return None
    if not isinstance(chunk, dict):
        return None
    return normalize_usage(chunk.get("usage"))


def parse_rodium_sse_line(line: str) -> tuple[str, str | None]:
    """Classify one upstream SSE line: skip | done | content."""
    if not line or line.startswith(":"):
        return "skip", None
    if not line.startswith("data:"):
        return "skip", None
    payload = line[5:].strip()
    if payload == SSE_DONE:
        return "done", None
    try:
        chunk = json.loads(payload)
    except json.JSONDecodeError:
        return "skip", None
    choices = chunk.get("choices") or []
    if not choices:
        return "skip", None
    delta = choices[0].get("delta") or {}
    content = delta.get("content")
    if isinstance(content, str) and content:
        return "content", content
    return "skip", None


def _stream_payload(model: str, messages: list[dict]) -> dict:
    return {
        "model": model,
        "messages": messages,
        "max_tokens": 512,
        "stream": True,
        # OpenAI-compatible: last SSE chunk carries usage when this flag is set.
        "stream_options": {"include_usage": True},
    }


async def _next_upstream_line(
    lines: AsyncIterator[str],
    cancelled: CancelledFn | None,
) -> str | None:
    """Wait for the next upstream line, polling client disconnect so Stop can abort mid-wait."""
    while True:
        if cancelled is not None and await cancelled():
            raise LLMStreamCancelled()
        try:
            return await asyncio.wait_for(anext(lines), timeout=0.25)
        except TimeoutError:
            continue
        except StopAsyncIteration:
            return None


async def iter_rodium_deltas(
    messages: list[dict],
    *,
    url: str,
    api_key: str,
    model: str,
    usage_holder: dict | None = None,
    cancelled: CancelledFn | None = None,
) -> AsyncIterator[str]:
    """Yield assistant text fragments from a RodiumAI stream=true completion."""
    last_usage = None
    try:
        async with httpx.AsyncClient(timeout=httpx.Timeout(60.0, connect=10.0)) as client:
            async with client.stream(
                "POST",
                url,
                headers={"Authorization": f"Bearer {api_key}"},
                json=_stream_payload(model, messages),
            ) as response:
                if response.status_code >= 400:
                    raise LLMStreamError("The LLM API call failed.")
                finished = False
                lines = response.aiter_lines()
                while True:
                    line = await _next_upstream_line(lines, cancelled)
                    if line is None:
                        break
                    parsed_usage = usage_from_sse_line(line)
                    if parsed_usage is not None:
                        last_usage = parsed_usage
                    kind, value = parse_rodium_sse_line(line)
                    if kind == "skip":
                        continue
                    if kind == "done":
                        finished = True
                        break
                    if cancelled is not None and await cancelled():
                        raise LLMStreamCancelled()
                    yield value
                    if cancelled is not None and await cancelled():
                        raise LLMStreamCancelled()
                if not finished:
                    raise LLMStreamInterrupted()
    except LLMStreamCancelled:
        raise
    except httpx.HTTPError as exc:
        raise LLMStreamError("The LLM API call failed.") from exc
    if usage_holder is not None:
        usage_holder["usage"] = last_usage


async def iterate_text_deltas(source: Any) -> AsyncIterator[str]:
    """Accept production async generators or test doubles that yield synchronously."""
    if hasattr(source, "__aiter__"):
        async for item in source:
            yield item
        return
    if isinstance(source, Iterator):
        for item in source:
            yield item
        return
    raise TypeError("iter_rodium_deltas must return an async or sync iterator")
