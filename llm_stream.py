import json

import httpx

SSE_DONE = "[DONE]"


class LLMStreamError(Exception):
    """RodiumAI HTTP/protocol error while streaming."""


class LLMStreamInterrupted(Exception):
    """Upstream stream ended without the [DONE] sentinel."""


def format_sse(data: dict | str) -> str:
    payload = data if isinstance(data, str) else json.dumps(data, ensure_ascii=False)
    return f"data: {payload}\n\n"


def normalize_usage(raw: object) -> dict | None:
    if not isinstance(raw, dict):
        return None
    prompt = raw.get("prompt_tokens")
    completion = raw.get("completion_tokens")
    total = raw.get("total_tokens")
    if not all(isinstance(value, int) and not isinstance(value, bool) for value in (prompt, completion, total)):
        return None
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


def iter_rodium_deltas(
    messages: list[dict],
    *,
    url: str,
    api_key: str,
    model: str,
    usage_holder: dict | None = None,
):
    """Yield assistant text fragments from a RodiumAI stream=true completion."""
    last_usage = None
    try:
        with httpx.stream(
            "POST",
            url,
            headers={"Authorization": f"Bearer {api_key}"},
            json={
                "model": model,
                "messages": messages,
                "max_tokens": 512,
                "stream": True,
            },
            timeout=60.0,
        ) as response:
            if response.status_code >= 400:
                raise LLMStreamError("The LLM API call failed.")
            finished = False
            for line in response.iter_lines():
                parsed_usage = usage_from_sse_line(line)
                if parsed_usage is not None:
                    last_usage = parsed_usage
                kind, value = parse_rodium_sse_line(line)
                if kind == "skip":
                    continue
                if kind == "done":
                    finished = True
                    break
                yield value
            if not finished:
                raise LLMStreamInterrupted()
    except httpx.HTTPError as exc:
        raise LLMStreamError("The LLM API call failed.") from exc
    if usage_holder is not None:
        usage_holder["usage"] = last_usage
