import asyncio

import httpx

from llm_stream import LLMStreamError, iter_rodium_deltas, normalize_usage
from tests.test_streaming import _parse_sse_body, _stream_chat


def _patch_async_stream(monkeypatch, lines: list[str], seen: dict | None = None):
    class FakeResponse:
        status_code = 200

        async def aiter_lines(self):
            for line in lines:
                yield line

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

    class FakeAsyncClient:
        def __init__(self, *args, **kwargs):
            pass

        async def __aenter__(self):
            return self

        async def __aexit__(self, *args):
            return False

        def stream(self, method, url, **kwargs):
            if seen is not None:
                seen["json"] = kwargs.get("json")
            return FakeResponse()

    monkeypatch.setattr(httpx, "AsyncClient", FakeAsyncClient)


def test_done_includes_usage_from_last_chunk(client, monkeypatch):
    _patch_async_stream(
        monkeypatch,
        [
            'data: {"choices":[{"delta":{"content":"Bonjour"}}]}',
            'data: {"choices":[],"usage":{"prompt_tokens":100,"completion_tokens":25}}',
            "data: [DONE]",
        ],
    )

    conversation_id = client.post("/conversations").json()["conversation_id"]
    status, body = _stream_chat(client, conversation_id, "Salut")
    assert status == 200
    events = _parse_sse_body(body)
    done = next(e for e in events if isinstance(e, dict) and e.get("type") == "done")
    assert done["reply"] == "Bonjour"
    assert done["usage"] == {
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
    }


def test_done_usage_null_when_absent(client, monkeypatch):
    def fake_deltas(messages, **kwargs):
        yield "ok"

    monkeypatch.setattr("main.iter_rodium_deltas", fake_deltas)

    conversation_id = client.post("/conversations").json()["conversation_id"]
    status, body = _stream_chat(client, conversation_id, "Salut")
    assert status == 200
    events = _parse_sse_body(body)
    done = next(e for e in events if isinstance(e, dict) and e.get("type") == "done")
    assert done["usage"] is None
    stored = client.get(f"/conversations/{conversation_id}/messages").json()
    assert len(stored) == 2
    assert all("usage" not in m and "prompt_tokens" not in m for m in stored)


def test_partial_or_invalid_usage_becomes_null():
    assert normalize_usage({"prompt_tokens": 100}) is None
    assert normalize_usage({"prompt_tokens": "100", "completion_tokens": 25, "total_tokens": 125}) is None
    assert normalize_usage(None) is None
    assert normalize_usage("nope") is None


def test_usage_without_total_tokens_is_normalized():
    # RodiumAI stream example: usage chunk may omit total_tokens.
    assert normalize_usage({"prompt_tokens": 100, "completion_tokens": 25}) == {
        "prompt_tokens": 100,
        "completion_tokens": 25,
        "total_tokens": 125,
    }


def test_invalid_usage_in_stream_does_not_fail_chat(client, monkeypatch):
    _patch_async_stream(
        monkeypatch,
        [
            'data: {"choices":[{"delta":{"content":"Hi"}}]}',
            'data: {"choices":[{"delta":{}}],"usage":{"prompt_tokens":100}}',
            "data: [DONE]",
        ],
    )

    conversation_id = client.post("/conversations").json()["conversation_id"]
    status, body = _stream_chat(client, conversation_id, "Salut")
    assert status == 200
    events = _parse_sse_body(body)
    done = next(e for e in events if isinstance(e, dict) and e.get("type") == "done")
    assert done["reply"] == "Hi"
    assert done["usage"] is None


def test_error_after_deltas_has_no_done_usage_or_persist(client, monkeypatch):
    def fake_deltas(messages, usage_holder=None, **kwargs):
        if usage_holder is not None:
            usage_holder["usage"] = {
                "prompt_tokens": 10,
                "completion_tokens": 2,
                "total_tokens": 12,
            }
        yield "Bonjour"
        raise LLMStreamError("fail")

    monkeypatch.setattr("main.iter_rodium_deltas", fake_deltas)

    conversation_id = client.post("/conversations").json()["conversation_id"]
    status, body = _stream_chat(client, conversation_id, "Salut")
    assert status == 200
    events = _parse_sse_body(body)
    assert not any(isinstance(e, dict) and e.get("type") == "done" for e in events)
    assert not any(isinstance(e, dict) and "usage" in e and e.get("type") != "error" for e in events)
    assert client.get(f"/conversations/{conversation_id}/messages").json() == []


def test_iter_rodium_deltas_keeps_last_valid_usage(monkeypatch):
    _patch_async_stream(
        monkeypatch,
        [
            'data: {"choices":[{"delta":{"content":"A"}}],"usage":{"prompt_tokens":1,"completion_tokens":1,"total_tokens":2}}',
            'data: {"choices":[],"usage":{"prompt_tokens":10,"completion_tokens":5}}',
            "data: [DONE]",
        ],
    )
    holder: dict = {"usage": None}

    async def collect():
        return [
            chunk
            async for chunk in iter_rodium_deltas(
                [{"role": "user", "content": "hi"}],
                url="https://example.invalid",
                api_key="test-key",
                model="openai/gpt-4o",
                usage_holder=holder,
            )
        ]

    chunks = asyncio.run(collect())
    assert chunks == ["A"]
    assert holder["usage"] == {
        "prompt_tokens": 10,
        "completion_tokens": 5,
        "total_tokens": 15,
    }
