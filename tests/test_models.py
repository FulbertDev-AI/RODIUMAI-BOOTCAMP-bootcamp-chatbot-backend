import httpx

from llm_stream import iter_rodium_deltas
from main import ALLOWED_MODELS, DEFAULT_MODEL

from tests.test_streaming import _parse_sse_body, _stream_chat

EXPECTED_MODELS = [
    {"id": "openai/gpt-4o", "label": "GPT-4o"},
    {"id": "openai/gpt-4o-mini", "label": "GPT-4o Mini"},
    {"id": "anthropic/claude-sonnet-4-5-20250929", "label": "Claude Sonnet 4.5"},
    {"id": "anthropic/claude-haiku-4-5-20251001", "label": "Claude Haiku 4.5"},
    {"id": "google/gemini-2.5-flash", "label": "Gemini 2.5 Flash"},
]


def test_get_models_returns_five_allowed_models(client):
    response = client.get("/models")
    assert response.status_code == 200
    body = response.json()
    assert body["models"] == EXPECTED_MODELS
    assert len(body["models"]) == 5
    assert body["default"] == DEFAULT_MODEL
    assert body["default"] in ALLOWED_MODELS


def test_valid_model_is_forwarded_with_stream_true(client, monkeypatch):
    captured: dict = {}

    def fake_deltas(messages, **kwargs):
        captured["kwargs"] = kwargs
        yield "Bonjour"

    monkeypatch.setattr("main.iter_rodium_deltas", fake_deltas)

    conversation_id = client.post("/conversations").json()["conversation_id"]
    status, body = _stream_chat(
        client,
        conversation_id,
        "Explique les listes Python.",
        model="openai/gpt-4o",
    )
    assert status == 200
    events = _parse_sse_body(body)
    assert {"type": "delta", "content": "Bonjour"} in events
    assert any(isinstance(e, dict) and e.get("type") == "done" for e in events)
    assert captured["kwargs"]["model"] == "openai/gpt-4o"


def test_iter_rodium_deltas_sends_stream_true(monkeypatch):
    seen: dict = {}

    class FakeResponse:
        status_code = 200

        def iter_lines(self):
            yield 'data: {"choices":[{"delta":{"content":"Hi"}}]}'
            yield "data: [DONE]"

        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

    def fake_stream(method, url, **kwargs):
        seen["json"] = kwargs["json"]
        return FakeResponse()

    monkeypatch.setattr(httpx, "stream", fake_stream)
    chunks = list(
        iter_rodium_deltas(
            [{"role": "user", "content": "hi"}],
            url="https://api.rodiumai.io/v1/chat/completions",
            api_key="test-key",
            model="openai/gpt-4o",
        )
    )
    assert chunks == ["Hi"]
    assert seen["json"]["model"] == "openai/gpt-4o"
    assert seen["json"]["stream"] is True


def test_each_allowed_model_can_be_used(client, monkeypatch):
    used: list[str] = []

    def fake_deltas(messages, **kwargs):
        used.append(kwargs["model"])
        yield "ok"

    monkeypatch.setattr("main.iter_rodium_deltas", fake_deltas)

    conversation_id = client.post("/conversations").json()["conversation_id"]
    for model_id in ALLOWED_MODELS:
        status, body = _stream_chat(client, conversation_id, f"msg {model_id}", model=model_id)
        assert status == 200
        events = _parse_sse_body(body)
        assert any(isinstance(e, dict) and e.get("type") == "done" for e in events)
    assert used == list(ALLOWED_MODELS)


def test_unsupported_model_returns_400_without_calling_llm(client, monkeypatch):
    calls = {"n": 0}

    def fake_deltas(messages, **kwargs):
        calls["n"] += 1
        yield "should-not-run"

    monkeypatch.setattr("main.iter_rodium_deltas", fake_deltas)

    conversation_id = client.post("/conversations").json()["conversation_id"]
    response = client.post(
        "/chat",
        json={
            "conversation_id": conversation_id,
            "message": "hello",
            "model": "totally/fake-model",
        },
    )
    assert response.status_code == 400
    assert response.json()["detail"] == "Unsupported model."
    assert calls["n"] == 0


def test_missing_model_returns_422(client, monkeypatch):
    calls = {"n": 0}

    def fake_deltas(messages, **kwargs):
        calls["n"] += 1
        yield "should-not-run"

    monkeypatch.setattr("main.iter_rodium_deltas", fake_deltas)

    conversation_id = client.post("/conversations").json()["conversation_id"]
    response = client.post(
        "/chat",
        json={"conversation_id": conversation_id, "message": "hello"},
    )
    assert response.status_code == 422
    assert calls["n"] == 0


def test_model_can_change_between_turns(client, monkeypatch):
    used: list[str] = []
    payloads: list[list[dict]] = []

    def fake_deltas(messages, **kwargs):
        used.append(kwargs["model"])
        payloads.append(messages)
        yield f"reply-{kwargs['model']}"

    monkeypatch.setattr("main.iter_rodium_deltas", fake_deltas)

    conversation_id = client.post("/conversations").json()["conversation_id"]
    status1, body1 = _stream_chat(client, conversation_id, "premier", model="openai/gpt-4o")
    status2, body2 = _stream_chat(
        client, conversation_id, "deuxième", model="google/gemini-2.5-flash"
    )
    assert status1 == 200 and status2 == 200
    assert used == ["openai/gpt-4o", "google/gemini-2.5-flash"]
    assert [m["role"] for m in payloads[0]] == ["system", "user"]
    assert [m["role"] for m in payloads[1]] == ["system", "user", "assistant", "user"]
    assert payloads[1][1]["content"] == "premier"
    assert payloads[1][2]["content"] == "reply-openai/gpt-4o"

    stored = client.get(f"/conversations/{conversation_id}/messages").json()
    assert [(m["role"], m["content"]) for m in stored] == [
        ("user", "premier"),
        ("assistant", "reply-openai/gpt-4o"),
        ("user", "deuxième"),
        ("assistant", "reply-google/gemini-2.5-flash"),
    ]
    assert any(isinstance(e, dict) and e.get("type") == "done" for e in _parse_sse_body(body1))
    assert any(isinstance(e, dict) and e.get("type") == "done" for e in _parse_sse_body(body2))
