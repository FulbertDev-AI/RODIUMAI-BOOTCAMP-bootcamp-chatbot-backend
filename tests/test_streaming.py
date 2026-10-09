import json

from database.models import Conversation, Message
from llm_stream import (
    LLMStreamCancelled,
    LLMStreamError,
    LLMStreamInterrupted,
    parse_rodium_sse_line,
)
from main import NOTE_ROLE, NOTIFICATION_ROLE, NOTIFICATION_TEXT, load_system_prompt


def _parse_sse_body(text: str) -> list[dict | str]:
    events: list[dict | str] = []
    for block in text.split("\n\n"):
        line = block.strip()
        if not line.startswith("data:"):
            continue
        payload = line[5:].strip()
        if payload == "[DONE]":
            events.append("[DONE]")
        else:
            events.append(json.loads(payload))
    return events


def _stream_chat(
    client, conversation_id: int, message: str, model: str = "openai/gpt-4o"
) -> tuple[int, str]:
    with client.stream(
        "POST",
        "/chat",
        json={"conversation_id": conversation_id, "message": message, "model": model},
    ) as response:
        return response.status_code, response.read().decode()


def test_parse_ignores_chunk_without_text_content():
    kind, value = parse_rodium_sse_line(
        'data: {"choices":[{"delta":{}}]}'
    )
    assert kind == "skip"
    assert value is None


def test_streaming_success_persists_user_and_assistant(client, monkeypatch):
    def fake_deltas(messages, **kwargs):
        yield "Bonjour"
        yield " "
        yield "étudiant"

    monkeypatch.setattr("main.iter_rodium_deltas", fake_deltas)

    conversation_id = client.post("/conversations").json()["conversation_id"]
    status, body = _stream_chat(client, conversation_id, "Salut")
    assert status == 200
    events = _parse_sse_body(body)
    assert events[0] == {"type": "delta", "content": "Bonjour"}
    assert events[1] == {"type": "delta", "content": " "}
    assert events[2] == {"type": "delta", "content": "étudiant"}
    assert events[3] == {
        "type": "done",
        "reply": "Bonjour étudiant",
        "notification": None,
        "usage": None,
    }
    assert events[4] == "[DONE]"
    assert not any(isinstance(e, dict) and e.get("type") == "error" for e in events)

    stored = client.get(f"/conversations/{conversation_id}/messages").json()
    assert [(m["role"], m["content"]) for m in stored] == [
        ("user", "Salut"),
        ("assistant", "Bonjour étudiant"),
    ]


def test_error_during_stream_persists_nothing(client, monkeypatch):
    def fake_deltas(messages, **kwargs):
        yield "Bonjour"
        raise LLMStreamError("fail")

    monkeypatch.setattr("main.iter_rodium_deltas", fake_deltas)

    conversation_id = client.post("/conversations").json()["conversation_id"]
    status, body = _stream_chat(client, conversation_id, "Salut")
    assert status == 200
    events = _parse_sse_body(body)
    assert events[0] == {"type": "delta", "content": "Bonjour"}
    assert {"type": "error", "message": "The LLM API call failed."} in events
    assert "[DONE]" in events
    assert not any(isinstance(e, dict) and e.get("type") == "done" for e in events)

    stored = client.get(f"/conversations/{conversation_id}/messages").json()
    assert stored == []


def test_interrupted_stream_before_done_persists_nothing(client, monkeypatch):
    def fake_deltas(messages, **kwargs):
        yield "Bonjour"
        raise LLMStreamInterrupted()

    monkeypatch.setattr("main.iter_rodium_deltas", fake_deltas)

    conversation_id = client.post("/conversations").json()["conversation_id"]
    status, body = _stream_chat(client, conversation_id, "Salut")
    assert status == 200
    events = _parse_sse_body(body)
    assert events[0] == {"type": "delta", "content": "Bonjour"}
    assert not any(isinstance(e, dict) and e.get("type") == "done" for e in events)
    assert {"type": "error", "message": "The LLM stream was interrupted."} in events
    assert events[-1] == "[DONE]"

    stored = client.get(f"/conversations/{conversation_id}/messages").json()
    assert stored == []


def test_cancelled_generation_is_not_persisted(client, monkeypatch):
    def fake_deltas(messages, **kwargs):
        yield "Bonjour"
        raise LLMStreamCancelled()

    monkeypatch.setattr("main.iter_rodium_deltas", fake_deltas)

    conversation_id = client.post("/conversations").json()["conversation_id"]
    client.post(
        f"/conversations/{conversation_id}/notes",
        json={"content": "déjà là"},
    )
    status, body = _stream_chat(client, conversation_id, "Salut")
    assert status == 200
    events = _parse_sse_body(body)
    assert events[0] == {"type": "delta", "content": "Bonjour"}
    assert not any(isinstance(e, dict) and e.get("type") == "done" for e in events)
    stored = client.get(f"/conversations/{conversation_id}/messages").json()
    assert [(m["role"], m["content"]) for m in stored] == [("note", "déjà là")]


def test_note_excluded_from_llm_and_notification_counter(
    client, sqlite_sessionmaker, monkeypatch
):
    captured: dict = {}

    def fake_deltas(messages, **kwargs):
        captured["messages"] = messages
        yield "ok"

    monkeypatch.setattr("main.iter_rodium_deltas", fake_deltas)

    with sqlite_sessionmaker() as db:
        conversation = Conversation()
        db.add(conversation)
        db.commit()
        cid = conversation.id
        for seq in range(1, 9):
            role = "user" if seq % 2 == 1 else "assistant"
            db.add(Message(conversation_id=cid, seq=seq, role=role, content=f"m{seq}"))
        db.add(Message(conversation_id=cid, seq=9, role=NOTE_ROLE, content="réviser les listes"))
        db.commit()

    status, body = _stream_chat(client, cid, "suite")
    assert status == 200
    events = _parse_sse_body(body)
    done = next(e for e in events if isinstance(e, dict) and e.get("type") == "done")
    assert done["notification"] == NOTIFICATION_TEXT

    llm_messages = captured["messages"]
    assert llm_messages[0]["role"] == "system"
    assert llm_messages[0]["content"] == load_system_prompt()
    assert not any(m.get("role") == NOTE_ROLE for m in llm_messages)
    assert not any(m.get("content") == "réviser les listes" for m in llm_messages)
    assert [m["role"] for m in llm_messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]

    stored = client.get(f"/conversations/{cid}/messages").json()
    roles = [m["role"] for m in stored]
    assert NOTE_ROLE in roles
    assert NOTIFICATION_ROLE in roles
    assert roles.count("user") == 5
    assert roles.count("assistant") == 5


def test_stream_sends_system_prompt_from_file(client, monkeypatch):
    captured: dict = {}

    def fake_deltas(messages, **kwargs):
        captured["messages"] = messages
        yield "réponse"

    monkeypatch.setattr("main.iter_rodium_deltas", fake_deltas)

    conversation_id = client.post("/conversations").json()["conversation_id"]
    _stream_chat(client, conversation_id, "Qu'est-ce qu'une liste ?")
    assert captured["messages"][0]["role"] == "system"
    assert captured["messages"][0]["content"] == load_system_prompt()
