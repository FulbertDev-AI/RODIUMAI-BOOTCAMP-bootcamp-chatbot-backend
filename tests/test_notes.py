from datetime import datetime

from database.models import Message
from main import (
    LLM_ROLES,
    NOTE_ROLE,
    NOTIFICATION_EVERY,
    NOTIFICATION_ROLE,
    build_llm_history,
)


def _msg(seq: int, role: str, content: str) -> Message:
    return Message(
        conversation_id=1,
        seq=seq,
        role=role,
        content=content,
        created_at=datetime(2026, 1, 1),
    )


def test_create_note(client):
    conversation_id = client.post("/conversations").json()["conversation_id"]
    response = client.post(
        f"/conversations/{conversation_id}/notes",
        json={"content": "Je dois revoir les listes Python ce soir."},
    )
    assert response.status_code == 201
    body = response.json()
    assert body["role"] == NOTE_ROLE
    assert body["content"] == "Je dois revoir les listes Python ce soir."
    assert body["seq"] == 1
    assert "created_at" in body


def test_note_appears_in_message_history(client):
    conversation_id = client.post("/conversations").json()["conversation_id"]
    client.post(
        f"/conversations/{conversation_id}/notes",
        json={"content": "Réviser les listes"},
    )
    response = client.get(f"/conversations/{conversation_id}/messages")
    assert response.status_code == 200
    messages = response.json()
    assert len(messages) == 1
    assert messages[0]["role"] == "note"
    assert messages[0]["content"] == "Réviser les listes"


def test_create_note_unknown_conversation_returns_404(client):
    response = client.post("/conversations/99999/notes", json={"content": "hello"})
    assert response.status_code == 404
    assert response.json()["detail"] == "Conversation not found."


def test_create_note_rejects_blank_content(client):
    conversation_id = client.post("/conversations").json()["conversation_id"]
    response = client.post(
        f"/conversations/{conversation_id}/notes",
        json={"content": "   "},
    )
    assert response.status_code == 422


def test_build_llm_history_excludes_note_and_notification():
    rows = [
        _msg(1, "user", "Q1"),
        _msg(2, "assistant", "A1"),
        _msg(3, NOTE_ROLE, "Je dois revoir les listes Python ce soir."),
        _msg(4, "user", "Q2"),
        _msg(5, NOTIFICATION_ROLE, "Notification système : Une dizaine de messages écrits."),
        _msg(6, "assistant", "A2"),
    ]
    history = build_llm_history(rows)
    assert history == [
        {"role": "user", "content": "Q1"},
        {"role": "assistant", "content": "A1"},
        {"role": "user", "content": "Q2"},
        {"role": "assistant", "content": "A2"},
    ]
    assert NOTE_ROLE not in LLM_ROLES
    assert all(item["role"] in LLM_ROLES for item in history)
    assert not any(item["content"].startswith("Je dois revoir") for item in history)


def test_note_does_not_change_notification_counter():
    dialogue = [
        _msg(1, "user", "u1"),
        _msg(2, "assistant", "a1"),
        _msg(3, "user", "u2"),
        _msg(4, "assistant", "a2"),
        _msg(5, "user", "u3"),
        _msg(6, "assistant", "a3"),
        _msg(7, "user", "u4"),
        _msg(8, "assistant", "a4"),
    ]
    with_note = [
        *dialogue,
        _msg(9, NOTE_ROLE, "note personnelle"),
    ]
    history_without = build_llm_history(dialogue)
    history_with = build_llm_history(with_note)
    assert len(history_without) == len(history_with) == 8
    # Next chat turn adds user+assistant (+2). Same notify decision with or without the note.
    assert (len(history_without) + 2) % NOTIFICATION_EVERY == 0
    assert (len(history_with) + 2) % NOTIFICATION_EVERY == (
        len(history_without) + 2
    ) % NOTIFICATION_EVERY
    assert (len(with_note) + 2) % NOTIFICATION_EVERY != (
        len(history_with) + 2
    ) % NOTIFICATION_EVERY
