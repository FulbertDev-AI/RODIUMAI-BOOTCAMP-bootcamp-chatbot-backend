from datetime import datetime
from pathlib import Path

import pytest

from database.models import Message
from main import (
    NOTE_ROLE,
    NOTIFICATION_ROLE,
    SYSTEM_PROMPT_PATH,
    build_llm_messages,
    load_system_prompt,
)


def _msg(seq: int, role: str, content: str) -> Message:
    return Message(
        conversation_id=1,
        seq=seq,
        role=role,
        content=content,
        created_at=datetime(2026, 1, 1),
    )


def test_load_system_prompt_missing_file_is_explicit(monkeypatch, tmp_path):
    missing = tmp_path / "absent.md"
    monkeypatch.setattr("main.SYSTEM_PROMPT_PATH", missing)
    with pytest.raises(FileNotFoundError, match="System prompt file not found"):
        load_system_prompt()


def test_system_prompt_file_exists():
    assert SYSTEM_PROMPT_PATH.is_file()
    assert SYSTEM_PROMPT_PATH.name == "system.md"


def test_load_system_prompt_reads_file_contents():
    expected = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    loaded = load_system_prompt()
    assert loaded == expected
    assert "tuteur" in loaded.lower()
    assert "Python" in loaded


def test_llm_payload_starts_with_system_prompt_from_file():
    rows = [
        _msg(1, "user", "Q1"),
        _msg(2, "assistant", "A1"),
        _msg(3, NOTE_ROLE, "Je dois revoir les listes Python ce soir."),
        _msg(4, NOTIFICATION_ROLE, "Notification système : Une dizaine de messages écrits."),
        _msg(5, "user", "Q2"),
        _msg(6, "assistant", "A2"),
    ]
    prompt = load_system_prompt()
    messages = build_llm_messages(rows, "Comment fonctionne une liste ?")

    assert messages[0]["role"] == "system"
    assert messages[0]["content"] == prompt
    assert messages[0]["content"] == Path(SYSTEM_PROMPT_PATH).read_text(encoding="utf-8").strip()
    assert messages[-1] == {"role": "user", "content": "Comment fonctionne une liste ?"}
    assert [m["role"] for m in messages] == [
        "system",
        "user",
        "assistant",
        "user",
        "assistant",
        "user",
    ]
    assert not any(m["role"] == NOTE_ROLE for m in messages)
    assert not any(m["role"] == NOTIFICATION_ROLE for m in messages)
    assert not any("Je dois revoir les listes Python ce soir." in m["content"] for m in messages)
    assert not any(m["content"].startswith("Notification système") for m in messages)
