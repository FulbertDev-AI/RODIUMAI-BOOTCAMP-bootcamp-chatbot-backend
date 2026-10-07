import os
from collections.abc import Iterator
from datetime import datetime
from pathlib import Path

from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel, Field, field_validator
from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import Conversation, Message
from llm_stream import (
    LLMStreamError,
    LLMStreamInterrupted,
    SSE_DONE,
    format_sse,
    iter_rodium_deltas,
)

load_dotenv()

RODIUMAI_URL = "https://api.rodiumai.io/v1/chat/completions"
RODIUMAI_API_KEY = os.environ["RODIUMAI_API_KEY"]
PREVIEW_LENGTH = 60

# Single source of truth: ids sent to RodiumAI and returned by GET /models (order is stable).
ALLOWED_MODELS = {
    "openai/gpt-4o": "GPT-4o",
    "openai/gpt-4o-mini": "GPT-4o Mini",
    "anthropic/claude-sonnet-4-5-20250929": "Claude Sonnet 4.5",
    "anthropic/claude-haiku-4-5-20251001": "Claude Haiku 4.5",
    "google/gemini-2.5-flash": "Gemini 2.5 Flash",
}


def _default_model() -> str:
    # RODIUMAI_MODEL is optional config for GET /models "default" only. POST /chat always
    # requires an explicit allowed model; env cannot bypass the allow-list.
    fallback = next(iter(ALLOWED_MODELS))
    configured = os.getenv("RODIUMAI_MODEL")
    if not configured:
        return fallback
    if configured not in ALLOWED_MODELS:
        allowed = ", ".join(ALLOWED_MODELS)
        raise RuntimeError(
            f"RODIUMAI_MODEL={configured!r} is not in the allow-list. "
            f"Use one of: {allowed}"
        )
    return configured


DEFAULT_MODEL = _default_model()

# Roles the LLM understands. Stored roles outside this set (note, system-notification) stay out of its prompt.
LLM_ROLES = {"user", "assistant"}
NOTE_ROLE = "note"
NOTIFICATION_ROLE = "system-notification"
NOTIFICATION_EVERY = 10  # a notification each time the dialogue reaches a multiple of this many messages
NOTIFICATION_TEXT = "Notification système : Une dizaine de messages écrits."

SYSTEM_PROMPT_PATH = Path(__file__).resolve().parent / "prompts" / "system.md"


def load_system_prompt() -> str:
    try:
        content = SYSTEM_PROMPT_PATH.read_text(encoding="utf-8").strip()
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"System prompt file not found: {SYSTEM_PROMPT_PATH}"
        ) from exc
    if not content:
        raise ValueError(f"System prompt file is empty: {SYSTEM_PROMPT_PATH}")
    return content

app = FastAPI(title="Study Buddy Chatbot")


class ConversationResponse(BaseModel):
    conversation_id: int


class ConversationSummary(BaseModel):
    id: int
    created_at: datetime
    preview: str | None  # first user message, truncated; None while the conversation is empty


class ChatRequest(BaseModel):
    conversation_id: int
    message: str
    model: str


class ModelOption(BaseModel):
    id: str
    label: str


class ModelsResponse(BaseModel):
    default: str
    models: list[ModelOption]


class ChatResponse(BaseModel):
    reply: str
    notification: str | None = None  # set when this turn also stored a system-notification


class MessageResponse(BaseModel):
    seq: int
    role: str
    content: str
    created_at: datetime


class NoteRequest(BaseModel):
    content: str = Field(min_length=1)

    @field_validator("content")
    @classmethod
    def content_must_not_be_blank(cls, value: str) -> str:
        stripped = value.strip()
        if not stripped:
            raise ValueError("content must not be empty")
        return stripped


def load_messages(db: Session, conversation_id: int) -> list[Message]:
    # A conversation's messages in order; 404 if the conversation doesn't exist.
    if db.get(Conversation, conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.seq)
    ).all()


def build_llm_history(rows: list[Message]) -> list[dict]:
    # The DB/UI history is not what the LLM sees: only user and assistant turns (LLM_ROLES).
    # Roles such as note and system-notification are persisted and listed in GET .../messages, never sent to RodiumAI.
    return [{"role": m.role, "content": m.content} for m in rows if m.role in LLM_ROLES]


def build_llm_messages(rows: list[Message], user_content: str) -> list[dict]:
    # Full payload for RodiumAI: specialized system prompt, then filtered history, then the new user turn.
    return [
        {"role": "system", "content": load_system_prompt()},
        *build_llm_history(rows),
        {"role": "user", "content": user_content},
    ]


def require_allowed_model(model_id: str) -> str:
    if model_id not in ALLOWED_MODELS:
        raise HTTPException(status_code=400, detail="Unsupported model.")
    return model_id


@app.get("/models")
def list_models() -> ModelsResponse:
    return ModelsResponse(
        default=DEFAULT_MODEL,
        models=[ModelOption(id=model_id, label=label) for model_id, label in ALLOWED_MODELS.items()],
    )


@app.post("/conversations", status_code=201)
def create_conversation(db: Session = Depends(get_db)) -> ConversationResponse:
    conversation = Conversation()
    db.add(conversation)
    db.commit()
    return ConversationResponse(conversation_id=conversation.id)


@app.get("/conversations")
def list_conversations(db: Session = Depends(get_db)) -> list[ConversationSummary]:
    # Newest first, each joined to its first message (seq 1, always the user's) for the preview.
    rows = db.execute(
        select(Conversation, Message.content)
        .outerjoin(Message, and_(Message.conversation_id == Conversation.id, Message.seq == 1))
        .order_by(Conversation.id.desc())
    ).all()
    return [
        ConversationSummary(
            id=conversation.id,
            created_at=conversation.created_at,
            preview=content[:PREVIEW_LENGTH] if content else None,
        )
        for conversation, content in rows
    ]


@app.get("/conversations/{conversation_id}/messages")
def list_messages(conversation_id: int, db: Session = Depends(get_db)) -> list[MessageResponse]:
    return [
        MessageResponse(seq=m.seq, role=m.role, content=m.content, created_at=m.created_at)
        for m in load_messages(db, conversation_id)
    ]


@app.post("/conversations/{conversation_id}/notes", status_code=201)
def create_note(
    conversation_id: int, req: NoteRequest, db: Session = Depends(get_db)
) -> MessageResponse:
    rows = load_messages(db, conversation_id)
    next_seq = rows[-1].seq + 1 if rows else 1
    note = Message(
        conversation_id=conversation_id,
        seq=next_seq,
        role=NOTE_ROLE,
        content=req.content,
    )
    db.add(note)
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        raise HTTPException(
            status_code=409, detail="The conversation was updated concurrently, please retry."
        )
    db.refresh(note)
    return MessageResponse(
        seq=note.seq, role=note.role, content=note.content, created_at=note.created_at
    )


def _chat_sse(
    *,
    db: Session,
    conversation_id: int,
    user_content: str,
    messages: list[dict],
    history_len: int,
    next_seq: int,
    model: str,
) -> Iterator[str]:
    # Accumulate assistant text in memory only. Persist user/assistant/notification after a
    # complete upstream stream AND a successful commit; only then emit type=done.
    parts: list[str] = []
    try:
        for content in iter_rodium_deltas(
            messages,
            url=RODIUMAI_URL,
            api_key=RODIUMAI_API_KEY,
            model=model,
        ):
            parts.append(content)
            yield format_sse({"type": "delta", "content": content})
    except GeneratorExit:
        db.rollback()
        raise
    except LLMStreamError:
        yield format_sse({"type": "error", "message": "The LLM API call failed."})
        yield format_sse(SSE_DONE)
        return
    except LLMStreamInterrupted:
        yield format_sse({"type": "error", "message": "The LLM stream was interrupted."})
        yield format_sse(SSE_DONE)
        return

    reply = "".join(parts)
    notification = None
    db.add_all(
        [
            Message(conversation_id=conversation_id, seq=next_seq, role="user", content=user_content),
            Message(conversation_id=conversation_id, seq=next_seq + 1, role="assistant", content=reply),
        ]
    )
    # Count only LLM dialogue (user + assistant via history). Notes and notifications do not
    # increment this total; using len(rows) or seq would shift the multiples of 10.
    if (history_len + 2) % NOTIFICATION_EVERY == 0:
        notification = NOTIFICATION_TEXT
        db.add(
            Message(
                conversation_id=conversation_id,
                seq=next_seq + 2,
                role=NOTIFICATION_ROLE,
                content=notification,
            )
        )
    try:
        db.commit()
    except IntegrityError:
        db.rollback()
        yield format_sse(
            {
                "type": "error",
                "message": "The conversation was updated concurrently, please retry.",
            }
        )
        yield format_sse(SSE_DONE)
        return

    yield format_sse({"type": "done", "reply": reply, "notification": notification})
    yield format_sse(SSE_DONE)


@app.post("/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> StreamingResponse:
    # 400/404/422 happen before the SSE body. After StreamingResponse starts, errors are SSE events.
    model = require_allowed_model(req.model)
    rows = load_messages(db, req.conversation_id)
    history = build_llm_history(rows)
    next_seq = rows[-1].seq + 1 if rows else 1
    messages = build_llm_messages(rows, req.message)

    return StreamingResponse(
        _chat_sse(
            db=db,
            conversation_id=req.conversation_id,
            user_content=req.message,
            messages=messages,
            history_len=len(history),
            next_seq=next_seq,
            model=model,
        ),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "Connection": "keep-alive",
            "X-Accel-Buffering": "no",
        },
    )
