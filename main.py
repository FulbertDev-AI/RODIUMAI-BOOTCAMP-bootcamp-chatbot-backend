import os
from datetime import datetime

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import and_, select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import Conversation, Message

load_dotenv()

RODIUMAI_URL = "https://api.rodiumai.io/v1/chat/completions"
RODIUMAI_API_KEY = os.environ["RODIUMAI_API_KEY"]
MODEL = os.getenv("RODIUMAI_MODEL", "anthropic/claude-sonnet-4-5-20250929")
PREVIEW_LENGTH = 60

SYSTEM_PROMPT = (
    "Tu es Study Buddy, un tuteur bienveillant pour les étudiants."
    "Réponds aux questions de manière claire et concise."
)

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


class ChatResponse(BaseModel):
    reply: str


class MessageResponse(BaseModel):
    seq: int
    role: str
    content: str
    created_at: datetime


def load_messages(db: Session, conversation_id: int) -> list[Message]:
    # A conversation's messages in order; 404 if the conversation doesn't exist.
    if db.get(Conversation, conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")
    return db.scalars(
        select(Message)
        .where(Message.conversation_id == conversation_id)
        .order_by(Message.seq)
    ).all()


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


@app.post("/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    # Load this conversation from the database, in message order.
    rows = load_messages(db, req.conversation_id)
    history = [{"role": m.role, "content": m.content} for m in rows]
    next_seq = rows[-1].seq + 1 if rows else 1
    user_message = {"role": "user", "content": req.message}

    # The LLM is stateless: resend the system prompt + the whole conversation each turn.
    messages = [{"role": "system", "content": SYSTEM_PROMPT}, *history, user_message]

    try:
        response = httpx.post(
            RODIUMAI_URL,
            headers={"Authorization": f"Bearer {RODIUMAI_API_KEY}"},
            json={
                "model": MODEL,
                "messages": messages, 
                "max_tokens": 512,
                "stream": False,
            },
            timeout=30,
        )
        response.raise_for_status()
    except httpx.HTTPError as exc:
        raise HTTPException(status_code=502, detail="The LLM API call failed.") from exc

    reply = response.json()["choices"][0]["message"]["content"]

    # Only record the turn once the call succeeded, so a failure doesn't leave a dangling user message.
    db.add_all([
        Message(conversation_id=req.conversation_id, seq=next_seq, role="user", content=req.message),
        Message(conversation_id=req.conversation_id, seq=next_seq + 1, role="assistant", content=reply),
    ])
    try:
        db.commit()
    except IntegrityError:
        # Another request already wrote these seq numbers in this conversation while we waited for the LLM.
        db.rollback()
        raise HTTPException(
            status_code=409, detail="The conversation was updated concurrently, please retry."
        )
    print(history)
    return ChatResponse(reply=reply)
