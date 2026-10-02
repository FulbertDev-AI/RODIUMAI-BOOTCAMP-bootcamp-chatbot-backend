import os

import httpx
from dotenv import load_dotenv
from fastapi import Depends, FastAPI, HTTPException
from pydantic import BaseModel
from sqlalchemy import select
from sqlalchemy.exc import IntegrityError
from sqlalchemy.orm import Session

from database.db import get_db
from database.models import Conversation, Message

load_dotenv()

RODIUMAI_URL = "https://api.rodiumai.io/v1/chat/completions"
RODIUMAI_API_KEY = os.environ["RODIUMAI_API_KEY"]
MODEL = os.getenv("RODIUMAI_MODEL", "anthropic/claude-sonnet-4-5-20250929")

SYSTEM_PROMPT = (
    "Tu es Study Buddy, un tuteur bienveillant pour les étudiants."
    "Réponds aux questions de manière claire et concise."
)

app = FastAPI(title="Study Buddy Chatbot")


class ConversationResponse(BaseModel):
    conversation_id: int


class ChatRequest(BaseModel):
    conversation_id: int
    message: str


class ChatResponse(BaseModel):
    reply: str


@app.post("/conversations", status_code=201)
def create_conversation(db: Session = Depends(get_db)) -> ConversationResponse:
    conversation = Conversation()
    db.add(conversation)
    db.commit()
    return ConversationResponse(conversation_id=conversation.id)


@app.post("/chat")
def chat(req: ChatRequest, db: Session = Depends(get_db)) -> ChatResponse:
    if db.get(Conversation, req.conversation_id) is None:
        raise HTTPException(status_code=404, detail="Conversation not found.")

    # Load this conversation from the database, in message order.
    rows = db.scalars(
        select(Message)
        .where(Message.conversation_id == req.conversation_id)
        .order_by(Message.seq)
    ).all()
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
