import os

import httpx
from dotenv import load_dotenv
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

load_dotenv()

RODIUMAI_URL = "https://api.rodiumai.io/v1/chat/completions"
RODIUMAI_API_KEY = os.environ["RODIUMAI_API_KEY"]
MODEL = os.getenv("RODIUMAI_MODEL", "anthropic/claude-sonnet-4-5-20250929")

SYSTEM_PROMPT = (
    "Tu es Study Buddy, un tuteur bienveillant pour les étudiants."
    "Réponds aux questions de manière claire et concise."
)

app = FastAPI(title="Study Buddy Chatbot")


class ChatRequest(BaseModel):
    message: str


class ChatResponse(BaseModel):
    reply: str


# In-memory conversation history for a single session: a list of {role, content}.
history: list[dict] = []


@app.post("/chat")
def chat(req: ChatRequest) -> ChatResponse:
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
    history.append(user_message)
    history.append({"role": "assistant", "content": reply})
    print(history)
    return ChatResponse(reply=reply)
