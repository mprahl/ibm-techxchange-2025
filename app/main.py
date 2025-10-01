import os
from typing import Any, Dict

import httpx
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel


KSERVE_URL = os.environ.get(
    "KSERVE_URL",
    "https://host-name/openai/v1/chat/completions",
)
MODEL_ID = os.environ.get("MODEL_ID", "yoda-llm")

app = FastAPI(title="Yoda Translator App")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"]
)


class ChatRequest(BaseModel):
    model: str | None = None
    messages: list
    temperature: float = 0.0
    max_tokens: int = 64


@app.post("/api/chat")
async def proxy_chat(req: ChatRequest) -> Dict[str, Any]:
    # Force model from env if not provided by client
    if not req.model:
        req.model = MODEL_ID
    async with httpx.AsyncClient(timeout=30.0, verify=True) as client:
        try:
            resp = await client.post(KSERVE_URL, json=req.model_dump())
        except httpx.RequestError as e:
            raise HTTPException(status_code=502, detail=f"Upstream error: {e}")

    if resp.status_code >= 400:
        raise HTTPException(status_code=resp.status_code, detail=resp.text)

    return resp.json()


ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
STATIC_DIR = os.path.join(ROOT_DIR, "static")
if os.path.isdir(STATIC_DIR):
    app.mount("/", StaticFiles(directory=STATIC_DIR, html=True), name="static")


