import os
from datetime import datetime, timezone
from uuid import uuid4

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from qdrant_client import QdrantClient

from . import ingestion, query


class ChatRequest(BaseModel):
    question: str = Field(min_length=1, max_length=2000)
    chat_id: str | None = None


class Source(BaseModel):
    document: str
    page: int | None = None


class ChatResponse(BaseModel):
    chat_id: str
    answer: str
    sources: list[Source]


class ChatMessage(BaseModel):
    question: str
    answer: str
    sources: list[Source]
    created_at: datetime


class ChatSummary(BaseModel):
    chat_id: str
    title: str
    message_count: int
    created_at: datetime
    updated_at: datetime


class ChatDetail(ChatSummary):
    messages: list[ChatMessage]


app = FastAPI(title="HR Policy RAG API", version="1.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)

chat_history: dict[str, dict] = {}


def get_client() -> QdrantClient:
    config = query.load_config()
    return QdrantClient(
        host=os.getenv("QDRANT_HOST", config["qdrant"]["host"]),
        port=int(os.getenv("QDRANT_PORT", config["qdrant"]["port"])),
    )


def ensure_indexed() -> None:
    config = query.load_config()
    client = get_client()
    if not client.collection_exists(config["qdrant"]["collection_name"]):
        ingestion.main()


@app.on_event("startup")
def index_fixed_documents() -> None:
    ensure_indexed()


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok", "service": "rag-backend"}


@app.get("/api/chats", response_model=list[ChatSummary])
def list_chats() -> list[ChatSummary]:
    chats = []
    for chat_id, chat_data in chat_history.items():
        chats.append(
            ChatSummary(
                chat_id=chat_id,
                title=chat_data["messages"][0].question,
                message_count=len(chat_data["messages"]),
                created_at=chat_data["created_at"],
                updated_at=chat_data["updated_at"],
            )
        )
    return sorted(chats, key=lambda chat: chat.updated_at, reverse=True)


@app.get("/api/chats/{chat_id}", response_model=ChatDetail)
def get_chat(chat_id: str) -> ChatDetail:
    chat_data = chat_history.get(chat_id)
    if not chat_data:
        raise HTTPException(status_code=404, detail="Chat not found")

    return ChatDetail(
        chat_id=chat_id,
        title=chat_data["messages"][0].question,
        message_count=len(chat_data["messages"]),
        created_at=chat_data["created_at"],
        updated_at=chat_data["updated_at"],
        messages=chat_data["messages"],
    )


@app.post("/api/chat", response_model=ChatResponse)
def chat(request: ChatRequest) -> ChatResponse:
    try:
        config = query.load_config()
        client = get_client()
        query_vector = query.embed_query(
            request.question,
            config["ollama"]["embedding_model"],
        )
        results = client.query_points(
            collection_name=config["qdrant"]["collection_name"],
            query=query_vector,
            limit=5,
        )
        points = query.rerank_points(results.points, request.question, limit=3)
        context = "\n\n".join(
            f"[{index}] {point.payload.get('document_title', 'Unknown document')}, "
            f"page {point.payload.get('page_number', 'unknown')}:\n"
            f"{point.payload['text']}"
            for index, point in enumerate(points, start=1)
        )
        answer = query.ask_llm(
            context,
            request.question,
            config["ollama"]["llm_model"],
        )
        sources = [
            Source(
                document=point.payload.get("document_title", "Unknown document"),
                page=point.payload.get("page_number"),
            )
            for point in points
        ]
        chat_id = request.chat_id or str(uuid4())
        now = datetime.now(timezone.utc)
        if chat_id not in chat_history:
            chat_history[chat_id] = {
                "created_at": now,
                "updated_at": now,
                "messages": [],
            }
        chat_history[chat_id]["messages"].append(
            ChatMessage(
                question=request.question,
                answer=answer,
                sources=sources,
                created_at=now,
            )
        )
        chat_history[chat_id]["updated_at"] = now
        return ChatResponse(chat_id=chat_id, answer=answer, sources=sources)
    except Exception as error:
        raise HTTPException(status_code=503, detail=str(error)) from error