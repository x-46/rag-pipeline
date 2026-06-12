import asyncio
import time
import uuid
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException

from rag_core.mongodb import save_request_metrics
from rag_core.pipeline import RagPipeline, build_pipeline

from rag_api.schemas import (
    ChatCompletionRequest,
    ChatCompletionResponse,
    Choice,
    ChoiceMessage,
    ModelCard,
    ModelList,
    Source,
    Usage,
)

_pipeline: RagPipeline | None = None


@asynccontextmanager
async def lifespan(app: FastAPI):
    global _pipeline
    _pipeline = build_pipeline()
    yield
    


app = FastAPI(title="RAG API", version="1.0.0", lifespan=lifespan)


# Routes
@app.get("/v1/models", response_model=ModelList)
async def list_models():
    return ModelList(data=[ModelCard(id="rag-model", created=int(time.time()))])


@app.post("/v1/chat/completions", response_model=ChatCompletionResponse)
async def chat_completions(request: ChatCompletionRequest):
    if _pipeline is None:
        raise HTTPException(status_code=503, detail="Pipeline not ready")

    question = _get_user_message(request)
    if not question:
        raise HTTPException(status_code=400, detail="No user message found")

    loop = asyncio.get_running_loop()
    result = await loop.run_in_executor(None, _pipeline.invoke, question)

    loop.run_in_executor(None, save_request_metrics, result["metrics"], question)
    
    print(result["metrics"].report())

    return ChatCompletionResponse(
        id=f"chatcmpl-{uuid.uuid4().hex}",
        created=int(time.time()),
        model=request.model,
        choices=[
            Choice(
                index=0,
                message=ChoiceMessage(role="assistant", content=result["answer"]),
                finish_reason="stop",
            )
        ],
        sources=_build_sources(result["context"]),
    )


# Helpers
def _get_user_message(request: ChatCompletionRequest) -> str:
    """Return the content of the last user message in the request."""
    for msg in reversed(request.messages):
        if msg.role == "user":
            return msg.content
    return ""


def _build_sources(context_docs: list) -> list[Source]:
    """Deduplicate and extract source metadata from retrieved context documents."""
    seen: set[str] = set()
    sources: list[Source] = []
    for doc in context_docs:
        src = doc.metadata.get("source", "unknown")
        if src not in seen:
            seen.add(src)
            sources.append(Source(title=src.split("/")[-1], source=src))
    return sources


# Dev entrypoint 

def main():
    import uvicorn
    uvicorn.run("rag_api.main:app", host="0.0.0.0", port=8000, reload=True)


if __name__ == "__main__":
    main()
