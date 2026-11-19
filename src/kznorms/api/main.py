"""FastAPI-сервис: тот же RagPipeline, что в CLI и Gradio, но как HTTP API.

Зачем, если есть Gradio: Gradio — витрина для человека, API — контракт для других систем
(проверка раздела проекта, бот, фронтенд). Ответ — та же Pydantic-схема `AnswerResponse`,
что и внутри пайплайна: клиент получает статус, цитаты, флаг verify и тайминги.

Решения и ловушки:
- модели поиска грузятся ~30 с — один раз при старте (lifespan), а не на первый запрос;
- обработчики синхронные (`def`): FastAPI выполняет их в пуле потоков, event loop не
  блокируется. Но поиск на GPU и реранкер не рассчитаны на параллельные вызовы из разных
  потоков, поэтому запросы к пайплайну идут по очереди (`Lock`). Для одной видеокарты это
  честный потолок: параллельность дала бы только борьбу за VRAM;
- llama-server не запущен → 503 с понятным текстом, а не 500 с трейсбеком;
- вход ограничен схемой (длина вопроса, k ≤ 20): API нельзя попросить вернуть весь корпус.
"""

from collections.abc import Callable
from contextlib import asynccontextmanager
from threading import Lock
from typing import Literal

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from kznorms.schemas import AnswerResponse, Clause

Mode = Literal["bm25", "dense", "hybrid", "hybrid_rerank"]


class AskRequest(BaseModel):
    question: str = Field(min_length=3, max_length=1000, examples=["Ширина марша лестницы?"])
    k: int = Field(5, ge=1, le=20, description="сколько пунктов отдать модели")


class SearchRequest(AskRequest):
    mode: Mode = "hybrid_rerank"


class SearchHit(BaseModel):
    clause: Clause
    score: float


class SearchResponse(BaseModel):
    hits: list[SearchHit]
    timings_ms: dict[str, float]


def default_pipeline():
    from kznorms.rag.llm_client import get_llm
    from kznorms.rag.pipeline import RagPipeline

    pipe = RagPipeline(get_llm("base"))
    pipe.retriever.warmup("hybrid_rerank")
    return pipe


def create_app(pipeline_factory: Callable = default_pipeline) -> FastAPI:
    """pipeline_factory() → RagPipeline; в тестах — подставной пайплайн без GPU."""
    lock = Lock()

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        app.state.pipe = pipeline_factory()
        yield

    app = FastAPI(
        title="kz-norms-llm",
        description="Ответы по строительным нормам РК с дословными цитатами пунктов",
        version="0.1.0",
        lifespan=lifespan,
    )

    @app.get("/health")
    def health() -> dict[str, str]:
        return {"status": "ok", "model": app.state.pipe.llm.model}

    @app.post("/search", response_model=SearchResponse)
    def search(req: SearchRequest) -> SearchResponse:
        """Только поиск, без LLM: какие пункты норм относятся к вопросу."""
        pipe = app.state.pipe
        with lock:
            hits = pipe.retriever.retrieve(req.question, k=req.k, mode=req.mode)
            timings = dict(pipe.retriever.timings_ms)
        return SearchResponse(
            hits=[SearchHit(clause=h.clause, score=h.score) for h in hits], timings_ms=timings
        )

    @app.post("/ask", response_model=AnswerResponse)
    def ask(req: AskRequest) -> AnswerResponse:
        """Поиск → LLM → verify. `verified=false` — ответ не прошёл проверку цитат и чисел."""
        from openai import APIConnectionError, APITimeoutError

        with lock:
            try:
                return app.state.pipe.answer(req.question, k=req.k)
            except (APIConnectionError, APITimeoutError) as e:
                raise HTTPException(503, "LLM недоступна: запусти llama-server (см. README)") from e

    return app


app = create_app()  # uvicorn kznorms.api.main:app — модели грузятся только при старте
