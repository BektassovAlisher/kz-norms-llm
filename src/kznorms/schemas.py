"""Контракты данных, на которые опираются все модули."""

from typing import Literal

from pydantic import BaseModel, Field


class Source(BaseModel):
    """Строка из data/sources.csv."""

    doc_slug: str
    doc_code: str
    title: str
    url: str = ""  # пусто — файл кладётся в data/raw/ вручную
    format: Literal["docx", "html", "pdf"]
    status: str = "действует"
    version_date: str = ""
    lang: Literal["ru", "kk"] = "ru"
    topic: str = ""


class Block(BaseModel):
    """Единица потока после парсинга (до сегментации)."""

    type: Literal["heading", "paragraph", "table"]
    text: str


class Clause(BaseModel):
    """Пункт нормы — единица поиска и цитирования."""

    clause_id: str = Field(description='"<doc_slug>:4.2.5"')
    doc_code: str
    doc_title: str
    section: str = ""
    number: str
    text: str
    status: str = "действует"
    source_url: str = ""


class Citation(BaseModel):
    clause_id: str
    quote: str = Field(description="Дословный фрагмент пункта")


class Answer(BaseModel):
    """То, что генерирует LLM (и на чём она обучается)."""

    status: Literal["answered", "partial", "not_found"]
    answer: str
    citations: list[Citation] = []


class AnswerResponse(BaseModel):
    """Ответ сервиса: ответ модели + контекст + тайминги."""

    answer: Answer
    contexts: list[Clause]
    verified: bool
    timings_ms: dict[str, float] = {}
    model: str = ""


class EvalItem(BaseModel):
    """Строка из data/eval/qa_test.jsonl (ручная разметка)."""

    id: str
    question: str
    answerable: bool
    gold_clause_ids: list[str] = []
    gold_answer: str = ""
    type: str = ""
    lang: Literal["ru", "kk"] = "ru"
