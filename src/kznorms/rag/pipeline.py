"""answer(question) -> AnswerResponse: поиск → промпт → LLM → verify.

Две половины разделены: `generate` отвечает по готовому контексту. Так оценка в режиме
oracle (контекст собран из gold-пунктов руками) идёт через тот же код, что и прод.
"""

from dataclasses import dataclass

from pydantic import ValidationError

from kznorms.rag.llm_client import LLM, get_llm
from kznorms.rag.prompts import Passage, build_messages
from kznorms.rag.verify import verify
from kznorms.retrieval.retriever import Hit, Retriever
from kznorms.schemas import Answer, AnswerResponse


def to_passage(hit: Hit) -> Passage:
    c = hit.clause
    body = hit.chunk.split("\n", 1)[-1]  # без заголовка «документ. раздел»
    return Passage(c.clause_id, c.doc_code, c.number, body)


@dataclass
class Generation:
    answer: Answer
    problems: list[str]
    llm_ms: float


class RagPipeline:
    def __init__(self, llm: LLM | None = None, retriever: Retriever | None = None) -> None:
        self.llm = llm or get_llm("base")
        self.retriever = retriever or Retriever()

    def generate(self, question: str, passages: list[Passage]) -> Generation:
        raw = self.llm.chat_json(build_messages(question, passages), Answer.model_json_schema())
        try:
            ans = Answer.model_validate_json(raw)
        except ValidationError:
            ans = Answer(status="not_found", answer="Модель вернула ответ не по схеме.")
            # грамматика не даёт сломать JSON — значит, почти всегда упёрлись в max_tokens
            cut = getattr(self.llm, "last_finish", "") == "length"
            why = "ответ обрезан по max_tokens" if cut else "ответ не по JSON-схеме"
            problems = [f"{why}: {raw[:200]!r}"]
        else:
            problems = verify(ans, passages, question).problems
        return Generation(ans, problems, self.llm.last_ms)

    def answer(self, question: str, k: int | None = None) -> AnswerResponse:
        hits = self.retriever.retrieve(question, k=k)
        timings = dict(self.retriever.timings_ms)
        gen = self.generate(question, [to_passage(h) for h in hits])
        timings["llm"] = gen.llm_ms
        return AnswerResponse(
            answer=gen.answer,
            contexts=[h.clause for h in hits],
            verified=not gen.problems,
            problems=gen.problems,
            timings_ms=timings,
            model=self.llm.model,
        )
