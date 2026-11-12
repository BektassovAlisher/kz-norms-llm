"""answer(question) -> AnswerResponse: поиск → промпт → LLM → verify."""

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


class RagPipeline:
    def __init__(self, llm: LLM | None = None, retriever: Retriever | None = None) -> None:
        self.llm = llm or get_llm("base")
        self.retriever = retriever or Retriever()

    def answer(self, question: str, k: int | None = None) -> AnswerResponse:
        hits = self.retriever.retrieve(question, k=k)
        timings = dict(self.retriever.timings_ms)
        passages = [to_passage(h) for h in hits]

        raw = self.llm.chat_json(build_messages(question, passages), Answer.model_json_schema())
        timings["llm"] = self.llm.last_ms
        try:
            ans = Answer.model_validate_json(raw)
        except ValidationError:
            ans = Answer(status="not_found", answer="Модель вернула ответ не по схеме.")
            problems = [f"ответ не по JSON-схеме: {raw[:200]!r}"]
        else:
            problems = verify(ans, passages, question).problems

        return AnswerResponse(
            answer=ans,
            contexts=[h.clause for h in hits],
            verified=not problems,
            problems=problems,
            timings_ms=timings,
            model=self.llm.model,
        )
