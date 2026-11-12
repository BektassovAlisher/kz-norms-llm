"""Целевые JSON-ответы от учителя — по тому же промпту, что в проде.

Учитель получает ровно те сообщения, которые потом станут входом ученика
(`rag.prompts.build_messages`). Если учителю дать «подсказку» (например, сказать, какой пункт
gold), его ответы будут опираться на то, чего ученик не увидит, — и ученик выучит угадывать.
"""

from pydantic import ValidationError

from kznorms.rag.prompts import Passage, build_messages
from kznorms.schemas import Answer, SynthExample
from kznorms.synth.gen_questions import pmap


def gen_answers(teacher, examples: list[SynthExample], cfg: dict) -> list[SynthExample]:
    schema = Answer.model_json_schema()

    def one(ex: SynthExample) -> SynthExample:
        passages = [Passage(**p) for p in ex.passages]
        raw = teacher.chat_json(
            build_messages(ex.q.question, passages), schema, cfg["temperature_answers"]
        )
        try:
            ex.answer = Answer.model_validate_json(raw) if raw else None
        except ValidationError:
            ex.answer = None
        return ex

    return pmap(one, examples, cfg["workers"], "ответы")
