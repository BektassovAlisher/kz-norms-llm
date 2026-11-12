"""Проверка ответа: clause_id из контекста, цитата — дословная подстрока, числа из цитат.

Это проверка «не выдумано ли», а не «правильно ли»: несуществующий пункт, цитата, которой
нет в тексте, число из головы. Используется трижды: фильтр синтетики, рантайм, метрика.
"""

import re
from dataclasses import dataclass, field

from kznorms.rag.prompts import Passage
from kznorms.schemas import Answer

NUMBER = re.compile(r"\d+(?:[.,]\d+)*")


def normalize(text: str) -> str:
    """Без учёта пробелов/переносов и ё: фрагменты длинных пунктов склеены через пробел."""
    return " ".join(text.replace("ё", "е").replace("Ё", "Е").split())


def numbers(text: str) -> set[str]:
    return {n.replace(",", ".") for n in NUMBER.findall(text)}


@dataclass
class Verdict:
    problems: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.problems


def verify(answer: Answer, passages: list[Passage], question: str = "") -> Verdict:
    """question — чтобы числа из самого вопроса («на 20-м метре») не считались выдуманными."""
    v = Verdict()
    by_id = {p.clause_id: p for p in passages}

    if answer.status == "not_found":
        if answer.citations:
            v.problems.append("not_found, но есть цитаты")
        return v
    if not answer.citations:
        v.problems.append(f"status={answer.status}, но нет ни одной цитаты")

    allowed = numbers(question)  # числа, на которые ответ имеет право опираться
    for c in answer.citations:
        p = by_id.get(c.clause_id)
        if p is None:
            v.problems.append(f"пункта {c.clause_id} нет в контексте")
            continue
        if not c.quote.strip() or normalize(c.quote) not in normalize(p.text):
            v.problems.append(f"цитата не найдена дословно в {c.clause_id}: {c.quote[:60]!r}")
            continue
        allowed |= numbers(c.quote) | numbers(p.number) | numbers(p.doc_code)

    if extra := numbers(answer.answer) - allowed:
        v.problems.append(f"числа в ответе без опоры на цитату: {sorted(extra)}")
    return v
