"""Проверка ответа: clause_id из контекста, цитата — дословная подстрока, числа из цитат.

Это проверка «не выдумано ли», а не «правильно ли»: несуществующий пункт, цитата, которой
нет в тексте, число из головы. Используется трижды: фильтр синтетики, рантайм, метрика.

«Дословно» — с точностью до пробелов, пунктуации, регистра и ё: в PDF-тексте норм
«I -теплоизолирующей» и «стены, - непрерывный», и требовать от модели воспроизводить такие
артефакты бессмысленно.
Но слова и цифры должны совпасть: если модель «исправила» опечатку нормы («зоной» → «зоне»),
это уже не цитата.
"""

import re
from dataclasses import dataclass, field

from kznorms.rag.prompts import Passage
from kznorms.schemas import Answer

# число, но не маркер списка «1)»; вторая проверка не даёт откусить «1» от «12)»
NUMBER = re.compile(r"\d+(?:[.,]\d+)*(?![.,]?\d)(?!\))")
_NON_WORD = re.compile(r"[^\w]+")


def normalize(text: str) -> str:
    """Только слова и цифры через один пробел, в нижнем регистре: пунктуация, переносы,
    ё и регистр не важны (цитата из начала пункта «не менее…» при «Не менее…» в тексте)."""
    return " ".join(_NON_WORD.sub(" ", text.lower().replace("ё", "е")).split())


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
    texts = {p.clause_id: normalize(p.text) for p in passages}

    if answer.status == "not_found":
        if answer.citations:
            v.problems.append("not_found, но есть цитаты")
        return v
    if not answer.citations:
        v.problems.append(f"status={answer.status}, но нет ни одной цитаты")

    # числа, на которые ответ имеет право опираться: из вопроса и идентификаторы контекста
    # («п. 4.2.25», «СП РК 3.02-101-2012») — это ссылки, а не значения
    allowed = numbers(question)
    for p in passages:
        allowed |= numbers(p.number) | numbers(p.doc_code)
    for c in answer.citations:
        if c.clause_id not in by_id:
            v.problems.append(f"пункта {c.clause_id} нет в контексте")
            continue
        q = normalize(c.quote)
        if not q:
            v.problems.append(f"пустая цитата в {c.clause_id}")
            continue
        if q not in texts[c.clause_id]:
            elsewhere = [cid for cid, t in texts.items() if q in t]
            where = f" — она из {elsewhere[0]}" if elsewhere else ""
            v.problems.append(f"цитата не найдена в {c.clause_id}{where}: {c.quote[:60]!r}")
            continue
        allowed |= numbers(c.quote)

    if extra := numbers(answer.answer) - allowed:
        # разные ошибки: число есть в контексте, но цитата его не покрывает (неполная
        # цитата) — или числа нет нигде (выдумано / из памяти модели)
        in_ctx = extra & set().union(*(numbers(p.text) for p in passages))
        if in_ctx:
            v.problems.append(f"числа из контекста без цитаты: {sorted(in_ctx)}")
        if invented := extra - in_ctx:
            v.problems.append(f"числа, которых нет в контексте: {sorted(invented)}")
    return v
