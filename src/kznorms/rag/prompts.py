"""Единственное место с шаблоном промпта — его используют synth, train и инференс.

Модель учится на конкретном формате: если при обучении контекст оформлен «[id] текст»,
а на инференсе иначе, дообученная модель увидит незнакомый формат (train/serve skew).
"""

from dataclasses import dataclass

SYSTEM = """Ты — помощник инженера по нормоконтролю. Отвечаешь на вопросы о строительных
нормах Республики Казахстан.

Правила:
1. Отвечай только по пунктам из раздела КОНТЕКСТ. Своими знаниями не дополняй.
2. Каждое утверждение подкрепи цитатой: clause_id пункта — точно как в квадратных скобках
   (например, zhilye-zdaniya:6.2.10) — и дословный фрагмент его текста: копируй символ
   в символ, не пересказывай.
3. Числа в ответе бери только из цитат.
4. status:
   - "answered" — контекст полностью отвечает на вопрос;
   - "partial" — отвечает лишь на часть вопроса; в answer скажи, чего не хватает;
   - "not_found" — ответа в контексте нет; citations пустой.
5. Отвечай кратко: сам ответ, без пересказа соседних пунктов.
6. Формат — строго JSON:
   {"status": ..., "answer": ..., "citations": [{"clause_id": ..., "quote": ...}]}."""


@dataclass(frozen=True)
class Passage:
    clause_id: str
    doc_code: str
    number: str
    text: str  # ровно то, что видит модель; по нему verify проверяет цитаты


def format_context(passages: list[Passage]) -> str:
    blocks = [f"[{p.clause_id}] {p.doc_code}, п. {p.number}\n{p.text}" for p in passages]
    return "\n\n".join(blocks) or "(пусто)"


def build_messages(question: str, passages: list[Passage]) -> list[dict[str, str]]:
    user = f"КОНТЕКСТ:\n{format_context(passages)}\n\nВОПРОС: {question}"
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]
