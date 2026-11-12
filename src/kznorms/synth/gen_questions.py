"""Генерация вопросов к пунктам: число, «можно ли», условие, разговорный.

Учитель видит один фрагмент пункта (тот же, что потом станет gold в контексте) и пишет
вопросы, ответ на которые целиком в нём. Главная ловушка — вопросы, списанные с текста
пункта: тогда модель учится сопоставлять строки, а не понимать вопрос. Поэтому в промпте
«перефразируй», а `lexical_overlap` в отчёте показывает, насколько это удалось.
"""

import json
import random
from collections import Counter
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor

from kznorms.rag.verify import normalize
from kznorms.retrieval.chunking import Chunk
from kznorms.schemas import SynthQuestion
from kznorms.splits import doc_of

QTYPES = {
    "число": "ответ — конкретное число или величина из пункта (размер, площадь, расстояние, "
    "количество, предел огнестойкости)",
    "можно ли": "«можно ли / допускается ли / обязательно ли …», ответ — да или нет с условием",
    "условие": "в каких случаях / при каком условии действует требование",
    "разговорный": "как спросил бы заказчик или прораб простыми словами, без терминов из пункта",
}

SYSTEM = """Ты составляешь вопросы для обучения помощника по строительным нормам Республики
Казахстан. Тебе дают один пункт нормы. Придумай вопросы, ответ на которые целиком содержится
в этом пункте.

Требования к каждому вопросу:
- понятен без пункта: если требование относится к определённому типу здания или помещения,
  назови его в вопросе;
- не копирует формулировки пункта — перефразируй, как спросил бы инженер-проектировщик;
- не упоминает номер пункта и название документа;
- одно предложение на русском языке.

Типы вопросов:
{types}

Каждый вопрос — своего типа. Если пункт не подходит для типа (например, в нём нет чисел),
пропусти этот тип: лучше меньше вопросов, чем натянутые.
Ответ — JSON: {{"questions": [{{"type": "<тип>", "question": "<вопрос>"}}]}}"""


def body(chunk_text: str) -> str:
    """Текст фрагмента без заголовка «документ. раздел» — так его видит модель."""
    return chunk_text.split("\n", 1)[-1]


def letter_share(text: str) -> float:
    return sum(c.isalpha() for c in text) / max(len(text), 1)


def select_chunks(
    retriever, split_of: dict[str, str], n: int, val_share: float, min_chars: int,
    min_letters: float, rng: random.Random,
) -> list[tuple[str, Chunk]]:  # fmt: skip
    """n фрагментов из train/val-документов: по одному случайному фрагменту на пункт.

    Пункты-таблицы (мало букв) и обрывки (короче min_chars) пропускаем — вопросы по ним
    выходят мусорные. Test-документы не трогаем вовсе.
    """
    pools: dict[str, list[Chunk]] = {"train": [], "val": []}
    for cid in sorted(retriever.clause_chunks):
        split = split_of.get(doc_of(cid))
        if split not in pools:
            continue
        ch = rng.choice(retriever.clause_chunks[cid])
        text = body(ch.text)
        if len(text) >= min_chars and letter_share(text) >= min_letters:
            pools[split].append(ch)
    n_val = min(round(n * val_share), len(pools["val"]))
    n_train = min(n - n_val, len(pools["train"]))
    return [("train", c) for c in rng.sample(pools["train"], n_train)] + [
        ("val", c) for c in rng.sample(pools["val"], n_val)
    ]


def question_schema(types: list[str], n: int) -> dict:
    item = {
        "type": "object",
        "properties": {"type": {"enum": types}, "question": {"type": "string"}},
        "required": ["type", "question"],
    }
    return {
        "type": "object",
        "properties": {"questions": {"type": "array", "items": item, "maxItems": n}},
        "required": ["questions"],
    }


def question_messages(clause, chunk: Chunk, types: list[str], n: int) -> list[dict]:
    type_lines = "\n".join(f"- {t}: {QTYPES[t]}" for t in types)
    user = (
        f"Документ: {clause.doc_code} «{clause.doc_title}»\n"
        f"Раздел: {clause.section}\nПункт {clause.number}:\n{body(chunk.text)}\n\n"
        f"Составь до {n} вопросов."
    )
    return [
        {"role": "system", "content": SYSTEM.format(types=type_lines)},
        {"role": "user", "content": user},
    ]


def parse_questions(raw: str | None, types: list[str], n: int) -> list[tuple[str, str]]:
    """JSON учителя → [(тип, вопрос)]: неизвестные типы, повторы типа и пустые — выкидываем."""
    if not raw:
        return []
    try:
        items = json.loads(raw).get("questions", [])
    except (json.JSONDecodeError, AttributeError):
        return []
    out, seen = [], set()
    for it in items if isinstance(items, list) else []:
        t, q = str(it.get("type", "")), " ".join(str(it.get("question", "")).split())
        if t in types and t not in seen and len(q) >= 10:
            out.append((t, q))
            seen.add(t)
    return out[:n]


def pmap(fn: Callable, items: list, workers: int, desc: str) -> list:
    """Параллельно (запросы к API — это ожидание сети, потоки тут подходят), с прогрессом."""
    from rich.progress import track

    with ThreadPoolExecutor(max_workers=workers) as ex:
        return list(track(ex.map(fn, items), total=len(items), description=desc))


def gen_questions(teacher, retriever, picked, cfg: dict) -> list[SynthQuestion]:
    types, n = cfg["question_types"], cfg["questions_per_clause"]
    schema = question_schema(types, n)

    def one(item: tuple[str, Chunk]) -> list[SynthQuestion]:
        split, ch = item
        clause = retriever.clauses[ch.clause_id]
        raw = teacher.chat_json(
            question_messages(clause, ch, types, n), schema, cfg["temperature_questions"]
        )
        return [
            SynthQuestion(
                qid=f"{ch.chunk_id}/{i}",
                split=split,
                clause_id=ch.clause_id,
                chunk_id=ch.chunk_id,
                qtype=t,
                question=q,
            )  # fmt: skip
            for i, (t, q) in enumerate(parse_questions(raw, types, n))
        ]

    return [q for qs in pmap(one, picked, cfg["workers"], "вопросы") for q in qs]


def lexical_overlap(question: str, passage: str) -> float:
    """Доля слов вопроса, которые есть в пункте: близко к 1 — вопрос списан с текста."""
    qw = set(normalize(question).split())
    return len(qw & set(normalize(passage).split())) / max(len(qw), 1)


def type_counts(questions: list[SynthQuestion]) -> dict[str, int]:
    return dict(Counter(q.qtype for q in questions))
