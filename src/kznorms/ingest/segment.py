"""list[Block] → list[Clause].

Нарезка по номерам пунктов. Всё, что идёт после пункта и не начинается с нового номера
(списки, примечания, строки таблиц), приклеивается к нему.

Главная идея: номер в начале строки считается новым пунктом, только если он правдоподобно
продолжает нумерацию (5.2.3 → 5.2.4, 5.2.20 → 5.3.1, В.7.3 → В.8.1). Так в пункты не
попадают даты «26.05.2026», значения из таблиц «12.5» и ссылки «согласно 5.3.2».
"""

import re
from collections import deque

from kznorms.schemas import Block, Clause, Source

# «*» до или после номера («*5.3.3.10», «5.5.30*») — пометка изменённого пункта
CLAUSE = re.compile(r"^((?:[А-ЯA-Z]\.)?\d+(?:\.\d+)+|[А-ЯA-Z]\.\d+)\.?\*?(?:\s+(.*))?$", re.S)
SECTION = re.compile(r"^(\d+)\.?\s+([А-ЯЁ].*)$", re.S)
APPENDIX = re.compile(r"^(?:ПРИЛОЖЕНИЕ|Приложение)\s+([А-ЯA-Z])\b(.*)$", re.S)
EMBEDDED = re.compile(r"(?<=[.;:])\s+\*?((?:[А-ЯA-Z]\.)?\d+(?:\.\d+)+)\.?\*?\s+(?=[А-ЯЁ])")
REF_WORDS = (
    "п.", "пп.", "пункт", "пункта", "пункте", "пунктом", "пунктов", "пунктах", "пункту",
    "табл.", "таблица", "таблицы", "таблице", "таблицей", "таблицам", "таблицах",
    "рис.", "рисунок", "рисунка", "рисунке", "рисунку", "согласно",  # не «см.»: это и сантиметры
)  # fmt: skip
SKIP_SECTIONS = ("НОРМАТИВНЫЕ ССЫЛКИ",)
EXCLUDED = re.compile(r"^\(?\s*исключ", re.IGNORECASE)
# Выходные данные в конце издания («УДК 728.4 МКС 93.040.10 Ключевые слова: …») — конец норм
END_MATTER = re.compile(r"\b(?:УДК\s+\d|МКС\s+\d+\.\d+)")


def split_number(number: str) -> tuple[str, list[int]]:
    """«Д.3.2» → ("Д", [3, 2]); «5.2.1» → ("", [5, 2, 1]); «Б» → ("Б", [])."""
    prefix = ""
    if number[0].isalpha():
        prefix, number = number[0], number[2:]
    return prefix, [int(x) for x in number.split(".") if x]


def is_successor(prev: str | None, cand: str, tolerance: int = 1) -> bool:
    """Может ли пункт cand идти сразу после prev.

    tolerance=1 — строго: вглубь (5.2 → 5.2.1) или к следующему соседу на любом уровне
    с первыми подпунктами (5.2.20 → 5.3.1). tolerance=3 — допускаем пропуск до двух номеров
    (в нормах бывают исключённые пункты: 3.7 → 3.9).
    """
    if any(len(x) > 3 for x in re.split(r"[.]", cand)):  # 26.05.2026 — это дата
        return False
    if prev is None:
        return True
    (pp, p), (cp, c) = split_number(prev), split_number(cand)
    if not c:
        return False
    if pp != cp:
        return cp != "" and c[0] <= tolerance  # начало приложения: «Б.1»
    if len(c) > len(p) and c[: len(p)] == p:
        return all(x <= tolerance for x in c[len(p) :])
    for k in range(1, min(len(c), len(p)) + 1):
        if c[: k - 1] == p[: k - 1] and 0 < c[k - 1] - p[k - 1] <= tolerance:
            return all(x <= tolerance for x in c[k:])
    return False


def split_embedded(text: str, prev: str | None) -> list[str]:
    """Режем абзац, если внутри начинается следующий пункт: «…ограждения. 5.3.16.10 Каждая…»."""
    for m in EMBEDDED.finditer(text):
        before = text[: m.start()].rstrip()
        last_word = before.rsplit(" ", 1)[-1].lower()
        if last_word not in REF_WORDS and is_successor(prev, m.group(1)):
            return [before, text[m.start() :].strip()]
    return [text]


def segment(blocks: list[Block], src: Source) -> list[Clause]:
    clauses: list[Clause] = []
    section, subsection = "", ""
    last_section = 0
    skip = False
    current: dict | None = None  # пункт, который сейчас собираем
    last_number: str | None = None

    def flush() -> None:
        nonlocal current
        if current and current["text"].strip() and not EXCLUDED.match(current["text"]):
            clauses.append(_make_clause(current, src))
        current = None

    queue = deque(b.text for b in blocks)
    while queue:
        # «*2 НОРМАТИВНЫЕ ССЫЛКИ» — звёздочкой помечены изменённые разделы и пункты
        text = queue.popleft().lstrip("*").strip()
        if m := END_MATTER.search(text):
            text, _ = text[: m.start()].strip(), queue.clear()
            if not text:
                continue
        # Если абзац сам начинается с пункта («5.3.3.9 … *5.3.3.10 …»), вложенный номер
        # сравниваем с ним, а не с предыдущим пунктом — иначе 5.3.3.8 → 5.3.3.10 выглядит пропуском.
        head = CLAUSE.match(text)
        own = head.group(1) if head and is_successor(last_number, head.group(1), 3) else None
        parts = split_embedded(text, own or last_number)
        if len(parts) > 1:
            queue.appendleft(parts[1])
            text = parts[0]

        # Приложение А, Б, …
        if m := APPENDIX.match(text):
            flush()
            section, subsection, skip = f"Приложение {m.group(1)}", "", False
            last_number = m.group(1)
            current = {"number": m.group(1), "text": m.group(2).strip(), "section": section}
            continue

        # Раздел верхнего уровня: «5 ТРЕБОВАНИЯ К …» (только следующий по порядку)
        m = SECTION.match(text)
        if m and int(m.group(1)) == last_section + 1 and not section.startswith("Приложение"):
            title = m.group(2)
            words = title.split()
            n_upper = next((i for i, w in enumerate(words) if not w.isupper()), len(words))
            heading = " ".join(words[:n_upper])
            if len(heading) >= 4:
                flush()
                last_section = int(m.group(1))
                last_number = m.group(1)
                section, subsection = f"{m.group(1)} {heading}", ""
                skip = any(s in heading for s in SKIP_SECTIONS)
                rest = " ".join(words[n_upper:])
                current = {"number": m.group(1), "text": rest, "section": section}
                continue

        if skip:
            continue

        # Пункт: «5.2.1 Текст…»
        m = CLAUSE.match(text)
        # «В.8 как для плит…» — это хвост ссылки «…по Таблице В.8», а не пункт:
        # текст настоящего пункта не начинается со строчной буквы
        if m and (m.group(2) or "")[:1].islower():
            m = None
        if m and is_successor(last_number, m.group(1), tolerance=3):
            number, body = m.group(1), (m.group(2) or "").strip()
            flush()
            last_number = number
            nxt = queue[0] if queue else ""
            if _looks_like_subheading(number, body, nxt):
                subsection = f"{number} {body}"
                continue
            current = {
                "number": number,
                "text": body,
                "section": " / ".join(x for x in (section, subsection) if x),
            }
            continue

        if current is not None:
            current["text"] = (current["text"] + "\n" + text).strip()
    flush()
    return _dedupe_ids(clauses)


def _looks_like_subheading(number: str, body: str, next_text: str) -> bool:
    """«5.4 Лифты и лифтовые шахты», за которым идёт 5.4.1, — заголовок, а не пункт."""
    return (
        0 < len(body) < 100
        and not body.endswith((".", ":", ";"))
        and next_text.startswith(number + ".")
    )


def _make_clause(cur: dict, src: Source) -> Clause:
    return Clause(
        clause_id=f"{src.doc_slug}:{cur['number']}",
        doc_code=src.doc_code,
        doc_title=src.title,
        section=cur["section"],
        number=cur["number"],
        text=cur["text"],
        status=src.status,
        source_url=src.url,
    )


def _dedupe_ids(clauses: list[Clause]) -> list[Clause]:
    seen: dict[str, int] = {}
    for c in clauses:
        n = seen.get(c.clause_id, 0) + 1
        seen[c.clause_id] = n
        if n > 1:
            c.clause_id = f"{c.clause_id}#{n}"
    return clauses
