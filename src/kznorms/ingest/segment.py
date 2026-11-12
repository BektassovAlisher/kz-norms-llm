"""list[Block] → list[Clause].

Нарезка по номерам пунктов. Всё, что идёт после пункта и не начинается с нового номера
(списки, примечания, строки таблиц), приклеивается к нему.

Главная идея: номер в начале строки считается новым пунктом, только если он правдоподобно
продолжает нумерацию (5.2.3 → 5.2.4, 5.2.20 → 5.3.1, В.7.3 → В.8.1). Так в пункты не
попадают даты «26.05.2026», значения из таблиц «12.5» и ссылки «согласно 5.3.2».

Решение принимается глобально, а не жадно (см. `numbering_chain`): одна ложная «удачная»
строка таблицы («9.8», «17.15») при жадном подходе становится пунктом, и все настоящие
пункты после неё выглядят шагом назад и теряются — в СП РК 3.02-02 так пропадал весь
конец документа.
"""

import re
from collections import deque
from functools import lru_cache

from kznorms.schemas import Block, Clause, Source

# «*» до или после номера («*5.3.3.10», «5.5.30*») — пометка изменённого пункта
# Номер пункта: «5.2.1», «Д.3.2», «Д.38», а в приложениях с цифровым номером — «П4.1.1».
# Букву с цифрой разрешаем только «П»: «Ф1.3» в начале строки — класс пожарной опасности.
NUMBER = r"(?:[А-ЯA-Z]\.)?\d+(?:\.\d+)+|[А-ЯA-Z]\.\d+|П\d+(?:\.\d+)+"
CLAUSE = re.compile(rf"^({NUMBER})\.?\*?(?:\s+(.*))?$", re.S)
# «7 * СИСТЕМЫ ВНУТРЕННЕГО ВОДОПРОВОДА» — звёздочка изменённого раздела после номера
SECTION = re.compile(r"^(\d+)\.?(?:\s*\*)?\s+([А-ЯЁA-Z].*)$", re.S)
# «3 TЕРМИНЫ» — латинская T в русском слове. Чиним только в названиях разделов:
# в тексте пунктов «смешанные» слова почти всегда законные (tн, Rпр, IIIб, REI)
_LATIN_TO_CYR = str.maketrans("ABCEHKMOPTXY", "АВСЕНКМОРТХУ")
# «Приложение Б», а в части СП — «Приложение 4» (пункты в нём — П4.1.1)
APPENDIX = re.compile(r"^(?:ПРИЛОЖЕНИЕ|Приложение)\s+([А-ЯA-Z]|\d{1,2})\b(.*)$", re.S)
EMBEDDED = re.compile(
    r"(?<=[.;:])\s+\*?((?:[А-ЯA-Z]\.)?\d+(?:\.\d+)+|П\d+(?:\.\d+)+)\.?\*?\s+(?=[А-ЯЁ])"
)
REF_WORDS = (
    "п.", "пп.", "пункт", "пункта", "пункте", "пунктом", "пунктов", "пунктах", "пункту",
    "табл.", "таблица", "таблицы", "таблице", "таблицей", "таблицам", "таблицах",
    "рис.", "рисунок", "рисунка", "рисунке", "рисунку", "согласно",  # не «см.»: это и сантиметры
)  # fmt: skip
SKIP_SECTIONS = ("НОРМАТИВНЫЕ ССЫЛКИ",)
EXCLUDED = re.compile(r"^\(?\s*исключ", re.IGNORECASE)
# Выходные данные в конце издания («УДК 728.4 МКС 93.040.10 Ключевые слова: …») — конец норм
END_MATTER = re.compile(r"\b(?:УДК\s+\d|МКС\s+\d+\.\d+)")


@lru_cache(maxsize=100_000)
def split_number(number: str) -> tuple[str, list[int]]:
    """«Д.3.2» → ("Д", [3, 2]); «5.2.1» → ("", [5, 2, 1]); «Б» → ("Б", []);
    «П4.1.1» → ("П4", [1, 1]) — пункт 1.1 приложения 4."""
    m = re.match(r"([А-ЯA-Z]\d*)\.?(.*)$", number)
    prefix, number = (m.group(1), m.group(2)) if m else ("", number)
    return prefix, [int(x) for x in number.split(".") if x]


def appendix_number(letter_or_digit: str) -> str:
    """«Б» → «Б»; «4» → «П4»: иначе приложение 4 и раздел 4 получат один clause_id."""
    return f"П{letter_or_digit}" if letter_or_digit.isdigit() else letter_or_digit


def plausible_number(number: str) -> bool:
    """В нормах нет пунктов «5.0», «7.00», «4.05» — так выглядят значения из таблиц."""
    _, parts = split_number(number)
    raw = re.sub(r"^[А-ЯA-Z]\d*\.?", "", number).split(".")
    return all(p > 0 for p in parts) and not any(len(x) > 1 and x[0] == "0" for x in raw)


@lru_cache(maxsize=500_000)
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


def _clean(text: str) -> str:
    # «*2 НОРМАТИВНЫЕ ССЫЛКИ» — звёздочкой помечены изменённые разделы и пункты
    return text.lstrip("*").strip()


def _section_heading(text: str) -> tuple[int, str, str] | None:
    """«5 ТРЕБОВАНИЯ К ЗДАНИЯМ Текст…» → (5, "ТРЕБОВАНИЯ К ЗДАНИЯМ", "Текст…")."""
    m = SECTION.match(text)
    if not m:
        return None
    words = [
        w.translate(_LATIN_TO_CYR) if re.search("[А-ЯЁа-яё]", w) else w for w in m.group(2).split()
    ]
    n_upper = next((i for i, w in enumerate(words) if not w.isupper()), len(words))
    heading = " ".join(words[:n_upper])
    if len(heading) < 4:
        return None
    return int(m.group(1)), heading, " ".join(words[n_upper:])


def _clause_head(text: str) -> tuple[str, str] | None:
    m = CLAUSE.match(text)
    # «В.8 как для плит…» — это хвост ссылки «…по Таблице В.8», а не пункт:
    # текст настоящего пункта не начинается со строчной буквы
    if not m or (m.group(2) or "")[:1].islower() or not plausible_number(m.group(1)):
        return None
    return m.group(1), (m.group(2) or "").strip()


def head_candidate(text: str) -> tuple[str, str] | None:
    """Номер в начале абзаца: ("appendix", "Б") | ("section", "5") | ("clause", "5.2.1")."""
    if m := APPENDIX.match(text):
        return "appendix", appendix_number(m.group(1))
    if sec := _section_heading(text):
        return "section", str(sec[0])
    if cl := _clause_head(text):
        return "clause", cl[0]
    return None


def numbering_chain(cands: list[tuple[str, str]], window: int = 400) -> set[int]:
    """Самая длинная цепочка номеров, где каждый правдоподобно продолжает предыдущий.

    Это задача о наибольшей возрастающей подпоследовательности: dp[i] — длина лучшей
    цепочки, кончающейся на i-м кандидате. Ложный номер из таблицы может «продолжить»
    нумерацию, но обрывает цепочку — настоящие пункты дальше от него не продолжаются,
    поэтому цепочка в обход него длиннее и выигрывает. Приложения начинают новую ветку
    нумерации (Б.1, Б.2 …) и могут стоять после чего угодно.
    """
    n = len(cands)
    dp, prev = [1] * n, [-1] * n
    for i, (kind, num) in enumerate(cands):
        for j in range(max(0, i - window), i):
            if dp[j] + 1 > dp[i] or (dp[j] + 1 == dp[i] and j > prev[i]):
                if kind == "appendix" or is_successor(cands[j][1], num, tolerance=3):
                    dp[i], prev[i] = dp[j] + 1, j
    chain: set[int] = set()
    i = max(range(n), key=lambda k: (dp[k], k)) if n else -1
    while i >= 0:
        chain.add(i)
        i = prev[i]
    return chain


def segment(blocks: list[Block], src: Source) -> list[Clause]:
    clauses: list[Clause] = []
    section, subsection = "", ""
    skip = False
    current: dict | None = None  # пункт, который сейчас собираем
    last_number: str | None = None

    def flush() -> None:
        nonlocal current
        if current and current["text"].strip() and not EXCLUDED.match(current["text"]):
            clauses.append(_make_clause(current, src))
        current = None

    # Выходные данные в конце издания — дальше текста норм нет
    texts: list[str] = []
    for b in blocks:
        text = _clean(b.text)
        if m := END_MATTER.search(text):
            if head := text[: m.start()].strip():
                texts.append(head)
            break
        texts.append(text)

    # Глобальный выбор: какие номера в начале абзацев — настоящие пункты
    heads = [head_candidate(t) for t in texts]
    cand_idx = [i for i, h in enumerate(heads) if h]
    chain = {cand_idx[k] for k in numbering_chain([heads[i] for i in cand_idx])}

    # (текст, индекс исходного абзаца); у кусков после split_embedded индекса нет
    queue: deque[tuple[str, int | None]] = deque((t, i) for i, t in enumerate(texts))
    while queue:
        text, idx = queue.popleft()
        in_chain = idx in chain if idx is not None else None
        # Если абзац сам начинается с пункта («5.3.3.9 … *5.3.3.10 …»), вложенный номер
        # сравниваем с ним, а не с предыдущим пунктом — иначе 5.3.3.8 → 5.3.3.10 выглядит пропуском.
        if idx is not None:
            own = heads[idx][1] if in_chain and heads[idx][0] == "clause" else None
        else:  # кусок после split_embedded: цепочка о нём не знает, проверяем локально
            h = _clause_head(text)
            own = h[0] if h and is_successor(last_number, h[0], tolerance=3) else None
        parts = split_embedded(text, own or last_number)
        if len(parts) > 1:
            queue.appendleft((_clean(parts[1]), None))
            text = parts[0]

        # Приложение А, Б, …
        if m := APPENDIX.match(text):
            flush()
            section, subsection, skip = f"Приложение {m.group(1)}", "", False
            last_number = appendix_number(m.group(1))
            current = {"number": last_number, "text": m.group(2).strip(), "section": section}
            continue

        # Раздел верхнего уровня: «5 ТРЕБОВАНИЯ К …»
        sec = _section_heading(text)
        if sec and (in_chain if in_chain is not None else False):
            num, heading, rest = sec
            flush()
            last_number = str(num)
            section, subsection = f"{num} {heading}", ""
            skip = any(s in heading for s in SKIP_SECTIONS)
            current = {"number": str(num), "text": rest, "section": section}
            continue

        if skip:
            continue

        # Пункт: «5.2.1 Текст…»
        cl = _clause_head(text)
        accepted = (
            in_chain
            if in_chain is not None
            else (cl is not None and is_successor(last_number, cl[0], tolerance=3))
        )
        if cl and accepted:
            number, body = cl
            flush()
            last_number = number
            nxt = queue[0][0] if queue else ""
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
    return _dedupe_ids(_drop_toc_appendices(clauses))


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


def _drop_toc_appendices(clauses: list[Clause]) -> list[Clause]:
    """«Приложение 1 (обязательное). Перечень…» из оглавления без отточий даёт короткий
    «пункт» с тем же номером, что и настоящее приложение дальше. Короткие ранние копии
    номера приложения выбрасываем — остаётся последняя, с содержимым."""
    last = {c.clause_id: i for i, c in enumerate(clauses)}
    return [
        c
        for i, c in enumerate(clauses)
        if not (split_number(c.number)[1] == [] and i < last[c.clause_id] and len(c.text) < 300)
    ]


def _dedupe_ids(clauses: list[Clause]) -> list[Clause]:
    seen: dict[str, int] = {}
    for c in clauses:
        n = seen.get(c.clause_id, 0) + 1
        seen[c.clause_id] = n
        if n > 1:
            c.clause_id = f"{c.clause_id}#{n}"
    return clauses
