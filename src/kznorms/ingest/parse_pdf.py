"""PDF → list[Block].

Официальные издания норм РК двуязычные: сначала казахская часть, потом русская.
Берём только русские страницы, срезаем колонтитулы и склеиваем строки в абзацы.
"""

import re
from collections import Counter
from pathlib import Path

import pymupdf

from kznorms.ingest.segment import REF_WORDS
from kznorms.schemas import Block

KAZAKH_LETTERS = set("әғқңөұүһіӘҒҚҢӨҰҮҺІ")

# Начало нового абзаца: номер пункта (5.2.1, А.3), раздел («5 ТРЕБОВАНИЯ»), пункт списка,
# примечание, таблица, приложение
NEW_PARAGRAPH = re.compile(
    r"^(?:"
    r"(?:[А-ЯA-Z]\.)?\d+(?:\.\d+)+\.?(?:\s|$)"  # 5.2.1 / А.3.1
    r"|[А-ЯA-Z]\.\d+\.?(?:\s|$)"  # Д.38
    r"|\d+\.?\s+[А-ЯЁ]{2}"  # «5 ТРЕБОВАНИЯ» — номер раздела только перед заглавными
    r"|\d+\)(?:\s|$)|[а-я]\)(?:\s|$)"  # 1) / а)
    r"|[-–—•](?:\s|$)"  # пункт списка
    r"|Примечани[ея]|Таблица\s|Продолжение таблицы|Окончание таблицы"
    r"|ПРИЛОЖЕНИЕ|Приложение\s+[А-Я]\b"
    r")"
)
PAGE_NUMBER = re.compile(r"^(?:\d{1,3}|[IVXLC]{1,6})$")
DOT_LEADER = re.compile(r"(?:…|\.{4,})")  # строки оглавления «4.1 Цели ……… 3»
SCOPE_HEADING = re.compile(r"^1\.?\s+ОБЛАСТЬ ПРИМЕНЕНИЯ", re.IGNORECASE)


def page_lang(text: str) -> str:
    letters = [c for c in text if c.isalpha()]
    if not letters:
        return "empty"
    kk_share = sum(c in KAZAKH_LETTERS for c in letters) / len(letters)
    return "kk" if kk_share > 0.01 else "ru"


Fragment = tuple[tuple[float, float, float, float], str]  # (x0, y0, x1, y1), текст


def visual_rows(fragments: list[Fragment]) -> list[str]:
    """Собрать строки так, как их видит глаз: по вертикали, внутри строки — слева направо.

    Порядок фрагментов в потоке PDF не обязан совпадать с визуальным. В пожарном СП номер
    пункта (x=85) лежит на 2–3 pt ниже первой строки своего текста (x=118) и в потоке идёт
    после неё — простой get_text() отдаёт «…первая строка пункта 5.1.3.7» и только потом
    «5.1.3.7 продолжение», и все пункты раздела съезжают на одну строку.
    Фрагменты, перекрывающиеся по высоте больше чем наполовину, — одна визуальная строка.
    """
    rows: list[tuple[float, float, list[tuple[float, str]]]] = []
    for (x0, y0, _, y1), text in sorted(fragments, key=lambda f: (f[0][1] + f[0][3]) / 2):
        if rows:
            ry0, ry1, parts = rows[-1]
            overlap = min(ry1, y1) - max(ry0, y0)
            if overlap > 0.5 * min(ry1 - ry0, y1 - y0):
                parts.append((x0, text))
                continue
        rows.append((y0, y1, [(x0, text)]))
    lines = (" ".join(t for _, t in sorted(parts, key=lambda p: p[0])) for _, _, parts in rows)
    return [" ".join(line.split()) for line in lines if line.strip()]


def page_lines(page: pymupdf.Page) -> list[str]:
    fragments: list[Fragment] = [
        (tuple(line["bbox"]), "".join(span["text"] for span in line["spans"]))
        for block in page.get_text("dict")["blocks"]
        for line in block.get("lines", [])
    ]
    return visual_rows(fragments)


def find_running_lines(pages: list[list[str]], min_share: float = 0.3) -> set[str]:
    """Строки, которые повторяются в начале или конце многих страниц, — колонтитулы."""
    counts: Counter[str] = Counter()
    for lines in pages:
        counts.update(set(lines[:3] + lines[-3:]))
    threshold = max(2, int(len(pages) * min_share))
    return {line for line, n in counts.items() if n >= threshold}


def is_heading(line: str) -> bool:
    """Заголовок набран заглавными: «2 НОРМАТИВНЫЕ ССЫЛКИ», «ПРИЛОЖЕНИЕ А»."""
    letters = [c for c in line if c.isalpha()]
    return len(letters) >= 4 and all(c.isupper() for c in letters)


def join_lines(lines: list[str]) -> list[str]:
    """Склеиваем строки, разорванные выравниванием по ширине, в абзацы."""
    paragraphs: list[str] = []
    after_heading = False
    for line in lines:
        # заголовок на двух строках: «5 ТРЕБОВАНИЯ К … ПРИ ПРОЕКТИРОВАНИИ И» + «СТРОИТЕЛЬСТВЕ»
        if paragraphs and after_heading and is_heading(line) and not line[0].isdigit():
            paragraphs[-1] += " " + line
            continue
        # «…принимается по Таблице» + «В.8 как для плит»: номер в начале строки — хвост ссылки
        after_ref = bool(paragraphs) and paragraphs[-1].rsplit(" ", 1)[-1].lower() in REF_WORDS
        starts_new = (
            after_heading or (NEW_PARAGRAPH.match(line) and not after_ref) or is_heading(line)
        )
        after_heading = is_heading(line)
        if paragraphs and not starts_new:
            prev = paragraphs[-1]
            # перенос слова: «противо-» + «пожарный»
            if prev.endswith("-") and len(prev) > 1 and prev[-2].isalpha() and line[:1].islower():
                paragraphs[-1] = prev[:-1] + line
            else:
                paragraphs[-1] = prev + " " + line
        else:
            paragraphs.append(line)
    return paragraphs


def parse_pdf(path: Path) -> list[Block]:
    doc = pymupdf.open(str(path))
    ru_pages = [page_lines(p) for p in doc if page_lang(p.get_text()) == "ru"]

    running = find_running_lines(ru_pages)
    lines: list[str] = []
    for page in ru_pages:
        lines += [
            ln
            for ln in page
            if ln not in running and not PAGE_NUMBER.match(ln) and not DOT_LEADER.search(ln)
        ]

    # всё до «1 ОБЛАСТЬ ПРИМЕНЕНИЯ» — обложка, предисловие, оглавление
    start = next((i for i, ln in enumerate(lines) if SCOPE_HEADING.match(ln)), 0)
    return [Block(type="paragraph", text=p) for p in join_lines(lines[start:])]
