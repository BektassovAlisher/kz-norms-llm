"""PDF → list[Block].

Официальные издания норм РК двуязычные: сначала казахская часть, потом русская.
Берём только русские страницы, срезаем колонтитулы и склеиваем строки в абзацы.
"""

import re
from collections import Counter
from pathlib import Path

import pymupdf

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


def page_lines(text: str) -> list[str]:
    return [" ".join(line.split()) for line in text.splitlines() if line.strip()]


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
        starts_new = after_heading or NEW_PARAGRAPH.match(line) or is_heading(line)
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
    ru_pages = [page_lines(p.get_text()) for p in doc if page_lang(p.get_text()) == "ru"]

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
