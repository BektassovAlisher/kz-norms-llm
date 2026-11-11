"""DOCX → list[Block].

Обходим document.element.body по порядку, чтобы таблицы не отрывались от пунктов.
"""

from collections.abc import Iterator
from pathlib import Path

from docx import Document
from docx.document import Document as DocumentObj
from docx.oxml.ns import qn
from docx.table import Table
from docx.text.paragraph import Paragraph

from kznorms.schemas import Block


def iter_block_items(doc: DocumentObj) -> Iterator[Paragraph | Table]:
    """Абзацы и таблицы в том порядке, в котором они стоят в документе."""
    for child in doc.element.body.iterchildren():
        if child.tag == qn("w:p"):
            yield Paragraph(child, doc)
        elif child.tag == qn("w:tbl"):
            yield Table(child, doc)


def clean(text: str) -> str:
    """Схлопываем пробелы, переносы и неразрывные пробелы в один пробел."""
    return " ".join(text.split())


def table_to_markdown(table: Table) -> str:
    rows = [[clean(cell.text).replace("|", "/") for cell in row.cells] for row in table.rows]
    rows = [r for r in rows if any(r)]
    if not rows:
        return ""
    width = max(len(r) for r in rows)
    rows = [r + [""] * (width - len(r)) for r in rows]
    lines = ["| " + " | ".join(rows[0]) + " |", "|" + "---|" * width]
    lines += ["| " + " | ".join(r) + " |" for r in rows[1:]]
    return "\n".join(lines)


def parse_docx(path: Path) -> list[Block]:
    doc = Document(str(path))
    blocks: list[Block] = []
    for item in iter_block_items(doc):
        if isinstance(item, Paragraph):
            text = clean(item.text)
            if not text:
                continue
            style = item.style.name.lower() if item.style is not None else ""
            kind = "heading" if style.startswith(("heading", "заголовок")) else "paragraph"
            blocks.append(Block(type=kind, text=text))
        else:
            md = table_to_markdown(item)
            if md:
                blocks.append(Block(type="table", text=md))
    return blocks
