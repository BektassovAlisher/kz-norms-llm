from docx import Document

from kznorms.ingest.parse_docx import parse_docx


def test_order_preserved(tmp_path):
    doc = Document()
    doc.add_paragraph("4.2.5 Первый пункт")
    table = doc.add_table(rows=2, cols=2)
    table.cell(0, 0).text = "Тип"
    table.cell(0, 1).text = "Значение"
    table.cell(1, 0).text = "А"
    table.cell(1, 1).text = "1"
    doc.add_paragraph("4.2.6 Второй пункт")
    path = tmp_path / "t.docx"
    doc.save(path)

    blocks = parse_docx(path)

    assert [b.type for b in blocks] == ["paragraph", "table", "paragraph"]
    assert blocks[1].text.splitlines()[0] == "| Тип | Значение |"


def test_spaces_collapsed(tmp_path):
    doc = Document()
    doc.add_paragraph("4.1\u00a0 Текст   с\tпробелами")
    doc.add_paragraph("   ")
    path = tmp_path / "t.docx"
    doc.save(path)

    blocks = parse_docx(path)

    assert len(blocks) == 1
    assert blocks[0].text == "4.1 Текст с пробелами"