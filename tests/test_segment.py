from kznorms.ingest.segment import is_successor, segment, split_embedded
from kznorms.schemas import Block, Source

SRC = Source(doc_slug="doc", doc_code="СН РК тест", title="Тест", format="pdf")


def blocks(*texts: str) -> list[Block]:
    return [Block(type="paragraph", text=t) for t in texts]


def test_successor_rules():
    assert is_successor("5.2.3", "5.2.4")  # следующий сосед
    assert is_successor("5.2", "5.2.1")  # вглубь
    assert is_successor("5.2.20", "5.3.1")  # следующий подраздел
    assert is_successor("5.4.11", "6.1.1")  # следующий раздел
    assert is_successor("В.7.3", "В.8.1")  # внутри приложения
    assert is_successor("В.7.3", "Г.1")  # новое приложение
    assert not is_successor("5.2.3", "5.2.7")  # пропуск — подозрительно
    assert is_successor("3.7", "3.9", tolerance=3)  # 3.8 исключён — допустимо
    assert not is_successor("5.3.3.1", "26.05.2026", tolerance=3)  # дата
    assert not is_successor("В.7.3", "12.5", tolerance=3)  # число из таблицы


def test_split_embedded_clause():
    text = "Ставить ограждения. 5.3.16.10 Каждая зона безопасности оснащается."
    assert split_embedded(text, "5.3.16.9") == [
        "Ставить ограждения.",
        "5.3.16.10 Каждая зона безопасности оснащается.",
    ]


def test_reference_is_not_split():
    text = "Требования приведены в п. 5.3.2 Настоящего свода."
    assert split_embedded(text, "5.3.1") == [text]


def test_segment_basic():
    clauses = segment(
        blocks(
            "1 ОБЛАСТЬ ПРИМЕНЕНИЯ",
            "1.1 Нормы распространяются на здания.",
            "2 НОРМАТИВНЫЕ ССЫЛКИ",
            "СН РК 3.03-05-2014 Стоянки автомобилей.",
            "3 ТРЕБОВАНИЯ",
            "3.1 Лестницы",
            "3.1.1 Ширина марша принимается по расчёту:",
            "- не менее указанной в таблице;",
            "3.1.2 (Исключен)",
            "3.1.3 Ограждения обязательны. Изменение от 26.05.2026 года.",
            "ПРИЛОЖЕНИЕ А",
            "А.1 Пункт приложения.",
        ),
        SRC,
    )
    ids = [c.clause_id for c in clauses]
    assert ids == ["doc:1.1", "doc:3.1.1", "doc:3.1.3", "doc:А.1"]

    c = clauses[1]
    assert c.section == "3 ТРЕБОВАНИЯ / 3.1 Лестницы"  # «3.1 Лестницы» стал заголовком
    assert c.text.endswith("- не менее указанной в таблице;")  # список приклеен к пункту
    assert c.doc_code == "СН РК тест"
