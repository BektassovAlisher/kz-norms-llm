from kznorms.ingest.segment import is_successor, segment, split_embedded, split_number
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


def test_split_embedded_amended_clause():
    text = "Не устанавливать на двери. *5.3.3.10 Входы следует выделять."
    assert split_embedded(text, "5.3.3.9") == [
        "Не устанавливать на двери.",
        "*5.3.3.10 Входы следует выделять.",
    ]


def test_segment_splits_clause_embedded_in_numbered_paragraph():
    clauses = segment(
        blocks(
            "5 ТРЕБОВАНИЯ",
            "5.1 Входы",
            "5.1.1 Первый пункт.",
            "5.1.2 Номера не ставить на двери. *5.1.3 Входы следует выделять.",
            "5.1.4 Четвёртый.",
        ),
        SRC,
    )
    assert [c.number for c in clauses] == ["5.1.1", "5.1.2", "5.1.3", "5.1.4"]


def test_split_embedded_asterisk_after_number():
    text = "Пороги не более 15 см. 5.5.30* Двери выходов из квартир."
    assert split_embedded(text, "5.5.29") == [
        "Пороги не более 15 см.",
        "5.5.30* Двери выходов из квартир.",
    ]
    clauses = segment(
        blocks(
            "5 ТРЕБОВАНИЯ",
            "5.1.1 Пороги не более 15 см. 5.1.2* Двери выходов из квартир.",
            "5.1.3 Ещё.",
        ),
        SRC,
    )
    assert [c.number for c in clauses] == ["5.1.1", "5.1.2", "5.1.3"]
    assert clauses[1].text == "Двери выходов из квартир."


def test_end_matter_is_cut():
    clauses = segment(
        blocks(
            "5 ТРЕБОВАНИЯ",
            "5.1 Входы",
            "5.1.1 Последний пункт. УДК 728.4 МКС 93.040.10 Ключевые слова: доступность",
            "Ещё что-то после выходных данных",
        ),
        SRC,
    )
    assert clauses[-1].text == "Последний пункт."


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


def test_number_followed_by_lowercase_is_not_clause():
    clauses = segment(
        blocks("5 ТРЕБОВАНИЯ", "5.1.1 Принимается по таблице", "5.4 как для плит.", "5.1.2 Далее."),
        SRC,
    )
    assert [c.number for c in clauses] == ["5.1.1", "5.1.2"]
    assert clauses[0].text == "Принимается по таблице\n5.4 как для плит."


def test_false_jump_from_table_row_does_not_swallow_clauses():
    # «9.8 Комплексные пункты…» — строка таблицы, формально «продолжает» 6.1.1;
    # жадный разбор принимал её, и 6.1.2 … 6.2.1 терялись как «шаг назад»
    clauses = segment(
        blocks(
            "6 ТРЕБОВАНИЯ",
            "6.1.1 Первый.",
            "Таблица 3",
            "9.8 Комплексные приемные пункты",
            "6.1.2 Второй.",
            "6.1.3 Третий.",
            "6.2.1 Четвёртый.",
        ),
        SRC,
    )
    assert [c.number for c in clauses] == ["6.1.1", "6.1.2", "6.1.3", "6.2.1"]
    assert "9.8 Комплексные" in clauses[0].text


def test_table_values_with_zero_are_not_clauses():
    clauses = segment(blocks("4 ТРЕБОВАНИЯ", "4.5.1 Пункт.", "5.0 0.6 0.5", "4.6 Освещение."), SRC)
    assert [c.number for c in clauses] == ["4.5.1", "4.6"]


def test_chain_of_embedded_clauses_in_one_paragraph():
    clauses = segment(
        blocks(
            "6 ТРЕБОВАНИЯ",
            "6.1.1 Выбросы через дымоходы.",
            "Дополнен подпунктами. 6.1.1.1* Первый. 6.1.1.2* Второй. 6.1.1.3* Третий.",
            "6.1.2 Дальше.",
        ),
        SRC,
    )
    assert [c.number for c in clauses] == ["6.1.1", "6.1.1.1", "6.1.1.2", "6.1.1.3", "6.1.2"]


def test_numeric_appendix_with_p_prefixed_clauses():
    assert split_number("П4.1.1") == ("П4", [1, 1])
    assert split_number("Д.3.2") == ("Д", [3, 2]) and split_number("Б") == ("Б", [])
    clauses = segment(
        blocks(
            "17 БЕЗОПАСНОСТЬ",
            "17.15 Последний пункт.",
            "Приложение 4",
            "(рекомендуемое)",
            "П4.1.1 Оборудование станции мониторинга.",
            "П4.1.2 Необходимо обеспечить места установки.",
            "- вблизи центральной оси;",
        ),
        SRC,
    )
    assert [c.number for c in clauses] == ["17.15", "П4", "П4.1.1", "П4.1.2"]
    assert clauses[-1].text.endswith("- вблизи центральной оси;")


def test_toc_appendix_entries_are_dropped():
    clauses = segment(
        blocks(
            "5 ТРЕБОВАНИЯ",
            "5.1.1 Пункт.",
            "Приложение А (обязательное). Перечень документов",
            "Приложение Б (рекомендуемое). Термины",
            "Приложение А",
            "(обязательное) Перечень документов. " + "СН РК 1.01-01 Основные положения. " * 10,
            "Приложение Б",
            "Термины. " + "Анализ уязвимости — оценка. " * 15,
        ),
        SRC,
    )
    assert [c.clause_id for c in clauses] == ["doc:5.1.1", "doc:А", "doc:Б"]
    assert len(clauses[1].text) > 300


def test_section_heading_with_asterisk_after_number():
    clauses = segment(
        blocks(
            "6 ТРЕБОВАНИЯ", "6.11 Пункт.", "7 * СИСТЕМЫ ВОДОПРОВОДА", "7.1 Первый.", "8.1 Дальше."
        ),
        SRC,
    )
    assert [c.number for c in clauses] == ["6.11", "7.1", "8.1"]
    assert clauses[1].section == "7 СИСТЕМЫ ВОДОПРОВОДА"


def test_section_heading_with_latin_lookalike_letter():
    clauses = segment(
        blocks(
            "1 ОБЛАСТЬ ПРИМЕНЕНИЯ",
            "1.1 Нормы.",
            "2* НОРМАТИВНЫЕ ССЫЛКИ",
            "СН РК 1.01-01 Основные положения.",
            "3 TЕРМИНЫ И ОПРЕДЕЛЕНИЯ",  # T — латинская
            "3.1 Канализация: система.",
        ),
        SRC,
    )
    assert [c.number for c in clauses] == ["1.1", "3.1"]
    assert clauses[1].section == "3 ТЕРМИНЫ И ОПРЕДЕЛЕНИЯ"
