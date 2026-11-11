from kznorms.ingest.parse_pdf import find_running_lines, is_heading, join_lines, page_lang


def test_page_lang():
    assert page_lang("Здания жилые многоквартирные") == "ru"
    assert page_lang("Көппәтерлі тұрғын үй ғимараттары") == "kk"


def test_running_lines_detected():
    pages = [["СН РК 3.02-01-2023", str(i), f"Текст страницы {i}"] for i in range(10)]
    assert "СН РК 3.02-01-2023" in find_running_lines(pages)


def test_join_lines():
    lines = [
        "5.2.11 Количество",
        "эвакуационных лестничных клеток принимается исходя из общей",
        "площади. Противо-",
        "пожарные двери обязательны.",
        "1) первый пункт;",
        "2 НОРМАТИВНЫЕ ССЫЛКИ",
        "Для применения необходимы документы:",
    ]
    assert join_lines(lines) == [
        "5.2.11 Количество эвакуационных лестничных клеток принимается исходя из общей "
        "площади. Противопожарные двери обязательны.",
        "1) первый пункт;",
        "2 НОРМАТИВНЫЕ ССЫЛКИ",
        "Для применения необходимы документы:",
    ]


def test_is_heading():
    assert is_heading("5 ТРЕБОВАНИЯ К РАБОЧИМ ХАРАКТЕРИСТИКАМ")
    assert not is_heading("5.4 Лифты и лифтовые шахты")
