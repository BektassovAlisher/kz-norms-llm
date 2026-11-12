"""Сплит документов train/val/test.

Делим не вопросы и не пункты, а документы — точнее, группы документов. СН задаёт требование
в общем виде, СП к нему повторяет и раскрывает то же самое, часто почти дословно. Если СН
попадёт в train, а его СП — в test, модель на тесте увидит «знакомый» текст, и мы померим
заучивание вместо навыка. Поэтому единица сплита — группа (`group` в sources.csv), и вся
группа целиком лежит в одном сплите.

- train — из этих документов генерируем синтетику для обучения;
- val   — синтетика для валидации во время обучения (ранняя остановка, выбор чекпойнта);
- test  — только ручные вопросы, ни один пункт отсюда не попадает в обучение — ни как
          gold, ни как дистрактор в контексте.

Индекс поиска при этом содержит все документы: в проде RAG ищет по всему корпусу.
"""

from collections import defaultdict

from kznorms.schemas import EvalItem, Source

SPLITS = ("train", "val", "test")


def doc_of(clause_id: str) -> str:
    """'sp-zhilye-zdaniya:4.4.3.3' → 'sp-zhilye-zdaniya'."""
    return clause_id.split(":", 1)[0]


def check_sources(sources: list[Source]) -> list[str]:
    """Ошибки разметки сплита: пустые поля, группа, разорванная между сплитами."""
    errors = [f"{s.doc_slug}: не указаны group/split" for s in sources if not (s.group and s.split)]
    by_group: dict[str, set[str]] = defaultdict(set)
    for s in sources:
        if s.group:
            by_group[s.group].add(s.split)
    errors += [
        f"группа {g} разорвана между сплитами: {sorted(sp)}"
        for g, sp in by_group.items()
        if len(sp) > 1
    ]
    return errors


def load_splits(sources: list[Source] | None = None) -> dict[str, str]:
    """doc_slug → split. Падает, если разметка неполная: молча отнести документ
    куда-нибудь — ровно та ошибка, ради которой сплит заводился."""
    if sources is None:
        from kznorms.ingest.download import load_sources

        sources = load_sources()
    if errors := check_sources(sources):
        raise ValueError("сплит в sources.csv размечен с ошибками:\n" + "\n".join(errors))
    return {s.doc_slug: s.split for s in sources}


def item_split(item: EvalItem, splits: dict[str, str]) -> str:
    """Сплит вопроса — сплит документов его gold-пунктов.

    Gold из разных сплитов → "mixed": такой вопрос не годится для честного теста (один из
    правильных пунктов модель видела при обучении). Вопрос без ответа ни на какой пункт не
    опирается, утечки нет — он тестовый.
    """
    if not item.answerable:
        return "test"
    found = {splits.get(doc_of(g), "?") for g in item.gold_clause_ids}
    return found.pop() if len(found) == 1 else "mixed"
