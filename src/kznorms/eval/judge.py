"""LLM-судья по рубрике: верно / неполно / неверно — по сохранённым ответам, без перегенерации.

Зачем: verify ловит выдумку («нет ли в ответе того, чего нет в пункте»), но не смысл.
«1,80 м» дословно верно, но без «для школ в 4–5 этажей» может ввести в заблуждение —
это видит только тот, кто сравнивает ответ с эталоном по смыслу.

Ловушки LLM-судей и что с ними делаем:
- любовь к длинному (verbosity bias) — в рубрике прямо: длина не важна, важны условия;
- рассуждение до решения — в схеме `reason` идёт первым (грамматика llama.cpp выдаёт поля
  в порядке схемы): сначала сравнение, потом ответы;
- общая оценка «по смыслу» маленькому судье не по силам (у Qwen3-8B всё было «неполно»):
  решение разложено на два вопроса — «противоречит ли?» и «чего из эталона нет?»,
  а вердикт из них собирает код;
- самопредпочтение — судья Qwen3-8B из того же семейства, что ученик и учитель: в выводах
  оговариваем, вердикты выборочно проверяем глазами;
- судья — не ученик: ответы 4B не может оценивать сама 4B (`check_judge_model`).
"""

import json
from collections import Counter
from dataclasses import asdict, dataclass
from pathlib import Path

from kznorms.eval.eval_generation import GenRecord

VERDICTS = ("верно", "неполно", "неверно")

SYSTEM = """Ты проверяешь ответы помощника по строительным нормам Республики Казахстан.
Тебе дают вопрос, эталонный ответ, текст пункта нормы и ответ помощника.
Ответь на два простых вопроса.

1. contradicts — противоречит ли ответ помощника эталону? true, если ответ утверждает
   обратное («да» вместо «нет»), называет другое значение, отвечает не на тот вопрос
   или пишет, что ответа нет / он «не указан», когда эталон ответ даёт. Иначе false.
2. missing — какие значения и условия из эталона отсутствуют в ответе помощника
   (число, единица, тип здания, этажность, высота, класс, исключение). Перечисли каждое
   коротко; если в ответе есть всё, что есть в эталоне, — пустой список.
   Лишние сведения в ответе не считай. Формулировки могут отличаться — важен смысл.

Сначала в reason коротко сравни ответ с эталоном, потом заполни поля.
Ответ — JSON: {"reason": "...", "contradicts": true|false, "missing": ["...", ...]}."""

SCHEMA = {
    "type": "object",
    "properties": {
        "reason": {"type": "string"},  # первым: рассуждение до решения
        "contradicts": {"type": "boolean"},
        "missing": {"type": "array", "items": {"type": "string"}},
    },
    "required": ["reason", "contradicts", "missing"],
}


@dataclass
class Judgment:
    id: str
    mode: str
    verdict: str  # верно | неполно | неверно | отказ | ошибка судьи
    reason: str
    answer: str
    gold_answer: str


def judge_messages(r: GenRecord) -> list[dict]:
    gold_texts = [p["text"] for p in r.passages if p["clause_id"] in r.gold_clause_ids]
    clause = "\n\n".join(gold_texts) or "(текст пункта недоступен)"
    user = (
        f"ВОПРОС: {r.question}\n\nЭТАЛОН: {r.gold_answer}\n\nПУНКТ НОРМЫ:\n{clause}\n\n"
        f"ОТВЕТ ПОМОЩНИКА (статус {r.status}): {r.answer}"
    )
    return [{"role": "system", "content": SYSTEM}, {"role": "user", "content": user}]


def parse_verdict(raw: str | None) -> tuple[str, str]:
    """Вердикт считает код, а не судья: противоречит → неверно; есть пропуски → неполно.

    Маленькому судье проще ответить на два конкретных вопроса, чем вынести общую
    оценку «по смыслу» — та у Qwen3-8B скатывалась в «неполно» почти на всё.
    """
    try:
        data = json.loads(raw or "")
        contradicts, missing = data["contradicts"], data.get("missing") or []
        reason = str(data.get("reason", ""))
    except (json.JSONDecodeError, KeyError, TypeError, AttributeError):
        return "ошибка судьи", (raw or "")[:200]
    missing = [m for m in missing if str(m).strip()]
    if contradicts:
        return "неверно", reason
    if missing:
        return "неполно", "нет: " + "; ".join(map(str, missing))
    return "верно", reason


def judge_record(teacher, r: GenRecord) -> Judgment | None:
    """Только вопросы с ответом: у вопросов без ответа эталона нет — их меряет refusal_ok."""
    if not r.answerable or not r.gold_answer:
        return None
    if r.status == "not_found":
        return Judgment(r.id, r.mode, "отказ", "модель отказалась", r.answer, r.gold_answer)
    verdict, reason = parse_verdict(teacher.chat_json(judge_messages(r), SCHEMA, 0.0, 512))
    return Judgment(r.id, r.mode, verdict, reason, r.answer, r.gold_answer)


def summarize_judgments(js: list[Judgment], mode: str) -> dict[str, float]:
    rows = [j for j in js if j.mode == mode]
    counts = Counter(j.verdict for j in rows)
    return {v: counts[v] / max(len(rows), 1) for v in (*VERDICTS, "отказ", "ошибка судьи")}


def check_judge_model(model_name: str, hint: str = "8B") -> None:
    """Ответы 4B-ученика не должна оценивать сама 4B — нужен другой (больший) судья."""
    if hint not in model_name:
        raise SystemExit(
            f"судья — {model_name}, а нужен Qwen3-8B: запусти llama-server с "
            "artifacts/models/Qwen3-8B-Q4_K_M.gguf (без --lora)"
        )


def save_judgments(js: list[Judgment], path: Path) -> None:
    with open(path, "w", encoding="utf-8") as f:
        for j in js:
            f.write(json.dumps(asdict(j), ensure_ascii=False) + "\n")
