"""Gradio-интерфейс: вопрос → ответ с цитатами, проверкой verify и контекстом.

Отрисовка — чистые функции (AnswerResponse → HTML), без Gradio: их можно проверить тестом,
а сам Gradio — тонкая обёртка. Пайплайн создаётся один раз на процесс: загрузка BGE-M3 и
реранкера занимает ~10 с, на каждый вопрос её повторять нельзя.
"""

import html

from kznorms.schemas import AnswerResponse, Clause

STATUS = {
    "answered": ("ответ найден", "ok"),
    "partial": ("ответ неполный", "warn"),
    "not_found": ("в нормах не найдено", "muted"),
}

EXAMPLES = [
    "Можно ли сделать вход в гардеробную прямо из спальни?",
    "Какой просвет между маршами нужен в лестничной клетке Л2?",
    "Можно ли в новой многоэтажке поставить складной пандус?",
    "Нужны ли эвакуационные выходы с каждого уровня двухуровневой квартиры на 20-м метре?",
    "Какая максимальная площадь квартиры-студии?",
]

CSS = """
.kz-badges { display: flex; gap: 8px; flex-wrap: wrap; margin-bottom: 12px; }
.kz-badge { padding: 2px 10px; border-radius: 999px; font-size: 13px; font-weight: 600; }
.kz-ok { background: rgba(34,160,90,.15); color: #1f8a50; }
.kz-warn { background: rgba(220,150,20,.18); color: #a86b00; }
.kz-bad { background: rgba(210,60,60,.15); color: #c03535; }
.kz-muted { background: rgba(128,128,128,.15); color: inherit; }
.kz-answer { font-size: 16px; line-height: 1.55; margin: 4px 0 14px; }
.kz-cite { border-left: 3px solid rgba(34,120,220,.6); padding: 4px 0 4px 12px; margin: 8px 0; }
.kz-cite small, .kz-ctx small { opacity: .7; }
.kz-problems { margin-top: 10px; font-size: 14px; color: #c03535; }
.kz-ctx { border: 1px solid rgba(128,128,128,.25); border-radius: 8px; padding: 8px 12px;
          margin-bottom: 8px; }
.kz-ctx summary { cursor: pointer; }
.kz-ctx-text { white-space: pre-wrap; font-size: 14px; line-height: 1.5; margin-top: 6px; }
.kz-ctx mark { background: rgba(255,210,0,.45); color: inherit; padding: 0 1px; }
/* тёмная тема Gradio вешает класс .dark: тёмные оттенки на тёмном фоне не читаются */
.dark .kz-ok { color: #5fd394; }
.dark .kz-warn { color: #f0b54a; }
.dark .kz-bad, .dark .kz-problems { color: #ff8a8a; }
.dark .kz-ctx mark { background: rgba(255,210,0,.28); }
"""


def _badge(text: str, kind: str) -> str:
    return f'<span class="kz-badge kz-{kind}">{html.escape(text)}</span>'


def highlight(text: str, quotes: list[str]) -> str:
    """Экранировать текст пункта и подсветить в нём цитаты (без учёта регистра).

    Ищем точное вхождение: verify сравнивает мягче (без пунктуации), но для подсветки
    нужна позиция в исходном тексте. Не нашли — показываем без подсветки.
    """
    spans = []
    low = text.lower()
    for q in quotes:
        q = q.strip()
        i = low.find(q.lower()) if q else -1
        if i >= 0:
            spans.append((i, i + len(q)))
    out, pos = [], 0
    for a, b in sorted(spans):
        if a < pos:  # перекрывающиеся цитаты — склеиваем
            a = pos
        if a >= b:
            continue
        out += [html.escape(text[pos:a]), f"<mark>{html.escape(text[a:b])}</mark>"]
        pos = b
    out.append(html.escape(text[pos:]))
    return "".join(out)


def render_answer(resp: AnswerResponse) -> str:
    a = resp.answer
    label, kind = STATUS[a.status]
    badges = [_badge(label, kind)]
    if a.status != "not_found":
        badges.append(
            _badge("verify ✓" if resp.verified else "verify ✗", "ok" if resp.verified else "bad")
        )
    body = html.escape(a.answer) if a.answer else "<i>В переданных пунктах норм ответа нет.</i>"
    parts = [
        f'<div class="kz-badges">{"".join(badges)}</div>',
        f'<div class="kz-answer">{body}</div>',
    ]
    by_id = {c.clause_id: c for c in resp.contexts}
    for c in a.citations:
        cl = by_id.get(c.clause_id)
        where = f"{cl.doc_code}, п. {cl.number}" if cl else c.clause_id
        parts.append(
            f'<div class="kz-cite">«{html.escape(c.quote)}»<br>'
            f"<small>{html.escape(where)} · {html.escape(c.clause_id)}</small></div>"
        )
    if resp.problems:
        items = "".join(f"<li>{html.escape(p)}</li>" for p in resp.problems)
        parts.append(f'<div class="kz-problems">Проверка не пройдена:<ul>{items}</ul></div>')
    return "\n".join(parts)


def render_contexts(resp: AnswerResponse) -> str:
    """Пункты, которые видела модель; процитированные раскрыты, цитаты подсвечены."""
    quotes: dict[str, list[str]] = {}
    for c in resp.answer.citations:
        quotes.setdefault(c.clause_id, []).append(c.quote)
    blocks = []
    for i, cl in enumerate(resp.contexts, 1):
        blocks.append(_context_block(i, cl, quotes.get(cl.clause_id, [])))
    return "\n".join(blocks) or "<i>Контекст пуст.</i>"


def _context_block(i: int, cl: Clause, quotes: list[str]) -> str:
    mark = " ★" if quotes else ""
    head = f"{i}. {cl.doc_code}, п. {cl.number}{mark}"
    section = f"<br><small>{html.escape(cl.section)}</small>" if cl.section else ""
    return (
        f'<details class="kz-ctx"{" open" if quotes else ""}><summary><b>{html.escape(head)}</b>'
        f"{section}</summary>"
        f'<div class="kz-ctx-text">{highlight(cl.text, quotes)}</div></details>'
    )


def render_timings(resp: AnswerResponse) -> str:
    t = " · ".join(f"{k} {v:.0f} мс" for k, v in resp.timings_ms.items())
    return f"модель `{resp.model}` · {t}"


def build_app(pipeline_factory):
    """pipeline_factory() → RagPipeline; вызывается лениво, один раз."""
    import gradio as gr
    from openai import APIConnectionError

    state: dict = {}

    def ask(question: str):
        question = (question or "").strip()
        if not question:
            return "<i>Введите вопрос.</i>", "", ""
        if "pipe" not in state:
            state["pipe"] = pipeline_factory()
        try:
            resp = state["pipe"].answer(question)
        except APIConnectionError:
            msg = (
                '<div class="kz-problems">LLM недоступна: запусти llama-server '
                "(команда в CLAUDE.md) на порту 8080.</div>"
            )
            return msg, "", ""
        return render_answer(resp), render_contexts(resp), render_timings(resp)

    with gr.Blocks(title="Нормы РК — помощник нормоконтролёра") as demo:
        gr.Markdown(
            "## Строительные нормы РК\n"
            "Ответ строится только по найденным пунктам норм; каждая цитата проверяется "
            "на дословность (verify)."
        )
        with gr.Row():
            q = gr.Textbox(
                label="Вопрос",
                placeholder="Например: какой ширины должен быть марш лестницы в жилом доме?",
                scale=5,
                lines=1,  # однострочное: Enter отправляет вопрос (в многострочном — перенос)
            )
            btn = gr.Button("Спросить", variant="primary", scale=1)
        gr.Examples(EXAMPLES, inputs=q, label="Примеры")
        answer = gr.HTML()
        timings = gr.Markdown()
        with gr.Accordion("Контекст: пункты, которые видела модель", open=True):
            contexts = gr.HTML()
        outputs = [answer, contexts, timings]
        btn.click(ask, inputs=q, outputs=outputs)
        q.submit(ask, inputs=q, outputs=outputs)
    return demo
