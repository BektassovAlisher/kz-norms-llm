"""Нарезка пунктов на фрагменты (chunks) для индексации.

Единица ответа — пункт (clause), а единица индексации — фрагмент: 99% пунктов короче 1800
символов и становятся одним фрагментом, а редкие гиганты (таблицы приложений по 10 тыс.
символов) режутся на куски с перекрытием. Иначе эмбеддер обрежет текст по max_seq_length,
а BM25 «размажет» вес слов по огромному документу.

К каждому фрагменту спереди добавляется заголовок «документ. раздел» — фрагмент из середины
таблицы сам по себе не говорит, к чему он относится (contextual chunk header).
"""

import re
from collections.abc import Iterable
from dataclasses import dataclass

from kznorms.schemas import Clause

# Граница предложения: знак конца + пробел + заглавная буква/кавычка.
# Заглавная нужна, чтобы не резать «п. 5.3.2» и «т. е.».
_SENT_END = re.compile(r"(?<=[.;!?])\s+(?=[А-ЯЁA-Z«])")


@dataclass(frozen=True)
class Chunk:
    chunk_id: str  # "<clause_id>#<i>"
    clause_id: str
    text: str  # то, что индексируем: заголовок + фрагмент текста пункта


def chunk_header(c: Clause) -> str:
    return ". ".join(p for p in (c.doc_title, c.section) if p)


def _pieces(text: str, max_chars: int) -> list[str]:
    """Строки → предложения → (если предложение огромное) куски по словам."""
    out: list[str] = []
    for line in text.split("\n"):
        for sent in _SENT_END.split(line):
            sent = sent.strip()
            if not sent:
                continue
            while len(sent) > max_chars:
                cut = sent.rfind(" ", 0, max_chars)
                cut = cut if cut > 0 else max_chars
                out.append(sent[:cut].strip())
                sent = sent[cut:].strip()
            if sent:
                out.append(sent)
    return out


def split_text(text: str, max_chars: int = 1500, overlap: int = 200) -> list[str]:
    """Жадно упаковать предложения в куски ≤ max_chars; хвост предыдущего куска
    (≤ overlap символов) повторяется в начале следующего, чтобы мысль на стыке не терялась."""
    if len(text) <= max_chars:
        return [text]
    chunks: list[str] = []
    cur: list[str] = []
    for p in _pieces(text, max_chars):
        if cur and len(" ".join([*cur, p])) > max_chars:
            chunks.append(" ".join(cur))
            tail: list[str] = []
            for s in reversed(cur):
                if len(" ".join([s, *tail])) > overlap:
                    break
                tail.insert(0, s)
            cur = tail if len(" ".join([*tail, p])) <= max_chars else []
        cur.append(p)
    if cur:
        chunks.append(" ".join(cur))
    return chunks


def make_chunks(
    clauses: Iterable[Clause], max_chars: int = 1500, overlap: int = 200
) -> list[Chunk]:
    chunks: list[Chunk] = []
    for c in clauses:
        head = chunk_header(c)
        for i, part in enumerate(split_text(c.text, max_chars, overlap)):
            text = f"{head}\n{part}" if head else part
            chunks.append(Chunk(f"{c.clause_id}#{i}", c.clause_id, text))
    return chunks
