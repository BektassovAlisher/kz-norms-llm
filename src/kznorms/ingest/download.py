"""Скачивание норм из data/sources.csv в data/raw/ (с паузой, без повторной загрузки)."""

import csv
import time
from pathlib import Path

import httpx
from rich import print
from tenacity import retry, stop_after_attempt, wait_exponential

from kznorms.paths import RAW, SOURCES_CSV
from kznorms.schemas import Source

HEADERS = {"User-Agent": "kz-norms-llm/0.1 (study project)"}


def load_sources(path: Path = SOURCES_CSV) -> list[Source]:
    with open(path, encoding="utf-8") as f:
        # пустые ячейки выкидываем, чтобы сработали значения по умолчанию из схемы
        return [Source(**{k: v for k, v in row.items() if v}) for row in csv.DictReader(f)]


def raw_path(src: Source) -> Path:
    return RAW / f"{src.doc_slug}.{src.format}"


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=20))
def _fetch(client: httpx.Client, url: str) -> bytes:
    resp = client.get(url)
    resp.raise_for_status()
    return resp.content


def download_all(sources: list[Source], delay: float = 2.0) -> None:
    RAW.mkdir(parents=True, exist_ok=True)
    with httpx.Client(headers=HEADERS, timeout=60, follow_redirects=True) as client:
        for src in sources:
            path = raw_path(src)
            if path.exists():
                print(f"[dim]есть[/dim] {path.name}")
                continue
            if not src.url:
                print(f"[yellow]нет url[/yellow] {src.doc_slug}: положи файл в {path} вручную")
                continue
            try:
                data = _fetch(client, src.url)
            except Exception as e:  # один недоступный файл не должен ронять весь ingest
                cause = e.last_attempt.exception() if hasattr(e, "last_attempt") else e
                print(
                    f"[red]не скачан[/red] {src.doc_slug}: {type(cause).__name__} — "
                    "запусти ingest ещё раз позже"
                )
                continue
            # DOCX — это zip-архив, он всегда начинается с байтов PK
            # PDF начинается с «%PDF»; иначе сайт отдал HTML-страницу ошибки с кодом 200
            magic = {"docx": b"PK", "pdf": b"%PDF"}.get(src.format)
            if magic and not data.startswith(magic):
                print(
                    f"[red]не {src.format.upper()}[/red] {src.doc_slug}: сайт вернул другое, "
                    "проверь url"
                )
                continue
            path.write_bytes(data)
            print(f"[green]скачан[/green] {path.name} ({len(data) // 1024} КБ)")
            time.sleep(delay)
