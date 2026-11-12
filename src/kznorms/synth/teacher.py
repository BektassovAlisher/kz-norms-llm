"""Клиент к модели-учителю с кэшем ответов на диск.

Кэш — главное в синтетике: тысячи платных запросов, и любой сбой (сеть, лимиты, Ctrl+C)
не должен стоить денег повторно. Ключ — хэш всего запроса (адрес, модель, сообщения, схема,
температура): изменил промпт — получишь новые ответы, не изменил — возьмутся старые.
"""

import hashlib
import json
import time
from pathlib import Path

from openai import BadRequestError, OpenAI

from kznorms.config import Settings
from kznorms.paths import SYNTHETIC

CACHE = SYNTHETIC / "cache"


class Teacher:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "none",
        cache_dir: Path = CACHE,
        retries: int = 4,
    ) -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=180)
        self.base_url, self.model = base_url, model
        self.cache_dir = cache_dir
        self.retries = retries
        self.json_mode = "json_schema"  # не все провайдеры умеют схему → откат на json_object
        self.calls = self.cached = self.failed = 0

    def _key(self, messages, schema, temperature) -> str:
        payload = [self.base_url, self.model, messages, schema, temperature]
        raw = json.dumps(payload, ensure_ascii=False, sort_keys=True)
        return hashlib.sha256(raw.encode("utf-8")).hexdigest()

    def _request(self, messages, schema, temperature, max_tokens) -> str:
        if self.json_mode == "json_schema":
            fmt = {"type": "json_schema", "json_schema": {"name": "out", "schema": schema}}
        else:
            fmt = {"type": "json_object"}
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            response_format=fmt,
        )
        return resp.choices[0].message.content or ""

    def chat_json(
        self, messages: list[dict], schema: dict, temperature: float = 0.0, max_tokens: int = 1024
    ) -> str | None:
        """JSON-ответ учителя или None, если запрос так и не удался."""
        path = self.cache_dir / f"{self._key(messages, schema, temperature)}.json"
        if path.exists():
            self.cached += 1
            return json.loads(path.read_text(encoding="utf-8"))["content"]
        for attempt in range(self.retries):
            try:
                content = self._request(messages, schema, temperature, max_tokens)
                break
            except BadRequestError:
                if self.json_mode == "json_schema":
                    self.json_mode = "json_object"
                    continue
                self.failed += 1
                return None
            except Exception:  # сеть, 429, 5xx — подождать и повторить
                time.sleep(2**attempt)
        else:
            self.failed += 1
            return None
        self.calls += 1
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        record = {"content": content, "model": self.model}
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return content


def get_teacher(name: str) -> Teacher:
    s = Settings()
    if name == "big":
        if not (s.teacher_base_url and s.teacher_api_key and s.teacher_model):
            raise SystemExit(
                "Учитель не настроен: скопируй .env.example в .env и заполни "
                "TEACHER_BASE_URL, TEACHER_API_KEY, TEACHER_MODEL"
            )
        return Teacher(s.teacher_base_url, s.teacher_model, s.teacher_api_key)
    if name == "base":  # пилот: проверить конвейер на локальной модели, данные не для обучения
        return Teacher(s.llm_base_url, "local")
    raise ValueError(f"неизвестный учитель {name!r}: big | base")
