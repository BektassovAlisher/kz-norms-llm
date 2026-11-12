"""Клиент к модели-учителю с кэшем ответов на диск.

Кэш — главное в синтетике: тысячи платных запросов, и любой сбой (сеть, лимиты, Ctrl+C)
не должен стоить денег повторно. Ключ — хэш всего запроса (адрес, модель, сообщения, схема,
температура, параметры): изменил промпт — получишь новые ответы, не изменил — возьмутся старые.

Лимиты бесплатных тарифов (Gemini: порядка 10–15 запросов в минуту и ~1500 в день):
- `rpm` — не чаще N запросов в минуту на все потоки вместе, иначе сразу 429;
- 429 — ждём дольше (минутный лимит сбрасывается за минуту) и повторяем;
- много неудач подряд — значит, кончилась дневная квота: останавливаемся (`QuotaExhausted`),
  а не тратим часы на заведомо проваленные запросы. Перезапуск завтра продолжит с кэша.
"""

import hashlib
import json
import threading
import time
from pathlib import Path

from openai import (
    APITimeoutError,
    BadRequestError,
    InternalServerError,
    OpenAI,
    RateLimitError,
)

from kznorms.config import Settings
from kznorms.paths import SYNTHETIC

CACHE = SYNTHETIC / "cache"


class QuotaExhausted(RuntimeError):
    """Учитель перестал отвечать (квота / ключ) — продолжать бессмысленно."""


class RateLimiter:
    """Не чаще rpm вызовов в минуту на все потоки: запросы разносятся на 60/rpm секунд."""

    def __init__(self, rpm: float) -> None:
        self.interval = 60.0 / rpm if rpm > 0 else 0.0
        self.next_at = 0.0
        self.lock = threading.Lock()

    def wait(self) -> None:
        if not self.interval:
            return
        with self.lock:
            now = time.monotonic()
            at = max(now, self.next_at)
            self.next_at = at + self.interval
        time.sleep(max(0.0, at - now))


class Teacher:
    def __init__(
        self,
        base_url: str,
        model: str,
        api_key: str = "none",
        cache_dir: Path = CACHE,
        retries: int = 4,
        rpm: float = 0,
        params: dict | None = None,
        max_consecutive_failures: int = 5,
    ) -> None:
        # max_retries=0: повторами управляем сами — со своим ожиданием и счётчиком неудач
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=90, max_retries=0)
        self.base_url, self.model = base_url, model
        self.cache_dir = cache_dir
        self.retries = retries
        self.limiter = RateLimiter(rpm)
        self.params = params or {}  # например {"reasoning_effort": "low"} для «думающих» моделей
        self.max_consecutive_failures = max_consecutive_failures
        self.json_mode = "json_schema"  # не все провайдеры умеют схему → откат на json_object
        self.calls = self.cached = self.failed = 0
        self.consecutive_failures = 0
        self.last_error = ""
        self.lock = threading.Lock()

    def _key(self, messages, schema, temperature) -> str:
        payload = [self.base_url, self.model, messages, schema, temperature, self.params]
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
            extra_body=self.params or None,  # поля сверх OpenAI-API: llama-server, Gemini
        )
        return resp.choices[0].message.content or ""

    def _fail(self, err: Exception | str) -> None:
        with self.lock:
            self.failed += 1
            self.consecutive_failures += 1
            self.last_error = str(err)[:300]

    def chat_json(
        self, messages: list[dict], schema: dict, temperature: float = 0.0, max_tokens: int = 2048
    ) -> str | None:
        """JSON-ответ учителя или None, если запрос так и не удался."""
        path = self.cache_dir / f"{self._key(messages, schema, temperature)}.json"
        if path.exists():
            self.cached += 1
            return json.loads(path.read_text(encoding="utf-8"))["content"]
        if self.consecutive_failures >= self.max_consecutive_failures:
            raise QuotaExhausted(self.last_error)
        content = None
        for attempt in range(self.retries):
            self.limiter.wait()
            try:
                content = self._request(messages, schema, temperature, max_tokens)
                break
            except BadRequestError as e:
                if self.json_mode == "json_schema":
                    self.json_mode = "json_object"  # схему не приняли — пробуем без неё
                    continue
                self._fail(e)
                return None
            except RateLimitError as e:
                body = getattr(getattr(e, "response", None), "text", "") or str(e)
                if "PerDay" in body:  # дневная квота (Gemini: ...PerDayPerProjectPerModel...)
                    self.last_error = "дневная квота исчерпана: " + body[:200]
                    self.consecutive_failures = self.max_consecutive_failures
                    raise QuotaExhausted(self.last_error) from e
                self.last_error = str(e)[:300]
                time.sleep(15 * (attempt + 1))
            except (InternalServerError, APITimeoutError) as e:
                # 503 «high demand» и таймауты — перегрузка у провайдера: проходит сама
                self.last_error = str(e)[:300]
                time.sleep(15 * (attempt + 1))  # 15, 30, 45 с …
            except Exception as e:  # сеть, 5xx
                self.last_error = str(e)[:300]
                time.sleep(2**attempt)
        if content is None:
            self._fail(self.last_error)
            return None
        with self.lock:
            self.calls += 1
            self.consecutive_failures = 0
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        record = {"content": content, "model": self.model}
        path.write_text(json.dumps(record, ensure_ascii=False), encoding="utf-8")
        return content


def get_teacher(name: str, cfg: dict | None = None) -> Teacher:
    cfg = cfg or {}
    s = Settings()
    if name == "big":
        if not (s.teacher_base_url and s.teacher_api_key and s.teacher_model):
            raise SystemExit(
                "Учитель не настроен: в .env нужны TEACHER_BASE_URL, TEACHER_API_KEY, "
                "TEACHER_MODEL (образец — .env.example). Файл сохранён?"
            )
        return Teacher(
            s.teacher_base_url,
            s.teacher_model,
            s.teacher_api_key,
            rpm=cfg.get("rpm", 0),
            params=cfg.get("teacher_params") or {},
        )
    if name in ("local", "base"):  # то, что сейчас загружено в llama-server
        return Teacher(
            s.llm_base_url,
            served_model(s.llm_base_url),
            params=cfg.get("local_teacher_params") or {},
        )
    raise ValueError(f"неизвестный учитель {name!r}: big | local")


def served_model(base_url: str) -> str:
    """Имя файла модели, загруженной в llama-server.

    Сервер принимает любое `model` в запросе и отвечает той, что загружена. Но кэш различает
    учителей по имени: если всегда слать "local", ответы 4B и 8B перемешаются в одном кэше.
    """
    from pathlib import PurePath

    try:
        data = OpenAI(base_url=base_url, api_key="none", max_retries=0).models.list()
        return PurePath(data.data[0].id).name
    except Exception as e:
        raise SystemExit(f"llama-server на {base_url} не отвечает: {e}") from e
