"""OpenAI-совместимый клиент: base / ft (llama-server) и big (API).

llama-server, vLLM и облачные API говорят на одном протоколе (/v1/chat/completions),
поэтому клиент один — модели различаются адресом и именем.
"""

import time
from pathlib import Path

from openai import OpenAI

from kznorms.config import Settings


class LLM:
    def __init__(self, base_url: str, model: str = "local", api_key: str = "none") -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)
        self.model = model
        self.last_ms = 0.0
        self.last_finish = ""  # "length" — ответ обрезан по max_tokens

    def chat_json(
        self, messages: list[dict[str, str]], schema: dict, max_tokens: int = 1024
    ) -> str:
        """Ответ строго по JSON-схеме: сервер ограничивает генерацию грамматикой."""
        t = time.perf_counter()
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.0,  # воспроизводимость: base и ft сравниваем без шума сэмплирования
            max_tokens=max_tokens,
            response_format={
                "type": "json_schema",
                "json_schema": {"name": "answer", "schema": schema},
            },
        )
        self.last_ms = (time.perf_counter() - t) * 1000
        self.last_finish = resp.choices[0].finish_reason or ""
        return resp.choices[0].message.content or ""


def active_adapters(base_url: str) -> list[str]:
    """Пути LoRA-адаптеров, включённых в llama-server (scale > 0); [] — чистая база.

    Одна база Q4_K_M + адаптер через `--lora`: сравнение base/ft идёт при одинаковой
    квантизации, а для абляций хватает одной базы и разных адаптеров по 66 МБ.
    """
    import httpx

    root = base_url.rstrip("/").removesuffix("/v1")
    try:
        items = httpx.get(f"{root}/lora-adapters", timeout=5).json()
    except Exception:
        return []  # не llama-server или эндпоинта нет — адаптеров нет
    return [a["path"] for a in items if a.get("scale", 0) > 0]


def get_llm(name: str = "base") -> LLM:
    """base — llama-server без адаптера; ft — тот же сервер с адаптером; big — API учителя.

    base и ft сверяются с сервером: «оценка базы» на сервере с подключённым адаптером
    сравнила бы дообученную модель саму с собой — такую ошибку не заметишь по цифрам.
    """
    s = Settings()
    if name in ("base", "ft"):
        adapters = active_adapters(s.llm_base_url)
        if name == "base" and adapters:
            raise SystemExit(f"запрошена база, но в llama-server подключён адаптер: {adapters}")
        if name == "ft" and not adapters:
            raise SystemExit("запрошена ft, но llama-server запущен без --lora <адаптер.gguf>")
        label = "ft:" + Path(adapters[0]).stem if adapters else "local"
        return LLM(s.llm_base_url, label)
    if name == "big":
        return LLM(s.teacher_base_url, s.teacher_model, s.teacher_api_key)
    raise ValueError(f"неизвестная модель {name!r}: base | ft | big")
