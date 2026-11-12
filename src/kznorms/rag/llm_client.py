"""OpenAI-совместимый клиент: base / ft (llama-server) и big (API).

llama-server, vLLM и облачные API говорят на одном протоколе (/v1/chat/completions),
поэтому клиент один — модели различаются адресом и именем.
"""

import time

from openai import OpenAI

from kznorms.config import Settings


class LLM:
    def __init__(self, base_url: str, model: str = "local", api_key: str = "none") -> None:
        self.client = OpenAI(base_url=base_url, api_key=api_key, timeout=120)
        self.model = model
        self.last_ms = 0.0

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
        return resp.choices[0].message.content or ""


def get_llm(name: str = "base") -> LLM:
    s = Settings()
    if name == "base":
        return LLM(s.llm_base_url)
    if name == "big":
        return LLM(s.teacher_base_url, s.teacher_model, s.teacher_api_key)
    raise ValueError(f"неизвестная модель {name!r}: base | big")  # ft добавим после обучения
