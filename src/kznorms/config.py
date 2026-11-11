"""Загрузка YAML-конфигов и переменных окружения."""

from pathlib import Path
from typing import Any

import yaml
from pydantic_settings import BaseSettings, SettingsConfigDict

from kznorms.paths import CONFIGS, ROOT


class Settings(BaseSettings):
    """Секреты и адреса сервисов из .env."""

    model_config = SettingsConfigDict(env_file=ROOT / ".env", extra="ignore")

    teacher_base_url: str = ""
    teacher_api_key: str = ""
    teacher_model: str = ""
    llm_base_url: str = "http://localhost:8080/v1"
    qdrant_url: str = "http://localhost:6333"


def load_yaml(name: str) -> dict[str, Any]:
    path = Path(name) if name.endswith(".yaml") else CONFIGS / f"{name}.yaml"
    with open(path, encoding="utf-8") as f:
        return yaml.safe_load(f)
