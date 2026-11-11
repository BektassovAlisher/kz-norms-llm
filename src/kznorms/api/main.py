"""FastAPI-сервис."""

from fastapi import FastAPI

app = FastAPI(title="kz-norms-llm")


@app.get("/health")
def health() -> dict[str, str]:
    return {"status": "ok"}
