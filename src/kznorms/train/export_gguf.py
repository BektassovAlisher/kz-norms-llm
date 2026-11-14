"""LoRA-адаптер → GGUF для llama-server (`--lora`), без слияния с моделью.

Почему не сливать адаптер в модель:
- база в проде — та же Q4_K_M, что у базовой модели: сравнение base/ft идёт при одинаковой
  квантизации, разница — только адаптер;
- для абляций хватает одной базы и нескольких адаптеров по ~66 МБ вместо полных моделей
  по 2,5 ГБ, и не нужно качать полные bf16-веса (8 ГБ) ради слияния.

Конвертер — `convert_lora_to_gguf.py` из llama.cpp той же сборки, что llama-server
(формат GGUF развивается; конвертер из другого релиза может записать метаданные, которых
сервер не понимает). От базы конвертеру нужен только config.json — имена и формы тензоров.
Только Linux/WSL2: нужен train-набор зависимостей (torch, transformers).
"""

import shutil
import subprocess
from pathlib import Path
from typing import Any

from kznorms.paths import ROOT


def lora_gguf_path(run: str, cfg: dict[str, Any]) -> Path:
    return ROOT / cfg["lora_gguf_dir"] / f"{run}-f16.gguf"


def export_lora(run: str, cfg: dict[str, Any], copy_to: Path | None = None) -> Path:
    adapter = ROOT / cfg["output_dir"] / run / "adapter"
    if not (adapter / "adapter_config.json").exists():
        raise SystemExit(f"нет адаптера {adapter}: сначала kznorms train-llm --run-name {run}")
    llama_cpp = Path(cfg["llama_cpp_dir"]).expanduser()
    script = llama_cpp / "convert_lora_to_gguf.py"
    if not script.exists():
        raise SystemExit(f"нет {script}: склонируй скрипты llama.cpp (см. HANDOFF, шаг 8)")
    uv = shutil.which("uv") or str(Path("~/.local/bin/uv").expanduser())
    out = lora_gguf_path(run, cfg)
    out.parent.mkdir(parents=True, exist_ok=True)
    cmd = [
        uv, "run", "--with", str(llama_cpp / "gguf-py"), "python", str(script), str(adapter),
        "--base-model-id", cfg["base_model_id"], "--outtype", "f16", "--outfile", str(out),
    ]  # fmt: skip
    subprocess.run(cmd, check=True, cwd=ROOT)
    if copy_to:
        copy_to.mkdir(parents=True, exist_ok=True)
        shutil.copy2(out, copy_to / out.name)
    return out
