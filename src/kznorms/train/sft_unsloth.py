"""QLoRA Qwen3-4B через Unsloth + TRL (train_on_responses_only).

Что важно и почему:
- QLoRA: база в 4 битах (NF4) заморожена, учатся только адаптеры LoRA во всех линейных
  слоях — 4B-модель помещается в 8 ГБ, а адаптер весит десятки мегабайт.
- Текст примера = ровно то, что модель видит и пишет в проде: промпт для генерации
  (`apply_chat_template(..., add_generation_prompt=True)` — сверено с `/apply-template`
  llama-server, совпадает символ в символ) + ответ + `<|im_end|>`. Полный шаблон диалога
  НЕ годится: шаблон Qwen3 вставляет перед ответом пустой `<think>

</think>

`, а
  в проде модель сразу пишет JSON (грамматика llama-server). Обучи на think — получишь
  train/serve skew на уровне спецтокенов (поймано `check_mask` на пробном запуске).
- Loss только по ответу ассистента: системный промпт и контекст — ~95% токенов примера;
  учить их — значит тратить ёмкость на запоминание норм и промпта, а не навыка ответа.
  Метки всего, кроме ответа, = −100 (игнорируются в loss). `check_mask` проверяет, что
  маска не съехала: незамаскированный текст обязан начинаться с ответа-JSON.
- Подвыборки для абляций вложенные: данные перемешаны один раз с фиксированным зерном,
  500/1k/2k/4k — первые N. Меньшая всегда входит в большую — кривая обучения честная.

Только Linux/WSL2: Unsloth опирается на Triton, которого под Windows нет.
"""

import json
import random
import time
from pathlib import Path
from typing import Any

from kznorms.paths import ROOT, SFT

INSTRUCTION_PART = "<|im_start|>user\n"  # шаблон чата Qwen (ChatML)
RESPONSE_PART = "<|im_start|>assistant\n"
IGNORE = -100


def load_sft(path: Path, n: int | None = None, seed: int = 42) -> list[dict]:
    """Строки SFT-набора: перемешать с зерном и взять первые n (вложенные подвыборки)."""
    rows = [json.loads(x) for x in path.read_text(encoding="utf-8").splitlines() if x.strip()]
    random.Random(seed).shuffle(rows)
    return rows[:n] if n else rows


def build_text(messages: list[dict], tokenizer) -> str:
    """Промпт прода (system+user, «ассистент, начинай») + ответ + конец реплики."""
    prompt = tokenizer.apply_chat_template(
        messages[:-1], tokenize=False, add_generation_prompt=True
    )
    return prompt + messages[-1]["content"] + tokenizer.eos_token


def response_text(input_ids: list[int], labels: list[int], tokenizer) -> str:
    """Текст, по которому считается loss (метки ≠ −100)."""
    return tokenizer.decode([t for t, lab in zip(input_ids, labels, strict=True) if lab != IGNORE])


def check_mask(dataset, tokenizer, n: int = 3) -> float:
    """Проверить маску на первых примерах; вернуть долю токенов, идущих в loss.

    Если маска съехала (например, шаблон добавил что-то перед ответом), обучение молча
    пойдёт не на том тексте — поэтому падаем сразу, а не через час.
    """
    shares = []
    for row in dataset.select(range(min(n, len(dataset)))):
        text = response_text(row["input_ids"], row["labels"], tokenizer).strip()
        if not text.startswith('{"status"'):
            raise RuntimeError(f"маска loss съехала: обучаемый текст начинается с {text[:80]!r}")
        shares.append(sum(lab != IGNORE for lab in row["labels"]) / len(row["labels"]))
    return sum(shares) / len(shares)


def train(
    cfg: dict[str, Any],
    n_train: int | None = None,
    max_steps: int | None = None,
    run_name: str | None = None,
) -> dict[str, Any]:
    # unsloth — первым импортом: он патчит transformers/trl под свои быстрые ядра
    # (поэтому сортировку импортов здесь отключаем)
    from unsloth import FastLanguageModel  # noqa: I001
    from unsloth.chat_templates import train_on_responses_only

    import torch
    from datasets import Dataset
    from trl import SFTConfig, SFTTrainer

    seed, tcfg, max_len = cfg["seed"], cfg["train"], cfg["max_seq_length"]
    run_name = run_name or f"n{n_train or 'all'}_r{cfg['lora']['r']}"
    out = ROOT / cfg["output_dir"] / run_name
    started = time.time()

    model, tok = FastLanguageModel.from_pretrained(
        model_name=cfg["model"], max_seq_length=max_len, load_in_4bit=cfg["load_in_4bit"]
    )
    model = FastLanguageModel.get_peft_model(
        model,
        r=cfg["lora"]["r"],
        lora_alpha=cfg["lora"]["alpha"],
        lora_dropout=cfg["lora"]["dropout"],
        target_modules=cfg["lora"]["target_modules"],
        bias="none",
        use_gradient_checkpointing="unsloth",  # активации — в RAM: длинные примеры влезают
        random_state=seed,
    )

    def to_dataset(rows: list[dict]) -> tuple[Dataset, int]:
        texts = [build_text(r["messages"], tok) for r in rows]
        fits = [t for t in texts if len(tok(t, add_special_tokens=False).input_ids) <= max_len]
        return Dataset.from_dict({"text": fits}), len(texts) - len(fits)

    train_ds, cut_train = to_dataset(load_sft(SFT / "train.jsonl", n_train, seed))
    val_ds, cut_val = to_dataset(load_sft(SFT / "val.jsonl", cfg["val_examples"], seed))

    steps_hint = max_steps or 0
    eval_steps = min(tcfg["eval_steps"], steps_hint) if steps_hint else tcfg["eval_steps"]
    args = SFTConfig(
        output_dir=str(out / "checkpoints"),
        dataset_text_field="text",
        max_length=max_len,
        packing=False,  # с packing маска «только ответ» сложнее; примеры и так длинные
        per_device_train_batch_size=tcfg["batch_size"],
        per_device_eval_batch_size=tcfg["batch_size"],
        gradient_accumulation_steps=tcfg["grad_accum"],
        num_train_epochs=tcfg["epochs"],
        max_steps=max_steps or -1,
        learning_rate=tcfg["lr"],
        lr_scheduler_type=tcfg["scheduler"],
        warmup_steps=tcfg["warmup_ratio"],  # transformers 5: доля < 1 — это доля шагов
        weight_decay=tcfg["weight_decay"],
        optim=tcfg["optim"],
        bf16=tcfg["bf16"],
        logging_steps=tcfg["logging_steps"],
        eval_strategy="steps",
        eval_steps=eval_steps,
        save_strategy="no",  # сохраняем один адаптер в конце — чекпойнты съедят диск
        report_to="none",
        seed=seed,
    )
    trainer = SFTTrainer(
        model=model, processing_class=tok, train_dataset=train_ds, eval_dataset=val_ds, args=args
    )
    if cfg["train_on_responses_only"]:
        trainer = train_on_responses_only(
            trainer, instruction_part=INSTRUCTION_PART, response_part=RESPONSE_PART
        )
    loss_share = check_mask(trainer.train_dataset, tok)

    torch.cuda.reset_peak_memory_stats()
    result = trainer.train()
    final_eval = trainer.evaluate()

    out.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out / "adapter")
    tok.save_pretrained(out / "adapter")
    history = [h for h in trainer.state.log_history if "loss" in h or "eval_loss" in h]
    (out / "log_history.json").write_text(json.dumps(history, indent=1), encoding="utf-8")
    summary = {
        "run": run_name,
        "n_train": len(train_ds),
        "n_val": len(val_ds),
        "cut_too_long": {"train": cut_train, "val": cut_val},
        "loss_token_share": round(loss_share, 3),
        "steps": trainer.state.global_step,
        "train_loss": round(result.training_loss, 4),
        "eval_loss": round(final_eval["eval_loss"], 4),
        "peak_vram_gb": round(torch.cuda.max_memory_reserved() / 2**30, 2),
        "minutes": round((time.time() - started) / 60, 1),
        "config": cfg,
    }
    (out / "summary.json").write_text(
        json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8"
    )
    return summary
