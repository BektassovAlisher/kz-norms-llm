# Итерация 2 на ночь: ответы учителя с промптом v4 → обучение full_v2 в WSL → экспорт адаптера.
#
# Вопросы и контексты берутся из кэша (они не зависят от промпта ответа), заново
# генерируются только ответы учителя (~3,5 ч). Обучение ~4 ч. Лог — artifacts/iteration2.log.
# Запуск из корня проекта:
#   powershell -ExecutionPolicy Bypass -File scripts\iteration2.ps1
param([string]$RunName = "full_v2")

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$started = Get-Date

Write-Host "== 1/2 синтетика с промптом v4"
& powershell -ExecutionPolicy Bypass -File scripts\synth_local.ps1
if ($LASTEXITCODE -ne 0) { throw "синтетика завершилась с ошибкой ($LASTEXITCODE)" }

Write-Host "== 2/2 обучение $RunName в WSL и экспорт адаптера"
$win = "/mnt/c/Users/user/Desktop/kz-norms-llm"
$cmd = "cd ~/kz-norms-llm && git pull -q win main && cp $win/data/sft/*.jsonl data/sft/ " +
       "&& PYTHONUNBUFFERED=1 ~/.local/bin/uv run kznorms train-llm --run-name $RunName " +
       "2>&1 | tee artifacts/train_$RunName.log && ~/.local/bin/uv run kznorms export " +
       "--run $RunName --copy-to $win/artifacts/lora"
wsl -d Ubuntu-24.04 -- bash -lc "set -o pipefail; $cmd"
if ($LASTEXITCODE -ne 0) { throw "обучение или экспорт завершились с ошибкой ($LASTEXITCODE)" }

Write-Host ("== итерация 2 готова за {0:hh\:mm\:ss}; адаптер artifacts/lora/$RunName-f16.gguf" -f ((Get-Date) - $started))
