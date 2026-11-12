# Синтетика с локальным учителем (Qwen3-8B в llama-server) — все этапы по очереди.
#
# Учитель (~6 ГБ) и модели поиска (~2 ГБ) вместе не влезают в 8 ГБ видеопамяти, поэтому:
#   сервер вкл → вопросы → сервер выкл → контексты (GPU у поиска) → сервер вкл → ответы.
# Всё кэшируется: прерванный прогон при перезапуске продолжит с места.
#
# Запуск из корня проекта:
#   powershell -ExecutionPolicy Bypass -File scripts\synth_local.ps1              # полный
#   powershell -ExecutionPolicy Bypass -File scripts\synth_local.ps1 -NClauses 40 # пилот
param([int]$NClauses = 0)

$ErrorActionPreference = "Stop"
Set-Location (Split-Path -Parent $PSScriptRoot)
$env:PYTHONIOENCODING = "utf-8"

$model = "artifacts/models/Qwen3-8B-Q4_K_M.gguf"
# 3 слота по 4096 токенов, KV-кэш в 8 битах: модель + кэш ≈ 6,4 ГБ
$serverArgs = "-m $model -c 12288 -np 3 -ngl 99 -fa on -ctk q8_0 -ctv q8_0 " +
              "--port 8085 --host 127.0.0.1 --jinja"
$n = @()
if ($NClauses -gt 0) { $n = @("--n-clauses", "$NClauses") }

function Start-Teacher {
    Write-Host "== llama-server: $model"
    $p = Start-Process -FilePath "artifacts/llama/llama-server.exe" -ArgumentList $serverArgs `
        -RedirectStandardError "artifacts/llama/server8b.log" `
        -RedirectStandardOutput "artifacts/llama/server8b.out.log" -WindowStyle Hidden -PassThru
    for ($i = 0; $i -lt 180; $i++) {
        try {
            $r = Invoke-WebRequest -UseBasicParsing "http://127.0.0.1:8085/health" -TimeoutSec 2
            if ($r.StatusCode -eq 200) { return $p }
        } catch { }
        if ($p.HasExited) { throw "llama-server упал при запуске: см. artifacts/llama/server8b.log" }
        Start-Sleep -Seconds 1
    }
    throw "llama-server не поднялся за 3 минуты"
}

function Stop-Teacher($p) {
    if ($p -and -not $p.HasExited) { Stop-Process -Id $p.Id -Force }
    Start-Sleep -Seconds 3  # дать драйверу освободить видеопамять
}

function Invoke-Stage($stage) {
    Write-Host "== этап $stage"
    uv run kznorms synth @n --stage $stage
    if ($LASTEXITCODE -ne 0) { throw "этап $stage завершился с ошибкой ($LASTEXITCODE)" }
}

$started = Get-Date
$t = Start-Teacher
try { Invoke-Stage "questions" } finally { Stop-Teacher $t }
Invoke-Stage "contexts"
$t = Start-Teacher
try { Invoke-Stage "answers" } finally { Stop-Teacher $t }
Write-Host ("== готово за {0:hh\:mm\:ss}" -f ((Get-Date) - $started))
