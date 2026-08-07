# Установка окружения (Windows). Запуск в PowerShell:
#   powershell -ExecutionPolicy Bypass -File .\setup.ps1
$ErrorActionPreference = "Stop"
Set-Location $PSScriptRoot

if (-not (Get-Command python -ErrorAction SilentlyContinue)) {
    Write-Error "Не найден python. Установите Python 3.11+ (с галочкой 'Add to PATH') и повторите."
}

Write-Host "==> Виртуальное окружение .venv"
python -m venv .venv
& .\.venv\Scripts\python.exe -m pip install --upgrade pip | Out-Null

Write-Host "==> Зависимости из requirements-windows.txt"
& .\.venv\Scripts\pip.exe install -r requirements-windows.txt

Write-Host ""
Write-Host "Проверка окружения:"
try {
    Invoke-WebRequest -Uri "http://127.0.0.1:11434/api/version" -UseBasicParsing -TimeoutSec 3 | Out-Null
    Write-Host "  OK  Ollama запущена"
} catch {
    Write-Host "  !   Ollama не отвечает — установите с https://ollama.com, запустите, затем: ollama pull qwen3:8b"
}
$cable = Get-CimInstance Win32_SoundDevice -ErrorAction SilentlyContinue |
         Where-Object { $_.Name -match "CABLE|VB-Audio" }
if (-not $cable) {
    Write-Host "  !   VB-Cable не найден — см. раздел «Виртуальное аудиоустройство» в README.md"
}

Write-Host ""
Write-Host "Готово. Запуск:"
Write-Host "  .\.venv\Scripts\Activate.ps1"
Write-Host "  python -m app.main"
