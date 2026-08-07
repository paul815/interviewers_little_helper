#!/usr/bin/env bash
# Установка окружения (macOS / Linux). Запуск: ./setup.sh
set -euo pipefail
cd "$(dirname "$0")"

PY="${PYTHON:-python3}"
if ! command -v "$PY" >/dev/null 2>&1; then
  echo "Не найден python3. Установите Python 3.11+ и повторите." >&2
  exit 1
fi

echo "==> Виртуальное окружение .venv"
"$PY" -m venv .venv
# shellcheck disable=SC1091
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null

if [ "$(uname -s)" = "Darwin" ]; then
  REQ=requirements-macos.txt
else
  REQ=requirements-common.txt   # Linux: без mlx, ASR пойдёт через faster-whisper
fi
echo "==> Зависимости из $REQ"
pip install -r "$REQ"

echo
echo "Проверка окружения:"
if command -v ollama >/dev/null 2>&1; then
  if curl -sf http://127.0.0.1:11434/api/version >/dev/null 2>&1; then
    echo "  ✓ Ollama запущена"
  else
    echo "  ! Ollama установлена, но не запущена — запустите приложение Ollama"
  fi
else
  echo "  ! Ollama не найдена — установите с https://ollama.com, затем: ollama pull qwen3:8b"
fi
if [ "$(uname -s)" = "Darwin" ] && ! system_profiler SPAudioDataType 2>/dev/null | grep -qi blackhole; then
  echo "  ! BlackHole не найден — см. раздел «Виртуальное аудиоустройство» в README.md"
fi

echo
echo "Готово. Запуск:"
echo "  source .venv/bin/activate"
echo "  python -m app.main"
