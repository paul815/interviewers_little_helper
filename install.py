"""Установка окружения — один скрипт на все платформы.

    python3 install.py        # macOS / Linux
    python install.py         # Windows

Создаёт `.venv`, ставит зависимости под вашу ОС, скачивает веса ASR и
проверяет то, что чаще всего забывают: запущенную Ollama и виртуальный
аудиокабель.

Запускается системным Python (3.11+) до появления venv, поэтому обходится
стандартной библиотекой. Всё остальное — уже интерпретатором из venv.

Файл называется install.py, а не setup.py: setup.py в корне репозитория —
сигнал setuptools, из-за которого `pip install .` попытался бы собрать проект
как пакет.
"""
from __future__ import annotations

import argparse
import json
import platform
import subprocess
import sys
import urllib.error
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parent
MIN_PYTHON = (3, 11)
OLLAMA_URL = "http://127.0.0.1:11434/api/version"

IS_WINDOWS = sys.platform == "win32"
IS_MACOS = sys.platform == "darwin"


def fail(message: str) -> None:
    print(f"\nОшибка: {message}", file=sys.stderr)
    raise SystemExit(1)


def requirements_file() -> str:
    if IS_WINDOWS:
        return "requirements-windows.txt"
    if IS_MACOS:
        return "requirements-macos.txt"
    # Linux: без mlx; Parakeet поедет через onnx-asr, Whisper — через faster-whisper на CPU.
    return "requirements-common.txt"


def venv_python(venv: Path) -> Path:
    return venv / ("Scripts/python.exe" if IS_WINDOWS else "bin/python")


def run(cmd: list[str], *, what: str, allow_failure: bool = False) -> bool:
    """Прогнать команду, показывая её вывод как есть."""
    result = subprocess.run(cmd, cwd=ROOT)
    if result.returncode == 0:
        return True
    if allow_failure:
        return False
    fail(f"{what} завершилось с кодом {result.returncode}")
    return False


def create_venv(venv: Path) -> Path:
    python = venv_python(venv)
    if python.exists():
        print(f"==> Виртуальное окружение {venv.name} уже есть — переиспользую")
    else:
        print(f"==> Виртуальное окружение {venv.name}")
        run([sys.executable, "-m", "venv", str(venv)], what="создание venv")
        if not python.exists():
            fail(f"venv создан, но {python} не появился")
    subprocess.run(
        [str(python), "-m", "pip", "install", "--upgrade", "pip"],
        cwd=ROOT, stdout=subprocess.DEVNULL,
    )
    return python


def install_requirements(python: Path, req: str) -> None:
    print(f"\n==> Зависимости из {req}")
    run([str(python), "-m", "pip", "install", "-r", req], what=f"установка {req}")


def fetch_model(python: Path) -> None:
    print("\n==> Веса ASR (Parakeet, ~670 МБ) — чтобы первое интервью не ждало загрузку")
    ok = run([str(python), "-m", "tools.fetch_asr_model"],
             what="загрузка весов", allow_failure=True)
    if not ok:
        print("  !  Не удалось скачать. Приложение доскачает при первом старте сессии — "
              "или повторите вручную: python -m tools.fetch_asr_model")


def check_ollama() -> None:
    # Явно без прокси: адрес локальный, а системный прокси только помешает.
    opener = urllib.request.build_opener(urllib.request.ProxyHandler({}))
    try:
        with opener.open(OLLAMA_URL, timeout=3) as response:
            version = json.loads(response.read()).get("version", "")
        print(f"  ok Ollama запущена{f' (версия {version})' if version else ''}")
    except (urllib.error.URLError, OSError, ValueError):
        print("  !  Ollama не отвечает. Установите с https://ollama.com, запустите, "
              "затем: ollama pull qwen3:8b")


def check_audio_cable() -> None:
    """Виртуальный кабель — самая частая причина «в канале респондента тишина»."""
    if IS_MACOS:
        found = _command_output(["system_profiler", "SPAudioDataType"], "blackhole")
        name, hint = "BlackHole", "brew install blackhole-2ch"
    elif IS_WINDOWS:
        found = _command_output(
            ["powershell", "-NoProfile", "-Command",
             "Get-CimInstance Win32_SoundDevice | Select-Object -ExpandProperty Name"],
            "cable",
        )
        name, hint = "VB-Cable", "https://vb-audio.com/Cable/"
    else:
        return  # Linux: разделение каналов настраивается через PulseAudio/PipeWire вручную
    if found:
        print(f"  ok {name} найден")
    else:
        print(f"  !  {name} не найден ({hint}) — см. «Виртуальное аудиоустройство» в README.md")


def _command_output(cmd: list[str], needle: str) -> bool:
    try:
        out = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
    except (OSError, subprocess.SubprocessError):
        return False
    return needle in (out.stdout or "").lower()


def print_next_steps(venv: Path) -> None:
    activate = (
        rf"{venv.name}\Scripts\Activate.ps1" if IS_WINDOWS else f"source {venv.name}/bin/activate"
    )
    print("\nГотово. Запуск:")
    print(f"  {activate}")
    print("  python -m app.main")
    if IS_MACOS and platform.machine() == "arm64":
        print("\nПодсказка для Apple Silicon: Parakeet идёт через onnx-asr на CPU. "
              "Если хочется быстрее, попробуйте CoreML — в config.json:")
        print('  "asr": { "providers": ["CoreMLExecutionProvider", "CPUExecutionProvider"] }')
        print("  и сравните: python -m tools.bench_asr <запись.wav>")


def main() -> None:
    parser = argparse.ArgumentParser(description="Установка окружения проекта")
    parser.add_argument("--venv", default=".venv", help="каталог venv (по умолчанию .venv)")
    parser.add_argument("--no-model", action="store_true",
                        help="не скачивать веса ASR (докачаются при первом старте)")
    parser.add_argument("--requirements", help="файл зависимостей вместо выбранного по платформе")
    args = parser.parse_args()

    if sys.version_info < MIN_PYTHON:
        fail(f"нужен Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+, а запущен "
             f"{sys.version_info.major}.{sys.version_info.minor}")

    print(f"Платформа: {platform.system()} {platform.machine()}, Python {platform.python_version()}")
    venv = (ROOT / args.venv).resolve()
    python = create_venv(venv)
    install_requirements(python, args.requirements or requirements_file())
    if not args.no_model:
        fetch_model(python)

    print("\nПроверка окружения:")
    check_ollama()
    check_audio_cable()
    print_next_steps(venv)


if __name__ == "__main__":
    main()
