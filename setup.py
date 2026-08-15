#!/usr/bin/env python3
"""Environment setup — one script for every platform.

    python  setup.py     # Windows
    python3 setup.py     # macOS / Linux

Creates .venv, installs the dependencies for the current OS, downloads the ASR
weights up front, and brings in Ollama with a language model plus a virtual audio
cable. This is not a setuptools script: no package is built from it and
`pip install .` is not intended.

Everything that can be installed silently, the script installs itself (after
asking). Of the external components, only VB-Cable on Windows is left to be done
by hand: it is a driver, it needs administrator rights and a reboot, and the
vendor offers no silent install.
"""
from __future__ import annotations

import argparse
import json
import os
import platform
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
import webbrowser
from pathlib import Path
from typing import NoReturn

ROOT = Path(__file__).resolve().parent
VENV = ROOT / ".venv"
MIN_PYTHON = (3, 11)
OLLAMA_BASE = "http://127.0.0.1:11434"

# What the virtual audio cable is called on each platform: (substrings in the
# device name, human-readable name, how to install it).
CABLE = {
    "win32": (("cable", "vb-audio"), "VB-Cable", "https://vb-audio.com/Cable/"),
    "darwin": (("blackhole",), "BlackHole", "brew install blackhole-2ch"),
}

# Where the installers put ollama. Needed because right after installation our
# process still has the old PATH: which() will not see it, though the file is there.
OLLAMA_PATHS = {
    "win32": (
        Path(os.environ.get("LOCALAPPDATA", "")) / "Programs" / "Ollama" / "ollama.exe",
        Path(r"C:\Program Files\Ollama\ollama.exe"),
    ),
    "darwin": (
        Path("/opt/homebrew/bin/ollama"),
        Path("/usr/local/bin/ollama"),
        Path("/Applications/Ollama.app/Contents/Resources/ollama"),
    ),
}


def fix_stdio() -> None:
    """So that non-ASCII text does not take the script down with UnicodeEncodeError.

    The output below contains em dashes and arrows. When it goes to a pipe or a
    log, Python picks an ANSI encoding (cp1252 in western locales) and the first
    such character raises. In that case we switch the stream to UTF-8.
    """
    for stream in (sys.stdout, sys.stderr):
        if not hasattr(stream, "reconfigure"):
            continue
        try:
            "— probe →".encode(stream.encoding or "ascii")
        except (LookupError, UnicodeEncodeError):
            stream.reconfigure(encoding="utf-8", errors="replace")
        else:
            stream.reconfigure(errors="replace")


def say(msg: str = "") -> None:
    print(msg, flush=True)


def step(msg: str) -> None:
    say(f"\n==> {msg}")


def ok(msg: str) -> None:
    say(f"  [OK] {msg}")


def warn(msg: str) -> None:
    say(f"  [!]  {msg}")


def die(msg: str) -> NoReturn:
    say(f"\nError: {msg}")
    raise SystemExit(1)


def ask(question: str, assume_yes: bool = False) -> bool:
    """Ask yes/no. With --yes it does not ask; without a terminal (CI) it answers no.

    Installing something into the system is not a thing to do silently, so by
    default every external component is confirmed separately.
    """
    if assume_yes:
        say(f"  {question} — yes (--yes)")
        return True
    if not sys.stdin or not sys.stdin.isatty():
        return False
    try:
        return input(f"  {question} [y/N]: ").strip().lower() in {"y", "yes"}
    except EOFError:
        return False


def venv_python() -> Path:
    """The interpreter inside .venv (Scripts on Windows, bin elsewhere)."""
    return VENV / ("Scripts/python.exe" if os.name == "nt" else "bin/python")


def requirements_file() -> Path:
    if sys.platform == "win32":
        return ROOT / "requirements-windows.txt"
    if sys.platform == "darwin":
        return ROOT / "requirements-macos.txt"
    # Linux: no mlx and none of the CUDA packages from the Windows set
    return ROOT / "requirements-common.txt"


def run(cmd: list[str], **kw) -> subprocess.CompletedProcess:
    return subprocess.run(cmd, cwd=ROOT, **kw)


def run_in_venv(args: list[str], capture: bool = False) -> subprocess.CompletedProcess:
    """Running a module or a snippet with the interpreter from .venv.

    PYTHONIOENCODING is needed on Windows: audio device names and ASR logs do not
    fit the console encoding and without it crash with UnicodeEncodeError.
    """
    env = {**os.environ, "PYTHONIOENCODING": "utf-8"}
    return run(
        [str(venv_python()), *args],
        env=env,
        text=True,
        encoding="utf-8",
        errors="replace",
        capture_output=capture,
    )


# --- installation steps ---------------------------------------------------


def check_python() -> None:
    if sys.version_info < MIN_PYTHON:
        die(
            f"Python {MIN_PYTHON[0]}.{MIN_PYTHON[1]}+ is required, "
            f"but this one is {platform.python_version()} ({sys.executable})"
        )


def create_venv(recreate: bool) -> None:
    if recreate and VENV.exists():
        step("Removing the previous .venv")
        shutil.rmtree(VENV)
    step(f"Virtual environment {VENV.name}")
    if venv_python().exists() and not recreate:
        ok("already there — reusing it (recreate with --recreate)")
    else:
        if run([sys.executable, "-m", "venv", str(VENV)]).returncode != 0:
            die("could not create .venv")
    if not venv_python().exists():
        die(f".venv has no interpreter: {venv_python()}")
    run_in_venv(["-m", "pip", "install", "--upgrade", "pip", "--quiet"])


def install_requirements(req: Path) -> None:
    step(f"Dependencies from {req.name}")
    if not req.exists():
        die(f"file not found: {req}")
    if run_in_venv(["-m", "pip", "install", "-r", str(req)]).returncode != 0:
        die(f"pip install -r {req.name} failed — see the output above")


def fetch_asr_model() -> None:
    step("ASR weights (Parakeet, ~670 MB) — so the first interview does not wait on a download")
    if run_in_venv(["-m", "tools.fetch_asr_model"]).returncode != 0:
        warn("could not download them — the application will fetch them on first start")


# --- environment checks ---------------------------------------------------


def llm_model_name() -> str:
    """The model from config.json, otherwise from config.example.json, otherwise the default."""
    for name in ("config.json", "config.example.json"):
        try:
            cfg = json.loads((ROOT / name).read_text(encoding="utf-8"))
            model = cfg.get("llm", {}).get("model")
            if model:
                return str(model)
        except (OSError, ValueError):
            continue
    return "qwen3:8b"


def http_json(url: str, timeout: float = 3.0):
    with urllib.request.urlopen(url, timeout=timeout) as resp:  # noqa: S310 — localhost only
        return json.loads(resp.read().decode("utf-8"))


def ollama_exe() -> str | None:
    found = shutil.which("ollama")
    if found:
        return found
    for path in OLLAMA_PATHS.get(sys.platform, ()):
        if path.exists():
            return str(path)
    return None


def ollama_alive(timeout: float = 2.0) -> bool:
    try:
        http_json(f"{OLLAMA_BASE}/api/version", timeout)
        return True
    except (urllib.error.URLError, OSError, ValueError):
        return False


def install_ollama(assume_yes: bool) -> str | None:
    """Install Ollama with the platform's own package manager."""
    if sys.platform == "win32":
        winget = shutil.which("winget")
        if not winget:
            warn("no winget — install Ollama by hand from https://ollama.com")
            return None
        manager = "winget"
        cmd = [winget, "install", "--id", "Ollama.Ollama", "--exact", "--silent",
               "--accept-package-agreements", "--accept-source-agreements"]
    elif sys.platform == "darwin":
        brew = shutil.which("brew")
        if not brew:
            warn("no Homebrew — install Ollama by hand from https://ollama.com")
            return None
        manager = "Homebrew"
        cmd = [brew, "install", "--cask", "ollama"]
    else:
        # The official Linux installer is a script off the network; we will not
        # run it silently on the user's behalf — let them decide.
        warn("install Ollama: curl -fsSL https://ollama.com/install.sh | sh")
        return None

    if not ask(f"Ollama was not found. Install it via {manager}?", assume_yes):
        return None
    if run(cmd).returncode != 0:
        warn("the installation failed — install it by hand from https://ollama.com")
        return None
    return ollama_exe()


def start_ollama(exe: str, timeout: float = 40.0) -> bool:
    """Bring the server up in the background and wait until the API answers."""
    kw: dict = {"stdout": subprocess.DEVNULL, "stderr": subprocess.DEVNULL}
    if os.name == "nt":
        # So the server survives setup.py exiting and does not flash a console window.
        kw["creationflags"] = subprocess.CREATE_NO_WINDOW | subprocess.DETACHED_PROCESS
    else:
        kw["start_new_session"] = True
    try:
        subprocess.Popen([exe, "serve"], cwd=ROOT, **kw)
    except OSError:
        return False
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        if ollama_alive():
            return True
        time.sleep(0.5)
    return False


def has_model(model: str) -> bool:
    try:
        tags = http_json(f"{OLLAMA_BASE}/api/tags").get("models", [])
    except (urllib.error.URLError, OSError, ValueError):
        return False
    names = {str(m.get("name", "")) for m in tags}
    return any(n == model or n.startswith(f"{model}-") for n in names)


def ensure_ollama(assume_yes: bool) -> None:
    model = llm_model_name()
    step(f"Ollama and the model {model}")

    exe = ollama_exe()
    if exe:
        ok(f"Ollama is in place: {exe}")
    elif ollama_alive():
        ok("Ollama answers (started elsewhere)")  # in Docker, say, or on another machine
    else:
        exe = install_ollama(assume_yes)
        if not exe:
            warn(f"without Ollama there will be no hints; later: ollama pull {model}")
            return
        ok("Ollama installed")

    if not ollama_alive():
        say("  bringing the Ollama server up…")
        if not exe or not start_ollama(exe):
            warn("the server did not come up — start the Ollama application "
                 f"and: ollama pull {model}")
            return
    ok(f"the server answers at {OLLAMA_BASE}")

    if has_model(model):
        ok(f"the model {model} is already downloaded")
        return
    if not ask(f"Download the model {model} (a few GB)?", assume_yes):
        warn(f"skipping — later: ollama pull {model}")
        return
    if not exe or run([exe, "pull", model]).returncode != 0:
        warn(f"could not download it — run by hand: ollama pull {model}")
    else:
        ok(f"the model {model} is ready")


def input_devices() -> str | None:
    """The recording device names as one lower-case string (None if it did not work)."""
    probe = (
        "import sounddevice as sd\n"
        "print('\\n'.join(d['name'] for d in sd.query_devices()"
        " if d['max_input_channels'] > 0))\n"
    )
    res = run_in_venv(["-c", probe], capture=True)
    return res.stdout.lower() if res.returncode == 0 else None


def loopback_ready() -> bool:
    """Windows can offer system audio for recording without any cable at all."""
    probe = (
        "from app.audio import loopback\n"
        "print(len(loopback.list_loopback_devices()) if loopback.available() else 0)\n"
    )
    res = run_in_venv(["-c", probe], capture=True)
    if res.returncode != 0:
        return False
    return res.stdout.strip().isdigit() and int(res.stdout.strip()) > 0


def ensure_audio_cable(assume_yes: bool) -> None:
    patterns, human, howto = CABLE.get(sys.platform, ((), "", ""))
    if not patterns:
        return  # Linux: the cable is made with PulseAudio/PipeWire, nothing to detect
    step("System audio (the respondent channel)")

    if sys.platform == "win32" and loopback_ready():
        ok("taken from the output device via WASAPI loopback — no cable needed")
        say("   In the application, «System audio» = your headphones; Zoom needs no setup.")
        return

    devices = input_devices()
    if devices is None:
        warn(f"could not enumerate the audio devices — check {human} by hand")
        return
    if any(p in devices for p in patterns):
        ok(f"{human} was found among the recording devices")
        return

    if sys.platform == "darwin":
        brew = shutil.which("brew")
        if brew and ask(f"{human} was not found. Install it ({howto})?", assume_yes):
            if run([brew, "install", "blackhole-2ch"]).returncode == 0:
                ok(f"{human} installed — it appears among the devices after you log back in")
                return
            warn("the installation failed")
        warn(f"{human} was not found — install it: {howto}")
        return

    # We only get here when loopback is unavailable (no PyAudioWPatch, or WASAPI
    # offered no devices). VB-Cable is a signed driver: it is not in winget, its
    # installer needs administrator rights and a reboot, there is nothing to
    # automate — so we walk the user up to the button.
    warn(f"{human} was not found. It is a driver and cannot be installed silently:")
    say(f"   1. Download {howto} and unpack the archive")
    say("   2. VBCABLE_Setup_x64.exe → right-click → «Run as administrator»")
    say("   3. Reboot, then run setup.py again to check")
    say("   Setting up Zoom and the headphones — the «Virtual audio device» section in README.md")
    if ask("Open the download page in the browser?"):
        webbrowser.open(howto)


def print_next_steps() -> None:
    activate = (
        r".\.venv\Scripts\Activate.ps1" if os.name == "nt" else "source .venv/bin/activate"
    )
    say("\nDone. To run it:")
    say(f"  {activate}")
    say("  python -m app.main")
    if sys.platform == "darwin":
        say(
            "\nOn the first start macOS will ask for microphone access — allow it,\n"
            "otherwise both channels stay silent (System Settings → Privacy →\n"
            "Microphone, for the terminal you launch it from)."
        )


def main() -> None:
    fix_stdio()
    ap = argparse.ArgumentParser(description="Environment setup for Interviewer's Little Helper")
    ap.add_argument("--recreate", action="store_true", help="recreate .venv from scratch")
    ap.add_argument("--skip-model", action="store_true",
                    help="do not download the ASR weights up front")
    ap.add_argument("--requirements", type=Path, default=None, help="your own requirements file")
    ap.add_argument("-y", "--yes", action="store_true",
                    help="install external components without asking")
    ap.add_argument("--no-extras", action="store_true",
                    help="the Python environment only: leave Ollama and the audio cable alone")
    args = ap.parse_args()

    check_python()
    say(f"Platform: {platform.system()} {platform.machine()}, Python {platform.python_version()}")

    create_venv(args.recreate)
    install_requirements(args.requirements or requirements_file())
    if not args.skip_model:
        fetch_asr_model()

    if not args.no_extras:
        ensure_ollama(args.yes)
        ensure_audio_cable(args.yes)
    print_next_steps()


if __name__ == "__main__":
    main()
