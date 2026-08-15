"""Tests of the helper the launchers use to find the server port.

A narrow purpose, but an important one: `tools/launch_win.bat` uses this port to
decide whether the application came up. If the helper lies, the launcher waits on
an empty port and ends up showing the user a bogus startup error.
"""
from __future__ import annotations

import json
import socket
import subprocess
import sys
import time

import pytest

from app import main
from tools import launcher_port


@pytest.fixture
def config(tmp_path, monkeypatch):
    """Stubs the root so the helper reads config.json from tmp_path."""
    monkeypatch.setattr(launcher_port, "ROOT", tmp_path)

    def write(payload, *, encoding="utf-8"):
        (tmp_path / "config.json").write_text(payload, encoding=encoding)

    return write


def test_default_when_no_config(config):
    assert launcher_port.port() == launcher_port.DEFAULT_PORT


def test_reads_port_from_config(config):
    config(json.dumps({"server": {"host": "127.0.0.1", "port": 9999}}))
    assert launcher_port.port() == 9999


def test_survives_bom(config):
    # Notepad and PowerShell (Out-File -Encoding utf8) prepend a BOM.
    config(json.dumps({"server": {"port": 7777}}), encoding="utf-8-sig")
    assert launcher_port.port() == 7777


@pytest.mark.parametrize(
    "payload",
    [
        "not json at all",
        "{}",
        '{"server": {}}',
        '{"server": {"port": "abc"}}',
        '{"server": {"port": 0}}',
        '{"server": {"port": 70000}}',
        '{"server": null}',
    ],
)
def test_falls_back_on_garbage(config, payload):
    """The launcher must not fail over a broken config — that gets fixed by hand anyway."""
    config(payload)
    assert launcher_port.port() == launcher_port.DEFAULT_PORT


def test_prints_single_line():
    """The batch file reads the output with SET /P — exactly one line holding a number."""
    out = subprocess.run(
        [sys.executable, "-m", "tools.launcher_port"],
        capture_output=True, text=True, check=True,
    ).stdout
    assert out.strip().isdigit()
    assert len(out.strip().splitlines()) == 1


# ------------------------------------------------- waiting on a port, startup

def test_wait_port_sees_a_live_listener():
    with socket.socket() as srv:
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        assert main.wait_port("127.0.0.1", srv.getsockname()[1], timeout=2.0) is True


def test_wait_port_gives_up_on_closed_port():
    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    started = time.monotonic()
    assert main.wait_port("127.0.0.1", port, timeout=0.5) is False
    assert time.monotonic() - started < 5.0  # it does not hang past its own timeout


def test_busy_port_stops_with_a_readable_message(monkeypatch):
    """A second launch must not fail silently: the window may be hiding behind Zoom."""
    with socket.socket() as srv:
        srv.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        srv.bind(("127.0.0.1", 0))
        srv.listen(1)
        port = srv.getsockname()[1]

        monkeypatch.setattr(sys, "argv", ["ilh", "--port", str(port)])
        monkeypatch.setattr(main, "configure_logging", lambda *a, **kw: None)
        monkeypatch.setattr(main, "build_server",
                            lambda cfg: pytest.fail("the server must not start"))

        with pytest.raises(SystemExit) as e:
            main.main()
    assert str(port) in str(e.value) and "is busy" in str(e.value)
