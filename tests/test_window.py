"""Window geometry: the right half of the screen at full height.

The working pose is Zoom on the left, the helper on the right. A mistake here
does not blow up with a traceback; it quietly produces a window of the wrong size
or one that slid under the taskbar, so the size arithmetic is kept under test.
pywebview itself is not needed: `screen_area` is stubbed and the rest is
arithmetic.
"""
from __future__ import annotations

import json

import pytest

from app import main
from app.config import WindowConfig


@pytest.fixture
def screen(monkeypatch):
    """Stubs the screen query: (x, y, width, height) of the work area."""

    def use(area):
        monkeypatch.setattr(main, "screen_area", lambda: area)
        monkeypatch.setattr(main, "screen_size",
                            lambda: None if area is None else (area[2], area[3]))

    return use


def test_half_of_2k_screen(screen):
    """2560x1440 with a taskbar: half the width, the full work-area height."""
    screen((0, 0, 2560, 1392))
    assert main.default_geometry(WindowConfig()) == (1280, 1392)


def test_docks_to_right_edge(screen):
    screen((0, 0, 2560, 1392))
    win = WindowConfig()
    width, _ = main.default_geometry(win)
    assert main.dock_position(win, width) == (1280, 0)


def test_respects_second_monitor_origin(screen):
    """The work area need not start at zero — the window hugs its edge."""
    screen((2560, 0, 1920, 1040))
    win = WindowConfig()
    width, height = main.default_geometry(win)
    assert (width, height) == (960, 1040)
    assert main.dock_position(win, width) == (2560 + 960, 0)


def test_tiny_screen_keeps_minimum_width(screen):
    """On a narrow screen half is below the minimum — the interface would break."""
    screen((0, 0, 700, 500))
    win = WindowConfig()
    assert main.default_geometry(win)[0] == win.min_width


def test_falls_back_to_config_without_screen(screen):
    """The screen could not be queried — take the sizes from the config, with no position."""
    screen(None)
    win = WindowConfig()
    assert main.default_geometry(win) == (win.width, win.height)
    assert main.dock_position(win, win.width) == (None, None)


def test_half_screen_off_uses_config(screen):
    screen((0, 0, 2560, 1392))
    win = WindowConfig(half_screen=False, width=800, height=600)
    assert main.default_geometry(win) == (800, 600)
    assert main.dock_position(win, 800) == (None, None)


def test_restore_size_defaults_when_no_state(screen, tmp_path):
    screen((0, 0, 2560, 1392))
    assert main.restore_size(WindowConfig(), tmp_path / "no-such-file.json") == (1280, 1392)


def test_restore_size_prefers_remembered(screen, tmp_path):
    """A window stretched by hand survives a restart — the default does not override it."""
    screen((0, 0, 2560, 1392))
    state = tmp_path / "window.json"
    state.write_text(json.dumps({"width": 1000, "height": 900}), encoding="utf-8")
    assert main.restore_size(WindowConfig(), state) == (1000, 900)


def test_restore_size_clamps_to_smaller_monitor(screen, tmp_path):
    """A window from a large monitor must not spill over the edges of a small one."""
    screen((0, 0, 1366, 768))
    state = tmp_path / "window.json"
    state.write_text(json.dumps({"width": 2400, "height": 1400}), encoding="utf-8")
    assert main.restore_size(WindowConfig(), state) == (1366, 768)


@pytest.mark.parametrize("payload", ["not json", "{}", '{"width": "wide"}', '{"height": 900}'])
def test_restore_size_survives_broken_state(screen, tmp_path, payload):
    screen((0, 0, 2560, 1392))
    state = tmp_path / "window.json"
    state.write_text(payload, encoding="utf-8")
    assert main.restore_size(WindowConfig(), state) == (1280, 1392)
