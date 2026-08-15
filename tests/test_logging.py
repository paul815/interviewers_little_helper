"""Logging setup.

What is checked is exactly what an incident investigation on a user's machine
depends on: the rotation is bounded (the log does not eat the disk over a long
interview), a repeat call does not multiply handlers (or every line is doubled),
and a garbage level does not take startup down.
"""
from __future__ import annotations

import logging
import logging.handlers

import pytest

from app import logging_setup


@pytest.fixture()
def clean_logging(tmp_path, monkeypatch):
    """The root logger is global state: we clean our handlers up after ourselves."""
    root = logging.getLogger()
    saved_handlers, saved_level = list(root.handlers), root.level
    root.handlers.clear()
    monkeypatch.setattr(logging_setup, "ROOT", tmp_path)
    monkeypatch.setattr(logging_setup, "_configured", False)
    yield tmp_path
    for h in root.handlers:
        h.close()
    root.handlers[:] = saved_handlers
    root.setLevel(saved_level)


# The class reference is taken before the tests: one of them stubs it in logging.handlers.
_ROTATING = logging.handlers.RotatingFileHandler


def _file_handlers(root):
    return [h for h in root.handlers if isinstance(h, _ROTATING)]


def test_writes_rotating_file_log(clean_logging):
    logging_setup.configure("INFO")
    root = logging.getLogger()

    handlers = _file_handlers(root)
    assert len(handlers) == 1
    h = handlers[0]
    assert h.maxBytes == 2_000_000 and h.backupCount == 3
    assert (clean_logging / "logs" / "app.log").exists()


def test_level_is_applied(clean_logging):
    logging_setup.configure("debug")
    assert logging.getLogger().level == logging.DEBUG


def test_garbage_level_falls_back_to_info(clean_logging):
    logging_setup.configure("VERY VERBOSE INDEED")
    assert logging.getLogger().level == logging.INFO


def test_second_call_does_not_duplicate_handlers(clean_logging):
    logging_setup.configure("INFO")
    before = len(logging.getLogger().handlers)
    logging_setup.configure("DEBUG")
    assert len(logging.getLogger().handlers) == before


def test_noisy_http_libraries_are_muted(clean_logging):
    logging_setup.configure("DEBUG")
    assert logging.getLogger("httpx").level == logging.WARNING
    assert logging.getLogger("httpcore").level == logging.WARNING


def test_unwritable_log_dir_does_not_break_startup(clean_logging, monkeypatch):
    """The file log is not critical: the application must come up without it."""
    def boom(*a, **kw):
        raise OSError("the disk is read-only")

    monkeypatch.setattr(logging.handlers, "RotatingFileHandler", boom)
    logging_setup.configure("INFO")

    root = logging.getLogger()
    assert _file_handlers(root) == []
    assert any(isinstance(h, logging.StreamHandler) for h in root.handlers)
