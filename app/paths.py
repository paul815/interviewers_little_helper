"""Turning identifiers from HTTP into paths on disk.

Guides and projects are addressed by a folder/file name that comes straight from
the URL. The naive check "no slashes and no .." is leaky on Windows:
`Path("guides") / "C:evil"` gives `C:evil` — a drive-relative path that bypasses
the root. Hence a whitelist of characters here, not a blacklist of separators.
"""
from __future__ import annotations

import re
from pathlib import Path


class UnsafeName(ValueError):
    """An identifier from HTTP that must not be turned into a path.

    It subclasses ValueError deliberately: the calling code already catches ValueError.
    """


# The first character is a letter or a digit: this rules out `.hidden`, `-flag`
# and the empty string. After that a dot, a hyphen and an underscore are allowed:
# enough for real identifiers (`2026-08-08_10-15-00`, `20260808-101500-onboarding`).
# Cyrillic stays in the whitelist: the interface is English, but a guide or a
# project may still be named in another language.
_SAFE_RE = re.compile(r"^[0-9A-Za-zА-Яа-яЁё][0-9A-Za-zА-Яа-яЁё._-]{0,99}$")

# DOS device names: Windows opens the device instead of the file, even with a suffix.
_WIN_RESERVED = {
    "con", "prn", "aux", "nul",
    *(f"com{i}" for i in range(1, 10)),
    *(f"lpt{i}" for i in range(1, 10)),
}


def safe_child(root: Path, name: str, suffix: str = "") -> Path:
    """A path inside root from an HTTP-supplied name. Anything but a plain name is refused.

    Three layers: a character whitelist, the DOS device names and — in case the
    first two are ever weakened — a check that the result physically lies inside root.
    """
    if not isinstance(name, str) or not _SAFE_RE.match(name):
        raise UnsafeName(f"Invalid identifier: {name!r}")
    if name.split(".", 1)[0].lower() in _WIN_RESERVED:
        raise UnsafeName(f"Invalid identifier: {name!r}")

    path = (root / f"{name}{suffix}").resolve()
    if not path.is_relative_to(root.resolve()):
        raise UnsafeName(f"The path escapes the directory: {name!r}")
    return path
