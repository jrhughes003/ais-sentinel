"""Small I/O helpers shared by every stage that writes text artifacts."""

from __future__ import annotations

from pathlib import Path


def write_text(path: str | Path, text: str) -> int:
    """Write UTF-8 text with LF line endings and a trailing newline (on every OS).

    Python's default on Windows is the locale code page plus CRLF, which corrupts non-ASCII
    characters and churns git diffs.

    Returns:
        Number of bytes written.
    """
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    if not text.endswith("\n"):
        text += "\n"
    data = text.encode("utf-8")
    p.write_bytes(data)
    return len(data)
