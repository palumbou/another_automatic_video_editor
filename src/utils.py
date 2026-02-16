"""Utility functions for Another Automatic Video Editor."""

import datetime as dt
import hashlib
import json
import logging
import os
import shlex
import subprocess
from pathlib import Path
from typing import Any


# File extensions
IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".webp", ".tif", ".tiff", ".bmp", ".gif", ".heic", ".heif"}
VIDEO_EXTS = {".mp4", ".mov", ".mkv", ".webm", ".avi", ".m4v", ".mts", ".m2ts"}
AUDIO_EXTS = {".mp3", ".wav", ".m4a", ".aac", ".flac", ".ogg"}


def now_utc() -> str:
    """Return current UTC time as ISO string."""
    return dt.datetime.now(dt.timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def sha1_short(s: str, n: int = 8) -> str:
    """Return short SHA1 hash of string."""
    return hashlib.sha1(s.encode("utf-8")).hexdigest()[:n]


def run_cmd(cmd: list[str], *, cwd: Path | None = None, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command and log it."""
    logging.info("RUN: %s", " ".join(shlex.quote(c) for c in cmd))
    return subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=check, text=True, capture_output=False)


def run_cmd_capture(cmd: list[str], *, cwd: Path | None = None) -> str:
    """Run a command and capture output."""
    logging.info("RUN: %s", " ".join(shlex.quote(c) for c in cmd))
    p = subprocess.run(cmd, cwd=str(cwd) if cwd else None, check=True, text=True, capture_output=True)
    return p.stdout.strip()


def which(cmd: str) -> str | None:
    """Find executable in PATH."""
    for p in os.environ.get("PATH", "").split(os.pathsep):
        cand = os.path.join(p, cmd)
        if os.path.isfile(cand) and os.access(cand, os.X_OK):
            return cand
    return None


def ensure_dir(p: Path) -> None:
    """Ensure directory exists."""
    p.mkdir(parents=True, exist_ok=True)


def safe_json_dump(obj: Any, path: Path) -> None:
    """Safely write JSON to file."""
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False), encoding="utf-8")


def format_ts(seconds: int) -> str:
    """Format seconds as timestamp (HH:MM:SS or MM:SS)."""
    seconds = max(0, int(seconds))
    h = seconds // 3600
    m = (seconds % 3600) // 60
    s = seconds % 60
    if h > 0:
        return f"{h:02d}:{m:02d}:{s:02d}"
    return f"{m:02d}:{s:02d}"


def escape_drawtext(text: str) -> str:
    """Escape text for ffmpeg drawtext filter."""
    t = text.replace("\\", "\\\\")
    t = t.replace(":", "\\:")
    t = t.replace("'", "\\'")
    t = t.replace("\n", " ")
    t = t.replace("\r", " ")
    return t
