"""ffmpeg/ffprobe discovery and thin subprocess wrappers.

The binary is taken from ``SINGSYNC_FFMPEG`` (or ``SINGSYNC_FFPROBE``) when set,
otherwise from ``PATH``. Nothing here imports torch or ComfyUI.
"""
from __future__ import annotations

import json
import os
import shutil
import subprocess
from pathlib import Path


class FFmpegMissing(RuntimeError):
    """Raised when no usable ffmpeg/ffprobe binary could be found."""


def _resolve(env_var: str, default_name: str, sibling_of: str | None = None) -> str | None:
    override = os.environ.get(env_var)
    if override:
        p = Path(override)
        if p.is_dir():
            cand = p / default_name
            if cand.exists():
                return str(cand)
        elif p.exists() or shutil.which(override):
            return override
    found = shutil.which(default_name)
    if found:
        return found
    if sibling_of:
        other = os.environ.get(sibling_of)
        if other:
            cand = Path(other).parent / default_name
            if cand.exists():
                return str(cand)
    return None


def ffmpeg_bin() -> str:
    """Absolute path (or bare name) of ffmpeg; raises if unavailable."""
    found = _resolve("SINGSYNC_FFMPEG", "ffmpeg", "SINGSYNC_FFPROBE")
    if not found:
        raise FFmpegMissing(
            "ffmpeg not found. Install it, or set SINGSYNC_FFMPEG to the binary.")
    return found


def ffprobe_bin() -> str:
    """Absolute path (or bare name) of ffprobe; raises if unavailable."""
    found = _resolve("SINGSYNC_FFPROBE", "ffprobe", "SINGSYNC_FFMPEG")
    if not found:
        raise FFmpegMissing(
            "ffprobe not found. Install it, or set SINGSYNC_FFPROBE to the binary.")
    return found


def have_ffmpeg() -> bool:
    """True when both ffmpeg and ffprobe can be located (for test skips)."""
    try:
        ffmpeg_bin()
        ffprobe_bin()
    except FFmpegMissing:
        return False
    return True


def run(args, *, capture: bool = True, check: bool = True) -> subprocess.CompletedProcess:
    """Run a command, raising RuntimeError with stderr tail on failure."""
    args = [str(a) for a in args]
    proc = subprocess.run(
        args,
        stdout=subprocess.PIPE if capture else None,
        stderr=subprocess.PIPE if capture else None,
    )
    if check and proc.returncode != 0:
        tail = (proc.stderr or b"").decode("utf-8", "replace").strip().splitlines()[-12:]
        raise RuntimeError("command failed: %s\n%s" % (" ".join(args[:3]), "\n".join(tail)))
    return proc


def ffmpeg(args, **kw) -> subprocess.CompletedProcess:
    """Run ffmpeg with error-level logging and ``-y``."""
    return run([ffmpeg_bin(), "-v", "error", "-nostdin", "-y", *args], **kw)


def probe(path: str | Path) -> dict:
    """ffprobe JSON for format and streams."""
    proc = run([ffprobe_bin(), "-v", "error", "-print_format", "json",
                "-show_format", "-show_streams", str(path)])
    return json.loads(proc.stdout.decode("utf-8", "replace") or "{}")


def media_duration(path: str | Path) -> float:
    """Container duration in seconds, falling back to the longest stream."""
    info = probe(path)
    try:
        return float(info.get("format", {}).get("duration"))
    except (TypeError, ValueError):
        pass
    best = 0.0
    for stream in info.get("streams", []):
        try:
            best = max(best, float(stream.get("duration")))
        except (TypeError, ValueError):
            continue
    if best <= 0:
        raise RuntimeError("could not determine duration of %s" % path)
    return best


def video_size(path: str | Path) -> tuple[int, int]:
    """(width, height) of the first video stream."""
    for stream in probe(path).get("streams", []):
        if stream.get("codec_type") == "video":
            return int(stream["width"]), int(stream["height"])
    raise RuntimeError("no video stream in %s" % path)
