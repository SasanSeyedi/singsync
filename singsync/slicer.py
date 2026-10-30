"""Cut the fed audio slices and write the per-cut render prompts.

Each cut gets its own audio file covering ``[audio_start, audio_end)`` -- its
owned range plus the shared overlap words -- because that slice is what the
video model sings to. The prompt is built from a template so a character and
scene description stay constant across the song while the sung line changes.
"""
from __future__ import annotations

import re
from pathlib import Path

from .ffmpeg_util import ffmpeg

#: Default prompt for a cut with singing.
DEFAULT_TEMPLATE = ('{style}. {character}. {scene}. '
                    'The character sings: "{line_text}". {vibe}')
#: Default prompt for an instrumental cut (nothing is sung).
DEFAULT_BROLL_TEMPLATE = ('{style}. {scene}. Instrumental section, no singing, '
                          'no visible lip movement. {vibe}')

_WS = re.compile(r"\s+")


def slice_audio(src: str | Path, start: float, end: float, out: str | Path, *,
                sample_rate: int = 44100, channels: int = 1,
                bitrate: str = "192k", pad: bool = True) -> str:
    """Write ``[start, end)`` of ``src`` to ``out`` (``.wav`` or ``.mp3``).

    With ``pad`` the output is padded with silence when the source ends early,
    so the slice is always exactly ``end - start`` seconds -- the renderer's
    duration bucket was chosen from that length.
    """
    src, out = Path(src), Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    start = max(0.0, float(start))
    length = float(end) - start
    if length <= 0:
        raise ValueError("slice end must be after its start")

    args = ["-ss", "%.6f" % start, "-i", str(src), "-vn",
            "-ac", str(int(channels)), "-ar", str(int(sample_rate))]
    if pad:
        args += ["-af", "apad"]
    args += ["-t", "%.6f" % length]
    suffix = out.suffix.lower()
    if suffix == ".mp3":
        args += ["-c:a", "libmp3lame", "-b:a", str(bitrate)]
    elif suffix in (".wav", ""):
        args += ["-c:a", "pcm_s16le"]
    elif suffix in (".m4a", ".aac"):
        args += ["-c:a", "aac", "-b:a", str(bitrate)]
    else:
        raise ValueError("unsupported slice format %r (use .wav, .mp3 or .m4a)" % suffix)
    ffmpeg([*args, str(out)])
    return str(out)


def build_prompt(cut: dict, *, style: str = "", character: str = "", scene: str = "",
                 template: str | None = None, broll_template: str | None = None) -> str:
    """Render a cut's prompt from the template and its planned metadata."""
    is_vocal = bool(cut.get("words"))
    text = template if template else DEFAULT_TEMPLATE
    if not is_vocal:
        text = broll_template if broll_template else DEFAULT_BROLL_TEMPLATE
    fields = {
        "style": style.strip(),
        "character": character.strip(),
        "scene": scene.strip(),
        "line_text": _WS.sub(" ", str(cut.get("line_text", ""))).strip(),
        "vibe": str(cut.get("vibe", "")).strip(),
        "section": str(cut.get("section", "")).strip(),
        "energy": cut.get("energy", 0.0),
        "idx": cut.get("idx", 0),
        "duration": cut.get("duration", 0),
        "owns_start": cut.get("owns_start", 0.0),
        "owns_end": cut.get("owns_end", 0.0),
    }
    try:
        prompt = text.format(**fields)
    except (KeyError, IndexError) as exc:
        raise ValueError("prompt template has an unknown field: %s" % exc) from None
    # Empty fields leave ". . " runs behind; tidy them rather than ship them.
    prompt = re.sub(r"(?:^|\s)\.(?=\s|$)", " ", prompt)
    prompt = re.sub(r"\.\s*\.", ".", prompt)
    return _WS.sub(" ", prompt).strip().strip(".").strip() + "."


def slice_plan(plan: dict, audio_path: str | Path, work_dir: str | Path, *,
               style: str = "", character: str = "", scene: str = "",
               template: str | None = None, broll_template: str | None = None,
               fmt: str = "wav", slice_broll: bool = True,
               sample_rate: int = 44100, channels: int = 1) -> dict:
    """Write every cut's audio slice and prompt into ``work_dir``.

    ``audio_path`` should be the isolated vocal when one is available -- that is
    what makes the character sing rather than mouth along to a full mix.
    """
    audio_path = Path(audio_path)
    if not audio_path.exists():
        raise FileNotFoundError("audio not found: %s" % audio_path)
    work = Path(work_dir)
    slices = work / "slices"
    slices.mkdir(parents=True, exist_ok=True)
    fmt = fmt.lstrip(".").lower()

    for cut in sorted(plan.get("cuts") or [], key=lambda c: int(c.get("idx", 0))):
        cut["prompt"] = build_prompt(cut, style=style, character=character, scene=scene,
                                     template=template, broll_template=broll_template)
        if not cut.get("words") and not slice_broll:
            cut["slice_path"] = ""
            continue
        out = slices / ("cut_%04d.%s" % (int(cut.get("idx", 0)), fmt))
        cut["slice_path"] = slice_audio(
            audio_path, cut["audio_start"], cut["audio_end"], out,
            sample_rate=sample_rate, channels=channels)
        cut["slice_duration"] = round(float(cut["audio_end"]) - float(cut["audio_start"]), 4)
    plan["slices"] = {
        "source": str(audio_path),
        "dir": str(slices),
        "format": fmt,
        "count": sum(1 for c in plan.get("cuts", []) if c.get("slice_path")),
    }
    return plan


def prompt_sheet(plan: dict) -> str:
    """One line per cut: index, clip duration and prompt."""
    rows = []
    for cut in sorted(plan.get("cuts") or [], key=lambda c: int(c.get("idx", 0))):
        rows.append("[%d] %gs  %s" % (int(cut.get("idx", 0)),
                                      float(cut.get("duration", 0)),
                                      cut.get("prompt", "")))
    return "\n".join(rows)


def slice_list(plan: dict) -> list[dict]:
    """Compact per-cut render jobs, for a ComfyUI loop or an external script."""
    jobs = []
    for cut in sorted(plan.get("cuts") or [], key=lambda c: int(c.get("idx", 0))):
        jobs.append({
            "idx": int(cut.get("idx", 0)),
            "slice_path": cut.get("slice_path", ""),
            "prompt": cut.get("prompt", ""),
            "duration": cut.get("duration", 0),
            "kind": cut.get("kind", "vocal"),
        })
    return jobs
