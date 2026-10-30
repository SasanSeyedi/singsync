"""Karaoke lyric captions: words -> ASS subtitles -> burnt into the video.

One lyric line is on screen at a time and the words light up as they are sung,
using ASS ``\\k`` karaoke tags (durations in centiseconds). ``write_ass`` is a
pure text transform, so the output can be unit-tested without ffmpeg.
"""
from __future__ import annotations

from pathlib import Path

from .ffmpeg_util import ffmpeg

#: Style presets: ``(font, size_frac, primary, highlight, outline, border, alignment)``
#: ``size_frac`` is the font size as a fraction of the video height.
STYLES = {
    "bold_bottom": {
        "font": "DejaVu Sans", "size_frac": 0.062, "bold": -1,
        "primary": "&H00FFFFFF", "highlight": "&H0040D8FF",
        "outline_colour": "&H00101010", "back": "&H80000000",
        "outline": 3.0, "shadow": 0.0, "alignment": 2, "margin_v_frac": 0.11,
    },
    "bold_centre": {
        "font": "DejaVu Sans", "size_frac": 0.075, "bold": -1,
        "primary": "&H00FFFFFF", "highlight": "&H0040D8FF",
        "outline_colour": "&H00101010", "back": "&H80000000",
        "outline": 3.0, "shadow": 0.0, "alignment": 5, "margin_v_frac": 0.0,
    },
    "boxed_bottom": {
        "font": "DejaVu Sans", "size_frac": 0.055, "bold": -1,
        "primary": "&H00FFFFFF", "highlight": "&H0040D8FF",
        "outline_colour": "&H00000000", "back": "&HA0000000",
        "outline": 0.0, "shadow": 0.0, "alignment": 2, "margin_v_frac": 0.09,
        "border_style": 3,
    },
}
DEFAULT_STYLE = "bold_bottom"

#: Words further apart than this start a new caption line when the alignment
#: carries no line indices.
LINE_GAP_S = 0.65
MAX_WORDS_PER_LINE = 9
#: Keep a line on screen a little after its last word.
HOLD_S = 0.35


def style_names() -> list[str]:
    """Preset names, with the default first (for the ComfyUI dropdown)."""
    rest = sorted(k for k in STYLES if k != DEFAULT_STYLE)
    return [DEFAULT_STYLE, *rest]


def _cs(seconds: float) -> int:
    """Seconds -> centiseconds, never negative."""
    return max(0, int(round(float(seconds) * 100.0)))


def ass_time(seconds: float) -> str:
    """Format a timestamp as ASS ``H:MM:SS.cc``."""
    total = max(0.0, float(seconds))
    hours = int(total // 3600)
    minutes = int((total % 3600) // 60)
    secs = total % 60
    return "%d:%02d:%05.2f" % (hours, minutes, secs)


def group_lines(words, *, line_gap_s: float = LINE_GAP_S,
                max_words: int = MAX_WORDS_PER_LINE) -> list[list[dict]]:
    """Group words into caption lines, using ``line_idx`` when present."""
    lines: list[list[dict]] = []
    current: list[dict] = []
    for i, word in enumerate(words):
        if not str(word.get("word", "")).strip():
            continue
        if current:
            prev = current[-1]
            same_line = (word.get("line_idx") is not None
                         and prev.get("line_idx") is not None
                         and int(word["line_idx"]) == int(prev["line_idx"]))
            gap = float(word["start"]) - float(prev["end"])
            if (not same_line and (word.get("line_idx") is not None
                                   and prev.get("line_idx") is not None)) \
                    or gap >= line_gap_s or len(current) >= max_words \
                    or prev.get("line_end"):
                lines.append(current)
                current = []
        current.append(word)
    if current:
        lines.append(current)
    return lines


def _escape(text: str) -> str:
    """Escape ASS markup characters in lyric text."""
    return (str(text).replace("\\", "\\\\").replace("{", "\\{")
            .replace("}", "\\}").replace("\n", " ").strip())


def karaoke_text(line: list[dict], *, highlight: bool = True) -> str:
    """Build the ``{\\k..}word`` body for one caption line."""
    parts: list[str] = []
    for i, word in enumerate(line):
        start, end = float(word["start"]), float(word["end"])
        if i:
            gap = start - float(line[i - 1]["end"])
            if gap > 0.02:
                # An un-highlighted spacer keeps later words in time with the vocal.
                parts.append("{\\k%d} " % _cs(gap))
            else:
                parts.append(" ")
        tag = "\\k" if highlight else "\\K"
        parts.append("{%s%d}%s" % (tag, _cs(end - start), _escape(word.get("word", ""))))
    return "".join(parts)


def build_ass(words, *, style: str = DEFAULT_STYLE, size=(1280, 720),
              line_gap_s: float = LINE_GAP_S, max_words: int = MAX_WORDS_PER_LINE,
              hold_s: float = HOLD_S, title: str = "SingSync") -> str:
    """Render a complete ASS subtitle document for ``words``."""
    preset = dict(STYLES.get(style) or STYLES[DEFAULT_STYLE])
    width, height = int(size[0]), int(size[1])
    font_size = max(12, int(round(preset["size_frac"] * height)))
    margin_v = max(0, int(round(preset.get("margin_v_frac", 0.1) * height)))
    margin_h = max(10, int(round(0.06 * width)))

    header = [
        "[Script Info]",
        "; Generated by SingSync",
        "Title: %s" % _escape(title),
        "ScriptType: v4.00+",
        "WrapStyle: 0",
        "ScaledBorderAndShadow: yes",
        "PlayResX: %d" % width,
        "PlayResY: %d" % height,
        "",
        "[V4+ Styles]",
        ("Format: Name, Fontname, Fontsize, PrimaryColour, SecondaryColour, "
         "OutlineColour, BackColour, Bold, Italic, Underline, StrikeOut, ScaleX, "
         "ScaleY, Spacing, Angle, BorderStyle, Outline, Shadow, Alignment, "
         "MarginL, MarginR, MarginV, Encoding"),
        ("Style: SingSync,%s,%d,%s,%s,%s,%s,%d,0,0,0,100,100,0,0,%d,%g,%g,%d,%d,%d,%d,1"
         % (preset["font"], font_size, preset["highlight"], preset["primary"],
            preset["outline_colour"], preset["back"], preset["bold"],
            int(preset.get("border_style", 1)), preset["outline"], preset["shadow"],
            preset["alignment"], margin_h, margin_h, margin_v)),
        "",
        "[Events]",
        "Format: Layer, Start, End, Style, Name, MarginL, MarginR, MarginV, Effect, Text",
    ]

    events = []
    lines = group_lines(words, line_gap_s=line_gap_s, max_words=max_words)
    for n, line in enumerate(lines):
        start = float(line[0]["start"])
        end = float(line[-1]["end"]) + float(hold_s)
        nxt = lines[n + 1] if n + 1 < len(lines) else None
        if nxt:
            end = min(end, float(nxt[0]["start"]))
        if end <= start:
            end = start + 0.05
        events.append("Dialogue: 0,%s,%s,SingSync,,0,0,0,,%s"
                      % (ass_time(start), ass_time(end), karaoke_text(line)))
    return "\n".join(header + events) + "\n"


def write_ass(words, path: str | Path, **kw) -> str:
    """Write :func:`build_ass` output to ``path``; returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(build_ass(words, **kw), encoding="utf-8")
    return str(out)


def words_from_plan(plan: dict) -> list[dict]:
    """All owned words of a plan, in song order (shared wings excluded)."""
    words = []
    for cut in sorted(plan.get("cuts") or [], key=lambda c: int(c.get("idx", 0))):
        words.extend(cut.get("words") or [])
    return words


def burn_captions(video: str | Path, ass_path: str | Path, out_path: str | Path, *,
                  crf: int = 18, preset: str = "medium") -> str:
    """Burn an ASS file into a video with ffmpeg's ``subtitles`` filter."""
    from .assemble import _escape_filter_path

    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(["-i", str(video),
            "-vf", "subtitles=filename='%s'" % _escape_filter_path(ass_path),
            "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p", "-c:a", "copy", str(out)])
    return str(out)
