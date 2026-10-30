"""Trim each rendered clip to the range it owns, concatenate, mux the master song.

Every cut was rendered from an audio slice starting at ``audio_start``, so song
time maps into clip time as::

    clip_t = song_t - audio_start + offset

where ``offset`` is the lip-sync error measured by :mod:`singsync.retime` (zero
when the clip has already been retimed). Each cut contributes exactly the frames
of its owned range, which means the finished video is the same length as the
song: owned ranges tile ``[0, duration]`` and segment lengths are counted in
whole frames off a shared grid, so rounding cannot accumulate.
"""
from __future__ import annotations

import json
from pathlib import Path

from .ffmpeg_util import ffmpeg, media_duration, video_size


def clip_time(song_t: float, cut: dict, offset_s: float = 0.0) -> float:
    """Map a song timestamp into the timeline of this cut's rendered clip."""
    return float(song_t) - float(cut["audio_start"]) + float(offset_s)


def frame_index(time_s: float, fps: float) -> int:
    """Frame number of a timestamp on the output grid."""
    return int(round(float(time_s) * float(fps)))


def source_clip(cut: dict) -> str | None:
    """Preferred rendered clip for a cut: retimed if present, else raw."""
    for key in ("retimed_path", "clip_path"):
        value = cut.get(key)
        if value and Path(value).exists():
            return str(value)
    return None


def cut_offset(cut: dict, clip_path: str | None = None) -> float:
    """Offset still to apply. A retimed clip already has the shift baked in."""
    if clip_path and cut.get("retimed_path") and str(clip_path) == str(cut["retimed_path"]):
        return float(cut.get("residual_offset_s", 0.0) or 0.0)
    retime = cut.get("retime") or {}
    return float(cut.get("offset_s", retime.get("offset_s", 0.0)) or 0.0)


def slide_short_trim(cut: dict, segment: dict, available_s: float) -> None:
    """Move a sufficiently long source window earlier to prevent edge freezing."""
    needed = float(segment["duration_s"])
    if float(segment["clip_start"]) + needed <= float(available_s) + 1e-6:
        return
    if float(available_s) >= needed:
        earlier = float(segment["clip_start"]) + needed - float(available_s)
        segment["clip_start"] = max(0.0, float(available_s) - needed)
        cut["assembly_warning"] = "slid %.2f s earlier to avoid freeze" % earlier


def segment_plan(plan: dict, fps: float = 24.0) -> list[dict]:
    """Per-cut trim instructions, in output order.

    Pure function (no ffmpeg): returns ``[{idx, clip, clip_start, frames,
    pad_start_s, owns_start, owns_end}]``. ``frames`` is counted on a single
    frame grid shared by all cuts, so the frame counts sum to the song length.
    """
    fps = float(fps)
    if fps <= 0:
        raise ValueError("fps must be positive")
    cuts = sorted(plan.get("cuts") or [], key=lambda c: int(c.get("idx", 0)))
    if not cuts:
        raise ValueError("plan has no cuts")
    segments = []
    for cut in cuts:
        clip = source_clip(cut)
        offset = cut_offset(cut, clip)
        start = frame_index(cut["owns_start"], fps)
        end = frame_index(cut["owns_end"], fps)
        frames = end - start
        if frames <= 0:
            continue
        splice = (cut.get("retime") or {}).get("segments")
        if splice:
            owned_lo, owned_hi = float(cut["owns_start"]), float(cut["owns_end"])
            fed_start = float(cut.get("audio_start", 0.0))
            bounds = [fed_start + float(s["slice_start"]) for s in splice[1:]]
            frame_edges = [start, *(frame_index(x, fps) for x in bounds), end]
            for n, splice_seg in enumerate(splice):
                a, b = max(start, frame_edges[n]), min(end, frame_edges[n+1])
                if b <= a:
                    continue
                song_a = a / fps
                song_splice_a = fed_start + float(splice_seg["slice_start"])
                clip_a = float(splice_seg["clip_start"]) + song_a - song_splice_a
                part_frames = b-a
                segments.append({"idx": int(cut.get("idx", 0)), "clip": clip,
                    "retimed": bool(clip and cut.get("retimed_path") and str(clip)==str(cut["retimed_path"])),
                    "clip_start": round(max(0.0, clip_a), 6), "pad_start_s": round(max(0.0, -clip_a), 6),
                    "frames": part_frames, "duration_s": round(part_frames/fps, 6),
                    "owns_start": max(owned_lo, song_a), "owns_end": min(owned_hi, b/fps), "offset_s": round(offset, 6)})
            continue
        clip_start = clip_time(float(cut["owns_start"]), cut, offset)
        pad_start = max(0.0, -clip_start)
        segments.append({
            "idx": int(cut.get("idx", 0)),
            "clip": clip,
            "retimed": bool(clip and cut.get("retimed_path")
                            and str(clip) == str(cut["retimed_path"])),
            "clip_start": round(max(0.0, clip_start), 6),
            "pad_start_s": round(pad_start, 6),
            "frames": frames,
            "duration_s": round(frames / fps, 6),
            "owns_start": float(cut["owns_start"]),
            "owns_end": float(cut["owns_end"]),
            "offset_s": round(offset, 6),
        })
    return segments


def _escape_filter_path(path: str | Path) -> str:
    """Escape a path for use inside an ffmpeg filter argument."""
    text = str(path).replace("\\", "\\\\").replace(":", "\\:").replace("'", "\\'")
    return text


def _even(value: int) -> int:
    value = int(value)
    return value if value % 2 == 0 else value + 1


def assemble(plan: dict, song_path: str | Path, out_path: str | Path, *,
             fps: float = 24.0, size: tuple[int, int] | None = None,
             captions: bool = False, caption_style: str = "bold_bottom",
             ass_path: str | Path | None = None, crf: int = 18,
             preset: str = "medium", audio_bitrate: str = "192k",
             work_dir: str | Path | None = None, keep_graph: bool = False) -> dict:
    """Build the final video. Returns a report dict (also stored on the plan)."""
    song_path = Path(song_path)
    if not song_path.exists():
        raise FileNotFoundError("song audio not found: %s" % song_path)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    work = Path(work_dir) if work_dir else out_path.parent
    work.mkdir(parents=True, exist_ok=True)

    fps = float(fps)
    segments = segment_plan(plan, fps)
    for seg in segments:
        if not seg["clip"]:
            continue
        available = media_duration(seg["clip"])
        cut = next((c for c in plan["cuts"] if int(c.get("idx", 0)) == seg["idx"]), None)
        if cut is not None:
            slide_short_trim(cut, seg, available)
    missing = [s["idx"] for s in segments if not s["clip"]]
    if missing:
        raise FileNotFoundError(
            "no rendered clip for cut(s) %s - run the provider node (or --dry-run) first"
            % ", ".join(str(i) for i in missing))

    if size is None:
        width, height = video_size(segments[0]["clip"])
    else:
        width, height = size
    width, height = _even(width), _even(height)

    total_frames = sum(s["frames"] for s in segments)
    total_s = total_frames / fps

    if captions and ass_path is None:
        ass_path = _build_captions(plan, work, caption_style, (width, height))

    inputs: list[str] = []
    chains: list[str] = []
    for k, seg in enumerate(segments):
        # Input seeking decodes and discards, which is accurate because the
        # output is re-encoded; tpad clones the edge frames when a clip is short
        # or the measured offset reaches before its first frame.
        inputs += ["-ss", "%.6f" % seg["clip_start"],
                   "-t", "%.6f" % (seg["duration_s"] + 1.0),
                   "-i", str(seg["clip"])]
        chains.append(
            "[{k}:v]fps={fps},scale={w}:{h}:force_original_aspect_ratio=decrease,"
            "pad={w}:{h}:(ow-iw)/2:(oh-ih)/2,setsar=1,"
            "tpad=start_mode=clone:start_duration={ps:.6f}:"
            "stop_mode=clone:stop_duration={sd:.6f},"
            "trim=end_frame={n},setpts=PTS-STARTPTS[v{k}]".format(
                k=k, fps=fps, w=width, h=height,
                ps=seg["pad_start_s"], sd=seg["duration_s"] + 1.0, n=seg["frames"]))
    audio_input = len(segments)
    inputs += ["-i", str(song_path)]

    concat = "".join("[v%d]" % k for k in range(len(segments)))
    chains.append("%sconcat=n=%d:v=1:a=0[vcat]" % (concat, len(segments)))
    if ass_path:
        chains.append("[vcat]subtitles=filename='%s'[vout]" % _escape_filter_path(ass_path))
    else:
        chains.append("[vcat]null[vout]")

    graph_path = work / ("%s.filtergraph.txt" % out_path.stem)
    graph_path.write_text(";\n".join(chains), encoding="utf-8")

    ffmpeg([
        *inputs,
        "-filter_complex_script", str(graph_path),
        "-map", "[vout]", "-map", "%d:a" % audio_input,
        "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
        "-pix_fmt", "yuv420p", "-r", "%.6f" % fps,
        "-c:a", "aac", "-b:a", str(audio_bitrate),
        "-t", "%.6f" % total_s,
        "-movflags", "+faststart",
        str(out_path),
    ])
    if not keep_graph:
        graph_path.unlink(missing_ok=True)

    actual = media_duration(out_path)
    report = {
        "path": str(out_path),
        "fps": fps,
        "width": width,
        "height": height,
        "segments": len(segments),
        "frames": total_frames,
        "expected_duration": round(total_s, 6),
        "actual_duration": round(float(actual), 6),
        "song_duration": round(float(plan.get("duration", total_s)), 6),
        "captions": str(ass_path) if ass_path else "",
    }
    report["drift_frames"] = round((report["actual_duration"] - report["expected_duration"]) * fps, 4)
    plan["video"] = report
    return report


def _build_captions(plan: dict, work: Path, style: str, size) -> str | None:
    """Write karaoke captions for the plan's words; None when there are none."""
    from . import captions as captions_mod

    words = [w for cut in plan.get("cuts", []) for w in cut.get("words", [])]
    if not words:
        return None
    path = work / "captions.ass"
    captions_mod.write_ass(words, path, style=style, size=size)
    return str(path)


def write_fcpxml(plan: dict, out_path: str | Path, *, fps: float = 24.0) -> str:
    """Minimal FCPXML edit list of the owned ranges, for re-cutting in an NLE."""
    segments = segment_plan(plan, fps)
    rate = int(round(fps))
    total = sum(s["frames"] for s in segments)

    def dur(frames: int) -> str:
        return "%d/%ds" % (frames, rate)

    assets, clips = [], []
    offset = 0
    for n, seg in enumerate(segments, start=1):
        src = Path(seg["clip"]).resolve().as_uri() if seg["clip"] else ""
        frames = seg["frames"]
        assets.append(
            '    <asset id="a%d" name="cut%04d" start="0s" hasVideo="1" format="r1" '
            'duration="%s"><media-rep kind="original-media" src="%s"/></asset>'
            % (n, seg["idx"], dur(frames), src))
        clips.append(
            '        <asset-clip ref="a%d" name="cut%04d" offset="%s" start="%s" duration="%s"/>'
            % (n, seg["idx"], dur(offset), dur(int(round(seg["clip_start"] * rate))), dur(frames)))
        offset += frames

    xml = """<?xml version="1.0" encoding="UTF-8"?>
<fcpxml version="1.9">
  <resources>
    <format id="r1" name="SingSync {rate}p" frameDuration="1/{rate}s"/>
{assets}
  </resources>
  <library>
    <event name="SingSync">
      <project name="SingSync cut">
        <sequence format="r1" duration="{total}" tcStart="0s" tcFormat="NDF">
          <spine>
{clips}
          </spine>
        </sequence>
      </project>
    </event>
  </library>
</fcpxml>
""".format(rate=rate, assets="\n".join(assets), clips="\n".join(clips), total=dur(total))
    out = Path(out_path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(xml, encoding="utf-8")
    return str(out)


def assembly_report(plan: dict, fps: float = 24.0) -> str:
    """Human-readable table of what assembly will take from each clip."""
    segments = segment_plan(plan, fps)
    rows = ["idx  owns            frames  clip in   offset  source"]
    rows.append("-" * len(rows[0]))
    for seg in segments:
        source = "MISSING"
        if seg["clip"]:
            source = Path(seg["clip"]).name
            if seg["retimed"]:
                source += " (retimed)"
        rows.append("%3d  %6.2f-%6.2f %6d  %7.3f  %+6.3f  %s" % (
            seg["idx"], seg["owns_start"], seg["owns_end"], seg["frames"],
            seg["clip_start"], seg["offset_s"], source))
    total = sum(s["frames"] for s in segments)
    rows.append("-" * len(rows[0]))
    rows.append("%d segments, %d frames = %.3fs at %gfps (song %.3fs)" % (
        len(segments), total, total / float(fps), fps,
        float(plan.get("duration", 0.0))))
    return "\n".join(rows)


def save_report(report: dict, path: str | Path) -> str:
    out = Path(path)
    out.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return str(out)
