"""Measure and correct a rendered clip's lip-sync offset.

Video models that sing to a supplied vocal track do not start exactly on time:
one clip is a third of a second early, the next is a second late. The error is
per clip, so it cannot be fixed with one global nudge.

What this module does: transcribe the rendered clip, transcribe the vocal slice
that was fed to the renderer, match the two word sequences, and read the offset
off the matched pairs::

    offset = median(clip_t - slice_t)

:func:`estimate_offset` is pure and unit-tested on synthetic word lists with
known shifts, dropped words and hallucinated extras. The offset is carried on
the plan and applied arithmetically at assembly time (no extra encode); baking
it into a new clip file is optional via :func:`retime_clip`.
"""
from __future__ import annotations

import statistics
from difflib import SequenceMatcher
from pathlib import Path

from .align import normalise_word
from .ffmpeg_util import ffmpeg, media_duration

#: Offsets beyond this are treated as a failed measurement, not a real shift.
MAX_ABS_OFFSET_S = 4.0
#: Anchors within this distance of the median are kept as inliers.
INLIER_S = 0.35
#: Anchors needed before a non-constant (piecewise) map is considered.
PIECEWISE_MIN_ANCHORS = 4
#: Inlier spread that justifies a piecewise map instead of one constant shift.
PIECEWISE_SPREAD_S = 0.4


def _prep(words) -> list[dict]:
    """Normalised, time-ordered tokens, dropping non-lexical items."""
    out = []
    for word in words or []:
        if isinstance(word, dict):
            text, start = word.get("word", ""), word.get("start")
            end = word.get("end", start)
        else:
            text, start, end = word[0], word[1], word[2] if len(word) > 2 else word[1]
        token = normalise_word(text)
        if not token or start is None:
            continue
        out.append({"token": token, "word": str(text).strip(),
                    "start": float(start), "end": float(end if end is not None else start)})
    out.sort(key=lambda w: w["start"])
    return out


def _match(clip: list[dict], slice_: list[dict]) -> list[dict]:
    """Pair up identical words in the two transcripts, in order."""
    matcher = SequenceMatcher(None, [w["token"] for w in clip],
                              [w["token"] for w in slice_], autojunk=False)
    anchors = []
    for i, j, size in matcher.get_matching_blocks():
        for k in range(size):
            cw, sw = clip[i + k], slice_[j + k]
            anchors.append({
                "word": sw["word"],
                "clip_t": round(cw["start"], 4),
                "slice_t": round(sw["start"], 4),
                # Held notes can make Whisper's word starts disagree even
                # when the word ends line up. Use whichever boundary is the
                # better match, retaining that boundary's signed offset.
                "delta": round(min((cw["start"] - sw["start"],
                                     cw["end"] - sw["end"]), key=abs), 4),
            })
    return anchors


def _monotone_map(anchors: list[dict]) -> list[list[float]]:
    """Piecewise-linear slice-time -> clip-time knots, forced non-decreasing."""
    knots: list[list[float]] = []
    for anchor in sorted(anchors, key=lambda a: a["slice_t"]):
        point = [round(anchor["slice_t"], 4), round(anchor["clip_t"], 4)]
        if knots and point[0] - knots[-1][0] < 1e-6:
            continue
        if knots and point[1] < knots[-1][1]:
            # A clip time that goes backwards would mean playing the clip in
            # reverse; keep the map flat instead.
            point[1] = knots[-1][1]
        knots.append(point)
    return knots


def estimate_offset(clip_words, slice_words, *,
                    max_abs_offset_s: float = MAX_ABS_OFFSET_S,
                    inlier_s: float = INLIER_S,
                    piecewise_min_anchors: int = PIECEWISE_MIN_ANCHORS,
                    piecewise_spread_s: float = PIECEWISE_SPREAD_S) -> dict:
    """Offset between a clip's sung words and the vocal slice it was fed.

    Returns ``{"offset_s", "anchors", "method", "n_anchors", "spread_s", "map"}``.
    ``method`` is ``"none"`` (nothing matched), ``"median"`` (one constant shift)
    or ``"piecewise"`` (the shift drifts within the clip; ``map`` holds
    ``[slice_t, clip_t]`` knots).
    """
    clip = _prep(clip_words)
    slice_ = _prep(slice_words)
    result = {"offset_s": 0.0, "anchors": [], "method": "none", "n_anchors": 0,
              "spread_s": 0.0, "map": [], "clamped": False}
    if not clip or not slice_:
        return result

    anchors = _match(clip, slice_)
    if not anchors:
        return result

    anchors = sorted(anchors, key=lambda a: a["slice_t"])
    # Only contiguous runs can describe a real timing regime. Dispersed
    # outliers are ignored as noise; nearby anchors within a run must agree.
    runs = []
    for anchor in anchors:
        if not runs or abs(anchor["delta"] - runs[-1][-1]["delta"]) > 0.25:
            runs.append([anchor])
        else:
            runs[-1].append(anchor)
    clusters = [run for run in runs if len(run) >= 3]
    if (len(clusters) > 1 and
            any(abs(statistics.median(a["delta"] for a in left) -
                    statistics.median(a["delta"] for a in right)) >= 0.8
                for left, right in zip(clusters, clusters[1:]))):
        centers = [statistics.median(a["delta"] for a in c) for c in clusters]
        segments = []
        for i, (cluster, center) in enumerate(zip(clusters, centers)):
            start = 0.0 if i == 0 else (clusters[i-1][-1]["slice_t"] + cluster[0]["slice_t"]) / 2
            end = (clusters[i+1][0]["slice_t"] + cluster[-1]["slice_t"]) / 2 if i+1 < len(clusters) else None
            clip_start = (0.0 + center if i == 0 else
                          (clusters[i-1][-1]["clip_t"] + cluster[0]["clip_t"]) / 2)
            segments.append({"slice_start": round(start, 4), "slice_end": round(end, 4) if end is not None else None,
                             "clip_start": round(clip_start, 4), "offset_s": round(center, 4)})
        offset = centers[0]
        result.update({"offset_s": round(offset, 4), "anchors": anchors,
                       "method": "splice", "n_anchors": len(anchors),
                       "n_inliers": len(anchors), "spread_s": round(max(a["delta"] for a in anchors)-min(a["delta"] for a in anchors), 4),
                       "segments": segments})
        return result

    deltas = [a["delta"] for a in anchors]
    median = statistics.median(deltas)
    inliers = [a for a in anchors if abs(a["delta"] - median) <= inlier_s] or anchors
    inlier_deltas = [a["delta"] for a in inliers]
    offset = statistics.median(inlier_deltas)
    spread = max(inlier_deltas) - min(inlier_deltas)

    result.update({
        "offset_s": round(float(offset), 4),
        "anchors": anchors,
        "method": "median",
        "n_anchors": len(anchors),
        "n_inliers": len(inliers),
        "spread_s": round(float(spread), 4),
    })
    if len(inliers) >= piecewise_min_anchors and spread >= piecewise_spread_s:
        result["method"] = "piecewise"
        result["map"] = _monotone_map(inliers)
    if abs(result["offset_s"]) > max_abs_offset_s:
        # Almost certainly a bad transcript rather than a four-second delay;
        # report it but do not move the clip.
        result["clamped"] = True
        result["measured_offset_s"] = result["offset_s"]
        result["offset_s"] = 0.0
        result["method"] = "rejected"
        result["map"] = []
    return result


def offset_at(result: dict, slice_t: float) -> float:
    """Local offset at a point in slice time (constant unless piecewise)."""
    knots = result.get("map") or []
    if result.get("method") != "piecewise" or len(knots) < 2:
        return float(result.get("offset_s", 0.0))
    slice_t = float(slice_t)
    if slice_t <= knots[0][0]:
        return knots[0][1] - knots[0][0]
    if slice_t >= knots[-1][0]:
        return knots[-1][1] - knots[-1][0]
    for (t0, c0), (t1, c1) in zip(knots, knots[1:]):
        if t0 <= slice_t <= t1:
            frac = 0.0 if t1 - t0 < 1e-9 else (slice_t - t0) / (t1 - t0)
            return (c0 + frac * (c1 - c0)) - slice_t
    return float(result.get("offset_s", 0.0))


# --------------------------------------------------------------------------- #
# measurement against real files
# --------------------------------------------------------------------------- #

def measure_clip(clip_path: str | Path, slice_path: str | Path, *,
                 model: str = "small.en", device: str = "auto",
                 slice_words=None, **kw) -> dict:
    """Transcribe clip and slice, then :func:`estimate_offset` them.

    ``slice_words`` lets a caller reuse word timings it already has for the
    slice (relative to the slice start), skipping one transcription.
    """
    from .align import transcribe_words

    clip_alignment = transcribe_words(clip_path, None, model=model, device=device)
    if slice_words is None:
        slice_words = transcribe_words(slice_path, None, model=model,
                                       device=device).get("words", [])
    result = estimate_offset(clip_alignment.get("words", []), slice_words, **kw)
    result["clip"] = str(clip_path)
    result["slice"] = str(slice_path)
    result["clip_words"] = clip_alignment.get("words", [])
    return result


def expected_slice_words(cut: dict) -> list[dict]:
    """The cut's words re-based to its fed audio slice (slice time starts at 0)."""
    base = float(cut.get("audio_start", 0.0))
    words = []
    for group in ("shared_words_before", "words", "shared_words_after"):
        for word in cut.get(group) or []:
            words.append({"word": word.get("word", ""),
                          "start": round(float(word["start"]) - base, 4),
                          "end": round(float(word["end"]) - base, 4)})
    words.sort(key=lambda w: w["start"])
    return words


def retime_clip(clip_path: str | Path, out_path: str | Path, *, offset_s: float = 0.0,
                result: dict | None = None, duration: float | None = None,
                fps: float | None = None, crf: int = 18,
                preset: str = "medium") -> str:
    """Write a copy of the clip shifted so its singing lands on slice time.

    A positive offset means the clip sang late, so its content is pulled
    earlier. Edge frames are cloned to keep the clip's length. Output has no
    audio: assembly muxes the master song, never the clip's own track.
    """
    clip_path, out_path = Path(clip_path), Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    if duration is None:
        duration = media_duration(clip_path)
    duration = float(duration)

    knots = (result or {}).get("map") or []
    if (result or {}).get("method") == "piecewise" and len(knots) >= 2:
        return _retime_piecewise(clip_path, out_path, result, duration, fps, crf, preset)

    offset = float(offset_s if result is None else result.get("offset_s", offset_s))
    chain = []
    if fps:
        chain.append("fps=%g" % float(fps))
    if offset > 1e-4:
        chain.append("trim=start=%.6f" % offset)
        chain.append("setpts=PTS-STARTPTS")
    elif offset < -1e-4:
        chain.append("tpad=start_mode=clone:start_duration=%.6f" % -offset)
    chain.append("tpad=stop_mode=clone:stop_duration=%.6f" % (abs(offset) + 0.5))
    chain.append("trim=duration=%.6f" % duration)
    chain.append("setpts=PTS-STARTPTS")
    ffmpeg(["-i", str(clip_path), "-an", "-vf", ",".join(chain),
            "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p", "-t", "%.6f" % duration, str(out_path)])
    return str(out_path)


def _retime_piecewise(clip_path: Path, out_path: Path, result: dict, duration: float,
                      fps: float | None, crf: int, preset: str) -> str:
    """Shift each stretch between anchors by its own local offset, then concat."""
    knots = result["map"]
    edges = [0.0]
    for (t0, _), (t1, _) in zip(knots, knots[1:]):
        edges.append((t0 + t1) / 2.0)
    edges.append(duration)
    edges = sorted({round(min(max(e, 0.0), duration), 6) for e in edges})

    parts, chains = [], []
    kept = 0
    for i, (lo, hi) in enumerate(zip(edges, edges[1:])):
        if hi - lo < 1e-3:
            continue
        local = offset_at(result, (lo + hi) / 2.0)
        src_lo = max(0.0, lo + local)
        pre = max(0.0, -(lo + local))
        parts += ["-ss", "%.6f" % src_lo, "-t", "%.6f" % (hi - lo + 1.0), "-i", str(clip_path)]
        chain = "[%d:v]" % kept
        if fps:
            chain += "fps=%g," % float(fps)
        chain += ("tpad=start_mode=clone:start_duration=%.6f:"
                  "stop_mode=clone:stop_duration=%.6f," % (pre, hi - lo + 1.0))
        chain += "trim=duration=%.6f,setpts=PTS-STARTPTS[s%d]" % (hi - lo, kept)
        chains.append(chain)
        kept += 1
    if not kept:
        return retime_clip(clip_path, out_path, offset_s=result.get("offset_s", 0.0),
                           duration=duration, fps=fps, crf=crf, preset=preset)
    chains.append("%sconcat=n=%d:v=1:a=0[vout]"
                  % ("".join("[s%d]" % i for i in range(kept)), kept))
    ffmpeg([*parts, "-filter_complex", ";".join(chains), "-map", "[vout]", "-an",
            "-c:v", "libx264", "-preset", preset, "-crf", str(int(crf)),
            "-pix_fmt", "yuv420p", "-t", "%.6f" % duration, str(out_path)])
    return str(out_path)


def retime_plan(plan: dict, *, model: str = "small.en", device: str = "auto",
                write_clips: bool = False, work_dir: str | Path | None = None,
                reuse_slice_words: bool = True, fps: float | None = None,
                on_progress=None) -> dict:
    """Measure every rendered cut and store its offset on the plan.

    With ``write_clips`` the shift is baked into new clip files; otherwise it is
    applied for free when the timeline is assembled.
    """
    cuts = sorted(plan.get("cuts") or [], key=lambda c: int(c.get("idx", 0)))
    work = Path(work_dir) if work_dir else None
    measured = 0
    for cut in cuts:
        clip = cut.get("clip_path")
        if not clip or not Path(clip).exists():
            cut.setdefault("retime", {"method": "skipped", "reason": "no clip"})
            cut.setdefault("offset_s", 0.0)
            continue
        if not cut.get("words"):
            # Nothing is sung in a b-roll cut, so there is nothing to align.
            cut["retime"] = {"method": "skipped", "reason": "no words"}
            cut["offset_s"] = 0.0
            continue
        slice_path = cut.get("slice_path")
        have_slice = bool(slice_path and Path(slice_path).exists())
        # The planner already knows when each word is sung in the fed slice, so
        # by default only the clip needs transcribing.
        slice_words = None if (have_slice and not reuse_slice_words) \
            else expected_slice_words(cut)
        try:
            result = measure_clip(clip, slice_path or "(planned word times)",
                                  model=model, device=device, slice_words=slice_words)
        except Exception as exc:                      # noqa: BLE001 - keep going
            cut["retime"] = {"method": "failed", "reason": "%s: %s"
                             % (type(exc).__name__, exc)}
            cut["offset_s"] = 0.0
            continue
        result.pop("clip_words", None)
        if result.get("segments"):
            owned_slice = float(cut.get("owns_start", cut.get("audio_start", 0.0))) - float(cut.get("audio_start", 0.0))
            owner = next((s for s in result["segments"]
                          if float(s["slice_start"]) <= owned_slice and
                          (s["slice_end"] is None or owned_slice < float(s["slice_end"]))),
                         result["segments"][-1])
            result["offset_s"] = float(owner["offset_s"])
        cut["retime"] = result
        cut["offset_s"] = float(result.get("offset_s", 0.0))
        measured += 1
        if write_clips:
            target = (work or Path(clip).parent) / ("cut_%04d_retimed.mp4" % cut["idx"])
            cut["retimed_path"] = retime_clip(clip, target, result=result, fps=fps)
            cut["residual_offset_s"] = 0.0
        if on_progress:
            on_progress(cut)
    plan["retime_summary"] = {
        "measured": measured,
        "cuts": len(cuts),
        "mean_abs_offset_s": round(statistics.fmean(
            [abs(float(c.get("offset_s", 0.0))) for c in cuts]), 4) if cuts else 0.0,
        "baked": bool(write_clips),
    }
    return plan


def offset_table(plan: dict) -> str:
    """Per-cut offset report for node display and the CLI."""
    rows = ["idx  offset   method     anchors  spread  line"]
    rows.append("-" * len(rows[0]))
    for cut in sorted(plan.get("cuts") or [], key=lambda c: int(c.get("idx", 0))):
        result = cut.get("retime") or {}
        rows.append("%3d  %+6.3f  %-10s %7s  %6s  %s" % (
            int(cut.get("idx", 0)), float(cut.get("offset_s", 0.0) or 0.0),
            str(result.get("method", "-"))[:10],
            result.get("n_anchors", "-"), result.get("spread_s", "-"),
            " ".join(str(cut.get("line_text", "")).split())[:40]))
    summary = plan.get("retime_summary") or {}
    rows.append("-" * len(rows[0]))
    rows.append("measured %s/%s cuts, mean |offset| %.3fs%s" % (
        summary.get("measured", 0), summary.get("cuts", 0),
        summary.get("mean_abs_offset_s", 0.0),
        ", baked into clips" if summary.get("baked") else ", applied at assembly"))
    return "\n".join(rows)
