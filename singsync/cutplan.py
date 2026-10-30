"""Overlap cut planner.

The planner turns word timings into a list of cuts that tile the whole song.
Every cut *owns* a contiguous range ``[owns_start, owns_end)`` -- that is the
part of the final video it provides -- but the audio it is *fed* reaches a few
words into each neighbour. Those shared words are what the assembler trims on,
so a cut can be rendered by a video model that starts singing slightly early or
late and still be joined without clipping a syllable.

Invariants the planner guarantees (see :func:`validate_plan`):

* owned ranges are contiguous, ordered, non-overlapping and cover ``[0, duration]``
* ``audio_start <= owns_start`` and ``owns_end <= audio_end`` for every cut,
  so the owned range is always inside the audio that was fed to the renderer
* ``audio_end - audio_start <= max_audio_s`` (the renderer's limit)
* ``duration`` is the smallest allowed clip length that covers the audio span
"""
from __future__ import annotations

import copy
import json
import math
from pathlib import Path

from . import beats

#: Planner defaults. Overridable per call and exposed on the ComfyUI node.
DEFAULTS = {
    "target_owned_s": 9.0,       # aim for cuts this long
    "max_audio_s": 12.0,         # hard renderer limit on the fed audio span
    "overlap_words": 2,          # shared words fed into each neighbour
    "allowed_durations": (5.0, 10.0, 15.0),
    "gap_broll_s": 4.0,          # longer vocal gap -> instrumental/b-roll cut
    "line_snap_frac": 0.6,       # accept a line end this far from the target
    "snap_tol_s": 0.25,          # beat snapping tolerance
    "snap_prefer": "downbeat",
    "min_owned_s": 2.0,          # merge shorter tail cuts into the previous one
    "min_broll_s": 1.0,          # shorter instrumental stretches get absorbed
}


def parse_durations(spec) -> tuple[float, ...]:
    """Parse ``"5,10,15"`` (or a sequence) into a sorted tuple of floats."""
    if isinstance(spec, str):
        values = [v.strip() for v in spec.replace(";", ",").split(",")]
    else:
        values = list(spec or [])
    out = sorted({float(v) for v in values if str(v).strip()})
    if not out:
        raise ValueError("no allowed clip durations given")
    if out[0] <= 0:
        raise ValueError("clip durations must be positive")
    return tuple(out)


def pick_duration(span_s: float, allowed) -> float:
    """Smallest allowed clip duration that covers ``span_s``."""
    allowed = parse_durations(allowed)
    for value in allowed:
        if value >= span_s - 1e-6:
            return value
    return allowed[-1]


def _config(overrides: dict) -> dict:
    cfg = dict(DEFAULTS)
    for key, value in (overrides or {}).items():
        if value is not None:
            cfg[key] = value
    cfg["allowed_durations"] = parse_durations(cfg["allowed_durations"])
    cfg["overlap_words"] = max(0, int(cfg["overlap_words"]))
    cfg["target_owned_s"] = float(cfg["target_owned_s"])
    cfg["max_audio_s"] = float(cfg["max_audio_s"])
    cfg["gap_broll_s"] = max(0.25, float(cfg["gap_broll_s"]))
    cfg["line_snap_frac"] = max(0.0, float(cfg["line_snap_frac"]))
    # The renderer cannot be fed more than it can output.
    cfg["max_span_s"] = min(cfg["max_audio_s"], cfg["allowed_durations"][-1])
    if cfg["max_span_s"] <= 0:
        raise ValueError("max_audio_s must be positive")
    cfg["edge_pad_s"] = min(1.5, cfg["gap_broll_s"] / 3.0)
    return cfg


def _clean_words(words) -> list[dict]:
    out = []
    for word in words or []:
        try:
            start, end = float(word["start"]), float(word["end"])
        except (KeyError, TypeError, ValueError):
            continue
        out.append({
            "word": str(word.get("word", "")).strip(),
            "start": start,
            "end": max(end, start),
            "line_idx": int(word.get("line_idx", 0) or 0),
            "line_end": bool(word.get("line_end", False)),
        })
    out.sort(key=lambda w: (w["start"], w["end"]))
    return out


# --------------------------------------------------------------------------- #
# regions
# --------------------------------------------------------------------------- #

def _word_runs(words: list[dict], gap_broll_s: float) -> list[tuple[int, int]]:
    """Index ranges of words separated by gaps no longer than ``gap_broll_s``."""
    if not words:
        return []
    runs = []
    start = 0
    for i in range(1, len(words)):
        if words[i]["start"] - words[i - 1]["end"] > gap_broll_s:
            runs.append((start, i - 1))
            start = i
    runs.append((start, len(words) - 1))
    return runs


def _regions(words: list[dict], duration: float, cfg: dict) -> list[dict]:
    """Split ``[0, duration]`` into alternating vocal and instrumental regions."""
    pad = cfg["edge_pad_s"]
    runs = _word_runs(words, cfg["gap_broll_s"])
    if not runs:
        return [{"kind": "broll", "start": 0.0, "end": duration}]

    regions: list[dict] = []
    cursor = 0.0
    for n, (lo, hi) in enumerate(runs):
        run_start, run_end = words[lo]["start"], words[hi]["end"]
        start = max(cursor, run_start - pad)
        if start > cursor:
            regions.append({"kind": "broll", "start": cursor, "end": start})
        else:
            start = cursor
        if n + 1 < len(runs):
            next_start = words[runs[n + 1][0]]["start"]
            end = min(run_end + pad, next_start - pad)
            end = max(end, run_end)
        else:
            end = min(duration, run_end + pad)
            if duration - end <= 1e-9:
                end = duration
        regions.append({"kind": "vocal", "start": start, "end": end, "lo": lo, "hi": hi})
        cursor = end
    if duration - cursor > 1e-9:
        regions.append({"kind": "broll", "start": cursor, "end": duration})

    # Absorb negligible instrumental stretches into the neighbouring vocal region
    # so the plan does not carry half-second cuts no renderer can use.
    merged: list[dict] = []
    for region in regions:
        length = region["end"] - region["start"]
        if region["kind"] == "broll" and length < cfg["min_broll_s"]:
            if merged and merged[-1]["kind"] == "vocal":
                merged[-1]["end"] = region["end"]
                continue
            if len(merged) == 0 and len(regions) > 1:
                nxt = regions[regions.index(region) + 1]
                if nxt["kind"] == "vocal":
                    nxt["start"] = region["start"]
                    continue
        merged.append(region)
    return [r for r in merged if r["end"] - r["start"] > 1e-9]


# --------------------------------------------------------------------------- #
# grouping
# --------------------------------------------------------------------------- #

def _wing_start(words, i, lo, overlap):
    """Start time of the word ``overlap`` words before index ``i``."""
    return words[max(lo, i - overlap)]["start"]


def _wing_end(words, j, hi, overlap):
    """End time of the word ``overlap`` words after index ``j``."""
    return words[min(hi, j + overlap)]["end"]


def _boundary(words, j, hi, region_end):
    """Owned boundary after word ``j``: the middle of the gap that follows."""
    if j >= hi:
        return region_end
    return (words[j]["end"] + words[j + 1]["start"]) / 2.0


def _group_region(words: list[dict], region: dict, cfg: dict) -> list[dict]:
    """Greedily group a vocal region's words into cuts.

    Returns ``[{"lo", "hi", "owns_start", "owns_end"}]`` covering the region.
    """
    lo, hi = region["lo"], region["hi"]
    overlap = cfg["overlap_words"]
    target = cfg["target_owned_s"]
    max_span = cfg["max_span_s"]
    tol = cfg["line_snap_frac"] * target

    groups: list[dict] = []
    i = lo
    owns_start = region["start"]
    while i <= hi:
        feasible: list[tuple[int, float, float]] = []   # (j, owns_end, content_span)
        for j in range(i, hi + 1):
            owns_end = _boundary(words, j, hi, region["end"])
            audio_start = min(owns_start, _wing_start(words, i, lo, overlap))
            audio_end = max(owns_end, _wing_end(words, j, hi, overlap))
            if audio_end - audio_start > max_span + 1e-9 and feasible:
                break
            feasible.append((j, owns_end, words[j]["end"] - words[i]["start"]))

        # Default: the longest group that stays at or under the target length.
        chosen = feasible[0]
        for cand in feasible:
            if cand[2] <= target + 1e-9:
                chosen = cand
            else:
                break
        # Prefer ending on a lyric line when one falls near the target.
        line_cands = [c for c in feasible
                      if words[c[0]]["line_end"] and abs(c[2] - target) <= tol + 1e-9]
        if line_cands:
            chosen = min(line_cands, key=lambda c: abs(c[2] - target))

        groups.append({"lo": i, "hi": chosen[0], "owns_start": owns_start,
                       "owns_end": chosen[1]})
        owns_start = chosen[1]
        i = chosen[0] + 1

    if groups:
        groups[-1]["owns_end"] = region["end"]
    # A stub tail cut is worse than a slightly long neighbour: merge it back when
    # the merged audio span still fits.
    if len(groups) >= 2:
        tail = groups[-1]
        if tail["owns_end"] - tail["owns_start"] < cfg["min_owned_s"]:
            prev = groups[-2]
            audio_start = min(prev["owns_start"], _wing_start(words, prev["lo"], lo, overlap))
            audio_end = max(tail["owns_end"], _wing_end(words, tail["hi"], hi, overlap))
            if audio_end - audio_start <= max_span + 1e-9:
                prev["hi"] = tail["hi"]
                prev["owns_end"] = tail["owns_end"]
                groups.pop()
    return groups


def _split_broll(region: dict, cfg: dict) -> list[dict]:
    """Chop an instrumental region into renderable chunks."""
    length = region["end"] - region["start"]
    parts = max(1, int(math.ceil(length / cfg["max_span_s"] - 1e-9)))
    step = length / parts
    out = []
    for k in range(parts):
        start = region["start"] + k * step
        end = region["end"] if k + 1 == parts else region["start"] + (k + 1) * step
        out.append({"owns_start": start, "owns_end": end})
    return out


# --------------------------------------------------------------------------- #
# plan
# --------------------------------------------------------------------------- #

def _section_at(analysis: dict, time_s: float) -> str:
    for section in (analysis or {}).get("sections", []) or []:
        if float(section.get("start", 0)) <= time_s < float(section.get("end", 0)):
            return str(section.get("label", ""))
    return ""


def _energy_at(analysis: dict, time_s: float) -> float:
    curve = (analysis or {}).get("energy_curve", []) or []
    for point in curve:
        if float(point.get("time", 0)) >= time_s:
            return round(float(point.get("energy", 0.0)), 4)
    return round(float(curve[-1].get("energy", 0.0)), 4) if curve else 0.0


def _line_text(words: list[dict], group_words: list[dict], alignment: dict | None) -> str:
    """Readable lyric text for a cut, preferring the supplied lyric lines."""
    if alignment and alignment.get("lyrics_used"):
        by_idx = {int(l["line_idx"]): str(l.get("text", ""))
                  for l in alignment.get("lyric_lines", []) or []}
        seen, parts = set(), []
        for word in group_words:
            idx = int(word.get("line_idx", 0) or 0)
            if idx in seen:
                continue
            seen.add(idx)
            text = by_idx.get(idx)
            if text:
                parts.append(text)
        if parts:
            return " / ".join(parts)
    return " ".join(w["word"] for w in group_words if w["word"]).strip()


def plan_cuts(words, analysis: dict | None = None, *, duration: float | None = None,
              alignment: dict | None = None, snap: bool = True, **overrides) -> dict:
    """Build the overlap cut plan.

    ``words`` is the ``words`` list of an alignment (see :mod:`singsync.align`);
    ``analysis`` is :func:`singsync.beats.analyse` output, used for the song
    duration, beat snapping and the per-cut vibe brief. Both may be omitted for
    testing, in which case the song is assumed to end with the last word.
    """
    cfg = _config(overrides)
    analysis = analysis or {}
    if alignment is None and isinstance(words, dict):
        alignment, words = words, words.get("words", [])
    clean = _clean_words(words)

    if duration is None:
        duration = analysis.get("duration")
    if duration is None:
        duration = clean[-1]["end"] if clean else 0.0
    duration = float(duration)
    if clean:
        duration = max(duration, clean[-1]["end"])
    if duration <= 0:
        raise ValueError("song duration is unknown (no analysis, no words)")

    cuts: list[dict] = []
    for region in _regions(clean, duration, cfg):
        if region["kind"] == "broll":
            groups = [dict(g, lo=None, hi=None) for g in _split_broll(region, cfg)]
        else:
            groups = _group_region(clean, region, cfg)
        for group in groups:
            cuts.append(_make_cut(clean, region, group, cfg, analysis, alignment))

    for idx, cut in enumerate(cuts):
        cut["idx"] = idx

    plan = {
        "duration": round(duration, 4),
        "cuts": cuts,
        "config": {
            "target_owned_s": cfg["target_owned_s"],
            "max_audio_s": cfg["max_audio_s"],
            "max_span_s": cfg["max_span_s"],
            "overlap_words": cfg["overlap_words"],
            "allowed_durations": list(cfg["allowed_durations"]),
            "gap_broll_s": cfg["gap_broll_s"],
            "line_snap_frac": cfg["line_snap_frac"],
            "snap_tol_s": float(cfg["snap_tol_s"]),
            "snap_prefer": cfg["snap_prefer"],
        },
    }

    if snap and analysis.get("beats"):
        plan, log = beats.snap_cuts_to_beats(
            plan, analysis, clean,
            tol_s=float(cfg["snap_tol_s"]), prefer=cfg["snap_prefer"])
        plan["beat_snap_log"] = log
        _refresh_after_snap(plan, cfg, analysis)
    else:
        plan["beat_snap_log"] = []
    return plan


def _make_cut(words, region, group, cfg, analysis, alignment) -> dict:
    overlap = cfg["overlap_words"]
    owns_start = float(group["owns_start"])
    owns_end = float(group["owns_end"])
    lo, hi = group.get("lo"), group.get("hi")

    if lo is None:
        group_words: list[dict] = []
        before: list[dict] = []
        after: list[dict] = []
        audio_start, audio_end = owns_start, owns_end
    else:
        rlo, rhi = region["lo"], region["hi"]
        group_words = [dict(w) for w in words[lo:hi + 1]]
        before = [dict(w) for w in words[max(rlo, lo - overlap):lo]]
        after = [dict(w) for w in words[hi + 1:min(rhi, hi + overlap) + 1]]
        audio_start = min(owns_start, before[0]["start"] if before else owns_start)
        audio_end = max(owns_end, after[-1]["end"] if after else owns_end)

    # Never feed more than the renderer accepts: drop wing words one at a time.
    while audio_end - audio_start > cfg["max_span_s"] + 1e-9 and (before or after):
        if after and (audio_end - owns_end) >= (owns_start - audio_start):
            after.pop()
            audio_end = max(owns_end, after[-1]["end"] if after else owns_end)
        elif before:
            before.pop(0)
            audio_start = min(owns_start, before[0]["start"] if before else owns_start)
        else:
            after.pop()
            audio_end = max(owns_end, after[-1]["end"] if after else owns_end)

    mid = (owns_start + owns_end) / 2.0
    cut = {
        "idx": 0,
        "kind": "vocal" if group_words else "broll",
        "owns_start": round(owns_start, 4),
        "owns_end": round(owns_end, 4),
        "owns": [round(owns_start, 4), round(owns_end, 4)],
        "audio_start": round(max(0.0, audio_start), 4),
        "audio_end": round(audio_end, 4),
        "duration": pick_duration(audio_end - audio_start, cfg["allowed_durations"]),
        "words": group_words,
        "shared_words_before": before,
        "shared_words_after": after,
        "line_text": _line_text(words, group_words, alignment),
        "line_indices": sorted({int(w.get("line_idx", 0) or 0) for w in group_words}),
        "section": _section_at(analysis, mid),
        "energy": _energy_at(analysis, mid),
    }
    cut["audio_span"] = round(cut["audio_end"] - cut["audio_start"], 4)
    cut["owns_span"] = round(cut["owns_end"] - cut["owns_start"], 4)
    cut["vibe"] = beats.vibe_brief(analysis, cut) if analysis else ""
    return cut


def _refresh_after_snap(plan: dict, cfg: dict, analysis: dict) -> None:
    """Re-derive the fields that a moved owned boundary invalidates."""
    for cut in plan.get("cuts", []):
        owns = cut.get("owns") or [cut["owns_start"], cut["owns_end"]]
        cut["owns_start"], cut["owns_end"] = round(float(owns[0]), 4), round(float(owns[1]), 4)
        cut["owns"] = [cut["owns_start"], cut["owns_end"]]
        # Beat snapping only moves boundaries inside word gaps, but it can move
        # one past a wing-less audio edge; keep the owned range inside the audio.
        cut["audio_start"] = round(max(0.0, min(cut["audio_start"], cut["owns_start"])), 4)
        cut["audio_end"] = round(max(cut["audio_end"], cut["owns_end"]), 4)
        cut["audio_span"] = round(cut["audio_end"] - cut["audio_start"], 4)
        cut["owns_span"] = round(cut["owns_end"] - cut["owns_start"], 4)
        cut["duration"] = pick_duration(cut["audio_span"], cfg["allowed_durations"])
        if analysis:
            cut["vibe"] = beats.vibe_brief(analysis, cut)


# --------------------------------------------------------------------------- #
# validation and reporting
# --------------------------------------------------------------------------- #

def validate_plan(plan: dict, words=None, *, tol: float = 1e-3) -> list[str]:
    """Return a list of invariant violations; empty means the plan is sound."""
    problems: list[str] = []
    cuts = plan.get("cuts") or []
    duration = float(plan.get("duration", 0.0))
    cfg = plan.get("config", {})
    max_span = float(cfg.get("max_span_s", cfg.get("max_audio_s", DEFAULTS["max_audio_s"])))
    allowed = parse_durations(cfg.get("allowed_durations", DEFAULTS["allowed_durations"]))

    if not cuts:
        return ["plan has no cuts"]
    if abs(float(cuts[0]["owns_start"])) > tol:
        problems.append("first cut starts at %.4f, not 0" % cuts[0]["owns_start"])
    if abs(float(cuts[-1]["owns_end"]) - duration) > tol:
        problems.append("last cut ends at %.4f, song is %.4f"
                        % (cuts[-1]["owns_end"], duration))
    for i, cut in enumerate(cuts):
        if cut.get("idx") != i:
            problems.append("cut %d has idx %r" % (i, cut.get("idx")))
        if float(cut["owns_end"]) <= float(cut["owns_start"]):
            problems.append("cut %d owns an empty range" % i)
        if float(cut["audio_start"]) > float(cut["owns_start"]) + tol:
            problems.append("cut %d audio starts after its owned range" % i)
        if float(cut["audio_end"]) < float(cut["owns_end"]) - tol:
            problems.append("cut %d audio ends before its owned range" % i)
        span = float(cut["audio_end"]) - float(cut["audio_start"])
        if span > max_span + tol:
            problems.append("cut %d audio span %.3fs exceeds %.3fs" % (i, span, max_span))
        if float(cut["duration"]) + tol < span:
            problems.append("cut %d duration %.1fs shorter than span %.3fs"
                            % (i, cut["duration"], span))
        if float(cut["duration"]) not in [float(v) for v in allowed]:
            problems.append("cut %d duration %r not in allowed set" % (i, cut["duration"]))
        if i:
            prev_end = float(cuts[i - 1]["owns_end"])
            if abs(prev_end - float(cut["owns_start"])) > tol:
                problems.append("gap/overlap between cuts %d and %d (%.4f vs %.4f)"
                                % (i - 1, i, prev_end, cut["owns_start"]))
    if words:
        clean = _clean_words(words)
        for cut in cuts:
            for word in cut.get("words", []):
                if not (float(cut["audio_start"]) - tol <= float(word["start"])
                        and float(word["end"]) <= float(cut["audio_end"]) + tol):
                    problems.append("cut %d word %r outside its audio span"
                                    % (cut["idx"], word.get("word")))
        owned = sum(1 for _ in clean)
        covered = sum(len(c.get("words", [])) for c in cuts)
        if covered != owned:
            problems.append("cuts own %d words, alignment has %d" % (covered, owned))
    return problems


def cut_sheet(plan: dict) -> str:
    """Fixed-width table of the plan, for node display and the CLI."""
    header = ("idx  kind   owns            span   audio           fed    clip  "
              "section     line")
    rows = [header, "-" * len(header)]
    for cut in plan.get("cuts", []):
        rows.append(
            "%3d  %-6s %6.2f-%6.2f %5.2f  %6.2f-%6.2f %5.2f %5.1f  %-10s  %s" % (
                cut.get("idx", 0), cut.get("kind", ""),
                cut["owns_start"], cut["owns_end"],
                float(cut["owns_end"]) - float(cut["owns_start"]),
                cut["audio_start"], cut["audio_end"],
                float(cut["audio_end"]) - float(cut["audio_start"]),
                float(cut.get("duration", 0)),
                (cut.get("section") or "-")[:10],
                _short(cut.get("line_text", ""), 44),
            ))
    total = sum(float(c["owns_end"]) - float(c["owns_start"]) for c in plan.get("cuts", []))
    rows.append("-" * len(header))
    rows.append("%d cuts, owned total %.2fs, song %.2fs, renderer seconds %.0f"
                % (len(plan.get("cuts", [])), total, float(plan.get("duration", 0)),
                   sum(float(c.get("duration", 0)) for c in plan.get("cuts", []))))
    return "\n".join(rows)


def _short(text: str, width: int) -> str:
    text = " ".join(str(text).split())
    return text if len(text) <= width else text[:width - 1] + "…"


def save_plan(plan: dict, path: str | Path) -> str:
    """Write a plan to ``path`` as JSON; returns the path."""
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    return str(out)


def load_plan(json_or_path) -> dict:
    """Load a plan from a dict, a JSON string or a path to a JSON file."""
    if isinstance(json_or_path, dict):
        return copy.deepcopy(json_or_path)
    text = str(json_or_path)
    if text.lstrip().startswith("{"):
        return json.loads(text)
    return json.loads(Path(text).read_text(encoding="utf-8"))
