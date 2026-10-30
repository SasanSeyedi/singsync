#!/usr/bin/env python3
"""Run the SingSync chain from the command line, without ComfyUI.

    analyze -> plan -> slice -> render -> retime -> assemble

Three ways to run it:

* ``--provider kie``   render the cuts for real through kie.ai Seedance
* ``--dry-run``        fake every render with an lavfi slate carrying the cut's
                       own vocal slice, delayed by a known offset, so retime and
                       assembly are exercised end to end on a real song
* ``--demo``           as ``--dry-run`` but the song is generated too, so the
                       whole pipeline can be smoke-tested with no inputs at all

Examples::

    python scripts/run_pipeline.py --demo --work-dir /tmp/singsync-demo
    python scripts/run_pipeline.py --song song.mp3 --vocals vocals.wav \\
        --lyrics lyrics.txt --dry-run --captions --work-dir out/
    python scripts/run_pipeline.py --plan out/plan.json --stages assemble
"""
from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from singsync import align, assemble, beats, captions, cutplan, retime, slicer  # noqa: E402
from singsync.ffmpeg_util import ffmpeg, have_ffmpeg  # noqa: E402

STAGES = ["analyze", "plan", "slice", "render", "retime", "assemble"]

DEMO_LINES = [
    "I walk the wire over bright water",
    "counting every light below me",
    "you said the morning would be kinder",
    "but the morning never answered",
    "so I keep the engine running",
    "and the radio talks to nobody",
    "hold the line until it breaks",
    "hold the line until it breaks again",
]


def log(message: str) -> None:
    print("[singsync] %s" % message, flush=True)


# --------------------------------------------------------------------------- #
# demo inputs
# --------------------------------------------------------------------------- #

def make_demo_song(path: Path, duration: float, bpm: float = 120.0) -> str:
    """A tone plus a click track, so beat tracking has something to find."""
    period = 60.0 / bpm
    click = ("aevalsrc='0.7*sin(2*PI*90*t)*exp(-14*mod(t,%(p).6f))"
             "+0.25*sin(2*PI*1800*t)*exp(-60*mod(t,%(p).6f))':s=44100:d=%(d).3f"
             % {"p": period, "d": duration})
    pad = ("aevalsrc='0.18*sin(2*PI*220*t)+0.12*sin(2*PI*277.2*t)"
           "+0.1*sin(2*PI*330*t)':s=44100:d=%.3f" % duration)
    path.parent.mkdir(parents=True, exist_ok=True)
    ffmpeg(["-f", "lavfi", "-i", click, "-f", "lavfi", "-i", pad,
            "-filter_complex", "[0:a][1:a]amix=inputs=2:normalize=0[a]",
            "-map", "[a]", "-ac", "2", "-ar", "44100", "-c:a", "pcm_s16le",
            "-t", "%.3f" % duration, str(path)])
    return str(path)


def make_demo_words(lines, *, start=2.0, word_s=0.42, gap_s=0.1, line_gap_s=0.6,
                    instrumental_after=3, instrumental_s=8.0):
    """Word timings for the demo song's imaginary vocal."""
    words = []
    t = float(start)
    for line_idx, line in enumerate(lines):
        tokens = line.split()
        for i, token in enumerate(tokens):
            words.append({"word": token, "start": round(t, 4),
                          "end": round(t + word_s, 4), "line_idx": line_idx,
                          "line_end": i + 1 == len(tokens)})
            t += word_s + gap_s
        t += line_gap_s - gap_s
        if line_idx == instrumental_after:
            t += instrumental_s
    return words


# --------------------------------------------------------------------------- #
# fake renders
# --------------------------------------------------------------------------- #

def fake_offsets(plan: dict, spread_s: float, seed: int, explicit: str = "") -> dict:
    """The lip-sync error each fake render will be given, per cut index."""
    if explicit.strip():
        values = [float(v) for v in explicit.replace(";", ",").split(",") if v.strip()]
        return {int(c["idx"]): values[n % len(values)]
                for n, c in enumerate(plan["cuts"])}
    rng = random.Random(seed)
    return {int(c["idx"]): round(rng.uniform(-spread_s, spread_s), 3)
            for c in plan["cuts"]}


def fake_render(cut: dict, out_path: Path, offset_s: float, *, size: str = "640x360",
                fps: float = 24.0) -> str:
    """An lavfi slate the length of the cut's clip bucket.

    The cut's own vocal slice is muxed in, delayed by ``offset_s`` -- that is the
    error a real video model makes, and what retime has to measure back out.
    ``testsrc2`` prints a running timecode, so a wrong trim is visible by eye.
    """
    duration = float(cut["duration"])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    inputs = ["-f", "lavfi", "-i",
              "testsrc2=size=%s:rate=%g:duration=%.3f" % (size, fps, duration)]
    chains = ["[0:v]hue=h=%d:s=0.8[v]" % ((int(cut.get("idx", 0)) * 47) % 360)]
    maps = ["-map", "[v]"]
    slice_path = cut.get("slice_path")
    if slice_path and Path(slice_path).exists():
        inputs += ["-i", str(slice_path)]
        chain = "[1:a]"
        if offset_s > 1e-4:
            chain += "adelay=%d:all=1," % int(round(offset_s * 1000.0))
        elif offset_s < -1e-4:
            chain += "atrim=start=%.4f,asetpts=N/SR/TB," % (-offset_s)
        chains.append(chain + "apad[a]")
        maps += ["-map", "[a]", "-c:a", "aac", "-b:a", "128k"]
    ffmpeg([*inputs, "-filter_complex", ";".join(chains), *maps,
            "-c:v", "libx264", "-preset", "veryfast", "-crf", "28",
            "-pix_fmt", "yuv420p", "-r", "%g" % fps, "-t", "%.6f" % duration,
            str(out_path)])
    return str(out_path)


def simulated_clip_words(cut: dict, offset_s: float, rng: random.Random, *,
                         jitter_s: float = 0.05, dropout: float = 0.15,
                         extra_chance: float = 0.25):
    """Stand in for transcribing a fake clip when faster-whisper is absent.

    The fake clip really does carry its vocal slice delayed by ``offset_s``, so
    this is what a transcript of it would look like: the same words at
    ``slice_t + offset``, with timing jitter, dropped words and the odd extra.
    It exercises the matching and median, not whisper itself.
    """
    words = []
    for word in retime.expected_slice_words(cut):
        if rng.random() < dropout:
            continue
        jitter = rng.uniform(-jitter_s, jitter_s)
        words.append({"word": word["word"],
                      "start": round(word["start"] + offset_s + jitter, 4),
                      "end": round(word["end"] + offset_s + jitter, 4)})
    if words and rng.random() < extra_chance:
        at = rng.randrange(len(words))
        words.insert(at, {"word": "yeah", "start": words[at]["start"] - 0.05,
                          "end": words[at]["start"]})
    return words


# --------------------------------------------------------------------------- #
# stages
# --------------------------------------------------------------------------- #

def stage_analyze(args, work: Path) -> dict:
    if args.demo:
        words = make_demo_words(DEMO_LINES)
        duration = round(words[-1]["end"] + 4.0, 3)
        song = make_demo_song(work / "demo_song.wav", duration)
        log("demo song generated: %s (%.2fs)" % (song, duration))
        analysis = beats.analyse(song)
        log("tempo %.1f BPM (confidence %.2f), %d sections"
            % (analysis["tempo_bpm"], analysis["tempo_confidence"],
               len(analysis["sections"])))
        alignment = align.build_alignment(
            words, "\n".join(DEMO_LINES),
            meta={"source": song, "model": "demo", "language": "en"})
        vocal = song
    else:
        song = str(Path(args.song).expanduser())
        if not Path(song).exists():
            raise SystemExit("song not found: %s" % song)
        vocal = str(Path(args.vocals).expanduser()) if args.vocals else song
        if args.vocals and not Path(vocal).exists():
            raise SystemExit("vocal stem not found: %s" % vocal)
        log("analysing %s" % song)
        analysis = beats.analyse(song)
        duration = analysis["duration"]
        log("%.2fs, tempo %.1f BPM (confidence %.2f), %d beats, %d sections"
            % (duration, analysis["tempo_bpm"], analysis["tempo_confidence"],
               len(analysis["beats"]), len(analysis["sections"])))
        lyrics = ""
        if args.lyrics:
            lyrics_path = Path(args.lyrics).expanduser()
            lyrics = lyrics_path.read_text(encoding="utf-8") if lyrics_path.exists() \
                else args.lyrics
        if args.no_transcribe:
            alignment = {"words": [], "lines": [], "lyrics_used": False,
                         "lyric_lines": []}
            log("transcription skipped: every cut will be instrumental")
        else:
            log("transcribing the vocal with faster-whisper (%s)" % args.whisper_model)
            alignment = align.transcribe_words(vocal, lyrics or None,
                                               model=args.whisper_model,
                                               device=args.device)
            if lyrics:
                alignment["words"] = align.apply_lyrics(alignment.get("words", []), lyrics)
            log("%d words, %d lines%s" % (
                len(alignment["words"]), len(alignment["lines"]),
                " aligned to the supplied lyrics" if alignment.get("lyrics_used")
                else ""))

    plan = {
        "run_id": work.name,
        "work_dir": str(work),
        "song_path": song,
        "vocal_path": vocal,
        "duration": duration,
        "analysis": analysis,
        "alignment": alignment,
        "cuts": [],
        "config": {},
    }
    return plan


def stage_plan(args, plan: dict) -> dict:
    alignment = plan.get("alignment") or {}
    built = cutplan.plan_cuts(
        alignment.get("words", []), plan.get("analysis") or {},
        duration=plan.get("duration"), alignment=alignment,
        snap=args.snap_prefer != "none",
        target_owned_s=args.target, overlap_words=args.overlap_words,
        max_audio_s=args.max_clip, allowed_durations=args.durations,
        headroom_s=args.headroom,
        snap_tol_s=args.snap_tolerance,
        snap_prefer="beat" if args.snap_prefer == "beat" else "downbeat",
        gap_broll_s=args.broll_gap)
    plan.update({"cuts": built["cuts"], "config": built["config"],
                 "duration": built["duration"],
                 "beat_snap_log": built.get("beat_snap_log", [])})
    problems = cutplan.validate_plan(plan, alignment.get("words", []))
    if problems:
        raise SystemExit("cut plan is not valid:\n  " + "\n  ".join(problems))
    print(cutplan.cut_sheet(plan))
    log("%d boundaries snapped to beats" % len(plan.get("beat_snap_log", [])))
    return plan


def stage_slice(args, plan: dict, work: Path) -> dict:
    source = plan.get("vocal_path") or plan.get("song_path")
    slicer.slice_plan(plan, source, work, style=args.style, character=args.character,
                      scene=args.scene,
                      template=(args.prompt_template or "").strip() or None,
                      fmt=args.slice_format)
    log("%d slices written to %s" % (plan["slices"]["count"], plan["slices"]["dir"]))
    if args.show_prompts:
        print(slicer.prompt_sheet(plan))
    return plan


def stage_render(args, plan: dict, work: Path) -> dict:
    if args.provider == "kie":
        from singsync.providers.kie import KieSeedance

        if not args.character_image:
            raise SystemExit("--character-image is required with --provider kie")
        client = KieSeedance(work_dir=work)
        clips = work / "clips"
        clips.mkdir(parents=True, exist_ok=True)
        for cut in plan["cuts"]:
            out = clips / ("cut_%04d.mp4" % cut["idx"])
            log("rendering cut %d (%gs)" % (cut["idx"], cut["duration"]))
            task = client.create_reference_task(
                cut["prompt"], [Path(args.character_image)], Path(cut["slice_path"]),
                int(cut["duration"]), resolution=args.resolution)
            client.download(client.wait(task), out)
            cut["clip_path"] = str(out)
        return plan

    # Dry run: slates that carry the fed slice, delayed by a known error.
    offsets = fake_offsets(plan, args.fake_offset_spread, args.seed, args.fake_offsets)
    clips = work / "clips"
    for cut in plan["cuts"]:
        offset = offsets[int(cut["idx"])]
        out = clips / ("cut_%04d.mp4" % cut["idx"])
        fake_render(cut, out, offset, size=args.fake_size, fps=args.fps)
        cut["clip_path"] = str(out)
        cut["injected_offset_s"] = offset
    plan["dry_run"] = {"injected_offsets": offsets, "size": args.fake_size}
    log("%d slates rendered to %s (injected offsets %.2f..%.2fs)"
        % (len(plan["cuts"]), clips, min(offsets.values()), max(offsets.values())))
    return plan


def _whisper_available() -> bool:
    try:
        import faster_whisper  # noqa: F401
        return True
    except ImportError:
        return False


def stage_retime(args, plan: dict, work: Path) -> dict:
    mode = args.retime
    if mode == "auto":
        if _whisper_available():
            mode = "whisper"
        elif plan.get("dry_run"):
            mode = "simulated"
            log("faster-whisper is not installed: simulating the clip transcripts "
                "(the offsets are still measured by estimate_offset)")
        else:
            mode = "skip"
            log("faster-whisper is not installed: skipping retime, offsets stay 0")

    if mode == "skip":
        for cut in plan["cuts"]:
            cut["offset_s"] = 0.0
            cut["retime"] = {"method": "skipped", "reason": "retime disabled"}
        plan["retime_summary"] = {"measured": 0, "cuts": len(plan["cuts"]),
                                  "mean_abs_offset_s": 0.0, "baked": False}
        return plan

    if mode == "whisper":
        log("measuring per-clip lip-sync offsets with faster-whisper (%s)"
            % args.whisper_model)
        retime.retime_plan(plan, model=args.whisper_model, device=args.device,
                           write_clips=args.bake_retime, work_dir=work / "retimed",
                           fps=args.fps)
    else:
        rng = random.Random(args.seed + 1)
        measured = 0
        for cut in plan["cuts"]:
            if not cut.get("words"):
                cut["retime"] = {"method": "skipped", "reason": "no words"}
                cut["offset_s"] = 0.0
                continue
            injected = float(cut.get("injected_offset_s", 0.0))
            result = retime.estimate_offset(
                simulated_clip_words(cut, injected, rng),
                retime.expected_slice_words(cut))
            result["simulated"] = True
            cut["retime"] = result
            cut["offset_s"] = float(result["offset_s"])
            measured += 1
            if args.bake_retime:
                cut["retimed_path"] = retime.retime_clip(
                    cut["clip_path"], work / "retimed" / ("cut_%04d.mp4" % cut["idx"]),
                    result=result, fps=args.fps)
                cut["residual_offset_s"] = 0.0
        import statistics
        plan["retime_summary"] = {
            "measured": measured, "cuts": len(plan["cuts"]), "simulated": True,
            "mean_abs_offset_s": round(statistics.fmean(
                [abs(float(c.get("offset_s", 0.0))) for c in plan["cuts"]]), 4),
            "baked": bool(args.bake_retime)}

    print(retime.offset_table(plan))
    if plan.get("dry_run"):
        print(_offset_accuracy(plan))
    return plan


def _offset_accuracy(plan: dict) -> str:
    """Compare what the dry run injected with what retime measured."""
    rows = ["idx  injected  measured  error"]
    rows.append("-" * len(rows[0]))
    errors = []
    for cut in plan["cuts"]:
        if not cut.get("words"):
            continue
        injected = float(cut.get("injected_offset_s", 0.0))
        measured = float(cut.get("offset_s", 0.0))
        errors.append(abs(measured - injected))
        rows.append("%3d  %+8.3f  %+8.3f  %+6.3f"
                    % (cut["idx"], injected, measured, measured - injected))
    rows.append("-" * len(rows[0]))
    if errors:
        rows.append("mean |error| %.3fs, worst %.3fs over %d cuts"
                    % (sum(errors) / len(errors), max(errors), len(errors)))
    return "\n".join(rows)


def stage_assemble(args, plan: dict, work: Path) -> dict:
    out = work / args.out_name
    log("assembling %d cuts at %g fps" % (len(plan["cuts"]), args.fps))
    print(assemble.assembly_report(plan, args.fps))
    report = assemble.assemble(plan, plan["song_path"], out, fps=args.fps,
                               captions=args.captions,
                               caption_style=args.caption_style,
                               crf=args.crf, preset=args.preset, work_dir=work)
    if args.fcpxml:
        report["fcpxml"] = assemble.write_fcpxml(plan, work / "singsync.fcpxml",
                                                 fps=args.fps)
        log("edit list written to %s" % report["fcpxml"])
    log("video: %s" % report["path"])
    log("length %.3fs vs song %.3fs (%.2f frames of drift)"
        % (report["actual_duration"], report["song_duration"], report["drift_frames"]))
    if abs(report["drift_frames"]) > 1.0:
        log("WARNING: more than one frame of drift")
    return plan


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #

def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="run_pipeline.py", description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter)
    io_group = parser.add_argument_group("input")
    io_group.add_argument("--song", help="song audio (the master mix)")
    io_group.add_argument("--vocals", help="isolated vocal stem (recommended)")
    io_group.add_argument("--lyrics", help="lyrics file, or the lyrics themselves")
    io_group.add_argument("--plan", help="resume from a plan.json")
    io_group.add_argument("--work-dir", help="where runs are written "
                                             "(default ./output/singsync/<run>)")
    io_group.add_argument("--demo", action="store_true",
                          help="generate a song and a vocal timing, then dry-run")
    io_group.add_argument("--stages", default="all",
                          help="comma-separated subset of %s" % ",".join(STAGES))

    plan_group = parser.add_argument_group("cut plan")
    plan_group.add_argument("--target", type=float, default=9.0,
                            help="target owned seconds per cut (default 9)")
    plan_group.add_argument("--overlap-words", type=int, default=2,
                            help="shared words fed into each neighbour (default 2)")
    plan_group.add_argument("--max-clip", type=float, default=12.0,
                            help="renderer limit on the fed audio span (default 12)")
    plan_group.add_argument("--durations", default="5,10,15",
                            help="allowed clip durations (default 5,10,15)")
    plan_group.add_argument("--headroom", type=float, default=3.0,
                            help="extra clip seconds after fed audio to avoid freeze (default 3)")
    plan_group.add_argument("--snap-tolerance", type=float, default=0.25)
    plan_group.add_argument("--snap-prefer", default="downbeat",
                            choices=["downbeat", "beat", "none"])
    plan_group.add_argument("--broll-gap", type=float, default=4.0,
                            help="vocal gap that becomes a b-roll cut (default 4)")

    prompt_group = parser.add_argument_group("prompts")
    prompt_group.add_argument("--character", default="")
    prompt_group.add_argument("--scene", default="")
    prompt_group.add_argument("--style", default="")
    prompt_group.add_argument("--prompt-template", default="",
                              help="fields: style, character, scene, line_text, "
                                   "vibe, section, energy, idx, duration")
    prompt_group.add_argument("--slice-format", default="wav", choices=["wav", "mp3"])
    prompt_group.add_argument("--show-prompts", action="store_true")

    render_group = parser.add_argument_group("render")
    render_group.add_argument("--provider", default="none", choices=["none", "kie"])
    render_group.add_argument("--dry-run", action="store_true",
                              help="fake the renders with lavfi slates")
    render_group.add_argument("--character-image", help="reference image (--provider kie)")
    render_group.add_argument("--resolution", default="480p", choices=["480p", "720p"])
    render_group.add_argument("--fake-size", default="640x360")
    render_group.add_argument("--fake-offset-spread", type=float, default=1.2,
                              help="dry-run lip-sync error range, +/- seconds")
    render_group.add_argument("--fake-offsets", default="",
                              help="explicit dry-run offsets, e.g. '0.8,-0.4,1.7'")
    render_group.add_argument("--seed", type=int, default=7)

    out_group = parser.add_argument_group("retime and output")
    out_group.add_argument("--retime", default="auto",
                           choices=["auto", "whisper", "simulated", "skip"])
    out_group.add_argument("--bake-retime", action="store_true",
                           help="write shifted clip files instead of shifting at "
                                "assembly time")
    out_group.add_argument("--whisper-model", default="small.en")
    out_group.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    out_group.add_argument("--no-transcribe", action="store_true")
    out_group.add_argument("--fps", type=float, default=24.0)
    out_group.add_argument("--captions", action="store_true")
    out_group.add_argument("--caption-style", default=captions.DEFAULT_STYLE,
                           choices=captions.style_names())
    out_group.add_argument("--crf", type=int, default=18)
    out_group.add_argument("--preset", default="medium")
    out_group.add_argument("--fcpxml", action="store_true")
    out_group.add_argument("--out-name", default="singsync.mp4")
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    if not have_ffmpeg():
        raise SystemExit("ffmpeg and ffprobe are required (set SINGSYNC_FFMPEG)")
    if args.demo:
        args.dry_run = True
    if not (args.song or args.plan or args.demo):
        raise SystemExit("give --song, --plan or --demo (see --help)")
    if args.provider == "none" and not args.dry_run and "render" in _wanted(args):
        raise SystemExit("choose --provider kie to render, or --dry-run to fake it")

    stages = _wanted(args)
    if args.plan:
        plan = cutplan.load_plan(args.plan)
        work = Path(args.work_dir) if args.work_dir else Path(
            plan.get("work_dir") or Path(args.plan).parent)
    else:
        work = Path(args.work_dir) if args.work_dir else (
            Path.cwd() / "output" / "singsync" / time.strftime("%Y%m%d-%H%M%S"))
        plan = None
    work.mkdir(parents=True, exist_ok=True)
    log("work dir: %s" % work)

    started = time.time()
    for stage in STAGES:
        if stage not in stages:
            continue
        log("=== %s ===" % stage)
        if stage == "analyze":
            plan = stage_analyze(args, work)
        elif plan is None:
            raise SystemExit("stage %r needs a plan: run analyze or pass --plan" % stage)
        elif stage == "plan":
            plan = stage_plan(args, plan)
        elif stage == "slice":
            plan = stage_slice(args, plan, work)
        elif stage == "render":
            plan = stage_render(args, plan, work)
        elif stage == "retime":
            plan = stage_retime(args, plan, work)
        elif stage == "assemble":
            plan = stage_assemble(args, plan, work)
        plan["work_dir"] = str(work)
        cutplan.save_plan(plan, work / "plan.json")

    log("done in %.1fs; plan at %s" % (time.time() - started, work / "plan.json"))
    return 0


def _wanted(args) -> list[str]:
    if args.stages.strip().lower() in ("", "all"):
        return list(STAGES)
    wanted = [s.strip() for s in args.stages.split(",") if s.strip()]
    unknown = [s for s in wanted if s not in STAGES]
    if unknown:
        raise SystemExit("unknown stage(s): %s (choose from %s)"
                         % (", ".join(unknown), ", ".join(STAGES)))
    return wanted


if __name__ == "__main__":
    sys.exit(main())
