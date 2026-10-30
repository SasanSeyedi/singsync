"""ComfyUI wrappers around the SingSync core.

Nothing heavy happens here at import time: the modules that need numpy, ffmpeg
or faster-whisper are imported inside the node functions, so a missing optional
dependency shows up as a clear error on the node that needs it rather than
stopping ComfyUI from starting.

The nodes pass a ``SINGSYNC_PLAN``: a plain JSON-serialisable dict holding the
beat analysis, the word alignment, the cuts, and paths to the files written
under ``<ComfyUI output>/singsync/<run_id>/``.
"""
from __future__ import annotations

import copy
import json
import os
import time
import wave
from pathlib import Path

WHISPER_MODELS = ["small.en", "base.en", "tiny.en", "medium.en", "small", "base",
                  "medium", "large-v3", "large-v3-turbo"]
DEVICES = ["auto", "cpu", "cuda"]
X264_PRESETS = ["medium", "fast", "veryfast", "ultrafast", "slow"]

try:  # the preset list is only text; fall back if the module cannot be read
    from .captions import style_names as _style_names
    CAPTION_STYLES = _style_names()
except Exception:  # pragma: no cover - defensive
    CAPTION_STYLES = ["bold_bottom", "bold_centre", "boxed_bottom"]


# --------------------------------------------------------------------------- #
# work dir and plan plumbing
# --------------------------------------------------------------------------- #

def output_root() -> Path:
    """ComfyUI's output directory, or ``SINGSYNC_WORK_DIR`` outside ComfyUI."""
    override = os.environ.get("SINGSYNC_WORK_DIR")
    if override:
        return Path(override)
    try:
        import folder_paths  # provided by ComfyUI
        return Path(folder_paths.get_output_directory())
    except Exception:
        return Path.cwd() / "output"


def new_work_dir(prefix: str = "run") -> tuple[str, Path]:
    """Create ``<output>/singsync/<run_id>`` and return ``(run_id, path)``."""
    run_id = "%s_%s" % (time.strftime("%Y%m%d-%H%M%S"), prefix)
    path = output_root() / "singsync" / run_id
    path.mkdir(parents=True, exist_ok=True)
    return run_id, path


def plan_dir(plan: dict) -> Path:
    """Work directory of a plan, created afresh if the plan has none."""
    work = plan.get("work_dir")
    if work:
        path = Path(work)
        path.mkdir(parents=True, exist_ok=True)
        return path
    run_id, path = new_work_dir()
    plan["run_id"] = run_id
    plan["work_dir"] = str(path)
    return path


def save_plan(plan: dict, name: str = "plan.json") -> str:
    """Persist the plan next to its media, so a run can be inspected or resumed."""
    path = plan_dir(plan) / name
    try:
        path.write_text(json.dumps(plan, indent=2, ensure_ascii=False), encoding="utf-8")
    except (OSError, TypeError):
        return ""
    return str(path)


def _copy(plan) -> dict:
    """Defensive copy: a node must not mutate an upstream node's output."""
    if not isinstance(plan, dict) or "cuts" not in plan:
        raise ValueError("expected a SINGSYNC_PLAN from an upstream SingSync node")
    return copy.deepcopy(plan)


def audio_to_wav(audio: dict, path: str | Path) -> str:
    """Write a ComfyUI ``AUDIO`` dict to a 16-bit wav (no torchaudio needed)."""
    import numpy as np

    waveform = audio["waveform"]
    sample_rate = int(audio.get("sample_rate", 44100))
    array = waveform.detach().cpu().numpy() if hasattr(waveform, "detach") \
        else np.asarray(waveform)
    array = np.asarray(array, dtype=np.float32)
    if array.ndim == 3:
        array = array[0]
    if array.ndim == 1:
        array = array[None, :]
    array = np.clip(array, -1.0, 1.0)
    pcm = (array.T * 32767.0).astype("<i2")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with wave.open(str(path), "wb") as handle:
        handle.setnchannels(int(array.shape[0]))
        handle.setsampwidth(2)
        handle.setframerate(sample_rate)
        handle.writeframes(pcm.tobytes())
    return str(path)


def _resolve_audio(path: str, audio, work: Path, name: str) -> str:
    """Prefer an explicit path; otherwise materialise a connected AUDIO input."""
    path = (path or "").strip().strip('"')
    if path:
        if not Path(path).exists():
            raise FileNotFoundError("%s not found: %s" % (name, path))
        return str(Path(path))
    if audio is not None:
        return audio_to_wav(audio, work / ("%s.wav" % name))
    return ""


# --------------------------------------------------------------------------- #
# nodes
# --------------------------------------------------------------------------- #

class SingSyncSongAnalyze:
    """Analyse the song (tempo, beats, sections) and transcribe the vocal."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "song_path": ("STRING", {"default": "", "multiline": False}),
                "lyrics": ("STRING", {"default": "", "multiline": True,
                                      "placeholder": "One lyric line per line "
                                                     "(optional but recommended)"}),
                "whisper_model": (WHISPER_MODELS, {"default": "small.en"}),
                "device": (DEVICES, {"default": "auto"}),
                "transcribe": ("BOOLEAN", {"default": True}),
            },
            "optional": {
                "song": ("AUDIO",),
                "vocal_path": ("STRING", {"default": "", "multiline": False}),
                "vocals": ("AUDIO",),
            },
        }

    RETURN_TYPES = ("SINGSYNC_PLAN", "STRING")
    RETURN_NAMES = ("plan", "summary")
    FUNCTION = "analyze"
    CATEGORY = "SingSync"

    def analyze(self, song_path="", lyrics="", whisper_model="small.en", device="auto",
                transcribe=True, song=None, vocal_path="", vocals=None):
        from . import align, beats

        run_id, work = new_work_dir("analyze")
        song_file = _resolve_audio(song_path, song, work, "song")
        if not song_file:
            raise ValueError("connect an AUDIO input or set song_path")
        vocal_file = _resolve_audio(vocal_path, vocals, work, "vocals") or song_file

        analysis = beats.analyse(song_file)
        alignment = {"words": [], "lines": [], "lyrics_used": False, "lyric_lines": []}
        if transcribe:
            alignment = align.transcribe_words(
                vocal_file, lyrics or None, model=whisper_model, device=device)
            if lyrics:
                alignment["words"] = align.apply_lyrics(alignment.get("words", []), lyrics)

        plan = {
            "run_id": run_id,
            "work_dir": str(work),
            "song_path": song_file,
            "vocal_path": vocal_file,
            "lyrics": lyrics or "",
            "duration": analysis["duration"],
            "analysis": analysis,
            "alignment": alignment,
            "cuts": [],
            "config": {},
        }
        save_plan(plan)
        summary = (
            "song %.2fs | %.1f BPM (confidence %.2f) | %d beats, %d downbeats "
            "(confidence %.2f, weak by nature) | %d sections | %d words, %d lines%s"
            % (analysis["duration"], analysis["tempo_bpm"], analysis["tempo_confidence"],
               len(analysis["beats"]), len(analysis["downbeats"]),
               analysis["downbeat_confidence"], len(analysis["sections"]),
               len(alignment["words"]), len(alignment["lines"]),
               " (aligned to supplied lyrics)" if alignment.get("lyrics_used") else ""))
        return (plan, summary)


class SingSyncCutPlan:
    """Group the words into cuts that own the song between them."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("SINGSYNC_PLAN",),
                "target_seconds": ("FLOAT", {"default": 9.0, "min": 2.0, "max": 30.0,
                                             "step": 0.5}),
                "overlap_words": ("INT", {"default": 2, "min": 0, "max": 8}),
                "max_clip_seconds": ("FLOAT", {"default": 12.0, "min": 2.0, "max": 30.0,
                                               "step": 0.5}),
                "allowed_durations": ("STRING", {"default": "5,10,15"}),
                "headroom_s": ("FLOAT", {"default": 3.0, "min": 0.0, "max": 30.0, "step": 0.5}),
                "snap_tolerance": ("FLOAT", {"default": 0.25, "min": 0.0, "max": 2.0,
                                             "step": 0.05}),
                "snap_prefer": (["downbeat", "beat", "none"], {"default": "downbeat"}),
                "broll_gap_seconds": ("FLOAT", {"default": 4.0, "min": 1.0, "max": 30.0,
                                                "step": 0.5}),
            },
        }

    RETURN_TYPES = ("SINGSYNC_PLAN", "STRING")
    RETURN_NAMES = ("plan", "cut_sheet")
    FUNCTION = "build"
    CATEGORY = "SingSync"

    def build(self, plan, target_seconds=9.0, overlap_words=2, max_clip_seconds=12.0,
              allowed_durations="5,10,15", headroom_s=3.0, snap_tolerance=0.25, snap_prefer="downbeat",
              broll_gap_seconds=4.0):
        from . import cutplan

        plan = _copy(plan)
        alignment = plan.get("alignment") or {}
        built = cutplan.plan_cuts(
            alignment.get("words", []), plan.get("analysis") or {},
            duration=plan.get("duration"), alignment=alignment,
            snap=snap_prefer != "none",
            target_owned_s=target_seconds, overlap_words=overlap_words,
            max_audio_s=max_clip_seconds, allowed_durations=allowed_durations,
            headroom_s=headroom_s,
            snap_tol_s=snap_tolerance,
            snap_prefer="beat" if snap_prefer == "beat" else "downbeat",
            gap_broll_s=broll_gap_seconds)
        plan.update({"cuts": built["cuts"], "config": built["config"],
                     "duration": built["duration"],
                     "beat_snap_log": built.get("beat_snap_log", [])})

        problems = cutplan.validate_plan(plan, alignment.get("words", []))
        if problems:
            raise RuntimeError("cut plan failed its invariants: %s"
                               % "; ".join(problems[:5]))
        save_plan(plan)
        return (plan, cutplan.cut_sheet(plan))


class SingSyncCutSlicer:
    """Write each cut's vocal slice and its render prompt."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("SINGSYNC_PLAN",),
                "character": ("STRING", {"default": "", "multiline": True,
                                         "placeholder": "Who is singing, described "
                                                        "the same way in every cut"}),
                "scene": ("STRING", {"default": "", "multiline": True,
                                     "placeholder": "Where we are"}),
                "style": ("STRING", {"default": "", "multiline": True,
                                     "placeholder": "Look: lens, light, grade"}),
                "prompt_template": ("STRING", {"default": "", "multiline": True,
                                               "placeholder": "blank = default"}),
                "slice_format": (["wav", "mp3"], {"default": "wav"}),
                "use_vocal_stem": ("BOOLEAN", {"default": True}),
            },
        }

    RETURN_TYPES = ("SINGSYNC_PLAN", "STRING", "STRING", "STRING")
    RETURN_NAMES = ("plan", "prompt_sheet", "slice_paths", "prompts")
    OUTPUT_IS_LIST = (False, False, True, True)
    FUNCTION = "slice"
    CATEGORY = "SingSync"

    def slice(self, plan, character="", scene="", style="", prompt_template="",
              slice_format="wav", use_vocal_stem=True):
        from . import slicer

        plan = _copy(plan)
        if not plan.get("cuts"):
            raise ValueError("plan has no cuts - run SingSync Cut Plan first")
        work = plan_dir(plan)
        source = plan.get("vocal_path") if use_vocal_stem else plan.get("song_path")
        source = source or plan.get("song_path")
        if not source:
            raise ValueError("plan has no audio path")

        slicer.slice_plan(plan, source, work, style=style, character=character,
                          scene=scene, template=prompt_template.strip() or None,
                          fmt=slice_format)
        save_plan(plan)
        jobs = slicer.slice_list(plan)
        return (plan, slicer.prompt_sheet(plan),
                [j["slice_path"] for j in jobs], [j["prompt"] for j in jobs])


class SingSyncRetime:
    """Measure how early or late each rendered clip sings, per clip."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("SINGSYNC_PLAN",),
                "whisper_model": (WHISPER_MODELS, {"default": "small.en"}),
                "device": (DEVICES, {"default": "auto"}),
                "write_retimed_clips": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("SINGSYNC_PLAN", "STRING")
    RETURN_NAMES = ("plan", "offsets")
    FUNCTION = "measure"
    CATEGORY = "SingSync"

    def measure(self, plan, whisper_model="small.en", device="auto",
                write_retimed_clips=False):
        from . import retime

        plan = _copy(plan)
        rendered = [c for c in plan.get("cuts", []) if c.get("clip_path")]
        if not rendered:
            raise ValueError("no cut has a clip_path - render the cuts first")
        retime.retime_plan(plan, model=whisper_model, device=device,
                           write_clips=write_retimed_clips,
                           work_dir=plan_dir(plan) / "retimed")
        save_plan(plan)
        return (plan, retime.offset_table(plan))


class SingSyncAssemble:
    """Trim every clip to the range it owns, join, and mux the master song."""

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "plan": ("SINGSYNC_PLAN",),
                "fps": ("FLOAT", {"default": 24.0, "min": 1.0, "max": 120.0,
                                  "step": 1.0}),
                "captions": ("BOOLEAN", {"default": False}),
                "caption_style": (CAPTION_STYLES, {"default": CAPTION_STYLES[0]}),
                "filename": ("STRING", {"default": "singsync.mp4"}),
                "crf": ("INT", {"default": 18, "min": 0, "max": 51}),
                "preset": (X264_PRESETS, {"default": "medium"}),
            },
            "optional": {
                "write_fcpxml": ("BOOLEAN", {"default": False}),
            },
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("video_path", "report")
    FUNCTION = "assemble"
    CATEGORY = "SingSync"
    OUTPUT_NODE = True

    def assemble(self, plan, fps=24.0, captions=False, caption_style="bold_bottom",
                 filename="singsync.mp4", crf=18, preset="medium", write_fcpxml=False):
        from . import assemble as assemble_mod

        plan = _copy(plan)
        work = plan_dir(plan)
        name = Path(filename.strip() or "singsync.mp4").name
        if not name.lower().endswith(".mp4"):
            name += ".mp4"
        out = work / name
        report = assemble_mod.assemble(
            plan, plan["song_path"], out, fps=fps, captions=captions,
            caption_style=caption_style, crf=crf, preset=preset, work_dir=work)
        if write_fcpxml:
            report["fcpxml"] = assemble_mod.write_fcpxml(
                plan, work / (Path(name).stem + ".fcpxml"), fps=fps)
        save_plan(plan)
        text = "%s\n\n%s\n\n%s" % (
            assemble_mod.assembly_report(plan, fps),
            json.dumps(report, indent=2),
            "length %.3fs vs song %.3fs (%.2f frames)" % (
                report["actual_duration"], report["song_duration"],
                report["drift_frames"]))
        return {"ui": {"text": [text]}, "result": (str(out), text)}


NODE_CLASS_MAPPINGS = {
    "SingSyncSongAnalyze": SingSyncSongAnalyze,
    "SingSyncCutPlan": SingSyncCutPlan,
    "SingSyncCutSlicer": SingSyncCutSlicer,
    "SingSyncRetime": SingSyncRetime,
    "SingSyncAssemble": SingSyncAssemble,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "SingSyncSongAnalyze": "SingSync Song Analyze",
    "SingSyncCutPlan": "SingSync Cut Plan",
    "SingSyncCutSlicer": "SingSync Cut Slicer",
    "SingSyncRetime": "SingSync Retime",
    "SingSyncAssemble": "SingSync Assemble",
}

# Providers are separate and optional: a missing or broken provider must not
# take the rest of the pack down with it.
try:
    from .providers.kie_node import (NODE_CLASS_MAPPINGS as _KIE_CLASSES,
                                     NODE_DISPLAY_NAME_MAPPINGS as _KIE_NAMES)
    NODE_CLASS_MAPPINGS.update(_KIE_CLASSES)
    NODE_DISPLAY_NAME_MAPPINGS.update(_KIE_NAMES)
except ImportError:
    pass
except Exception as exc:  # pragma: no cover - defensive
    print("[SingSync] kie.ai provider node not loaded: %s" % exc)

try:
    from .providers.comfycloud_node import (NODE_CLASS_MAPPINGS as _CLOUD_CLASSES,
                                            NODE_DISPLAY_NAME_MAPPINGS as _CLOUD_NAMES)
    NODE_CLASS_MAPPINGS.update(_CLOUD_CLASSES)
    NODE_DISPLAY_NAME_MAPPINGS.update(_CLOUD_NAMES)
except ImportError:
    pass
except Exception as exc:  # pragma: no cover - defensive
    print("[SingSync] Comfy Cloud provider node not loaded: %s" % exc)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
