"""Create a short music video from two references and one sentence."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import re
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))  # run from a git checkout without installing

from singsync import align, assemble, cutplan, idea, retime, slicer
from singsync.providers.comfycloud import ComfyCloud, build_h3_r2v_prompt


def _save_json(path, value):
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def _output_item(outputs, key):
    for value in outputs.values():
        items = value.get(key) or []
        if items: return items[0]
    raise RuntimeError(f"Comfy Cloud job returned no {key} output")


def _duration(path):
    from singsync.ffmpeg_util import media_duration
    return media_duration(path)


def _shot_for(cut, shots):
    words = cut.get("words") or []
    if words:
        line = int(words[0].get("line_idx", 0))
        for shot in shots:
            lines = shot.get("lines", [])
            if isinstance(lines, str): lines = [lines]
            if any(str(x).strip() == str(cut.get("line_text", "")).strip() for x in lines): return shot
            if line < len(shots): return shots[min(line // 2, len(shots)-1)]
    return shots[int(cut.get("idx", 0)) % len(shots)]


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--character", required=True); p.add_argument("--scene", required=True)
    p.add_argument("--idea", required=True); p.add_argument("--seconds", type=float, default=30)
    p.add_argument("--turbo", action="store_true"); p.add_argument("--work", default="idea-video")
    p.add_argument("--seed", type=int, default=1)
    p.add_argument("--vibe", choices=sorted(idea.VIBES), help="mini preset for music style + visual energy")
    p.add_argument("--song-model", choices=["xl", "turbo"], default="xl")
    p.add_argument("--separate", action="store_true", help="vocal stem via Demucs on Comfy Cloud; slices + timing use the vocal")
    p.add_argument("--look", default="", help="fixed outfit/appearance text repeated in every shot")
    p.add_argument("--plan-only", action="store_true", help="stop after plan + song + cuts (no video renders)")
    a = p.parse_args(argv); work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    character, scene = Path(a.character), Path(a.scene)
    client = ComfyCloud(work_dir=work)

    plan_path = work / "plan.json"
    if plan_path.exists(): plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        refs = [client.upload(character), client.upload(scene)]
        outputs = client.run(idea.build_director_graph(*refs, idea.apply_vibe(a.idea, a.vibe), a.seed))
        plan = idea.parse_director_json(_output_item(outputs, "text"))
        plan["lyrics_text"] = idea.format_lyrics(plan.get("lyrics", ""))
        _save_json(plan_path, plan)

    song_path = work / "song.mp3"
    if not song_path.exists():
        tags = plan.get("style_tags", "")
        bpm = plan.get("bpm", 120)
        if a.vibe: tags, bpm = idea.VIBES[a.vibe][0] + ", " + str(tags), idea.VIBES[a.vibe][1]
        outputs = client.run(idea.build_song_graph(idea.song_tags(tags), plan["lyrics_text"], a.seconds, a.seed, bpm, model=a.song_model))
        client.download(_output_item(outputs, "audio"), song_path)

    voice_path = song_path
    if a.separate:
        voice_path = work / "vocals.flac"
        if not voice_path.exists():
            outputs = client.run(idea.build_stem_graph(client.upload(song_path)))
            client.download(_output_item(outputs, "audio"), voice_path)

    cuts_path = work / "cuts.json"
    if cuts_path.exists(): cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    else:
        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        # section tags like [Verse 1] are instructions for the song model, not sung words
        sung = "\n".join(l for l in plan["lyrics_text"].splitlines() if l.strip() and not re.match(r"^\s*\[.*\]\s*$", l))
        try: alignment = align.transcribe_words(voice_path, sung, device="cpu", vad_filter=False)  # VAD drops sung words over a full mix
        finally:
            if old_cuda is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else: os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
        words = align.apply_lyrics(alignment["words"], sung)
        alignment["words"] = words
        alignment["lines"] = align.build_alignment(words)["lines"]
        analysis = None
        duration = _duration(song_path)
        cuts = cutplan.plan_cuts(words, analysis, duration=duration, alignment=alignment,
                                 target_owned_s=8, max_audio_s=15, allowed_durations=tuple(range(4, 16)),
                                 headroom_s=0.5)
        slicer.slice_plan(cuts, voice_path, work, fmt="wav")
        # set prompts AFTER slicing: slice_plan writes its own template prompt per cut
        visual = idea.VIBES[a.vibe][2] if a.vibe else ""
        for cut in cuts["cuts"]:
            shot = dict(_shot_for(cut, plan.get("shots") or [{}]))
            text = re.sub(r"\[[^\]]*\]\s*/?\s*", "", str(cut.get("line_text") or "")).strip(" /")
            if cut.get("kind") == "broll" or not text:
                shot["lines"] = []
                shot["action"] = (shot.get("action") or "") + " Instrumental part: the character dances to the music with mouth closed, not singing."
            else:
                shot["lines"] = [text]   # the exact words in THIS cut's audio slice
            cut["prompt"] = idea.shot_prompt(shot, plan["lyrics_text"], plan.get("character", ""), visual, plan.get("scene", ""), a.look)
        _save_json(work / "alignment.json", alignment); _save_json(cuts_path, cuts)

    if a.plan_only:
        print("plan-only: %d cuts -> %s" % (len(cuts["cuts"]), cuts_path.resolve())); return
    images = [client.upload(character), client.upload(scene)]
    def render(cut):
        clip = work / "renders" / ("cut_%04d.mp4" % cut["idx"])
        if clip.exists(): return str(clip)
        audio = client.upload(cut["slice_path"])
        graph = build_h3_r2v_prompt(cut["prompt"], images, audio, cut["duration"], turbo=a.turbo,
                                    seed=a.seed+cut["idx"], ref_image_size="max")
        outputs = client.run(graph)
        return str(client.download(_output_item(outputs, "images"), clip))
    with ThreadPoolExecutor(max_workers=4) as pool:
        paths = list(pool.map(render, cuts["cuts"]))
    for cut, path in zip(cuts["cuts"], paths): cut["clip_path"] = path
    _save_json(cuts_path, cuts)
    if not (work / "retimed.json").exists():
        retime.retime_plan(cuts, device="cpu"); _save_json(work / "retimed.json", cuts)
    else: cuts = json.loads((work / "retimed.json").read_text(encoding="utf-8"))
    video = work / "music_video.mp4"
    if not video.exists():
        report = assemble.assemble(cuts, song_path, video, captions=True, work_dir=work)
    else: report = {"path": str(video)}
    _save_json(work / "report.json", {"video":report,"cuts":cuts.get("cuts"),"plan":str(plan_path)})
    print(video.resolve())


if __name__ == "__main__": main()
