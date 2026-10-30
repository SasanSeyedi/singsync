"""Create a short music video from two references and one sentence."""
import argparse
import json
import os
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

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
    a = p.parse_args(argv); work = Path(a.work); work.mkdir(parents=True, exist_ok=True)
    character, scene = Path(a.character), Path(a.scene)
    client = ComfyCloud(work_dir=work)

    plan_path = work / "plan.json"
    if plan_path.exists(): plan = json.loads(plan_path.read_text(encoding="utf-8"))
    else:
        refs = [client.upload(character), client.upload(scene)]
        outputs = client.run(idea.build_director_graph(*refs, a.idea, a.seed))
        plan = idea.parse_director_json(_output_item(outputs, "text"))
        plan["lyrics_text"] = idea.format_lyrics(plan.get("lyrics", ""))
        _save_json(plan_path, plan)

    song_path = work / "song.mp3"
    if not song_path.exists():
        outputs = client.run(idea.build_song_graph(idea.song_tags(plan.get("style_tags", "")), plan["lyrics_text"], a.seconds, a.seed, plan.get("bpm", 120)))
        client.download(_output_item(outputs, "audio"), song_path)

    cuts_path = work / "cuts.json"
    if cuts_path.exists(): cuts = json.loads(cuts_path.read_text(encoding="utf-8"))
    else:
        old_cuda = os.environ.get("CUDA_VISIBLE_DEVICES")
        os.environ["CUDA_VISIBLE_DEVICES"] = ""
        try: alignment = align.transcribe_words(song_path, plan["lyrics_text"], device="cpu")
        finally:
            if old_cuda is None: os.environ.pop("CUDA_VISIBLE_DEVICES", None)
            else: os.environ["CUDA_VISIBLE_DEVICES"] = old_cuda
        words = align.apply_lyrics(alignment["words"], plan["lyrics_text"])
        alignment["words"] = words
        alignment["lines"] = align.build_alignment(words)["lines"]
        analysis = None
        duration = _duration(song_path)
        cuts = cutplan.plan_cuts(words, analysis, duration=duration, alignment=alignment,
                                 target_owned_s=8, max_audio_s=15, allowed_durations=tuple(range(4, 16)),
                                 headroom_s=0.5)
        for cut in cuts["cuts"]:
            cut["prompt"] = idea.shot_prompt(_shot_for(cut, plan.get("shots") or [{}]), plan["lyrics_text"])
        slicer.slice_plan(cuts, song_path, work, fmt="wav")
        _save_json(work / "alignment.json", alignment); _save_json(cuts_path, cuts)

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
