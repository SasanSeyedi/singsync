"""Native ComfyUI graphs and parsing for sentence-led music videos."""
from __future__ import annotations

import json
import re

DIRECTOR_INSTRUCTION = '''Create a music-video plan for about 30 seconds from the two reference images and this idea: {idea!r}. Return ONLY JSON with keys title, style_tags, bpm, character, scene, lyrics, shots. character is one sentence describing image 1 exactly; scene is one sentence describing image 2. lyrics is [verse] with 4 short lines, then [chorus] with 4 short lines, suitable for about 30 seconds. shots has 3-4 objects with lines (exact lyric lines for this shot), action, camera. Camera rules: the camera always stays in FRONT of the character, face and mouth always visible, never behind or orbiting around the back; vary framing: close-up, medium, wide.'''


def build_director_graph(character_name, scene_name, idea, seed=1):
    return {
        "1":{"class_type":"CLIPLoader","inputs":{"clip_name":"qwen3vl_8b_fp8_scaled.safetensors","type":"ideogram4","device":"default"}},
        "2":{"class_type":"LoadImage","inputs":{"image":character_name}},
        "3":{"class_type":"LoadImage","inputs":{"image":scene_name}},
        "4":{"class_type":"ImageBatch","inputs":{"image1":["2",0],"image2":["3",0]}},
        "5":{"class_type":"TextGenerate","inputs":{"clip":["1",0],"prompt":DIRECTOR_INSTRUCTION.format(idea=idea),"max_length":1500,"sampling_mode":"on","sampling_mode.temperature":0.7,"sampling_mode.top_k":64,"sampling_mode.top_p":0.95,"sampling_mode.min_p":0.05,"sampling_mode.repetition_penalty":1.05,"sampling_mode.seed":int(seed),"image":["4",0]}},
        "6":{"class_type":"PreviewAny","inputs":{"source":["5",0]}},
    }


def build_song_graph(tags, lyrics, seconds, seed=1, bpm=120):
    return {
        "1":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"ace_step_1.5_turbo_aio.safetensors"}},
        "2":{"class_type":"ModelSamplingAuraFlow","inputs":{"model":["1",0],"shift":3}},
        "3":{"class_type":"TextEncodeAceStepAudio1.5","inputs":{"clip":["1",1],"tags":tags,"lyrics":lyrics,"seed":int(seed),"bpm":int(bpm),"duration":float(seconds),"timesignature":"4","language":"en","keyscale":"C major","generate_audio_codes":True,"cfg_scale":2.0,"temperature":0.85,"top_p":0.9,"top_k":0,"min_p":0.0}},
        "4":{"class_type":"ConditioningZeroOut","inputs":{"conditioning":["3",0]}},
        "5":{"class_type":"EmptyAceStep1.5LatentAudio","inputs":{"seconds":float(seconds),"batch_size":1}},
        "6":{"class_type":"KSampler","inputs":{"model":["2",0],"seed":int(seed),"steps":8,"cfg":1.0,"sampler_name":"euler","scheduler":"simple","positive":["3",0],"negative":["4",0],"latent_image":["5",0],"denoise":1.0}},
        "7":{"class_type":"VAEDecodeAudio","inputs":{"samples":["6",0],"vae":["1",2]}},
        "8":{"class_type":"SaveAudioMP3","inputs":{"audio":["7",0],"filename_prefix":"singsync/song","quality":"V0"}},
    }


def parse_director_json(value):
    """Accept model text or output records; tolerate fences and surrounding prose."""
    if isinstance(value, dict):
        for key in ("text", "string", "output", "value"):
            if key in value:
                value = value[key]; break
        else:
            return value
    if isinstance(value, (list, tuple)):
        value = value[0] if value else ""
    if isinstance(value, dict): return value
    text = str(value).strip()
    text = re.sub(r"^\s*```(?:json)?\s*|\s*```\s*$", "", text, flags=re.I)
    start = text.find("{")
    if start < 0: raise ValueError("director output contains no JSON object")
    decoder = json.JSONDecoder()
    try: result, _ = decoder.raw_decode(text[start:])
    except json.JSONDecodeError as exc: raise ValueError("director output contains invalid JSON") from exc
    if not isinstance(result, dict): raise ValueError("director JSON must be an object")
    return result


def format_lyrics(value):
    if isinstance(value, list):
        lines = [str(x).strip() for x in value if str(x).strip()]
        return "\n".join(lines)
    return str(value).strip()


def shot_prompt(shot, lyrics):
    lines = shot.get("lines", [])
    if isinstance(lines, str): lines = [lines]
    exact = "\n".join(str(x) for x in lines)
    return (f"Use <Picture 1> as the character and <Picture 2> as the location. "
            f"Use <Audio 1> exactly as the song slice. The lyrics he/she sings, in order: {exact}. "
            f"Action: {shot.get('action','')}. Camera: {shot.get('camera','')}. "
            "The camera always stays in FRONT of the character, face and mouth always visible, "
            "never behind or orbiting around the back; vary framing: close-up, medium, wide. Lip-synced.")
