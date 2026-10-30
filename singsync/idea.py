"""Native ComfyUI graphs and parsing for sentence-led music videos."""
from __future__ import annotations

import json
import re

DIRECTOR_INSTRUCTION = '''Create a music-video plan for about 30 seconds from the two reference images and this idea: {idea!r}. Return ONLY JSON with keys title, style_tags, bpm, character, scene, lyrics, shots. character is one sentence describing image 1 exactly; scene is one sentence describing image 2. lyrics is an object {{"verse": [4 full lyric lines], "chorus": [4 full lyric lines]}} - each item is one whole sung line (6-9 words), simple and catchy, about 30 seconds in total; never split a line at a comma. shots has 3-4 objects with lines (exact lyric lines for this shot), action, camera. Camera rules: the camera always stays in FRONT of the character, face and mouth always visible, never behind or orbiting around the back; vary framing: close-up, medium, wide.'''


def build_director_graph(character_name, scene_name, idea, seed=1):
    return {
        "1":{"class_type":"CLIPLoader","inputs":{"clip_name":"qwen3vl_8b_fp8_scaled.safetensors","type":"ideogram4","device":"default"}},
        "2":{"class_type":"LoadImage","inputs":{"image":character_name}},
        "3":{"class_type":"LoadImage","inputs":{"image":scene_name}},
        "4":{"class_type":"ImageBatch","inputs":{"image1":["2",0],"image2":["3",0]}},
        "5":{"class_type":"TextGenerate","inputs":{"clip":["1",0],"prompt":DIRECTOR_INSTRUCTION.format(idea=idea),"max_length":1500,"sampling_mode":"on","sampling_mode.temperature":0.7,"sampling_mode.top_k":64,"sampling_mode.top_p":0.95,"sampling_mode.min_p":0.05,"sampling_mode.repetition_penalty":1.05,"sampling_mode.seed":int(seed),"image":["4",0]}},
        "6":{"class_type":"PreviewAny","inputs":{"source":["5",0]}},
    }


def build_song_graph(tags, lyrics, seconds, seed=1, bpm=120, model="xl"):
    """ACE-Step 1.5 song graph. model="xl" (default; XL turbo, measured 41/44 lyric words recognised on a test song)
    or "turbo" (small all-in-one checkpoint, faster, 38/44)."""
    enc = {"tags":tags,"lyrics":lyrics,"seed":int(seed),"bpm":int(bpm),"duration":float(seconds),"timesignature":"4","language":"en","keyscale":"C major","generate_audio_codes":True,"cfg_scale":2.0,"temperature":0.85,"top_p":0.9,"top_k":0,"min_p":0.0}
    if model == "turbo":
        loaders = {"1":{"class_type":"CheckpointLoaderSimple","inputs":{"ckpt_name":"ace_step_1.5_turbo_aio.safetensors"}}}
        unet, clip, vae = ["1",0], ["1",1], ["1",2]
    else:
        loaders = {"1":{"class_type":"UNETLoader","inputs":{"unet_name":"acestep_v1.5_xl_turbo_bf16.safetensors","weight_dtype":"default"}},
                   "9":{"class_type":"DualCLIPLoader","inputs":{"clip_name1":"qwen_0.6b_ace15.safetensors","clip_name2":"qwen_4b_ace15.safetensors","type":"ace","device":"default"}},
                   "10":{"class_type":"VAELoader","inputs":{"vae_name":"ace_1.5_vae.safetensors"}}}
        unet, clip, vae = ["1",0], ["9",0], ["10",0]
    g = dict(loaders)
    g.update({
        "2":{"class_type":"ModelSamplingAuraFlow","inputs":{"model":unet,"shift":3}},
        "3":{"class_type":"TextEncodeAceStepAudio1.5","inputs":dict(enc, clip=clip)},
        "4":{"class_type":"ConditioningZeroOut","inputs":{"conditioning":["3",0]}},
        "5":{"class_type":"EmptyAceStep1.5LatentAudio","inputs":{"seconds":float(seconds),"batch_size":1}},
        "6":{"class_type":"KSampler","inputs":{"model":["2",0],"seed":int(seed),"steps":8,"cfg":1.0,"sampler_name":"euler","scheduler":"simple","positive":["3",0],"negative":["4",0],"latent_image":["5",0],"denoise":1.0}},
        "7":{"class_type":"VAEDecodeAudio","inputs":{"samples":["6",0],"vae":vae}},
        "8":{"class_type":"SaveAudioMP3","inputs":{"audio":["7",0],"filename_prefix":"singsync/song","quality":"V0"}},
    })
    return g


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


VIBES = {
    # name: (music style tags, bpm, visual energy for the shots)
    "happy": ("upbeat sunny pop, ukulele, handclaps, bouncy bass", 110, "joyful bouncing and spinning, big smiles, gentle handheld camera"),
    "party": ("dance-pop club banger, four-on-the-floor kick, punchy bass, handclaps", 124, "high-energy dancing and jumping, quick push-ins, bright colourful light"),
    "chill": ("lo-fi chill pop, soft keys, mellow drums, warm", 85, "relaxed swaying, slow dolly moves, golden-hour light"),
    "epic": ("cinematic pop anthem, big drums, soaring synths", 100, "powerful poses, slow motion, sweeping crane shots in front of the character"),
    "sad": ("emotional piano ballad, soft strings", 72, "slow heartfelt performance, close-ups, soft light, slow push-in"),
    "rap": ("hip-hop, hard 808s, trap hi-hats, confident rap vocal", 92, "swagger, hand gestures to camera, low-angle shots"),
    "kids": ("cheerful children's song, simple melody, xylophone, claps", 112, "playful silly dancing, colourful, bouncy camera"),
}


def apply_vibe(idea, vibe):
    """Fold a vibe preset into the one-sentence idea so the director, song and shots share it."""
    if not vibe: return idea
    if vibe not in VIBES: raise ValueError("unknown vibe %r; choose from %s" % (vibe, ", ".join(VIBES)))
    tags, bpm, visual = VIBES[vibe]
    return "%s (vibe: %s - music: %s, about %d BPM; visuals: %s)" % (idea, vibe, tags, bpm, visual)


VOCAL_TAGS = "clear lead vocal, english lyrics, sung clearly, simple melody, vocals loud in the mix"


def song_tags(style_tags):
    """ACE-Step mumbles unless the tags ask for a clear vocal (measured: 31/44 vs 38/44 lyric words recognised)."""
    tags = str(style_tags or "upbeat pop, catchy").strip().rstrip(",")
    return tags if "sung clearly" in tags else f"{tags}, {VOCAL_TAGS}"


def format_lyrics(value):
    """Whole lines under [Verse 1] / [Chorus] section tags - ACE-Step sings full lines far more clearly than fragments."""
    if isinstance(value, dict):
        parts = []
        for key, label in (("verse", "[Verse 1]"), ("pre_chorus", "[Pre-Chorus]"), ("chorus", "[Chorus]"), ("bridge", "[Bridge]")):
            lines = value.get(key) or []
            if isinstance(lines, str): lines = [lines]
            lines = [str(x).strip() for x in lines if str(x).strip()]
            if lines: parts.append(label + "\n" + "\n".join(lines))
        return "\n\n".join(parts)
    if isinstance(value, list):
        return "\n".join(str(x).strip() for x in value if str(x).strip())
    text = str(value).strip()
    import re as _re
    text = _re.sub(r"\s*\[(verse[^\]]*|chorus|pre-chorus|bridge)\]\s*", lambda m: "\n\n[" + m.group(1).title() + "]\n", text, flags=_re.I).strip()
    return text


def shot_prompt(shot, lyrics, character="", visual=""):
    lines = shot.get("lines", [])
    if isinstance(lines, str): lines = [lines]
    exact = "\n".join(str(x) for x in lines)
    who = f"The character ({character}) " if character else "The character "
    energy = f" Energy: {visual}." if visual else ""
    return (f"Use <Picture 1> as the character and <Picture 2> as the location. {who}looks exactly like <Picture 1>, same face, outfit and colours. "
            f"Use <Audio 1> exactly as the song slice. The lyrics he/she sings, in order: {exact}. "
            f"Action: {shot.get('action','')}. Camera: {shot.get('camera','')}. "
            "The camera always stays in FRONT of the character, face and mouth always visible, "
            "never behind or orbiting around the back; vary framing: close-up, medium, wide. Lip-synced." + energy)
