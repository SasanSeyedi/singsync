# SingSync

**One picture of a character. One picture of a place. One sentence.
Out comes a 30-second music video where your character actually sings the song.**

```bash
python scripts/idea_to_video.py \
  --character banana_guy.png --scene banana_field.png \
  --idea "a happy song about bananas raining from the sky" \
  --vibe happy
```

That's the whole interface. Behind it:

1. **A director** (Qwen3-VL 8B) looks at both pictures and writes the song, the
   style and a shot list.
2. **A songwriter** (ACE-Step 1.5 XL) turns that into a real song with vocals.
3. **SingSync** times every sung word, cuts the song on lyric lines and beats,
   and gives each shot its own slice of the vocal.
4. **A video model** (MiniMax H3) renders every shot with your character singing
   that slice, camera kept in front so you can see the mouth.
5. **SingSync again** measures how far each clip drifted off the vocal, fixes it,
   and stitches everything onto the master track with karaoke captions.

Every model in that chain is **open weights**, and the director, song and video
steps run through ComfyUI's own native nodes on Comfy Cloud. For the tightest
lip-sync today, render the video step on a hosted MiniMax H3 endpoint instead
(see Results): the open-weight route works end to end, but Comfy Cloud ships a
compressed H3 build that drifts more.

### Vibes

Pick a vibe and it steers the music, the tempo and how the shots move:

| `--vibe` | sounds like | looks like |
| --- | --- | --- |
| `happy` | sunny pop, ukulele, handclaps | bouncing, spinning, big smiles |
| `party` | dance-pop club banger | jumping, quick push-ins, bright light |
| `chill` | lo-fi, soft keys | swaying, slow dolly, golden hour |
| `epic` | cinematic anthem | slow motion, sweeping crane shots |
| `sad` | piano ballad, strings | close-ups, soft light |
| `rap` | hip-hop, 808s | swagger, low angles |
| `kids` | children's song, xylophone | silly dancing, colourful |

Or leave it off and let the director decide from your sentence.

### Already have a song?

Bring your own MP3, lyrics and a character image instead; SingSync does the
cutting, lip-sync correction and assembly. See
[Run it without ComfyUI](#run-it-without-comfyui) and the node list below.

### What you need

- A Comfy Cloud account on a paid plan (API keys don't work on the free tier),
  with `COMFY_API_KEY` set in your environment.
- `ffmpeg` on your PATH and `pip install faster-whisper` (runs on CPU).
- About 15–45 minutes per 30-second video, mostly waiting on the video model.
  `--plan-only` stops after the song and cut plan so you can check them first,
  and every run resumes where it left off.

## Why the singing lines up

Video models that take audio don't start singing exactly when you tell them to.
In our demo one model sang **up to 3.3 seconds late** and repeated a line.
SingSync listens to every rendered clip, finds the words it fed, measures the
offset and corrects it before anything is joined, so the mouth and the song
stay together across every cut.

## How it works

**Each cut is fed its own vocal slice.** Audio-driven video models sing what you
give them. SingSync cuts the vocal into per-shot slices and hands each shot the
seconds it needs, so the character sings its own line rather than mouthing along
to a whole track.

**It measures the lip-sync error per clip and corrects it.** A model asked to
sing from 0.0s does not always start at 0.0s; one clip lands a third of a second
early, the next a second late, and no single global nudge fixes both. SingSync
transcribes the rendered clip, matches its words against the slice it was fed,
and reads the offset straight off the matched pairs:

    offset = median(clip_t - slice_t)

That offset is applied when the clip is trimmed, so nothing is re-encoded to fix
it. If the error drifts inside one clip, the anchors become a piecewise-linear
map instead of one constant shift.

**Cuts land on lyric lines and then on beats.** Words are grouped to a target
length, preferring a break at the end of a lyric line; the resulting boundary is
then snapped to the nearest downbeat, but only when the move stays inside a gap
between words, so a snap can never cut a syllable in half.

**Overlap words make the joins sample-accurate.** Every cut *owns* a contiguous
range of the song, and the owned ranges tile `[0, duration]` exactly: no gaps, no
overlaps. The audio a cut is *fed* is wider than that — it reaches two or three
words into each neighbour. Both sides of a join therefore contain the same words,
so the trim happens on material both clips have, and the model has context for
the phrase it is starting and the one it is finishing. Segment lengths are
counted in whole frames off one shared grid, so the finished video matches the
song length to under a frame no matter how many cuts there are.

## Install

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/SasanSeyedi/singsync
pip install faster-whisper     # optional: needed for transcription and retime
```

`numpy` and `scipy` are already in ComfyUI. `ffmpeg` and `ffprobe` must be on
`PATH`, or point `SINGSYNC_FFMPEG` / `SINGSYNC_FFPROBE` at them. The core
(`singsync/`) imports no torch and no ComfyUI, so it runs and tests on its own.
Tested on Python 3.14.

## Nodes

All under the category **SingSync**. They pass a `SINGSYNC_PLAN`: a plain
JSON-serialisable dict plus files under `<ComfyUI output>/singsync/<run_id>/`.

| Node | Does |
| --- | --- |
| **SingSync Song Analyze** | Tempo, beats, downbeats, energy and sections; transcribes the vocal with faster-whisper and aligns it to your lyrics. |
| **SingSync Cut Plan** | Groups words into cuts, assigns owned ranges, shared overlap words, duration buckets; outputs the plan and a cut sheet. |
| **SingSync Cut Slicer** | Writes each cut's vocal slice and its render prompt; also outputs lists of slice paths and prompts for a loop. |
| **SingSync Seedance (kie.ai)** | Renders the cuts through kie.ai Seedance. Registered through a guarded import — a provider problem cannot stop the pack loading. |
| **SingSync Retime** | Measures the lip-sync offset of every rendered clip, optionally bakes it in; outputs an offsets table. |
| **SingSync Assemble** | Trims each clip to the range it owns, concatenates at constant fps, muxes the master song, optional burnt-in karaoke captions. |

Nothing heavy is imported and no work is done at import time, so a missing
optional dependency surfaces on the node that needs it rather than at start-up.

A minimal workflow chaining them is in
[`workflows/singsync_music_video.json`](workflows/singsync_music_video.json).

## Make the song too (ACE-Step 1.5, open weights)

ACE-Step 1.5 is ComfyUI-native and open weights: create the song in the workflow without an MP3 or a closed music service. The lyrics you write drive the music, captions and alignment, so Whisper mainly times known words instead of guessing them.

Open [`workflows/ace_step_to_singsync.json`](workflows/ace_step_to_singsync.json), then use ComfyUI Template Browser to load the official ACE-Step 1.5 template into its marked placeholder group. Connect generated AUDIO to SingSync Song Analyze → song, and use the same lyrics in ACE-Step and SingSync Song Analyze → lyrics. Set style tags in ACE-Step, then run the remaining SingSync chain.

ACE-Step outputs a full mix; SingSync aligns the known lyrics against it. For best lip-sync, add a vocal-separation node and connect its vocal stem to Song Analyze → vocals. The video model is the only partner/closed component: use the included kie.ai Seedance node or ComfyUI’s official Seedance 2.5 reference-to-video node, which accepts audio clips.

## Results from the demo

Same song section (30 s, 4 cuts: three sung, one instrumental b-roll), same SingSync plan, two video models. Offsets are what SingSync measured per clip by matching the words in the clip against the words in the vocal slice it was fed:

| Video model | Cut 0 | Cut 1 | Cut 2 | Notes |
|---|---|---|---|---|
| Seedance 2 (kie.ai) | +0.64 s | +3.28 s | +1.68 s | sings late; one clip repeated a line (handled by repeat-aware retime) |
| MiniMax H3 reference-to-video (kie.ai, hosted) | +0.02 s | +0.02 s | 0.00 s | follows the fed vocal closely; used for the final demo |
| MiniMax H3 native on Comfy Cloud (pruned int8 build) | +0.80 s | +0.40 s | +2.34 s | same inputs; re-performs with its own timing and repeats lines, Retime corrects part of it |

Final video: 30.000 s = 720 frames at 24 fps, master audio, karaoke captions from the real lyrics (Whisper only supplies the timing).

The Comfy Cloud route (native MiniMax H3 nodes) runs end to end: submit, poll, download. With four-step turbo and no lyrics in the prompt the character mouthed nonsense; with full steps and the cut's exact lyric lines in the prompt it sings the words, re-performed with some drift that Retime then corrects. That is why the one-sentence mode always puts each cut's lyrics in its prompt.

## Run it without ComfyUI

```bash
# No inputs at all: generates a song, fakes the renders, runs the whole chain.
python scripts/run_pipeline.py --demo --work-dir /tmp/singsync-demo

# A real song, renders faked:
python scripts/run_pipeline.py \
    --song song.mp3 --vocals vocals.wav --lyrics lyrics.txt \
    --character "a tall singer in a red coat" --scene "an empty car park at dusk" \
    --style "35mm, handheld, low sun" \
    --dry-run --captions --work-dir out/
```

`--dry-run` replaces every render with an lavfi slate that carries that cut's own
vocal slice, delayed by a known offset. Retime then has a real file to measure,
and the run prints what it injected next to what it measured:

```
idx  injected  measured  error
  0    -0.423    -0.432  -0.009
  1    -0.838    -0.822  +0.016
  3    -1.026    -1.038  -0.012
  4    +0.086    +0.070  -0.016
mean |error| 0.013s, worst 0.016s over 4 cuts
```

Runs are resumable (`--plan out/plan.json --stages assemble`) and can be run one
stage at a time. `--help` lists the planner and prompt options.

## Providers

**kie.ai Seedance** (`singsync/providers/kie.py`). The key is read from the
`KIE_API_KEY` environment variable and never written to the plan, the logs or
the work directory. Seedance 2 through kie accepts clip durations of 5, 10 or 15
seconds; SingSync defaults to a 12-second cap on the fed audio and picks the
smallest bucket covering the fed span plus 3 seconds of headroom. Configure it
with the Cut Plan `headroom_s` widget or CLI `--headroom`. Seedance 2.5 takes
4–30 seconds — pass
`--max-clip 30 --durations 4,...,30` (or set the two widgets) to use it.

**ComfyUI's own Seedance 2.5 reference-to-video node** also takes audio clips.
To use it instead, drop it in where the Seedance node sits: read
`slice_path`/`prompt`/`duration` from Cut Slicer's list outputs and write the
rendered file back as the cut's `clip_path`. Nothing else in the chain cares
where the clip came from.

## Limits

- **No source separation.** Bring your own vocal stem (Demucs, UVR, a stem
  export). Without one, the full mix is transcribed and fed, which transcribes
  less accurately and gives the renderer a backing track to lip-sync to.
- **Retime needs faster-whisper.** Without it, offsets stay at zero and the cuts
  are joined on their planned times. The dry run says so and simulates the clip
  transcripts instead, which exercises the estimator but not whisper.
- **Downbeat detection is weak** and is treated that way: it only ever nudges a
  boundary that is already inside a word gap, within a 0.25s tolerance.
  `downbeat_confidence` is reported; do not build on it.
- **Discrete clip buckets cost renderer seconds.** A 10.2s audio span becomes a
  15s clip. Narrowing the target (`--target 8 --max-clip 10 --durations 5,10`)
  trades shot length for render budget.
- **Drift inside one clip is only partly fixed.** Retime applies one offset per
  clip (or a splice for repeated lines). When a model changes tempo inside a
  clip, as the compressed H3 on Comfy Cloud does, part of the clip stays off.
- **A clip that starts singing early can freeze for a moment.** If the fix needs
  footage from before the clip's first frame, Assemble holds the first frame.
  Fixed-length padding is the next change.
- **Repeated sung material is splice-retimed** from anchor clusters; ordinary
  single-cluster cuts retain the constant offset behavior.
- **Character consistency** is whatever the reference image and the repeated
  prompt text give you. SingSync keeps the character and scene text identical
  across cuts; it does nothing else to hold a face steady.
- **Upscaling and interpolation** are left to ComfyUI's own nodes, downstream of
  Assemble.
- **What has been verified:** the planner, retime estimator, caption text and
  assembly length are covered by tests; the chain runs end to end with faked
  renders; and it has been run with real Seedance 2 and MiniMax H3 renders (see
  Results). No rendered media is shipped in this repository.

## Tests

```bash
python -m unittest discover -s tests
```

stdlib `unittest` only. Tests that need ffmpeg skip themselves when
it is absent; nothing needs ComfyUI, torch, a GPU or a network.

## Render on Comfy Cloud with MiniMax H3 (open weights)

The `SingSync Render (Comfy Cloud, MiniMax H3)` node sends each planned cut to
Comfy Cloud using ComfyUI's native MiniMax H3 reference-to-video nodes. It uses
the open-weight H3 model and optional four-step turbo LoRA. Set `COMFY_API_KEY`
in the environment, connect a SingSync plan, and provide one or more local
reference image paths as a semicolon-separated list. The cut vocal slices in
the plan are uploaded as audio references. You can also render one cut from a
saved plan with `python scripts/cloud_render_cut.py plan.json 0 --image face.png
--image outfit.png --turbo`. The node observes `SINGSYNC_MAX_CLIPS` (default 6)
and records submitted jobs in `singsync_spend.jsonl` under the plan work folder.

## Licence

MIT. All code in this repository was written for it.
