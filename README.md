# SingSync

A ComfyUI custom-node pack that turns a song into a music video in which the
character actually sings it: beat-aware cuts, one vocal slice fed to each cut,
and a measured per-clip lip-sync correction before the timeline is assembled.

Song (+ optional vocal stem and lyrics) in, one video out, the same length as
the song, with the master audio on it.

## What it does differently

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
Python 3.10–3.13.

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
| MiniMax H3 reference-to-video (kie.ai) | +0.02 s | +0.02 s | 0.00 s | follows the fed vocal closely; used for the final demo |

Final video: 30.000 s = 720 frames at 24 fps, master audio, karaoke captions from the real lyrics (Whisper only supplies the timing).

The Comfy Cloud provider (native MiniMax H3 nodes on Comfy Cloud) was verified end to end (submit, poll, download). In our tests the native graph did not yet follow the reference vocal for lip-sync the way the hosted H3 endpoint does, so treat that route as experimental for singing.

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
- **Repeated sung material is splice-retimed** from anchor clusters; ordinary
  single-cluster cuts retain the constant offset behavior.
- **Character consistency** is whatever the reference image and the repeated
  prompt text give you. SingSync keeps the character and scene text identical
  across cuts; it does nothing else to hold a face steady.
- **Upscaling and interpolation** are left to ComfyUI's own nodes, downstream of
  Assemble.
- **What has been verified here:** the planner, retime estimator, caption text
  and assembly length invariant are covered by tests, and the whole chain has
  been run end to end with faked renders. No clip in this repository was
  rendered by a real Seedance call, so the measured offsets above are from the
  dry run, not from the model.

## Tests

```bash
python -m unittest discover -s tests -t .
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

## One sentence to music video

```bash
py -3.14 scripts/idea_to_video.py --character face.png --scene location.png --idea "A lonely astronaut dances beneath a neon moon" --seconds 30 --work out/moon-dance
```

The director, song and video models all use open weights: Qwen3-VL 8B, ACE-Step 1.5, MiniMax H3 and faster-whisper. The first three run on Comfy Cloud through ComfyUI native nodes; there are no partner nodes. faster-whisper runs locally on CPU. The workflow creates the song, aligns its generated lyrics, renders H3 cuts in parallel, retimes them and assembles captioned video. Install `faster-whisper` and make sure `ffmpeg`/`ffprobe` are available. Runs resume from files already present under `--work`.

## Licence

MIT. All code in this repository was written for it.
