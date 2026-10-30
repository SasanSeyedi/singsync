# STATUS

Built per `BUILD_SPEC.md`, in the order the spec gives. Python 3.13, ffmpeg 6/7,
no ComfyUI, torch, GPU or network needed to run or test the core.

## Tests

```bash
python -m unittest discover -s tests -t .
```

**Result: `Ran 141 tests` — OK, 0 failures, 0 skipped** (on a machine with
ffmpeg; tests needing ffmpeg skip themselves without it).

| Module | Tests | Covers |
| --- | --- | --- |
| `test_cutplan` | 22 | Tiling, audio-span containment, max span, duration buckets, overlap words, line-end preference, b-roll regions, beat snapping, validator |
| `test_retime` | 24 | `estimate_offset` on ±1.7s shifts, dropouts, extras, outliers, drift→piecewise, rejection; clip baking keeps length |
| `test_align` | 18 | Word cleaning, monotonicity repair, lyric alignment with mishearings and dropouts, load/save |
| `test_captions` | 14 | ASS header, `\k` durations, gap spacers, line grouping, escaping, presets |
| `test_assemble` | 13 | Frame grid, clip-time mapping, and the length invariant on lavfi slates (24 and 30 fps, with and without offsets, with captions) |
| `test_beats` | 13 | Snapping rules on synthetic words; tempo and grid on a generated click track |
| `test_slicer` | 13 | Prompt templates (vocal and instrumental), slice lengths including padding past the end |
| `test_nodes` | 13 | Registration, node contract, plan plumbing, AUDIO→wav |
| `test_workflow` | 7 | The shipped workflow JSON matches the node definitions |
| `test_kie_provider` | 4 | kie.ai client (written by the provider agent) |

## What works, and how it was checked

- **`beats.py`** — tempo, beats, downbeats, energy, sections, `snap_cuts_to_beats`,
  `vibe_brief`. Kept the API; made the cut dicts accept `owns_start/owns_end`,
  vectorised the tempo search (it was minutes on a full song, now seconds), and
  took the ffmpeg path from the environment.
- **`align.py`** — faster-whisper word timestamps (imported lazily) aligned to
  supplied lyric lines with `difflib`. `build_alignment` is pure and tested;
  the whisper call itself is not exercised here (see below).
- **`cutplan.py`** — the overlap planner. Owned ranges tile `[0, duration]`
  exactly, the fed audio always contains the owned range, spans never exceed the
  renderer limit, and buckets are the smallest allowed that fit. `validate_plan`
  asserts all of it and the Cut Plan node refuses to emit a plan that fails.
- **`slicer.py`** — per-cut ffmpeg slices (wav/mp3/m4a, padded past the song end)
  and templated prompts.
- **`retime.py`** — `estimate_offset` matches clip and slice transcripts, takes a
  median over inlier anchors, rejects absurd measurements, and switches to a
  piecewise map when the error drifts inside one clip. Offsets are applied
  arithmetically at assembly (no re-encode); baking them into clip files is
  optional and was verified frame by frame: a clip measured at −0.822s has its
  content pushed 20 frames later at 24fps.
- **`assemble.py`** — trims each clip to its owned range, concatenates in one
  filtergraph at constant fps, muxes the master song. Segment lengths are whole
  frames off a shared grid, so length cannot drift: the demo run finishes at
  42.875s against a 42.880s song, **0.00 frames of drift**.
- **`captions.py`** — karaoke ASS with `\k` tags, three presets, burnt in with
  ffmpeg's `subtitles` filter. Verified visually on the demo output.
- **`nodes.py` + `__init__.py`** — five nodes under category SingSync, lazy
  imports, no work at import time, provider node behind a guarded import.
- **`scripts/run_pipeline.py`** — the whole chain without ComfyUI, resumable and
  stage-selectable. `--demo` needs no inputs at all.

### End-to-end evidence (dry run, no model calls)

```
python scripts/run_pipeline.py --demo --work-dir /tmp/singsync-demo --captions
```

6 cuts over a 42.88s generated song; each fake render carries its own vocal
slice delayed by a known offset, and retime measures it back out:

```
idx  injected  measured  error
  0    -0.423    -0.432  -0.009
  1    -0.838    -0.822  +0.016
  3    -1.026    -1.038  -0.012
  4    +0.086    +0.070  -0.016
mean |error| 0.013s, worst 0.016s over 4 cuts
length 42.875s vs song 42.880s (0.00 frames of drift)
```

Both the `--demo` and the `--song` paths were run, plus resuming from
`plan.json` with `--stages retime,assemble --bake-retime`.

## What is untested

- **No real Seedance render was made from this repo.** Every clip in the
  evidence above is an lavfi slate. The provider client has its own unit tests
  with a stub transport, but no live kie.ai call was made, so response-shape
  surprises are possible.
- **faster-whisper is not installed here**, so `transcribe_words` has never been
  run against audio in this environment, and the dry run simulates the clip
  transcripts (same estimator, synthetic input). The lyric-alignment logic that
  consumes whisper output *is* tested on synthetic word lists. This is the
  biggest untested seam: the first real run should check that whisper's word
  timings on a short clip are good enough for the median to be meaningful.
- **The nodes have not been loaded inside ComfyUI.** Registration, the input/
  output contract and the workflow JSON are checked against the classes, but no
  ComfyUI instance has imported the pack. The `AUDIO` tensor path is tested with
  a numpy array standing in for a torch tensor.
- **Piecewise retiming** is applied only when clips are baked; its segmented
  filtergraph is tested for length, not for per-segment content.
- **Beat snapping on real music.** The snapper is tested on synthetic grids. On
  the demo song no boundary was within tolerance of a downbeat, so zero snaps
  were exercised end to end; downbeat confidence is weak by design and only ever
  nudges a boundary already inside a word gap.
- **`--provider kie`** is wired in `run_pipeline.py` but was never run (no key,
  and no key will be committed).

## Not done

- `fcpxml` export exists (`--fcpxml`) but is a nice-to-have and was only checked
  for well-formed XML and clip count, not opened in an NLE.
- No upscaling or interpolation: left to ComfyUI's own nodes downstream of
  Assemble, as the spec says.
- No source separation. A vocal stem has to come from elsewhere.

No secrets, audio, video or characters are committed; `.gitignore` covers
`.env`, media and `_work/`.
