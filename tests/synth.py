"""Synthetic word timings and analyses, so core tests need no audio."""
from __future__ import annotations

LINES = [
    "I walk the wire over bright water",
    "counting every light below me",
    "you said the morning would be kinder",
    "but the morning never answered",
    "so I keep the engine running",
    "and the radio talks to nobody",
    "hold the line until it breaks",
    "hold the line until it breaks again",
]


def make_words(lines=None, *, start=1.0, word_s=0.42, gap_s=0.1,
               line_gap_s=0.55, instrumental_after=None, instrumental_s=8.0):
    """Build an alignment-style word list from lyric lines.

    ``instrumental_after`` inserts a silent stretch of ``instrumental_s`` seconds
    after that line index, which the planner should turn into a b-roll region.
    """
    lines = list(LINES if lines is None else lines)
    words = []
    t = float(start)
    for line_idx, line in enumerate(lines):
        tokens = line.split()
        for i, token in enumerate(tokens):
            words.append({
                "word": token,
                "start": round(t, 4),
                "end": round(t + word_s, 4),
                "line_idx": line_idx,
                "line_end": i + 1 == len(tokens),
            })
            t += word_s + gap_s
        t += line_gap_s - gap_s
        if instrumental_after is not None and line_idx == instrumental_after:
            t += instrumental_s
    return words


def song_duration(words, tail=2.0):
    """A plausible song length for a word list."""
    return round(float(words[-1]["end"]) + tail, 4)


def make_analysis(duration, bpm=120.0, *, downbeat_phase=0.0, sections=True):
    """A beats.analyse-shaped dict with an exact tempo grid."""
    period = 60.0 / bpm
    beats = []
    t = 0.0
    while t < duration:
        beats.append(round(t, 4))
        t += period
    downbeats = beats[int(round(downbeat_phase)) % 4::4]
    step = 0.5
    curve = []
    n = int(duration / step) + 1
    for i in range(n):
        time_s = round(min(duration, i * step + step / 2), 3)
        curve.append({"time": time_s, "rms": 0.1, "spectral_flux": 0.1,
                      "energy": round(0.3 + 0.4 * (i % 7) / 7.0, 4)})
    out = {
        "duration": round(float(duration), 4),
        "sample_rate": 22050,
        "tempo_bpm": bpm,
        "tempo_confidence": 0.8,
        "beats": beats,
        "downbeats": downbeats,
        "downbeat_confidence": 0.4,
        "energy_curve": curve,
        "sections": [],
        "drops": [],
    }
    if sections:
        third = duration / 3.0
        out["sections"] = [
            {"start": 0.0, "end": round(third, 3), "energy_mean": 0.3, "label": "intro"},
            {"start": round(third, 3), "end": round(2 * third, 3),
             "energy_mean": 0.6, "label": "verse"},
            {"start": round(2 * third, 3), "end": round(duration, 3),
             "energy_mean": 0.5, "label": "outro"},
        ]
    return out
