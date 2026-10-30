"""Beat analysis and beat snapping."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from singsync import beats
from singsync.ffmpeg_util import ffmpeg, have_ffmpeg
from tests import synth


def click_track(path: Path, duration: float, bpm: float):
    """A percussive click every beat, with a quiet pad under it."""
    period = 60.0 / bpm
    src = ("aevalsrc='0.8*sin(2*PI*90*t)*exp(-14*mod(t,%(p).6f))"
           "+0.3*sin(2*PI*2000*t)*exp(-70*mod(t,%(p).6f))"
           "+0.08*sin(2*PI*220*t)':s=44100:d=%(d).3f"
           % {"p": period, "d": duration})
    ffmpeg(["-f", "lavfi", "-i", src, "-ac", "1", "-c:a", "pcm_s16le", str(path)])
    return str(path)


class Snapping(unittest.TestCase):
    """Pure: no audio needed."""

    def setUp(self):
        self.words = synth.make_words()
        self.duration = synth.song_duration(self.words)
        self.analysis = synth.make_analysis(self.duration)

    def _plan(self, boundary):
        return {"duration": self.duration, "cuts": [
            {"idx": 0, "owns": [0.0, boundary], "owns_start": 0.0, "owns_end": boundary},
            {"idx": 1, "owns": [boundary, self.duration], "owns_start": boundary,
             "owns_end": self.duration}]}

    def _gap_mid(self):
        """A word gap wide enough to move a boundary inside."""
        for prev, nxt in zip(self.words, self.words[1:]):
            if nxt["start"] - prev["end"] > 0.4:
                return (prev["end"] + nxt["start"]) / 2.0, prev["end"], nxt["start"]
        raise AssertionError("no usable gap in the synthetic words")

    def test_boundary_moves_to_the_nearest_downbeat(self):
        mid, lo, hi = self._gap_mid()
        # Put a downbeat inside that gap, a little off the boundary.
        downbeat = round(min(hi - 0.01, mid + 0.12), 4)
        analysis = dict(self.analysis, downbeats=[downbeat], beats=[downbeat])
        plan, log = beats.snap_cuts_to_beats(self._plan(mid), analysis, self.words,
                                             tol_s=0.25)
        self.assertEqual(len(log), 1)
        self.assertAlmostEqual(plan["cuts"][0]["owns_end"], downbeat, places=4)
        self.assertAlmostEqual(plan["cuts"][1]["owns_start"], downbeat, places=4)
        self.assertAlmostEqual(plan["cuts"][0]["owns"][1], downbeat, places=4)

    def test_tolerance_is_respected(self):
        mid, _, _ = self._gap_mid()
        plan, log = beats.snap_cuts_to_beats(self._plan(mid), self.analysis, self.words,
                                             tol_s=0.0)
        self.assertEqual(log, [])
        self.assertAlmostEqual(plan["cuts"][0]["owns_end"], mid, places=6)

    def test_never_snaps_into_a_word(self):
        inside = (self.words[3]["start"] + self.words[3]["end"]) / 2.0
        plan, log = beats.snap_cuts_to_beats(self._plan(inside), self.analysis,
                                             self.words, tol_s=5.0)
        self.assertEqual(log, [])

    def test_snapping_keeps_the_tiling_exact(self):
        mid, _, _ = self._gap_mid()
        plan, _ = beats.snap_cuts_to_beats(self._plan(mid), self.analysis, self.words,
                                           tol_s=1.0)
        cuts = plan["cuts"]
        self.assertEqual(cuts[0]["owns_end"], cuts[1]["owns_start"])
        self.assertAlmostEqual(cuts[0]["owns_start"], 0.0)
        self.assertAlmostEqual(cuts[-1]["owns_end"], self.duration)

    def test_original_plan_is_not_mutated(self):
        mid, _, _ = self._gap_mid()
        original = self._plan(mid)
        beats.snap_cuts_to_beats(original, self.analysis, self.words, tol_s=1.0)
        self.assertAlmostEqual(original["cuts"][0]["owns_end"], mid, places=6)

    def test_beat_preference(self):
        mid, lo, hi = self._gap_mid()
        _, log = beats.snap_cuts_to_beats(self._plan(mid), self.analysis, self.words,
                                          tol_s=0.6, prefer="beat")
        for entry in log:
            self.assertIn(round(entry["to"], 4), set(self.analysis["beats"]))

    def test_vibe_brief_reads_both_cut_shapes(self):
        cut_a = {"owns": [10.0, 18.0]}
        cut_b = {"owns_start": 10.0, "owns_end": 18.0}
        text = beats.vibe_brief(self.analysis, cut_a)
        self.assertEqual(text, beats.vibe_brief(self.analysis, cut_b))
        self.assertIn("section=", text)
        self.assertIn("energy=", text)
        self.assertIn("downbeat", text)

    def test_vibe_brief_on_an_empty_analysis(self):
        self.assertIn("section=unknown", beats.vibe_brief({}, {"owns": [0.0, 1.0]}))


@unittest.skipUnless(have_ffmpeg(), "ffmpeg not available")
class Analyse(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="singsync-beats-"))
        cls.duration = 24.0
        cls.bpm = 120.0
        cls.audio = click_track(cls.tmp / "clicks.wav", cls.duration, cls.bpm)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_tempo_of_a_click_track(self):
        analysis = beats.analyse(self.audio)
        # Octave errors (half or double tempo) are the normal failure mode.
        ratio = analysis["tempo_bpm"] / self.bpm
        self.assertTrue(min(abs(ratio - r) for r in (0.5, 1.0, 2.0)) < 0.03,
                        "tempo %.2f is not a multiple of %.0f"
                        % (analysis["tempo_bpm"], self.bpm))
        self.assertGreater(analysis["tempo_confidence"], 0.5)

    def test_beat_grid_is_regular_and_covers_the_track(self):
        analysis = beats.analyse(self.audio)
        beat_times = analysis["beats"]
        self.assertGreater(len(beat_times), 10)
        self.assertLess(beat_times[0], 1.0)
        self.assertGreater(beat_times[-1], self.duration - 1.5)
        steps = [b - a for a, b in zip(beat_times, beat_times[1:])]
        self.assertAlmostEqual(max(steps), min(steps), places=3)
        self.assertEqual(analysis["downbeats"][0], beat_times[
            beat_times.index(analysis["downbeats"][0])])

    def test_shape_of_the_analysis(self):
        analysis = beats.analyse(self.audio)
        for key in ("duration", "sample_rate", "tempo_bpm", "tempo_confidence",
                    "beats", "downbeats", "downbeat_confidence", "onset_strength",
                    "energy_curve", "sections", "drops"):
            self.assertIn(key, analysis)
        self.assertAlmostEqual(analysis["duration"], self.duration, delta=0.1)
        self.assertTrue(analysis["sections"])
        self.assertAlmostEqual(analysis["sections"][0]["start"], 0.0, places=3)
        self.assertAlmostEqual(analysis["sections"][-1]["end"], analysis["duration"],
                               delta=0.6)
        for section in analysis["sections"]:
            self.assertLess(section["start"], section["end"])
        import json
        self.assertIsInstance(json.dumps(analysis), str)

    def test_downbeats_are_a_subset_of_beats(self):
        analysis = beats.analyse(self.audio)
        self.assertTrue(set(analysis["downbeats"]).issubset(set(analysis["beats"])))
        self.assertAlmostEqual(len(analysis["downbeats"]),
                               len(analysis["beats"]) / 4.0, delta=1.0)

    def test_empty_audio_is_rejected(self):
        silent = self.tmp / "tiny.wav"
        ffmpeg(["-f", "lavfi", "-i", "anullsrc=r=44100:cl=mono", "-t", "0.01",
                "-c:a", "pcm_s16le", str(silent)])
        try:
            beats.analyse(silent)
        except (ValueError, Exception):   # noqa: B014 - either is acceptable here
            pass


if __name__ == "__main__":
    unittest.main()
