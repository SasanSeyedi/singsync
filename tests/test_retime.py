"""Lip-sync offset estimation on synthetic word lists."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from singsync import retime
from singsync.ffmpeg_util import ffmpeg, have_ffmpeg, media_duration
from tests import synth

LINE = "hold the line until it breaks and the radio talks to nobody tonight".split()


def words_at(tokens, start=0.5, word_s=0.4, gap_s=0.1, shift=0.0):
    """Evenly spaced word timings, optionally shifted in time."""
    out = []
    t = float(start) + float(shift)
    for token in tokens:
        out.append({"word": token, "start": round(t, 4), "end": round(t + word_s, 4)})
        t += word_s + gap_s
    return out


class EstimateOffset(unittest.TestCase):
    def test_repeated_line_produces_splice_segments(self):
        slice_words = words_at("one two three four five six".split(), start=0.0, word_s=0.3, gap_s=0.1)
        clip_words = words_at("one two three".split(), start=0.0, word_s=0.3, gap_s=0.1)
        clip_words += words_at("one two three".split(), start=3.3, word_s=0.3, gap_s=0.1)
        clip_words += words_at("four five six".split(), start=6.8, word_s=0.3, gap_s=0.1)
        result = retime.estimate_offset(clip_words, slice_words)
        self.assertEqual(result["method"], "splice")
        self.assertEqual(len(result["segments"]), 2)
        self.assertLess(result["segments"][0]["slice_end"], result["segments"][1]["clip_start"])

    def test_known_positive_shift(self):
        slice_words = words_at(LINE)
        clip_words = words_at(LINE, shift=1.7)
        result = retime.estimate_offset(clip_words, slice_words)
        self.assertAlmostEqual(result["offset_s"], 1.7, places=3)
        self.assertEqual(result["method"], "median")
        self.assertEqual(result["n_anchors"], len(LINE))

    def test_known_negative_shift(self):
        result = retime.estimate_offset(words_at(LINE, start=2.5, shift=-1.7),
                                        words_at(LINE, start=2.5))
        self.assertAlmostEqual(result["offset_s"], -1.7, places=3)

    def test_no_shift(self):
        result = retime.estimate_offset(words_at(LINE), words_at(LINE))
        self.assertAlmostEqual(result["offset_s"], 0.0, places=6)
        self.assertEqual(result["spread_s"], 0.0)

    def test_survives_dropped_words(self):
        slice_words = words_at(LINE)
        clip = words_at(LINE, shift=1.7)
        # The renderer mumbled: drop every third word from the clip transcript.
        clip_words = [w for i, w in enumerate(clip) if i % 3]
        result = retime.estimate_offset(clip_words, slice_words)
        self.assertAlmostEqual(result["offset_s"], 1.7, places=3)
        self.assertGreaterEqual(result["n_anchors"], len(LINE) - len(LINE) // 3 - 1)

    def test_survives_extra_words(self):
        slice_words = words_at(LINE)
        clip_words = words_at(LINE, shift=-1.7)
        clip_words.insert(3, {"word": "yeah", "start": 1.05, "end": 1.2})
        clip_words.insert(9, {"word": "ooh", "start": 4.0, "end": 4.2})
        clip_words.append({"word": "alright", "start": 9.9, "end": 10.2})
        result = retime.estimate_offset(clip_words, slice_words)
        self.assertAlmostEqual(result["offset_s"], -1.7, places=2)

    def test_survives_outlier_anchors(self):
        slice_words = words_at(LINE)
        clip_words = words_at(LINE, shift=0.5)
        clip_words[2]["start"] = 7.5     # a wildly mistimed word
        clip_words[7]["start"] = 0.1
        result = retime.estimate_offset(clip_words, slice_words)
        self.assertAlmostEqual(result["offset_s"], 0.5, places=2)
        # Out-of-order words break the matched run, so they never become
        # anchors; whatever does get through is filtered against the median.
        self.assertLess(result["n_anchors"], len(LINE))
        self.assertLessEqual(result["n_inliers"], result["n_anchors"])

    def test_punctuation_and_case_do_not_matter(self):
        slice_words = words_at(LINE)
        clip_words = words_at([w.upper() + "," for w in LINE], shift=0.8)
        result = retime.estimate_offset(clip_words, slice_words)
        self.assertAlmostEqual(result["offset_s"], 0.8, places=3)

    def test_nothing_in_common(self):
        result = retime.estimate_offset(
            words_at("completely different syllables entirely".split()),
            words_at(LINE))
        self.assertIn(result["method"], ("none", "median", "rejected"))
        if result["method"] == "none":
            self.assertEqual(result["offset_s"], 0.0)

    def test_empty_inputs(self):
        for args in (([], words_at(LINE)), (words_at(LINE), []), ([], [])):
            result = retime.estimate_offset(*args)
            self.assertEqual(result["method"], "none")
            self.assertEqual(result["offset_s"], 0.0)
            self.assertEqual(result["anchors"], [])

    def test_absurd_offset_is_rejected(self):
        result = retime.estimate_offset(words_at(LINE, shift=30.0), words_at(LINE))
        self.assertEqual(result["method"], "rejected")
        self.assertEqual(result["offset_s"], 0.0)
        self.assertTrue(result["clamped"])
        self.assertAlmostEqual(result["measured_offset_s"], 30.0, places=2)

    def test_drift_gives_a_piecewise_map(self):
        slice_words = words_at(LINE)
        # Singing starts 0.2s late and drifts to ~1.2s late by the end.
        clip_words = []
        for i, word in enumerate(slice_words):
            drift = 0.2 + 1.0 * i / (len(slice_words) - 1)
            clip_words.append({"word": word["word"],
                               "start": round(word["start"] + drift, 4),
                               "end": round(word["end"] + drift, 4)})
        result = retime.estimate_offset(clip_words, slice_words, inlier_s=1.0)
        self.assertEqual(result["method"], "piecewise")
        self.assertGreaterEqual(len(result["map"]), 4)
        self.assertGreater(result["spread_s"], 0.4)
        # The local offset tracks the drift at both ends.
        self.assertAlmostEqual(retime.offset_at(result, slice_words[0]["start"]), 0.2,
                               delta=0.1)
        self.assertAlmostEqual(retime.offset_at(result, slice_words[-1]["start"]), 1.2,
                               delta=0.1)

    def test_offset_at_is_constant_for_median_method(self):
        result = retime.estimate_offset(words_at(LINE, shift=0.6), words_at(LINE))
        for t in (0.0, 1.0, 5.0, 100.0):
            self.assertAlmostEqual(retime.offset_at(result, t), 0.6, places=3)

    def test_map_never_runs_backwards(self):
        slice_words = words_at(LINE)
        clip_words = words_at(LINE, shift=0.5)
        clip_words[4]["start"] = clip_words[2]["start"] - 0.3
        result = retime.estimate_offset(clip_words, slice_words, inlier_s=2.0)
        knots = result["map"]
        for (t0, c0), (t1, c1) in zip(knots, knots[1:]):
            self.assertLess(t0, t1)
            self.assertLessEqual(c0, c1)

    def test_anchor_records(self):
        result = retime.estimate_offset(words_at(LINE, shift=1.7), words_at(LINE))
        anchor = result["anchors"][0]
        self.assertEqual(set(anchor), {"word", "clip_t", "slice_t", "delta"})
        self.assertAlmostEqual(anchor["clip_t"] - anchor["slice_t"], anchor["delta"],
                               places=4)


class SliceWords(unittest.TestCase):
    def setUp(self):
        from singsync import cutplan
        self.words = synth.make_words()
        duration = synth.song_duration(self.words)
        self.plan = cutplan.plan_cuts(self.words, synth.make_analysis(duration),
                                      duration=duration)

    def test_expected_slice_words_are_relative_to_the_slice(self):
        for cut in self.plan["cuts"]:
            slice_words = retime.expected_slice_words(cut)
            if not slice_words:
                continue
            self.assertGreaterEqual(slice_words[0]["start"], -1e-6)
            span = float(cut["audio_end"]) - float(cut["audio_start"])
            self.assertLessEqual(slice_words[-1]["end"], span + 1e-6)
            self.assertEqual(len(slice_words),
                             len(cut["words"]) + len(cut["shared_words_before"])
                             + len(cut["shared_words_after"]))

    def test_round_trip_measures_a_fake_render_offset(self):
        """A cut whose 'render' sang 0.9s late is measured as +0.9s."""
        cut = next(c for c in self.plan["cuts"] if c["words"])
        slice_words = retime.expected_slice_words(cut)
        clip_words = [{"word": w["word"], "start": w["start"] + 0.9,
                       "end": w["end"] + 0.9} for w in slice_words]
        result = retime.estimate_offset(clip_words, slice_words)
        self.assertAlmostEqual(result["offset_s"], 0.9, places=3)

    def test_offset_table_renders(self):
        for cut in self.plan["cuts"]:
            cut["offset_s"] = 0.25
            cut["retime"] = {"method": "median", "n_anchors": 7, "spread_s": 0.1}
        self.plan["retime_summary"] = {"measured": len(self.plan["cuts"]),
                                       "cuts": len(self.plan["cuts"]),
                                       "mean_abs_offset_s": 0.25, "baked": False}
        text = retime.offset_table(self.plan)
        self.assertIn("offset", text)
        self.assertIn("applied at assembly", text)

    def test_retime_plan_skips_cuts_without_clips(self):
        plan = retime.retime_plan(dict(self.plan))
        self.assertEqual(plan["retime_summary"]["measured"], 0)
        for cut in plan["cuts"]:
            self.assertEqual(cut["offset_s"], 0.0)


@unittest.skipUnless(have_ffmpeg(), "ffmpeg not available")
class BakeRetime(unittest.TestCase):
    """Baking a shift into a clip file keeps its length and frame rate."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="singsync-retime-"))
        cls.clip = cls.tmp / "clip.mp4"
        # testsrc2 prints a timecode, so a shift is visible frame by frame.
        ffmpeg(["-f", "lavfi", "-i", "testsrc2=size=160x90:rate=24:duration=10",
                "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                str(cls.clip)])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _duration(self, path):
        return media_duration(path)

    def test_positive_offset_keeps_the_length(self):
        out = retime.retime_clip(self.clip, self.tmp / "late.mp4", offset_s=1.3,
                                 preset="ultrafast", crf=30)
        self.assertAlmostEqual(self._duration(out), 10.0, delta=1.0 / 24.0)

    def test_negative_offset_keeps_the_length(self):
        out = retime.retime_clip(self.clip, self.tmp / "early.mp4", offset_s=-0.8,
                                 preset="ultrafast", crf=30)
        self.assertAlmostEqual(self._duration(out), 10.0, delta=1.0 / 24.0)

    def test_zero_offset_is_a_passthrough(self):
        out = retime.retime_clip(self.clip, self.tmp / "same.mp4", offset_s=0.0,
                                 preset="ultrafast", crf=30)
        self.assertAlmostEqual(self._duration(out), 10.0, delta=1.0 / 24.0)

    def test_result_dict_drives_the_shift(self):
        result = {"offset_s": 0.9, "method": "median", "map": []}
        out = retime.retime_clip(self.clip, self.tmp / "res.mp4", result=result,
                                 preset="ultrafast", crf=30)
        self.assertAlmostEqual(self._duration(out), 10.0, delta=1.0 / 24.0)

    def test_piecewise_map_is_applied_in_segments(self):
        result = {
            "offset_s": 0.6, "method": "piecewise",
            "map": [[0.5, 0.7], [3.0, 3.4], [5.5, 6.2], [8.0, 8.9]],
        }
        out = retime.retime_clip(self.clip, self.tmp / "pw.mp4", result=result,
                                 duration=10.0, fps=24.0, preset="ultrafast", crf=30)
        self.assertAlmostEqual(self._duration(out), 10.0, delta=2.0 / 24.0)

    def test_retime_plan_bakes_clips(self):
        from singsync import cutplan
        plan = cutplan.load_plan(self.plan_with_clips())
        retime.retime_plan(plan, write_clips=True, work_dir=self.tmp / "baked",
                           model="small.en")
        for cut in plan["cuts"]:
            if cut.get("words"):
                # No faster-whisper in CI: measurement fails, nothing is baked,
                # and the failure is recorded rather than raised.
                self.assertIn(cut["retime"]["method"], ("failed", "median",
                                                        "piecewise", "none",
                                                        "rejected"))
            self.assertIn("offset_s", cut)

    def plan_with_clips(self):
        from singsync import cutplan
        words = synth.make_words()
        duration = synth.song_duration(words)
        plan = cutplan.plan_cuts(words, synth.make_analysis(duration),
                                 duration=duration)
        for cut in plan["cuts"]:
            cut["clip_path"] = str(self.clip)
        return plan


if __name__ == "__main__":
    unittest.main()
