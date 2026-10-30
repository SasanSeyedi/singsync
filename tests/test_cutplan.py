"""Cut planner invariants."""
from __future__ import annotations

import unittest

from singsync import cutplan
from tests import synth


class PlanInvariants(unittest.TestCase):
    def setUp(self):
        self.words = synth.make_words()
        self.duration = synth.song_duration(self.words)
        self.analysis = synth.make_analysis(self.duration)

    def _plan(self, **kw):
        kw.setdefault("duration", self.duration)
        return cutplan.plan_cuts(self.words, self.analysis, **kw)

    def test_valid_plan(self):
        plan = self._plan()
        self.assertEqual(cutplan.validate_plan(plan, self.words), [])
        self.assertGreater(len(plan["cuts"]), 1)

    def test_owned_ranges_tile_the_song(self):
        plan = self._plan()
        cuts = plan["cuts"]
        self.assertAlmostEqual(cuts[0]["owns_start"], 0.0, places=6)
        self.assertAlmostEqual(cuts[-1]["owns_end"], plan["duration"], places=3)
        total = 0.0
        for prev, cut in zip(cuts, cuts[1:]):
            self.assertAlmostEqual(prev["owns_end"], cut["owns_start"], places=6)
        for cut in cuts:
            self.assertGreater(cut["owns_end"], cut["owns_start"])
            total += cut["owns_end"] - cut["owns_start"]
        self.assertAlmostEqual(total, plan["duration"], places=3)

    def test_audio_span_contains_owned_range(self):
        for plan in (self._plan(), self._plan(overlap_words=0),
                     self._plan(overlap_words=3)):
            for cut in plan["cuts"]:
                self.assertLessEqual(cut["audio_start"], cut["owns_start"] + 1e-6)
                self.assertGreaterEqual(cut["audio_end"], cut["owns_end"] - 1e-6)

    def test_max_audio_span_respected(self):
        for max_s in (8.0, 10.0, 12.0, 15.0):
            plan = self._plan(max_audio_s=max_s, allowed_durations="5,10,15,20")
            for cut in plan["cuts"]:
                span = cut["audio_end"] - cut["audio_start"]
                self.assertLessEqual(span, max_s + 1e-6,
                                     "cut %d span %.3f > %.1f" % (cut["idx"], span, max_s))
            self.assertEqual(cutplan.validate_plan(plan, self.words), [])

    def test_durations_are_allowed_buckets(self):
        plan = self._plan(allowed_durations="5,10,15")
        for cut in plan["cuts"]:
            self.assertIn(cut["duration"], (5.0, 10.0, 15.0))
            self.assertGreaterEqual(cut["duration"] + 1e-6,
                                    cut["audio_end"] - cut["audio_start"])
        # The bucket must be the *smallest* one that fits.
        for cut in plan["cuts"]:
            span = cut["audio_end"] - cut["audio_start"]
            smaller = [d for d in (5.0, 10.0, 15.0) if d < cut["duration"]]
            for value in smaller:
                self.assertLess(value + 1e-6, span)

    def test_pick_duration(self):
        self.assertEqual(cutplan.pick_duration(4.2, "5,10,15"), 5.0)
        self.assertEqual(cutplan.pick_duration(5.0, "5,10,15"), 5.0)
        self.assertEqual(cutplan.pick_duration(5.01, "5,10,15"), 10.0)
        self.assertEqual(cutplan.pick_duration(99.0, "5,10,15"), 15.0)
        self.assertEqual(cutplan.parse_durations("4, 8 ,8,12"), (4.0, 8.0, 12.0))

    def test_overlap_words_come_from_the_neighbours(self):
        plan = self._plan(overlap_words=2)
        vocal = [c for c in plan["cuts"] if c["kind"] == "vocal"]
        self.assertGreater(len(vocal), 2)
        for cut in vocal:
            self.assertLessEqual(len(cut["shared_words_before"]), 2)
            self.assertLessEqual(len(cut["shared_words_after"]), 2)
        for prev, cut in zip(vocal, vocal[1:]):
            if prev["kind"] != "vocal" or cut["kind"] != "vocal":
                continue
            if not cut["shared_words_before"]:
                continue
            # Words shared backwards are exactly the tail of the previous cut.
            tail = [w["word"] for w in prev["words"]][-len(cut["shared_words_before"]):]
            self.assertEqual([w["word"] for w in cut["shared_words_before"]], tail)
            # ...and the audio actually reaches them.
            self.assertLessEqual(cut["audio_start"],
                                 cut["shared_words_before"][0]["start"] + 1e-6)

    def test_overlap_zero_leaves_no_shared_words(self):
        plan = self._plan(overlap_words=0)
        for cut in plan["cuts"]:
            self.assertEqual(cut["shared_words_before"], [])
            self.assertEqual(cut["shared_words_after"], [])
        self.assertEqual(cutplan.validate_plan(plan, self.words), [])

    def test_every_word_is_owned_exactly_once(self):
        plan = self._plan()
        owned = [w for cut in plan["cuts"] for w in cut["words"]]
        self.assertEqual(len(owned), len(self.words))
        for mine, theirs in zip(owned, self.words):
            self.assertEqual(mine["word"], theirs["word"])
            self.assertAlmostEqual(mine["start"], theirs["start"], places=6)

    def test_prefers_line_ends(self):
        plan = self._plan(target_owned_s=9.0, line_snap_frac=0.6)
        vocal = [c for c in plan["cuts"] if c["kind"] == "vocal"]
        ends_on_line = sum(1 for c in vocal if c["words"] and c["words"][-1]["line_end"])
        self.assertGreaterEqual(ends_on_line, max(1, len(vocal) - 1))

    def test_broll_region_for_long_instrumental(self):
        words = synth.make_words(instrumental_after=2, instrumental_s=9.0)
        duration = synth.song_duration(words)
        plan = cutplan.plan_cuts(words, synth.make_analysis(duration), duration=duration,
                                 gap_broll_s=4.0)
        broll = [c for c in plan["cuts"] if c["kind"] == "broll"]
        self.assertTrue(broll, "a 9s instrumental gap should produce b-roll cuts")
        for cut in broll:
            self.assertEqual(cut["words"], [])
            self.assertLessEqual(cut["audio_end"] - cut["audio_start"],
                                 plan["config"]["max_span_s"] + 1e-6)
        self.assertEqual(cutplan.validate_plan(plan, words), [])

    def test_short_gaps_do_not_make_broll(self):
        words = synth.make_words(instrumental_after=2, instrumental_s=2.0)
        duration = synth.song_duration(words)
        plan = cutplan.plan_cuts(words, synth.make_analysis(duration), duration=duration,
                                 gap_broll_s=4.0)
        self.assertEqual([c for c in plan["cuts"] if c["kind"] == "broll"], [])

    def test_leading_and_trailing_silence_is_covered(self):
        words = synth.make_words(start=11.0)
        duration = synth.song_duration(words, tail=10.0)
        plan = cutplan.plan_cuts(words, synth.make_analysis(duration), duration=duration)
        self.assertEqual(cutplan.validate_plan(plan, words), [])
        self.assertEqual(plan["cuts"][0]["kind"], "broll")
        self.assertEqual(plan["cuts"][-1]["kind"], "broll")

    def test_beat_snapping_moves_boundaries_to_downbeats(self):
        plan_raw = self._plan(snap=False)
        plan = self._plan(snap=True, snap_tol_s=0.35)
        self.assertTrue(plan["beat_snap_log"], "expected at least one snapped boundary")
        downbeats = set(self.analysis["downbeats"])
        for entry in plan["beat_snap_log"]:
            self.assertIn(round(entry["to"], 4), downbeats)
            self.assertLessEqual(abs(entry["delta"]), 0.35 + 1e-9)
        self.assertEqual(cutplan.validate_plan(plan, self.words), [])
        self.assertEqual(len(plan["cuts"]), len(plan_raw["cuts"]))

    def test_snapping_never_splits_a_word(self):
        plan = self._plan(snap=True, snap_tol_s=1.0)
        for cut in plan["cuts"]:
            for edge in (cut["owns_start"], cut["owns_end"]):
                for word in self.words:
                    self.assertFalse(word["start"] + 1e-6 < edge < word["end"] - 1e-6,
                                     "boundary %.4f splits %r" % (edge, word["word"]))

    def test_no_words_gives_all_broll(self):
        plan = cutplan.plan_cuts([], None, duration=30.0)
        self.assertTrue(plan["cuts"])
        self.assertTrue(all(c["kind"] == "broll" for c in plan["cuts"]))
        self.assertEqual(cutplan.validate_plan(plan), [])

    def test_single_word_song(self):
        words = [{"word": "oh", "start": 0.2, "end": 0.9, "line_idx": 0, "line_end": True}]
        plan = cutplan.plan_cuts(words, None, duration=3.0)
        self.assertEqual(cutplan.validate_plan(plan, words), [])

    def test_validate_catches_broken_plans(self):
        plan = self._plan()
        plan["cuts"][1]["owns_start"] += 0.5
        plan["cuts"][1]["owns"][0] += 0.5
        self.assertTrue(cutplan.validate_plan(plan))

    def test_requires_a_duration(self):
        with self.assertRaises(ValueError):
            cutplan.plan_cuts([], None)

    def test_cut_sheet_renders(self):
        sheet = cutplan.cut_sheet(self._plan())
        self.assertIn("idx", sheet)
        self.assertIn("cuts, owned total", sheet)
        self.assertEqual(len(sheet.splitlines()),
                         len(self._plan()["cuts"]) + 4)

    def test_line_text_uses_lyrics_when_given(self):
        alignment = {
            "words": self.words,
            "lyrics_used": True,
            "lyric_lines": [{"line_idx": i, "text": t} for i, t in enumerate(synth.LINES)],
        }
        plan = cutplan.plan_cuts(self.words, self.analysis, duration=self.duration,
                                 alignment=alignment)
        texts = " ".join(c["line_text"] for c in plan["cuts"])
        self.assertIn(synth.LINES[0], texts)

    def test_plan_is_json_serialisable(self):
        import json
        plan = self._plan()
        self.assertEqual(json.loads(json.dumps(plan))["duration"], plan["duration"])


if __name__ == "__main__":
    unittest.main()
