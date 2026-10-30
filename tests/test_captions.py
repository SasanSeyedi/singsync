"""Karaoke caption generation (pure text)."""
from __future__ import annotations

import unittest

from singsync import captions
from tests import synth


class AssText(unittest.TestCase):
    def test_lyric_corrected_words_group_by_lyric_line(self):
        from singsync.align import apply_lyrics
        fixed = apply_lyrics([
            {"word": "helo", "start": 0.0, "end": 0.4},
            {"word": "wrld", "start": 0.6, "end": 1.0}], "Hello world!\nAgain")
        self.assertEqual([line[0]["line_idx"] for line in captions.group_lines(fixed)], [0, 1])
        ass = captions.build_ass(fixed)
        self.assertIn("Hello", ass)
        self.assertIn("world!", ass)
        self.assertIn("Again", ass)

    def setUp(self):
        self.words = synth.make_words()

    def test_header_and_style(self):
        text = captions.build_ass(self.words, size=(1920, 1080))
        self.assertIn("[Script Info]", text)
        self.assertIn("ScriptType: v4.00+", text)
        self.assertIn("PlayResX: 1920", text)
        self.assertIn("PlayResY: 1080", text)
        self.assertIn("[V4+ Styles]", text)
        self.assertIn("Style: SingSync,", text)
        self.assertIn("[Events]", text)
        self.assertTrue(text.endswith("\n"))

    def test_font_size_scales_with_height(self):
        small = captions.build_ass(self.words, size=(640, 360))
        large = captions.build_ass(self.words, size=(1920, 1080))
        def size_of(text):
            line = [l for l in text.splitlines() if l.startswith("Style: SingSync,")][0]
            return int(line.split(",")[2])
        self.assertLess(size_of(small), size_of(large))

    def test_one_dialogue_per_lyric_line(self):
        text = captions.build_ass(self.words)
        events = [l for l in text.splitlines() if l.startswith("Dialogue:")]
        self.assertEqual(len(events), len(synth.LINES))
        self.assertIn(synth.LINES[0].split()[0], events[0])

    def test_karaoke_durations_match_word_lengths(self):
        line = [
            {"word": "hold", "start": 1.0, "end": 1.4, "line_idx": 0},
            {"word": "the", "start": 1.4, "end": 1.6, "line_idx": 0},
            {"word": "line", "start": 2.0, "end": 2.75, "line_idx": 0},
        ]
        text = captions.karaoke_text(line)
        # Adjacent words share a plain space; the 0.4s gap gets its own \k
        # spacer so "line" lights up exactly when it is sung.
        self.assertEqual(text, "{\\k40}hold {\\k20}the{\\k40} {\\k75}line")

    def test_gaps_become_unsung_spacers(self):
        line = [
            {"word": "a", "start": 0.0, "end": 0.5, "line_idx": 0},
            {"word": "b", "start": 1.5, "end": 2.0, "line_idx": 0},
        ]
        text = captions.karaoke_text(line)
        # 1.0s gap -> a 100cs spacer that highlights nothing.
        self.assertIn("{\\k100} ", text)
        self.assertEqual(sum(int(v) for v in _k_values(text)), 200)

    def test_timestamps(self):
        self.assertEqual(captions.ass_time(0.0), "0:00:00.00")
        self.assertEqual(captions.ass_time(3.456), "0:00:03.46")
        self.assertEqual(captions.ass_time(61.5), "0:01:01.50")
        self.assertEqual(captions.ass_time(3723.25), "1:02:03.25")
        self.assertEqual(captions.ass_time(-5.0), "0:00:00.00")

    def test_dialogue_times_bracket_the_line(self):
        text = captions.build_ass(self.words)
        events = [l for l in text.splitlines() if l.startswith("Dialogue:")]
        first = events[0].split(",")
        self.assertEqual(first[1], captions.ass_time(self.words[0]["start"]))
        # Lines never overlap each other.
        starts = [e.split(",")[1] for e in events]
        ends = [e.split(",")[2] for e in events]
        for end, nxt in zip(ends, starts[1:]):
            self.assertLessEqual(end, nxt)

    def test_braces_are_escaped(self):
        words = [{"word": "{tricky}", "start": 0.0, "end": 0.5, "line_idx": 0}]
        text = captions.build_ass(words)
        self.assertIn("\\{tricky\\}", text)

    def test_line_grouping_falls_back_to_gaps(self):
        words = [
            {"word": "one", "start": 0.0, "end": 0.3},
            {"word": "two", "start": 0.35, "end": 0.6},
            {"word": "three", "start": 2.0, "end": 2.4},
        ]
        lines = captions.group_lines(words)
        self.assertEqual([len(l) for l in lines], [2, 1])

    def test_line_grouping_honours_max_words(self):
        words = [{"word": "w%d" % i, "start": i * 0.3, "end": i * 0.3 + 0.25,
                  "line_idx": 0} for i in range(20)]
        lines = captions.group_lines(words, max_words=4)
        self.assertTrue(all(len(l) <= 4 for l in lines))
        self.assertEqual(sum(len(l) for l in lines), 20)

    def test_empty_words_give_no_events(self):
        text = captions.build_ass([])
        self.assertNotIn("Dialogue:", text)
        self.assertIn("[Events]", text)

    def test_style_presets(self):
        names = captions.style_names()
        self.assertEqual(names[0], captions.DEFAULT_STYLE)
        for name in names:
            text = captions.build_ass(self.words, style=name)
            self.assertIn("Style: SingSync,", text)
        # An unknown preset falls back instead of raising.
        self.assertIn("Style: SingSync,", captions.build_ass(self.words, style="nope"))

    def test_words_from_plan(self):
        from singsync import cutplan
        duration = synth.song_duration(self.words)
        plan = cutplan.plan_cuts(self.words, synth.make_analysis(duration),
                                 duration=duration)
        self.assertEqual(len(captions.words_from_plan(plan)), len(self.words))

    def test_write_ass(self):
        import tempfile
        from pathlib import Path
        with tempfile.TemporaryDirectory() as tmp:
            path = captions.write_ass(self.words, Path(tmp) / "sub" / "c.ass")
            self.assertTrue(Path(path).exists())
            self.assertIn("Dialogue:", Path(path).read_text(encoding="utf-8"))


def _k_values(text):
    import re
    return re.findall(r"\\[kK](\d+)", text)


if __name__ == "__main__":
    unittest.main()
