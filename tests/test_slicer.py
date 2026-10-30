"""Prompt building and audio slicing."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from singsync import cutplan, slicer
from singsync.ffmpeg_util import ffmpeg, have_ffmpeg, media_duration
from tests import synth

CUT = {
    "idx": 3,
    "kind": "vocal",
    "owns_start": 24.0, "owns_end": 32.0,
    "audio_start": 23.1, "audio_end": 33.2, "duration": 15.0,
    "words": [{"word": "hold", "start": 24.2, "end": 24.6}],
    "line_text": "hold the line until it breaks",
    "section": "drop", "energy": 0.82,
    "vibe": "section=drop, energy=0.82, rising; cut lands on downbeat 1 of bar 9",
}
BROLL = dict(CUT, idx=4, kind="broll", words=[], line_text="")


class Prompts(unittest.TestCase):
    def test_default_template(self):
        prompt = slicer.build_prompt(
            CUT, style="35mm, handheld, dusk", character="a tall singer in a red coat",
            scene="an empty car park")
        self.assertTrue(prompt.startswith("35mm, handheld, dusk."))
        self.assertIn("a tall singer in a red coat", prompt)
        self.assertIn("an empty car park", prompt)
        self.assertIn('The character sings: "hold the line until it breaks"', prompt)
        self.assertIn("section=drop", prompt)
        self.assertTrue(prompt.endswith("."))

    def test_instrumental_cut_says_no_singing(self):
        prompt = slicer.build_prompt(BROLL, style="35mm", scene="a car park",
                                     character="a singer")
        self.assertIn("no singing", prompt)
        self.assertNotIn("sings:", prompt)

    def test_empty_fields_do_not_leave_stray_dots(self):
        prompt = slicer.build_prompt(CUT)
        self.assertNotIn(" . ", prompt)
        self.assertNotIn("..", prompt)
        self.assertTrue(prompt.startswith("The character sings:"))

    def test_custom_template_and_fields(self):
        prompt = slicer.build_prompt(
            CUT, character="her", scene="a rooftop",
            template="cut {idx} ({duration}s, {section}): {character} on {scene} "
                     "singing {line_text}")
        self.assertIn("cut 3 (15.0s, drop): her on a rooftop", prompt)

    def test_unknown_template_field_is_reported(self):
        with self.assertRaises(ValueError):
            slicer.build_prompt(CUT, template="{nope}")

    def test_line_text_whitespace_is_collapsed(self):
        cut = dict(CUT, line_text="hold   the\nline")
        self.assertIn('sings: "hold the line"', slicer.build_prompt(cut))

    def test_prompt_sheet_and_job_list(self):
        plan = {"cuts": [dict(CUT, prompt="p3", slice_path="/a/3.wav"),
                         dict(BROLL, prompt="p4", slice_path="/a/4.wav")]}
        sheet = slicer.prompt_sheet(plan)
        self.assertIn("[3] 15s  p3", sheet)
        jobs = slicer.slice_list(plan)
        self.assertEqual([j["idx"] for j in jobs], [3, 4])
        self.assertEqual(jobs[0]["kind"], "vocal")
        self.assertEqual(jobs[1]["kind"], "broll")


@unittest.skipUnless(have_ffmpeg(), "ffmpeg not available")
class Slicing(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="singsync-slicer-"))
        cls.audio = cls.tmp / "song.wav"
        ffmpeg(["-f", "lavfi", "-i", "sine=frequency=440:duration=30",
                "-c:a", "pcm_s16le", str(cls.audio)])

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def test_slice_length(self):
        out = slicer.slice_audio(self.audio, 4.0, 9.5, self.tmp / "s.wav")
        self.assertAlmostEqual(media_duration(out), 5.5, delta=0.05)

    def test_slice_past_the_end_is_padded(self):
        out = slicer.slice_audio(self.audio, 28.0, 36.0, self.tmp / "pad.wav")
        self.assertAlmostEqual(media_duration(out), 8.0, delta=0.05)

    def test_mp3_output(self):
        out = slicer.slice_audio(self.audio, 1.0, 4.0, self.tmp / "s.mp3")
        self.assertTrue(Path(out).exists())
        self.assertAlmostEqual(media_duration(out), 3.0, delta=0.1)

    def test_bad_format_and_range(self):
        with self.assertRaises(ValueError):
            slicer.slice_audio(self.audio, 0.0, 1.0, self.tmp / "s.ogg")
        with self.assertRaises(ValueError):
            slicer.slice_audio(self.audio, 5.0, 5.0, self.tmp / "s2.wav")

    def test_slice_plan_covers_every_cut(self):
        words = synth.make_words()
        duration = synth.song_duration(words)
        plan = cutplan.plan_cuts(words, synth.make_analysis(duration),
                                 duration=duration)
        slicer.slice_plan(plan, self.audio, self.tmp / "run", character="a singer",
                          scene="a rooftop", style="35mm")
        self.assertEqual(plan["slices"]["count"], len(plan["cuts"]))
        for cut in plan["cuts"]:
            path = Path(cut["slice_path"])
            self.assertTrue(path.exists())
            span = float(cut["audio_end"]) - float(cut["audio_start"])
            self.assertAlmostEqual(media_duration(path), span, delta=0.06)
            self.assertTrue(cut["prompt"])

    def test_missing_source(self):
        with self.assertRaises(FileNotFoundError):
            slicer.slice_plan({"cuts": []}, self.tmp / "nope.wav", self.tmp)


if __name__ == "__main__":
    unittest.main()
