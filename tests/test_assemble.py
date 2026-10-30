"""Assembly: the frame grid, and the end-to-end length invariant."""
from __future__ import annotations

import shutil
import tempfile
import unittest
from pathlib import Path

from singsync import assemble, cutplan
from singsync.ffmpeg_util import ffmpeg, have_ffmpeg, media_duration
from tests import synth

COLOURS = ["0x1b2a41", "0x324a5f", "0x3e5c76", "0x748cab", "0xf0ebd8",
           "0x5c374c", "0x985277", "0xff8c42"]


def _slate(path: Path, seconds: float, colour: str, label: str, fps: float = 24.0):
    """A solid-colour clip standing in for a rendered cut."""
    ffmpeg(["-f", "lavfi", "-i",
            "color=c=%s:s=320x180:r=%g:d=%.3f" % (colour, fps, seconds),
            "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
            str(path)])


def _tone(path: Path, seconds: float):
    ffmpeg(["-f", "lavfi", "-i", "sine=frequency=330:duration=%.4f" % seconds,
            "-c:a", "pcm_s16le", str(path)])


class FrameGrid(unittest.TestCase):
    """Pure tests: no ffmpeg needed."""

    def setUp(self):
        words = synth.make_words(instrumental_after=3, instrumental_s=9.0)
        self.duration = synth.song_duration(words)
        self.plan = cutplan.plan_cuts(words, synth.make_analysis(self.duration),
                                      duration=self.duration)

    def test_clip_time_mapping(self):
        cut = {"audio_start": 6.5, "owns_start": 8.0, "owns_end": 16.0}
        self.assertAlmostEqual(assemble.clip_time(8.0, cut), 1.5)
        self.assertAlmostEqual(assemble.clip_time(8.0, cut, 0.4), 1.9)
        self.assertAlmostEqual(assemble.clip_time(6.5, cut, -0.25), -0.25)

    def test_frames_sum_to_the_song(self):
        for fps in (24.0, 25.0, 30.0):
            segments = assemble.segment_plan(self.plan, fps)
            total = sum(s["frames"] for s in segments)
            self.assertEqual(total, assemble.frame_index(self.duration, fps))
            self.assertAlmostEqual(total / fps, self.duration, delta=1.0 / fps)

    def test_segments_are_contiguous_on_the_grid(self):
        fps = 24.0
        segments = assemble.segment_plan(self.plan, fps)
        self.assertEqual(len(segments), len(self.plan["cuts"]))
        frame = 0
        for seg in segments:
            self.assertEqual(frame, assemble.frame_index(seg["owns_start"], fps))
            frame += seg["frames"]

    def test_negative_clip_start_becomes_a_front_pad(self):
        plan = cutplan.load_plan(self.plan)
        cut = plan["cuts"][1]
        cut["offset_s"] = -1.5
        seg = assemble.segment_plan(plan, 24.0)[1]
        self.assertGreater(seg["pad_start_s"], 0.0)
        self.assertEqual(seg["clip_start"], 0.0)

    def test_fps_must_be_positive(self):
        with self.assertRaises(ValueError):
            assemble.segment_plan(self.plan, 0)

    def test_report_renders(self):
        text = assemble.assembly_report(self.plan, 24.0)
        self.assertIn("segments", text)
        self.assertIn("MISSING", text)

    def test_fcpxml_export(self):
        tmp = Path(tempfile.mkdtemp())
        try:
            path = assemble.write_fcpxml(self.plan, tmp / "edit.fcpxml", fps=24.0)
            text = Path(path).read_text(encoding="utf-8")
            self.assertIn("<fcpxml", text)
            self.assertEqual(text.count("<asset-clip"), len(self.plan["cuts"]))
        finally:
            shutil.rmtree(tmp, ignore_errors=True)


@unittest.skipUnless(have_ffmpeg(), "ffmpeg not available")
class AssembleLength(unittest.TestCase):
    """End-to-end: synthetic slates in, one video of exactly song length out."""

    @classmethod
    def setUpClass(cls):
        cls.tmp = Path(tempfile.mkdtemp(prefix="singsync-assemble-"))
        words = synth.make_words()
        cls.duration = synth.song_duration(words)
        cls.plan = cutplan.plan_cuts(words, synth.make_analysis(cls.duration),
                                     duration=cls.duration)
        cls.song = cls.tmp / "song.wav"
        _tone(cls.song, cls.duration)
        for cut in cls.plan["cuts"]:
            clip = cls.tmp / ("cut_%04d.mp4" % cut["idx"])
            _slate(clip, float(cut["duration"]), COLOURS[cut["idx"] % len(COLOURS)],
                   "cut %d" % cut["idx"])
            cut["clip_path"] = str(clip)

    @classmethod
    def tearDownClass(cls):
        shutil.rmtree(cls.tmp, ignore_errors=True)

    def _assemble(self, name, **kw):
        plan = cutplan.load_plan(self.plan)
        for cut, original in zip(plan["cuts"], self.plan["cuts"]):
            cut["clip_path"] = original["clip_path"]
        out = self.tmp / name
        kw.setdefault("crf", 30)
        kw.setdefault("preset", "ultrafast")
        report = assemble.assemble(plan, self.song, out, **kw)
        return plan, report, out

    def test_final_length_matches_the_song(self):
        fps = 24.0
        plan, report, out = self._assemble("out.mp4", fps=fps)
        self.assertTrue(out.exists())
        one_frame = 1.0 / fps
        self.assertAlmostEqual(report["actual_duration"], self.duration, delta=one_frame)
        self.assertAlmostEqual(media_duration(out), self.duration, delta=one_frame)
        self.assertEqual(report["frames"], assemble.frame_index(self.duration, fps))
        self.assertLessEqual(abs(report["drift_frames"]), 1.0)
        self.assertEqual(report["segments"], len(plan["cuts"]))

    def test_length_holds_with_measured_offsets(self):
        fps = 24.0
        plan = cutplan.load_plan(self.plan)
        for n, cut in enumerate(plan["cuts"]):
            cut["clip_path"] = self.plan["cuts"][n]["clip_path"]
            cut["offset_s"] = [0.0, 0.7, -0.45, 1.3, -1.1][n % 5]
        out = self.tmp / "offsets.mp4"
        report = assemble.assemble(plan, self.song, out, fps=fps, crf=30,
                                   preset="ultrafast")
        self.assertAlmostEqual(report["actual_duration"], self.duration, delta=1.0 / fps)

    def test_master_audio_is_used(self):
        _, report, out = self._assemble("audio.mp4", fps=24.0)
        from singsync.ffmpeg_util import probe
        streams = {s["codec_type"]: s for s in probe(out)["streams"]}
        self.assertIn("audio", streams)
        self.assertIn("video", streams)
        self.assertAlmostEqual(float(streams["audio"]["duration"]), self.duration,
                               delta=0.15)

    def test_missing_clip_is_reported(self):
        plan = cutplan.load_plan(self.plan)
        for cut in plan["cuts"]:
            cut.pop("clip_path", None)
        with self.assertRaises(FileNotFoundError):
            assemble.assemble(plan, self.song, self.tmp / "nope.mp4")

    def test_other_frame_rate(self):
        fps = 30.0
        _, report, out = self._assemble("fps30.mp4", fps=fps)
        self.assertAlmostEqual(report["actual_duration"], self.duration, delta=1.0 / fps)
        self.assertEqual(report["frames"], assemble.frame_index(self.duration, fps))

    def test_burnt_in_captions(self):
        plan, report, out = self._assemble("captions.mp4", fps=24.0, captions=True)
        self.assertTrue(report["captions"])
        self.assertTrue(Path(report["captions"]).exists())
        self.assertAlmostEqual(report["actual_duration"], self.duration, delta=1.0 / 24.0)


if __name__ == "__main__":
    unittest.main()
