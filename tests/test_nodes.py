"""Node registration and the plumbing that does not need ComfyUI."""
from __future__ import annotations

import os
import tempfile
import unittest
from pathlib import Path

from singsync import nodes


class Registration(unittest.TestCase):
    def test_expected_nodes_are_registered(self):
        for key in ("SingSyncSongAnalyze", "SingSyncCutPlan", "SingSyncCutSlicer",
                    "SingSyncRetime", "SingSyncAssemble"):
            self.assertIn(key, nodes.NODE_CLASS_MAPPINGS)
            self.assertIn(key, nodes.NODE_DISPLAY_NAME_MAPPINGS)
            self.assertTrue(nodes.NODE_DISPLAY_NAME_MAPPINGS[key].startswith("SingSync "))

    def test_node_contract(self):
        for key, cls in nodes.NODE_CLASS_MAPPINGS.items():
            spec = cls.INPUT_TYPES()
            self.assertIn("required", spec, key)
            self.assertIsInstance(spec["required"], dict, key)
            self.assertTrue(hasattr(cls, "RETURN_TYPES"), key)
            self.assertTrue(hasattr(cls, "FUNCTION"), key)
            self.assertEqual(getattr(cls, "CATEGORY", None), "SingSync", key)
            self.assertTrue(callable(getattr(cls, cls.FUNCTION, None)), key)
            if hasattr(cls, "RETURN_NAMES"):
                self.assertEqual(len(cls.RETURN_NAMES), len(cls.RETURN_TYPES), key)

    def test_plan_type_is_threaded_through(self):
        classes = nodes.NODE_CLASS_MAPPINGS
        for key in ("SingSyncCutPlan", "SingSyncCutSlicer", "SingSyncRetime",
                    "SingSyncAssemble"):
            self.assertEqual(classes[key].INPUT_TYPES()["required"]["plan"],
                             ("SINGSYNC_PLAN",))
        for key in ("SingSyncSongAnalyze", "SingSyncCutPlan", "SingSyncCutSlicer"):
            self.assertIn("SINGSYNC_PLAN", classes[key].RETURN_TYPES)

    def test_package_init_exports_mappings(self):
        # Imitate how ComfyUI loads custom_nodes/<folder>/__init__.py.
        import importlib.util
        import sys
        root = Path(__file__).resolve().parents[1] / "__init__.py"
        spec = importlib.util.spec_from_file_location(
            "singsync_pack", root, submodule_search_locations=[str(root.parent)])
        module = importlib.util.module_from_spec(spec)
        sys.modules["singsync_pack"] = module
        try:
            spec.loader.exec_module(module)
            self.assertIn("SingSyncCutPlan", module.NODE_CLASS_MAPPINGS)
            self.assertIn("SingSyncCutPlan", module.NODE_DISPLAY_NAME_MAPPINGS)
        finally:
            for name in [n for n in sys.modules if n.startswith("singsync_pack")]:
                sys.modules.pop(name, None)


class Plumbing(unittest.TestCase):
    def test_work_dir_follows_the_env_override(self):
        with tempfile.TemporaryDirectory() as tmp:
            old = os.environ.get("SINGSYNC_WORK_DIR")
            os.environ["SINGSYNC_WORK_DIR"] = tmp
            try:
                run_id, path = nodes.new_work_dir("test")
                self.assertTrue(path.is_dir())
                self.assertTrue(str(path).startswith(tmp))
                self.assertIn("test", run_id)
            finally:
                if old is None:
                    os.environ.pop("SINGSYNC_WORK_DIR", None)
                else:
                    os.environ["SINGSYNC_WORK_DIR"] = old

    def test_copy_rejects_non_plans(self):
        for bad in (None, 42, "plan", {}, {"cut": []}):
            with self.assertRaises(ValueError):
                nodes._copy(bad)

    def test_copy_is_deep(self):
        plan = {"cuts": [{"idx": 0, "words": [{"word": "a"}]}]}
        other = nodes._copy(plan)
        other["cuts"][0]["words"][0]["word"] = "b"
        self.assertEqual(plan["cuts"][0]["words"][0]["word"], "a")

    def test_audio_dict_to_wav(self):
        import numpy as np
        with tempfile.TemporaryDirectory() as tmp:
            audio = {"waveform": np.zeros((1, 2, 4410), dtype="float32"),
                     "sample_rate": 22050}
            path = nodes.audio_to_wav(audio, Path(tmp) / "a.wav")
            import wave
            with wave.open(path) as handle:
                self.assertEqual(handle.getnchannels(), 2)
                self.assertEqual(handle.getframerate(), 22050)
                self.assertEqual(handle.getnframes(), 4410)

    def test_missing_audio_path_is_reported(self):
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(FileNotFoundError):
                nodes._resolve_audio("/nope/missing.wav", None, Path(tmp), "song")
            self.assertEqual(nodes._resolve_audio("", None, Path(tmp), "song"), "")

    def test_caption_styles_are_offered(self):
        self.assertIn("bold_bottom", nodes.CAPTION_STYLES)
        spec = nodes.NODE_CLASS_MAPPINGS["SingSyncAssemble"].INPUT_TYPES()
        self.assertEqual(spec["required"]["caption_style"][0], nodes.CAPTION_STYLES)


class CutPlanNode(unittest.TestCase):
    """The planning node runs with no ffmpeg, no whisper and no ComfyUI."""

    def setUp(self):
        from tests import synth
        words = synth.make_words()
        duration = synth.song_duration(words)
        self.plan = {
            "run_id": "test", "work_dir": tempfile.mkdtemp(),
            "song_path": "song.wav", "vocal_path": "vocals.wav",
            "duration": duration, "analysis": synth.make_analysis(duration),
            "alignment": {"words": words, "lines": [], "lyrics_used": False,
                          "lyric_lines": []},
            "cuts": [], "config": {},
        }

    def test_builds_and_validates(self):
        node = nodes.NODE_CLASS_MAPPINGS["SingSyncCutPlan"]()
        plan, sheet = node.build(self.plan, target_seconds=9.0, overlap_words=2,
                                 max_clip_seconds=12.0, allowed_durations="5,10,15")
        self.assertTrue(plan["cuts"])
        self.assertIn("idx", sheet)
        self.assertEqual(plan["song_path"], "song.wav")
        self.assertEqual(self.plan["cuts"], [], "upstream plan must not be mutated")

    def test_snap_can_be_disabled(self):
        node = nodes.NODE_CLASS_MAPPINGS["SingSyncCutPlan"]()
        plan, _ = node.build(self.plan, snap_prefer="none")
        self.assertEqual(plan.get("beat_snap_log"), [])

    def test_downstream_nodes_refuse_an_unplanned_plan(self):
        slicer_node = nodes.NODE_CLASS_MAPPINGS["SingSyncCutSlicer"]()
        with self.assertRaises(ValueError):
            slicer_node.slice(self.plan)
        retime_node = nodes.NODE_CLASS_MAPPINGS["SingSyncRetime"]()
        with self.assertRaises(ValueError):
            retime_node.measure(self.plan)


if __name__ == "__main__":
    unittest.main()
