"""Word cleaning and lyric-line alignment (no faster-whisper needed)."""
from __future__ import annotations

import json
import tempfile
import unittest
from pathlib import Path

from singsync import align

LYRICS = """I walk the wire over bright water
counting every light below me
you said the morning would be kinder"""


def raw(tokens, start=0.0, word_s=0.4, gap_s=0.1):
    out = []
    t = start
    for token in tokens:
        out.append({"word": token, "start": round(t, 4), "end": round(t + word_s, 4)})
        t += word_s + gap_s
    return out


class Normalise(unittest.TestCase):
    def test_strips_case_and_punctuation(self):
        self.assertEqual(align.normalise_word(" Water, "), "water")
        self.assertEqual(align.normalise_word("WIRE!"), "wire")
        self.assertEqual(align.normalise_word("don’t"), "don't")
        self.assertEqual(align.normalise_word("café"), "cafe")

    def test_non_lexical_tokens_become_empty(self):
        for token in ("...", "--", " ", "♪", ""):
            self.assertEqual(align.normalise_word(token), "")

    def test_tokenise_lyrics(self):
        lines = align.tokenise_lyrics("One two\n\n  \nThree!\n")
        self.assertEqual([l["text"] for l in lines], ["One two", "Three!"])
        self.assertEqual([l["line_idx"] for l in lines], [0, 1])
        self.assertEqual(lines[1]["tokens"], ["three"])


class BuildAlignment(unittest.TestCase):
    def test_apply_lyrics_corrects_words_and_fills_unheard_tokens(self):
        words = raw(["I", "walk", "water"], start=0.0, word_s=0.4, gap_s=0.2)
        fixed = align.apply_lyrics(words, "I WALK the water!\nAgain.")
        self.assertEqual([w["word"] for w in fixed], ["I", "WALK", "the", "water!", "Again."])
        self.assertEqual([w["line_idx"] for w in fixed], [0, 0, 0, 0, 1])
        self.assertAlmostEqual(fixed[2]["start"], words[1]["end"])

    def test_drops_junk_and_keeps_order(self):
        words = raw(["hello", "...", "world"])
        out = align.build_alignment(words)
        self.assertEqual([w["word"] for w in out["words"]], ["hello", "world"])

    def test_swapped_times_are_repaired(self):
        out = align.build_alignment([{"word": "a", "start": 2.0, "end": 1.0}])
        self.assertEqual(out["words"][0]["start"], 1.0)
        self.assertEqual(out["words"][0]["end"], 2.0)

    def test_overlapping_words_are_made_monotonic(self):
        out = align.build_alignment([
            {"word": "a", "start": 0.0, "end": 1.0},
            {"word": "b", "start": 0.5, "end": 1.4},
        ])
        first, second = out["words"]
        self.assertGreaterEqual(second["start"], first["end"])
        self.assertGreaterEqual(second["end"], second["start"])

    def test_lines_from_gaps_without_lyrics(self):
        words = raw(["one", "two"]) + raw(["three", "four"], start=5.0)
        out = align.build_alignment(words)
        self.assertEqual([w["line_idx"] for w in out["words"]], [0, 0, 1, 1])
        self.assertEqual([w["line_end"] for w in out["words"]],
                         [False, True, False, True])
        self.assertEqual(len(out["lines"]), 2)
        self.assertFalse(out["lyrics_used"])

    def test_alignment_to_lyrics(self):
        tokens = [w for line in LYRICS.splitlines() for w in line.split()]
        out = align.build_alignment(raw(tokens), LYRICS)
        self.assertTrue(out["lyrics_used"])
        self.assertEqual(len(out["lines"]), 3)
        self.assertEqual([w["line_idx"] for w in out["words"]][:7], [0] * 7)
        self.assertEqual(out["lines"][0]["lyric_text"], LYRICS.splitlines()[0])
        ends = [i for i, w in enumerate(out["words"]) if w["line_end"]]
        self.assertEqual(ends, [6, 11, 18])   # 7, 5 and 7 words per line

    def test_mishearings_inherit_a_line(self):
        tokens = "I walk the wire over bright water yeah ooh counting every light " \
                 "below me".split()
        out = align.build_alignment(raw(tokens), LYRICS)
        self.assertTrue(all("line_idx" in w for w in out["words"]))
        indices = [w["line_idx"] for w in out["words"]]
        self.assertEqual(sorted(indices), indices, "line indices must not go backwards")
        self.assertEqual(indices[0], 0)
        self.assertEqual(indices[-1], 1)

    def test_dropped_words_still_align(self):
        tokens = "I walk the over bright water counting light below me".split()
        out = align.build_alignment(raw(tokens), LYRICS)
        indices = [w["line_idx"] for w in out["words"]]
        self.assertEqual(sorted(indices), indices)
        self.assertEqual(set(indices), {0, 1})

    def test_empty_input(self):
        out = align.build_alignment([])
        self.assertEqual(out["words"], [])
        self.assertEqual(out["lines"], [])

    def test_metadata_is_carried(self):
        out = align.build_alignment(raw(["a"]), meta={"model": "small.en"})
        self.assertEqual(out["model"], "small.en")

    def test_tuple_input(self):
        out = align.build_alignment([("hey", 0.0, 0.4), ("there", 0.5, 0.9)])
        self.assertEqual(len(out["words"]), 2)


class LoadSave(unittest.TestCase):
    def test_round_trip(self):
        out = align.build_alignment(raw(["one", "two"]), "One two")
        with tempfile.TemporaryDirectory() as tmp:
            path = align.save_alignment(out, Path(tmp) / "a.json")
            again = align.load_alignment(path)
            self.assertEqual(again["words"], out["words"])
            self.assertEqual(align.load_alignment(json.dumps(out))["words"],
                             out["words"])
            self.assertEqual(align.load_alignment(out)["words"], out["words"])

    def test_defaults_are_filled_in(self):
        loaded = align.load_alignment({"words": [{"word": "a", "start": 0, "end": 1}]})
        self.assertEqual(loaded["words"][0]["line_idx"], 0)
        self.assertTrue(loaded["lines"])

    def test_bad_payload(self):
        with self.assertRaises(ValueError):
            align.load_alignment({"words": "nope"})


class FakeWord:
    def __init__(self, word, start, end):
        self.word, self.start, self.end = word, start, end


class FakeSegment:
    def __init__(self, words):
        self.words = [FakeWord(*w) for w in words]


class FakeInfo:
    language = "en"
    duration = 6.0


class WhisperInterface(unittest.TestCase):
    """How ``transcribe_words`` consumes faster-whisper's objects.

    This pins the shape of the call and of the segment/word attributes we read.
    It is a stub, not a transcription: it says nothing about whisper's accuracy.
    """

    def setUp(self):
        import sys
        import types
        self.calls = []
        module = types.ModuleType("faster_whisper")
        test = self

        class WhisperModel:
            def __init__(self, model, device="cpu", compute_type="int8"):
                test.calls.append(("init", model, device, compute_type))

            def transcribe(self, path, **kw):
                test.calls.append(("transcribe", path, kw))
                return ([FakeSegment([("I", 0.5, 0.8), (" walk", 0.9, 1.3)]),
                         FakeSegment([(" the", 1.4, 1.6), (" wire", 1.7, 2.2)])],
                        FakeInfo())

        module.WhisperModel = WhisperModel
        self.previous = sys.modules.get("faster_whisper")
        sys.modules["faster_whisper"] = module
        align._MODEL_CACHE.clear()
        self.sys = sys

    def tearDown(self):
        if self.previous is None:
            self.sys.modules.pop("faster_whisper", None)
        else:
            self.sys.modules["faster_whisper"] = self.previous
        align._MODEL_CACHE.clear()

    def test_words_and_metadata_come_through(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "vocals.wav"
            audio.write_bytes(b"stub")
            out = align.transcribe_words(audio, "I walk the wire",
                                         model="tiny.en", device="cpu")
        self.assertEqual([w["word"] for w in out["words"]],
                         ["I", "walk", "the", "wire"])
        self.assertEqual(out["words"][0]["start"], 0.5)
        self.assertEqual(out["model"], "tiny.en")
        self.assertEqual(out["language"], "en")
        self.assertEqual(out["audio_duration"], 6.0)
        self.assertTrue(out["lyrics_used"])
        self.assertEqual(len(out["lines"]), 1)

    def test_word_timestamps_are_requested(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "v.wav"
            audio.write_bytes(b"stub")
            align.transcribe_words(audio, model="tiny.en", device="cpu")
        kinds = [c[0] for c in self.calls]
        self.assertEqual(kinds, ["init", "transcribe"])
        self.assertTrue(self.calls[1][2]["word_timestamps"])

    def test_the_model_is_cached_between_calls(self):
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "v.wav"
            audio.write_bytes(b"stub")
            align.transcribe_words(audio, model="tiny.en", device="cpu")
            align.transcribe_words(audio, model="tiny.en", device="cpu")
        self.assertEqual([c[0] for c in self.calls].count("init"), 1)


class MissingDependency(unittest.TestCase):
    def test_clear_error_when_faster_whisper_is_absent(self):
        try:
            import faster_whisper  # noqa: F401
        except ImportError:
            pass
        else:
            self.skipTest("faster-whisper is installed")
        with tempfile.TemporaryDirectory() as tmp:
            audio = Path(tmp) / "a.wav"
            audio.write_bytes(b"not really audio")
            with self.assertRaises(ImportError) as caught:
                align.transcribe_words(audio)
            self.assertIn("pip install faster-whisper", str(caught.exception))

    def test_missing_file_is_reported_first(self):
        with self.assertRaises(FileNotFoundError):
            align.transcribe_words("/nope/missing.wav")


if __name__ == "__main__":
    unittest.main()
