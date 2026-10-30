import json
import unittest

from singsync.idea import build_director_graph, build_song_graph, parse_director_json


class IdeaTests(unittest.TestCase):
    def test_director_graph_shape(self):
        graph = build_director_graph("face.png", "place.png", "a moonlit dance", 7)
        types = {n["class_type"] for n in graph.values()}
        self.assertEqual(types, {"CLIPLoader", "LoadImage", "ImageBatch", "TextGenerate", "PreviewAny"})
        self.assertEqual(graph["4"]["inputs"], {"image1": ["2", 0], "image2": ["3", 0]})
        inputs = graph["5"]["inputs"]
        self.assertEqual(inputs["image"], ["4", 0]); self.assertEqual(inputs["sampling_mode.seed"], 7)
        self.assertIn("never behind or orbiting", inputs["prompt"])

    def test_song_graph_native_model_and_wiring(self):
        graph = build_song_graph("indie pop", "[verse]\nhello", 30, 3, 100)
        self.assertEqual(graph["1"]["inputs"]["unet_name"], "acestep_v1.5_xl_turbo_bf16.safetensors")
        self.assertEqual(graph["3"]["inputs"]["clip"], ["9", 0])
        self.assertEqual(graph["7"]["inputs"]["vae"], ["10", 0])
        small = build_song_graph("indie pop", "[verse]\nhello", 30, 3, 100, model="turbo")
        self.assertEqual(small["1"]["inputs"]["ckpt_name"], "ace_step_1.5_turbo_aio.safetensors")
        self.assertEqual(graph["6"]["inputs"]["steps"], 8)
        self.assertEqual(graph["8"]["class_type"], "SaveAudioMP3")

    def test_lyrics_whole_lines_and_vocal_tags(self):
        from singsync.idea import format_lyrics, song_tags
        text = format_lyrics({"verse": ["Bananas falling, from the sky"], "chorus": ["Banana rain"]})
        self.assertEqual(text, "[Verse 1]\nBananas falling, from the sky\n\n[Chorus]\nBanana rain")
        self.assertIn("sung clearly", song_tags("pop"))

    def test_parse_fenced_and_surrounded_json(self):
        value = {"title": "song", "lyrics": ["a"]}
        self.assertEqual(parse_director_json("Here you go:\n```json\n" + json.dumps(value) + "\n```"), value)


if __name__ == "__main__": unittest.main()
