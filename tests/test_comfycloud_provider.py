import json
import os
import tempfile
import unittest
from pathlib import Path

from singsync.providers.comfycloud import ComfyCloud, build_h3_r2v_prompt, frame_length


class Fake:
    def __init__(self, responses): self.responses=list(responses); self.calls=[]
    def __call__(self,url,*,method="GET",headers=None,data=None):
        self.calls.append((url,method,headers or {},data))
        r=self.responses.pop(0)
        return r


class ProviderTests(unittest.TestCase):
    def test_graph_native_models_and_wiring_and_frame_math(self):
        graph=build_h3_r2v_prompt("Sing",["one.png","two.png"],"vocal.wav",10,turbo=True)
        self.assertEqual(frame_length(10),243); self.assertEqual(frame_length(5),124)
        types={n["class_type"] for n in graph.values()}
        self.assertTrue({"MiniMaxH3ReferenceToVideo","VAEDecodeAudio","CreateVideo","SaveVideo"} <= types)
        self.assertEqual(graph["40"]["inputs"]["ref_images"],{"ref_image_0":["10",0],"ref_image_1":["11",0]})
        self.assertEqual(graph["40"]["inputs"]["ref_audios"],{"ref_audio_0":["30",0]})
        alltxt=json.dumps(graph)
        for name in ("minimax_h3_ref2va_pruned_int8_convrot.safetensors","qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors","minimax_h3_video_vae_int8_convrot.safetensors","minimax_h3_audio_vae_fp32.safetensors","minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors"):
            self.assertIn(name,alltxt)

    def test_upload_multipart_and_key_not_in_errors(self):
        with tempfile.TemporaryDirectory() as d:
            p=Path(d)/"x.wav"; p.write_bytes(b"audio")
            fake=Fake([(200,b'{"name":"x.wav"}')])
            self.assertEqual(ComfyCloud("secret",transport=fake).upload(p),"x.wav")
            body=fake.calls[0][3]
            self.assertIn(b'name="image"',body); self.assertIn(b'name="type"',body); self.assertIn(b'input',body); self.assertIn(b'overwrite',body)
            bad=ComfyCloud("secret",transport=Fake([(403,b'{"message":"secret rejected"}')]))
            with self.assertRaises(RuntimeError) as e: bad.submit({})
            self.assertNotIn("secret",str(e.exception))

    def test_status_success_and_failure(self):
        with tempfile.TemporaryDirectory() as d:
            fake=Fake([(200,b'{"status":"completed"}'),(200,b'{"p":{"outputs":{"1":{"videos":[{"filename":"a.mp4","subfolder":"s","type":"output"}]}}}}'),(200,b'mp4')])
            client=ComfyCloud("k",transport=fake,work_dir=d)
            self.assertTrue(client.wait("p",poll_s=0).exists())
            self.assertIn("/api/view?",fake.calls[-1][0])
        failed=ComfyCloud("sk_test",transport=Fake([(200,b'{"status":"failed","message":"sk_test bad"}')]))
        with self.assertRaises(RuntimeError) as e: failed.wait("p",poll_s=0)
        self.assertNotIn("sk_test",str(e.exception))

    def test_spend_guard(self):
        with tempfile.TemporaryDirectory() as d:
            old=os.environ.get("SINGSYNC_MAX_CLIPS"); os.environ["SINGSYNC_MAX_CLIPS"]="1"
            try:
                ComfyCloud._created_count=1
                with self.assertRaisesRegex(RuntimeError,"clip limit"):
                    ComfyCloud("k",transport=Fake([]),work_dir=d).render_cut("x",[],"a.wav",5)
            finally:
                ComfyCloud._created_count=0
                if old is None: os.environ.pop("SINGSYNC_MAX_CLIPS",None)
                else: os.environ["SINGSYNC_MAX_CLIPS"]=old


if __name__ == "__main__": unittest.main()
