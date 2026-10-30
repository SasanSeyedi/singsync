import json
import os
import tempfile
import unittest
from pathlib import Path

from singsync.providers.kie import KieSeedance


class FakeTransport:
    def __init__(self, states=None):
        self.states = list(states or [])
        self.calls = []

    def __call__(self, url, *, method="GET", headers=None, data=None):
        self.calls.append((url, method, headers or {}, data))
        if "file-base64-upload" in url:
            return 200, json.dumps({"success": True, "data": {"downloadUrl": f"https://cdn.test/{len(self.calls)}"}}).encode()
        if "createTask" in url:
            return 200, b'{"code":200,"data":{"taskId":"t-123"}}'
        if "recordInfo" in url:
            state = self.states.pop(0) if self.states else {"state": "success", "resultJson": "{\"resultUrls\":[\"https://video.test/a.mp4\"]}"}
            return 200, json.dumps({"data": state}).encode()
        if url.startswith("https://video.test"):
            return 200, b"video"
        raise AssertionError("unexpected request URL")


class KieProviderTests(unittest.TestCase):
    def setUp(self):
        KieSeedance._created_count = 0
        self.tmp = tempfile.TemporaryDirectory(dir=Path(__file__).parent)
        self.root = Path(self.tmp.name)
        self.image = self.root / "face.png"
        self.audio = self.root / "voice.wav"
        self.image.write_bytes(b"image")
        self.audio.write_bytes(b"audio")

    def tearDown(self):
        self.tmp.cleanup()

    def make_client(self, transport, key="SECRET_TEST_KEY"):
        return KieSeedance(api_key=key, transport=transport, work_dir=self.root)

    def test_create_payload_polling_and_spend_log(self):
        transport = FakeTransport()
        client = self.make_client(transport)
        task = client.create_reference_task("singing", [self.image], self.audio, 10)
        self.assertEqual(task, "t-123")
        create = next(call for call in transport.calls if "createTask" in call[0])
        body = json.loads(create[3])
        self.assertEqual(body["input"]["prompt"], "singing")
        self.assertEqual(body["input"]["reference_image_urls"], ["https://cdn.test/1"])
        self.assertEqual(body["input"]["reference_audio_urls"], ["https://cdn.test/2"])
        self.assertEqual(body["input"]["duration"], 10)
        self.assertEqual(body["input"]["resolution"], "480p")
        self.assertEqual(client.wait(task, poll_s=0), "https://video.test/a.mp4")
        record = json.loads((self.root / "singsync_spend.jsonl").read_text())
        self.assertEqual({k: record[k] for k in ("task_id", "duration", "resolution")},
                         {"task_id": task, "duration": 10, "resolution": "480p"})

    def test_poll_failure_message(self):
        client = self.make_client(FakeTransport([{"state": "fail", "failCode": "NO", "failMsg": "blocked"}]))
        with self.assertRaisesRegex(RuntimeError, "blocked"):
            client.wait("bad", poll_s=0)

    def test_spend_guard_refuses_seventh_task(self):
        client = self.make_client(FakeTransport())
        for _ in range(6):
            client.create_reference_task("prompt", [self.image], self.audio, 5)
        with self.assertRaisesRegex(RuntimeError, "limit"):
            client.create_reference_task("prompt", [self.image], self.audio, 5)

    def test_key_not_in_provider_errors_or_log(self):
        fake = FakeTransport([{"state": "fail", "failMsg": "SECRET_TEST_KEY rejected"}])
        client = self.make_client(fake)
        with self.assertRaises(RuntimeError) as caught:
            client.wait("bad", poll_s=0)
        self.assertNotIn("SECRET_TEST_KEY", str(caught.exception))
        client.create_reference_task("prompt", [self.image], self.audio, 5)
        log = (self.root / "singsync_spend.jsonl").read_text()
        self.assertNotIn("SECRET_TEST_KEY", log)


if __name__ == "__main__":
    unittest.main()
