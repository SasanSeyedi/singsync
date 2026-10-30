"""Small stdlib-only client for kie.ai Seedance reference-to-video jobs."""
from __future__ import annotations

import base64
import json
import mimetypes
import os
import time
import urllib.request
import urllib.parse
from pathlib import Path
from typing import Any, Callable


class KieSeedance:
    base_url = "https://api.kie.ai"
    upload_url = "https://kieai.redpandaai.co"
    model_id = "bytedance/seedance-2"
    _created_count = 0

    def __init__(self, api_key: str | None = None, transport: Callable[..., Any] | None = None,
                 work_dir: str | os.PathLike[str] | None = None):
        self.api_key = api_key if api_key is not None else os.environ.get("KIE_API_KEY", "")
        self.base_url = os.environ.get("KIE_BASE_URL", type(self).base_url).rstrip("/")
        self.transport = transport or self._request
        self.work_dir = Path(work_dir) if work_dir is not None else Path.cwd()

    @staticmethod
    def _request(url: str, *, method: str = "GET", headers: dict[str, str] | None = None,
                 data: bytes | None = None) -> tuple[int, bytes]:
        req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=60) as response:
                return response.status, response.read()
        except Exception as exc:
            # Do not include request headers or the exception text, which may contain credentials.
            raise RuntimeError(f"kie request failed ({type(exc).__name__})") from None

    def _call(self, url: str, method: str = "GET", payload: dict | None = None,
              *, upload_data: bytes | None = None) -> dict:
        headers = {"Authorization": f"Bearer {self.api_key}"}
        data = upload_data
        if payload is not None:
            headers["Content-Type"] = "application/json"
            data = json.dumps(payload).encode("utf-8")
        try:
            response = self.transport(url, method=method, headers=headers, data=data)
            if isinstance(response, tuple):
                status, raw = response
                if status >= 400:
                    raise RuntimeError(f"kie HTTP error ({status})")
                return json.loads(raw.decode("utf-8") if isinstance(raw, bytes) else raw)
            if hasattr(response, "status_code"):
                if response.status_code >= 400:
                    raise RuntimeError(f"kie HTTP error ({response.status_code})")
                return response.json()
            return response
        except Exception as exc:
            msg = str(exc).replace(self.api_key, "[redacted]") if self.api_key else str(exc)
            if self.api_key and self.api_key in msg:
                msg = "kie request failed"
            raise RuntimeError(msg) from None

    def _redact(self, value: Any) -> str:
        message = str(value)
        return message.replace(self.api_key, "[redacted]") if self.api_key else message

    def upload(self, path: str | os.PathLike[str], kind: str) -> str:
        file_path = Path(path)
        kind = kind.lower()
        mime = mimetypes.guess_type(file_path.name)[0]
        allowed = {"image": {"image/png", "image/jpeg"}, "audio": {"audio/mpeg", "audio/wav", "audio/x-wav"}}
        if kind not in allowed or mime not in allowed[kind]:
            raise ValueError("kind must be image (png/jpg) or audio (mp3/wav)")
        encoded = base64.b64encode(file_path.read_bytes()).decode("ascii")
        body = self._call(f"{self.upload_url}/api/file-base64-upload", "POST", {
            "base64Data": f"data:{mime};base64,{encoded}",
            "uploadPath": "images/singsync" if kind == "image" else "audio/singsync",
            "fileName": file_path.name,
        })
        if not body.get("success"):
            raise RuntimeError("kie upload failed")
        try:
            return body["data"]["downloadUrl"]
        except (KeyError, TypeError):
            raise RuntimeError("kie upload response missing download URL") from None

    def create_reference_task(self, prompt: str, image_paths: list[str | os.PathLike[str]],
                              audio_path: str | os.PathLike[str], duration: int,
                              resolution: str = "480p", generate_audio: bool = True,
                              aspect: str = "16:9") -> str:
        if len(image_paths) < 1 or len(image_paths) > 4:
            raise ValueError("image_paths must contain 1 to 4 character references")
        if duration not in (5, 10, 15):
            raise ValueError("duration must be 5, 10, or 15 seconds")
        if duration > 15:
            raise ValueError("duration over 15 seconds is not allowed")
        if resolution not in ("480p", "720p") and os.environ.get("SINGSYNC_ALLOW_HD") != "1":
            raise ValueError("resolution must be 480p or 720p unless SINGSYNC_ALLOW_HD=1")
        limit = int(os.environ.get("SINGSYNC_MAX_CLIPS", "6"))
        cls = type(self)
        if cls._created_count >= limit:
            raise RuntimeError(f"SingSync task limit reached ({limit})")
        image_urls = [self.upload(path, "image") for path in image_paths]
        audio_url = self.upload(audio_path, "audio")
        payload = {"model": self.model_id, "input": {
            "prompt": prompt, "reference_image_urls": image_urls,
            "reference_audio_urls": [audio_url], "aspect_ratio": aspect,
            "resolution": resolution, "duration": duration,
            "generate_audio": generate_audio,
        }}
        response = self._call(f"{self.base_url}/api/v1/jobs/createTask", "POST", payload)
        if response.get("code") != 200:
            raise RuntimeError("kie create task rejected: " + self._redact(response.get("msg", "provider rejected request")))
        try:
            task_id = response["data"]["taskId"]
        except (KeyError, TypeError):
            raise RuntimeError("kie create task response missing task ID") from None
        cls._created_count += 1
        self.work_dir.mkdir(parents=True, exist_ok=True)
        row = {"time": time.time(), "task_id": self._redact(task_id), "duration": duration, "resolution": resolution}
        with (self.work_dir / "singsync_spend.jsonl").open("a", encoding="utf-8") as log:
            log.write(json.dumps(row, separators=(",", ":")) + "\n")
        return task_id

    def wait(self, task_id: str, timeout_s: int = 1800, poll_s: int = 15) -> str:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            response = self._call(
                f"{self.base_url}/api/v1/jobs/recordInfo?taskId={urllib.parse.quote(task_id)}")
            data = response.get("data", {})
            state = data.get("state")
            if state == "success":
                result = data.get("resultJson") or {}
                if isinstance(result, str):
                    try:
                        result = json.loads(result)
                    except json.JSONDecodeError:
                        result = {}
                urls = result.get("resultUrls") or []
                if urls:
                    return urls[0]
                raise RuntimeError("kie task succeeded without a result URL")
            if state in ("fail", "failed", "rejected"):
                detail = data.get("failMsg") or data.get("message") or data.get("failCode") or "provider did not supply a reason"
                raise RuntimeError(f"kie task {state}: {self._redact(detail)}")
            time.sleep(min(poll_s, max(0, deadline - time.monotonic())))
        raise TimeoutError(f"kie task {self._redact(task_id)} timed out after {timeout_s}s")

    def download(self, url: str, out_path: str | os.PathLike[str]) -> Path:
        try:
            response = self.transport(url, method="GET", headers={}, data=None)
            if isinstance(response, tuple):
                status, content = response
                if status >= 400:
                    raise RuntimeError(f"kie download HTTP error ({status})")
            elif hasattr(response, "content"):
                if response.status_code >= 400:
                    raise RuntimeError(f"kie download HTTP error ({response.status_code})")
                content = response.content
            else:
                content = response
            dest = Path(out_path)
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_bytes(content if isinstance(content, bytes) else bytes(content))
            return dest
        except Exception as exc:
            msg = str(exc).replace(self.api_key, "[redacted]") if self.api_key else str(exc)
            raise RuntimeError(msg) from None
