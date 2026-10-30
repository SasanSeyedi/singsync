"""Comfy Cloud client and native MiniMax H3 reference-to-video graph."""
from __future__ import annotations

import json
import mimetypes
import os
import time
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any, Callable


def frame_length(seconds: float) -> int:
    n = max(5, round(float(seconds) * 24))
    return n + (5 - n % 17) % 17


def build_h3_r2v_prompt(prompt: str, image_names: list[str], audio_name: str,
                        seconds: float, width: int = 1344, height: int = 768,
                        turbo: bool = True, seed: int = 1, ref_image_size: str = "match") -> dict:
    if not image_names or len(image_names) > 9:
        raise ValueError("image_names must contain 1 to 9 references")
    nodes: dict[str, dict] = {}
    def node(i, cls, inputs):
        nodes[str(i)] = {"class_type": cls, "inputs": inputs}
    node(1, "UNETLoader", {"unet_name":"minimax_h3_ref2va_pruned_int8_convrot.safetensors", "weight_dtype":"default"})
    model = ["1", 0]
    if turbo:
        node(2, "LoraLoaderModelOnly", {"model":model, "lora_name":"minimax_h3_ref2v_turbo_4step_v0.1_comfyui_bf16.safetensors", "strength_model":1})
        model = ["2", 0]
    node(3, "CLIPLoader", {"clip_name":"qwen3vl_32b_minimax_h3_nvfp4_awq.safetensors", "type":"minimax", "device":"default"})
    node(4, "VAELoader", {"vae_name":"minimax_h3_video_vae_int8_convrot.safetensors"})
    node(5, "VAELoader", {"vae_name":"minimax_h3_audio_vae_fp32.safetensors"})
    for i, filename in enumerate(image_names):
        node(10+i, "LoadImage", {"image":filename})
    node(30, "LoadAudio", {"audio":audio_name})
    ref_images = {f"ref_image_{i}":[str(10+i),0] for i in range(len(image_names))}
    rinputs = {"clip":["3",0], "vae":["4",0], "audio_vae":["5",0], "prompt":prompt,
       "width":int(width), "height":int(height), "length":frame_length(seconds), "ref_image_size":ref_image_size,
       "ref_images":ref_images, "ref_audios":{"ref_audio_0":["30",0]}}
    node(40,"MiniMaxH3ReferenceToVideo",rinputs)
    steps = 4 if turbo else 20
    node(41,"RandomNoise",{"noise_seed":int(seed)})
    node(42,"KSamplerSelect",{"sampler_name":"res_multistep"})
    node(43,"BasicScheduler",{"model":model,"scheduler":"simple","steps":steps,"denoise":1.0})
    node(44,"BasicGuider",{"model":model,"conditioning":["40",0]})
    node(45,"SamplerCustomAdvanced",{"noise":["41",0],"guider":["44",0],"sampler":["42",0],"sigmas":["43",0],"latent_image":["40",1]})
    node(46,"VAEDecode",{"samples":["45",0],"vae":["4",0]})
    node(47,"VAEDecodeAudio",{"samples":["45",0],"vae":["5",0]})
    node(48,"CreateVideo",{"images":["46",0],"audio":["47",0],"fps":24})
    node(49,"SaveVideo",{"video":["48",0],"filename_prefix":"singsync/h3","format":"auto","codec":"auto"})
    return nodes


class ComfyCloud:
    _created_count = 0
    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 transport: Callable[..., Any] | None = None, work_dir=None):
        self.api_key = api_key if api_key is not None else os.environ.get("COMFY_API_KEY", "")
        self.base_url = (base_url or os.environ.get("COMFY_CLOUD_URL", "https://cloud.comfy.org")).rstrip("/")
        self.transport = transport or self._request
        self.work_dir = Path(work_dir) if work_dir is not None else Path.cwd()

    @staticmethod
    def _request(url, *, method="GET", headers=None, data=None):
        req = urllib.request.Request(url, data=data, headers=headers or {}, method=method)
        try:
            with urllib.request.urlopen(req, timeout=120) as response:
                return response.status, response.read()
        except Exception as exc:
            raise RuntimeError(f"Comfy Cloud request failed ({type(exc).__name__})") from None

    def _call(self, path_or_url, method="GET", payload=None, data=None, headers=None):
        url = path_or_url if path_or_url.startswith("http") else self.base_url + path_or_url
        hdr = {"X-API-Key":self.api_key}
        if headers: hdr.update(headers)
        if payload is not None:
            data = json.dumps(payload).encode(); hdr["Content-Type"] = "application/json"
        try:
            resp = self.transport(url, method=method, headers=hdr, data=data)
            if isinstance(resp, tuple):
                status, raw = resp
                if status >= 400:
                    try: detail = json.loads(raw.decode() if isinstance(raw,bytes) else raw).get("message", "")
                    except Exception: detail = ""
                    raise RuntimeError(f"Comfy Cloud HTTP {status}: {detail}")
                return raw if isinstance(raw, bytes) else str(raw).encode()
            if hasattr(resp,"status_code"):
                if resp.status_code >= 400: raise RuntimeError(f"Comfy Cloud HTTP {resp.status_code}: {getattr(resp,'text','')}")
                return getattr(resp,"content",None) or json.dumps(resp.json()).encode()
            if isinstance(resp,dict): return json.dumps(resp).encode()
            return resp if isinstance(resp,bytes) else bytes(resp)
        except Exception as exc:
            msg = str(exc)
            if self.api_key: msg = msg.replace(self.api_key,"[redacted]")
            if self.api_key and self.api_key in msg: msg = "Comfy Cloud request failed"
            raise RuntimeError(msg) from None

    def upload(self, path):
        p=Path(path); boundary="----SingSync"+uuid.uuid4().hex
        mime=mimetypes.guess_type(p.name)[0] or "application/octet-stream"
        chunks=[]
        def field(name,val): chunks.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"{name}\"\r\n\r\n{val}\r\n".encode())
        field("type","input"); field("overwrite","true")
        chunks.append(f"--{boundary}\r\nContent-Disposition: form-data; name=\"image\"; filename=\"{p.name}\"\r\nContent-Type: {mime}\r\n\r\n".encode()+p.read_bytes()+b"\r\n")
        chunks.append(f"--{boundary}--\r\n".encode())
        try: result=json.loads(self._call("/api/upload/image","POST",data=b"".join(chunks),headers={"Content-Type":f"multipart/form-data; boundary={boundary}"}).decode())
        except (ValueError,AttributeError): raise RuntimeError("Comfy Cloud upload returned invalid JSON") from None
        name=result.get("name") or result.get("filename")
        if not name: raise RuntimeError("Comfy Cloud upload response missing filename")
        return name

    def submit(self, prompt_dict):
        result=json.loads(self._call("/api/prompt","POST",{"prompt":prompt_dict}).decode())
        pid=result.get("prompt_id") or result.get("promptId")
        if not pid: raise RuntimeError("Comfy Cloud submit response missing prompt_id")
        return str(pid)

    def run(self, graph, timeout_s=2400, poll_s=10):
        """Submit any native Comfy graph and return its output dictionary."""
        pid = self.submit(graph)
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                status = json.loads(self._call("/api/job/" + urllib.parse.quote(pid) + "/status").decode())
            except RuntimeError:
                status = json.loads(self._call("/api/v1/jobs/" + urllib.parse.quote(pid) + "/status").decode())
            state = str(status.get("status", status.get("state", ""))).lower()
            if state in ("completed", "success"):
                history = None
                for endpoint in (f"/api/history_v2/{urllib.parse.quote(pid)}", f"/api/history/{urllib.parse.quote(pid)}"):
                    try:
                        history = json.loads(self._call(endpoint).decode()); break
                    except RuntimeError: continue
                h = history.get(pid, history) if isinstance(history, dict) else {}
                if isinstance(h, list): h = h[0] if h else {}
                return h.get("outputs", {}) if isinstance(h, dict) else {}
            if state in ("failed", "error", "cancelled"):
                return self._raise_redacted(f"Comfy Cloud job {state}: {status.get('message') or status.get('error') or state}")
            time.sleep(min(poll_s, max(0, deadline-time.monotonic())))
        raise TimeoutError("Comfy Cloud job timed out")

    def download(self, item, destination=None):
        """Download a Comfy output item (or return bytes when destination is omitted)."""
        query = urllib.parse.urlencode({k:item[k] for k in ("filename", "subfolder", "type") if k in item})
        content = self._call("/api/view?" + query)
        if destination is None:
            return content
        path = Path(destination); path.parent.mkdir(parents=True, exist_ok=True); path.write_bytes(content)
        return path

    def wait(self, prompt_id, timeout_s=2400, poll_s=10):
        deadline=time.monotonic()+timeout_s
        while time.monotonic()<deadline:
            try: status=json.loads(self._call("/api/job/"+urllib.parse.quote(prompt_id)+"/status").decode())
            except RuntimeError:
                status=json.loads(self._call("/api/v1/jobs/"+urllib.parse.quote(prompt_id)+"/status").decode())
            state=str(status.get("status",status.get("state",""))).lower()
            if state in ("completed","success"):
                history=None
                for endpoint in (f"/api/history_v2/{urllib.parse.quote(prompt_id)}",f"/api/history/{urllib.parse.quote(prompt_id)}"):
                    try: history=json.loads(self._call(endpoint).decode()); break
                    except RuntimeError: continue
                out=self._find_video(history,prompt_id)
                if out is None: raise RuntimeError("Comfy Cloud job completed without a SaveVideo output")
                query=urllib.parse.urlencode(out)
                content=self._call("/api/view?"+query)
                dest=self.work_dir/f"singsync_cloud_{prompt_id}.mp4"; dest.parent.mkdir(parents=True,exist_ok=True); dest.write_bytes(content)
                return dest
            if state in ("failed","error","cancelled"):
                msg=status.get("message") or status.get("error") or status.get("detail") or state
                return self._raise_redacted(f"Comfy Cloud job {state}: {msg}")
            time.sleep(min(poll_s,max(0,deadline-time.monotonic())))
        raise TimeoutError("Comfy Cloud job timed out")

    @staticmethod
    def _find_video(history, prompt_id):
        if not isinstance(history,dict): return None
        h=history.get(prompt_id,history)
        if isinstance(h,list): h=h[0] if h else {}
        outputs=h.get("outputs",{}) if isinstance(h,dict) else {}
        for value in outputs.values():
            # Comfy Cloud lists SaveVideo results under "images" (with metadata.kind == "video");
            # local ComfyUI uses "videos"/"gifs". Accept all three.
            for item in value.get("videos", []) + value.get("gifs", []) + value.get("images", []):
                if not isinstance(item, dict):
                    continue
                kind = (item.get("metadata") or {}).get("kind")
                name = str(item.get("filename", ""))
                if kind == "video" or name.lower().endswith((".mp4", ".webm", ".mov", ".gif")):
                    return {k: item[k] for k in ("filename", "subfolder", "type") if k in item}
        return None

    def _raise_redacted(self, message):
        if self.api_key: message=str(message).replace(self.api_key,"[redacted]")
        raise RuntimeError(message)

    def render_cut(self, prompt_text, image_paths, audio_path, seconds, width=1344,height=768,turbo=True,seed=1):
        limit=int(os.environ.get("SINGSYNC_MAX_CLIPS","6"))
        self.work_dir.mkdir(parents=True,exist_ok=True)
        log=self.work_dir/"singsync_spend.jsonl"
        if type(self)._created_count>=limit: raise RuntimeError(f"SingSync Comfy Cloud clip limit reached ({limit})")
        names=[self.upload(p) for p in image_paths]; audio=self.upload(audio_path)
        refs=" and ".join(f"<Picture {i+1}>" for i in range(len(names)))
        prompt=f"Use {refs} as the character reference and <Audio 1> exactly as it is: {prompt_text}, lip-synced"
        pid=self.submit(build_h3_r2v_prompt(prompt,names,audio,seconds,width,height,turbo,seed))
        type(self)._created_count += 1
        with log.open("a",encoding="utf-8") as f: f.write(json.dumps({"time":time.time(),"prompt_id":pid,"seconds":seconds,"turbo":turbo})+"\n")
        return self.wait(pid)
