"""ComfyUI node wrapper for kie.ai Seedance generation."""
from __future__ import annotations

import os
import shutil
import subprocess
import tempfile
from pathlib import Path


def _save_image_tensor(tensor, path: Path) -> None:
    # ComfyUI supplies a batch tensor in BHWC format; convert only on execution.
    try:
        import torch
        from PIL import Image
    except ImportError as exc:
        raise RuntimeError("Saving a ComfyUI IMAGE requires torch and Pillow") from exc
    image = tensor[0] if getattr(tensor, "ndim", 0) == 4 else tensor
    image = image.detach().cpu().clamp(0, 1)
    pixels = (image * 255).byte().numpy()
    Image.fromarray(pixels).save(path)


class SingSyncSeedanceKie:
    @classmethod
    def INPUT_TYPES(cls):
        req = {"plan": ("SINGSYNC_PLAN",), "resolution": (["480p", "720p"],),
               "cut_indices": ("STRING", {"default": "all"}), "dry_run": ("BOOLEAN", {"default": False})}
        opt = {"character_image_path": ("STRING", {"default": ""})}
        try:
            import torch  # noqa: F401
            opt["character_image"] = ("IMAGE",)
        except ImportError:
            pass
        return {"required": req, "optional": opt}

    RETURN_TYPES = ("SINGSYNC_PLAN",)
    RETURN_NAMES = ("plan",)
    FUNCTION = "generate"
    CATEGORY = "SingSync"

    def generate(self, plan, resolution="480p", cut_indices="all", dry_run=False,
                 character_image_path="", character_image=None):
        if not isinstance(plan, dict) or not isinstance(plan.get("cuts"), list):
            raise ValueError("plan must be a dict with a cuts list")
        cuts = plan["cuts"]
        indices = list(range(len(cuts))) if cut_indices.strip().lower() == "all" else [
            int(piece.strip()) for piece in cut_indices.split(",") if piece.strip()]
        if any(index < 0 or index >= len(cuts) for index in indices):
            raise ValueError("cut_indices contains an index outside plan cuts")
        root = Path(tempfile.gettempdir()) / "singsync"
        root.mkdir(parents=True, exist_ok=True)
        image_path = Path(character_image_path) if character_image_path else None
        if image_path is None and character_image is not None:
            image_path = root / "character.png"
            _save_image_tensor(character_image, image_path)
        if not dry_run and (image_path is None or not image_path.is_file()):
            raise ValueError("provide character_image or character_image_path")

        client = None
        if not dry_run:
            from .kie import KieSeedance
            client = KieSeedance(work_dir=root)
        ffmpeg = shutil.which("ffmpeg")
        for index in indices:
            cut = cuts[index]
            audio_path = Path(cut["slice_path"])
            duration = int(cut["duration"])
            out = root / f"cut_{index:04d}.mp4"
            if dry_run:
                if not ffmpeg:
                    raise RuntimeError("ffmpeg is required for dry_run slate clips")
                cmd = [ffmpeg, "-y", "-f", "lavfi", "-i", f"color=c=0x26364a:s=1280x720:d={duration}",
                       "-i", str(audio_path), "-shortest", "-c:v", "libx264", "-pix_fmt", "yuv420p",
                       "-c:a", "aac", str(out)]
                subprocess.run(cmd, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE)
            else:
                task = client.create_reference_task(cut["prompt"], [image_path], audio_path,
                                                    duration, resolution=resolution)
                client.download(client.wait(task), out)
            cut["clip_path"] = str(out)
        return (plan,)


NODE_CLASS_MAPPINGS = {"SingSyncSeedanceKie": SingSyncSeedanceKie}
NODE_DISPLAY_NAME_MAPPINGS = {"SingSyncSeedanceKie": "SingSync Seedance (kie.ai)"}
