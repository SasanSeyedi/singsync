"""ComfyUI wrapper for Comfy Cloud MiniMax H3 rendering."""
from pathlib import Path


class SingSyncComfyCloudH3:
    @classmethod
    def INPUT_TYPES(cls):
        return {"required": {"plan": ("SINGSYNC_PLAN",), "turbo": ("BOOLEAN", {"default": True}),
            "width": ("INT", {"default": 1344, "min": 32, "max": 16384, "step": 32}),
            "height": ("INT", {"default": 768, "min": 32, "max": 16384, "step": 32}),
            "dry_run": ("BOOLEAN", {"default": False})},
            "optional": {"image_paths": ("STRING", {"default":"", "multiline":False})}}
    RETURN_TYPES = ("SINGSYNC_PLAN",)
    RETURN_NAMES = ("plan",)
    FUNCTION = "render"
    CATEGORY = "SingSync"

    def render(self, plan, turbo=True, width=1344, height=768, dry_run=False, image_paths=""):
        if not isinstance(plan, dict) or not isinstance(plan.get("cuts"), list):
            raise ValueError("plan must be a dict with a cuts list")
        if dry_run: return (plan,)
        images=[Path(x.strip().strip('"')) for x in image_paths.split(";") if x.strip()]
        if not images: raise ValueError("provide one or more semicolon-separated image_paths")
        from .comfycloud import ComfyCloud
        work=Path(plan.get("work_dir") or Path.cwd()); work.mkdir(parents=True,exist_ok=True)
        client=ComfyCloud(work_dir=work)
        for i,cut in enumerate(plan["cuts"]):
            cut["clip_path"]=str(client.render_cut(cut["prompt"],images,cut["slice_path"],cut["duration"],width,height,turbo,seed=i+1))
        return (plan,)


NODE_CLASS_MAPPINGS = {"SingSyncComfyCloudH3": SingSyncComfyCloudH3}
NODE_DISPLAY_NAME_MAPPINGS = {"SingSyncComfyCloudH3": "SingSync Render (Comfy Cloud, MiniMax H3)"}
