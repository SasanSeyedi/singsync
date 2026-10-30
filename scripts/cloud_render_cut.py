"""Render one SingSync plan cut with Comfy Cloud MiniMax H3."""
import argparse
import json
from pathlib import Path

from singsync.providers.comfycloud import ComfyCloud


def main():
    parser=argparse.ArgumentParser()
    parser.add_argument("plan_json")
    parser.add_argument("cut_idx", type=int)
    parser.add_argument("--image", action="append", required=True)
    parser.add_argument("--turbo", action="store_true")
    args=parser.parse_args()
    plan_path=Path(args.plan_json)
    plan=json.loads(plan_path.read_text(encoding="utf-8"))
    cut=plan["cuts"][args.cut_idx]
    work=Path(plan.get("work_dir") or plan_path.parent)
    out=ComfyCloud(work_dir=work).render_cut(cut["prompt"],args.image,cut["slice_path"],cut["duration"],turbo=args.turbo)
    print(out)


if __name__ == "__main__": main()
