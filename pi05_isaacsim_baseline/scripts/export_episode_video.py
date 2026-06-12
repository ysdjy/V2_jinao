"""Export a video (or image strip) for one episode from a normalized dataset's
saved images. Uses imageio if available, else ffmpeg, else writes a montage.

Usage:
  python scripts/export_episode_video.py \
      --dataset data/processed/normalized_dataset/<name> --camera front_rgb --episode 0
"""

from __future__ import annotations

import argparse
import glob
import os
import subprocess


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", required=True)
    p.add_argument("--camera", default="front_rgb")
    p.add_argument("--episode", type=int, default=0)
    p.add_argument("--fps", type=int, default=30)
    p.add_argument("--out", default=None)
    args = p.parse_args()

    cam_dir = os.path.join(args.dataset, "images", args.camera)
    pattern = os.path.join(cam_dir, f"ep{args.episode:04d}_step*.png")
    frames = sorted(glob.glob(pattern))
    if not frames:
        print(f"[video] no frames at {pattern} (state-only dataset?)")
        return

    proj = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    outdir = os.path.join(proj, "logs", "videos")
    os.makedirs(outdir, exist_ok=True)
    out = args.out or os.path.join(outdir, f"{os.path.basename(args.dataset)}_{args.camera}_ep{args.episode}.mp4")

    # 1. imageio
    try:
        import imageio.v2 as imageio

        with imageio.get_writer(out, fps=args.fps) as w:
            for fp in frames:
                w.append_data(imageio.imread(fp))
        print(f"[video] wrote {out} via imageio ({len(frames)} frames)")
        return
    except Exception as e:  # noqa
        print(f"[video] imageio path failed ({e}); trying ffmpeg")

    # 2. ffmpeg
    if subprocess.call(["bash", "-lc", "command -v ffmpeg"], stdout=subprocess.DEVNULL) == 0:
        cmd = ["ffmpeg", "-y", "-framerate", str(args.fps), "-pattern_type", "glob",
               "-i", pattern, "-c:v", "libx264", "-pix_fmt", "yuv420p", out]
        subprocess.call(cmd)
        print(f"[video] wrote {out} via ffmpeg")
        return

    print(f"[video] neither imageio nor ffmpeg available; {len(frames)} frames are at {cam_dir}")


if __name__ == "__main__":
    main()
