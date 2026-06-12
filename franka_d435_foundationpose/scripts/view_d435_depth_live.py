"""Live D435 viewer: real-time colorized depth (and RGB) in an OpenCV window.

Perception only — never controls the robot. Uses the project's
RealSenseD435Provider (color + aligned depth + intrinsics, depth in meters,
~30-frame warmup) and shows a JET-colorized depth map. Press q / ESC to quit.

Run (in the realsense env, on a machine with a display):
    cd /home1/banghai/Documents/IsaacLab
    conda run -n realsense python franka_d435_foundationpose/scripts/view_d435_depth_live.py

Headless / over SSH without X: pass --save_dir to dump rolling depth_vis frames
to disk instead of opening a window (or reconnect with `ssh -X`).
"""

import argparse
import os
import sys
import time

import numpy as np

import _bootstrap  # noqa: F401

from franka_d435_foundationpose.utils import depth_utils
from franka_d435_foundationpose.utils.logging_utils import get_logger

logger = get_logger("d435_viewer")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--max_depth_m", type=float, default=4.0,
                        help="upper bound for the depth colormap (meters)")
    parser.add_argument("--min_depth_m", type=float, default=0.2)
    parser.add_argument("--no_rgb", action="store_true", help="show depth only")
    parser.add_argument("--save_dir", default=None,
                        help="headless mode: write depth_vis frames here instead of a window")
    parser.add_argument("--save_every", type=int, default=5, help="save every Nth frame (headless)")
    parser.add_argument("--warmup_frames", type=int, default=30)
    args = parser.parse_args()

    # Pre-flight: pyrealsense2 must be importable.
    try:
        import pyrealsense2  # noqa: F401
    except Exception:
        logger.error(
            "pyrealsense2 not available. Use the realsense env:\n"
            "    conda run -n realsense python franka_d435_foundationpose/scripts/"
            "view_d435_depth_live.py"
        )
        return 1

    from franka_d435_foundationpose.camera.realsense_d435_provider import RealSenseD435Provider

    headless = args.save_dir is not None
    cv2 = None
    if not headless:
        try:
            import cv2 as _cv2

            cv2 = _cv2
            cv2.namedWindow("D435 depth (q=quit)", cv2.WINDOW_NORMAL)
        except Exception as e:
            logger.warning("could not open a window (%s); falling back to --save_dir mode", e)
            headless = True
            args.save_dir = args.save_dir or "franka_d435_foundationpose/outputs/realsense_live_view"

    if headless:
        os.makedirs(args.save_dir, exist_ok=True)
        logger.info("headless mode: writing depth frames to %s (Ctrl-C to stop)", args.save_dir)

    logger.info("opening D435 (%dx%d @ %dfps), warming up...", args.width, args.height, args.fps)
    provider = RealSenseD435Provider(args.width, args.height, args.fps,
                                     warmup_frames=args.warmup_frames)

    frame_idx = 0
    fps_t0, fps_n = time.time(), 0
    try:
        while True:
            frame = provider.get_frame()
            depth = frame.depth
            vis = depth_utils.colorize_depth(depth, min_m=args.min_depth_m, max_m=args.max_depth_m)

            # center-pixel depth readout
            cy, cx = depth.shape[0] // 2, depth.shape[1] // 2
            cz = float(depth[cy, cx])
            valid_pct = 100.0 * np.mean((depth > 0) & np.isfinite(depth))

            if cv2 is not None:
                txt = f"center {cz:.3f} m | valid {valid_pct:.0f}% | range [{args.min_depth_m},{args.max_depth_m}] m"
                cv2.putText(vis, txt, (10, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (255, 255, 255), 1, cv2.LINE_AA)
                cv2.drawMarker(vis, (cx, cy), (255, 255, 255), cv2.MARKER_CROSS, 14, 1)
                if args.no_rgb:
                    show = vis
                else:
                    bgr = frame.rgb[:, :, ::-1]  # RGB -> BGR for cv2
                    show = np.hstack([np.ascontiguousarray(bgr), vis])
                cv2.imshow("D435 depth (q=quit)", show)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
            else:
                if frame_idx % max(1, args.save_every) == 0:
                    from franka_d435_foundationpose.utils import image_io

                    image_io.save_bgr(os.path.join(args.save_dir, f"depth_{frame_idx:05d}.png"), vis)
                    logger.info("frame %d  center %.3f m  valid %.0f%%", frame_idx, cz, valid_pct)

            frame_idx += 1
            fps_n += 1
            if time.time() - fps_t0 >= 2.0:
                logger.info("~%.1f fps", fps_n / (time.time() - fps_t0))
                fps_t0, fps_n = time.time(), 0
    except KeyboardInterrupt:
        logger.info("stopped by user")
    finally:
        provider.close()
        if cv2 is not None:
            cv2.destroyAllWindows()
    return 0


if __name__ == "__main__":
    sys.exit(main())
