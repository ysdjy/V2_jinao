"""Regenerate the 'Wake-up Summary' block at the top of STATUS.md from artifacts
on disk. Conservative: only fills what it can verify; leaves the rest of STATUS.md
untouched (everything below the AUTO marker is preserved if present, else appended).
"""

from __future__ import annotations

import glob
import json
import os

PROJ = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
AUTO_MARK = "<!-- AUTO-SUMMARY -->"


def exists_nonempty(path):
    return os.path.exists(path) and os.path.getsize(path) > 0


def yn(b):
    return "yes" if b else "no"


def latest(pattern):
    fs = sorted(glob.glob(pattern), key=os.path.getmtime, reverse=True)
    return fs[0] if fs else None


def main():
    venv_py = os.path.join(PROJ, ".venv_openpi", "bin", "python")
    openpi_installed = os.path.exists(venv_py)

    dryrun_p = os.path.join(PROJ, "logs", "pi05_dryrun.txt")
    dry = {}
    if exists_nonempty(dryrun_p):
        try:
            dry = json.load(open(dryrun_p))
        except Exception:
            pass
    pi_real = dry.get("backend_used") == "openpi"

    eval_p = latest(os.path.join(PROJ, "logs", "eval_policy_*.json"))
    rollout_ok = False
    if eval_p:
        try:
            ev = json.load(open(eval_p))
            rollout_ok = ev.get("avg_episode_length", 0) > 0 or ev.get("notes") == "rollout completed"
        except Exception:
            pass

    demo = latest(os.path.join(PROJ, "data", "raw_hdf5", "*.hdf5"))
    hdf5_summary = exists_nonempty(os.path.join(PROJ, "logs", "hdf5_summary.json"))
    conv = os.path.join(PROJ, "logs", "conversion_report.json")
    lerobot_ok = False
    if exists_nonempty(conv):
        try:
            lerobot_ok = bool(json.load(open(conv)).get("lerobot_built"))
        except Exception:
            pass
    ckpt = latest(os.path.join(PROJ, "policies", "checkpoints", "*"))
    train_ok = ckpt is not None

    runtests = latest(os.path.join(PROJ, "logs", "run_tests_*.log"))
    mock_ok = False
    if runtests:
        mock_ok = "ALL NON-ISAAC TESTS PASSED" in open(runtests).read()

    summary = [
        AUTO_MARK,
        "## Wake-up Summary",
        f"_Last updated by update_status.py_",
        "",
        f"* Project path: `{PROJ}`",
        f"* IsaacLab root: `/home1/banghai/Documents/IsaacLab`",
        f"* OpenPI installed: {yn(openpi_installed)}",
        f"* pi0.5 real model loaded (dry-run): {yn(pi_real)}",
        f"* Mock policy server passed: {yn(mock_ok)}",
        f"* IsaacLab rollout passed: {yn(rollout_ok)}",
        f"* Demo HDF5 found: {yn(bool(demo))}{(' ('+os.path.basename(demo)+')') if demo else ''}",
        f"* HDF5 inspect passed: {yn(hdf5_summary)}",
        f"* LeRobot conversion passed: {yn(lerobot_ok)}",
        f"* pi0.5 training smoke test passed: {yn(train_ok)}",
        "",
        "### Next 3 commands to run",
        "```bash",
        "bash pi05_isaacsim_baseline/scripts/make_env_check.sh",
        "PYBIN=python3 bash pi05_isaacsim_baseline/scripts/start_mock_server.sh 8008 && bash pi05_isaacsim_baseline/scripts/test_pi05_dryrun.sh",
        "bash pi05_isaacsim_baseline/scripts/collect_demos.sh --task Isaac-Stack-Cube-Franka-IK-Rel-v0 --teleop_device <your_device> --num_demos 10 --enable_cameras",
        "```",
        AUTO_MARK,
    ]
    block = "\n".join(summary) + "\n"

    status_p = os.path.join(PROJ, "STATUS.md")
    if exists_nonempty(status_p):
        content = open(status_p).read()
        if content.count(AUTO_MARK) >= 2:
            pre = content.split(AUTO_MARK)[0]
            post = content.split(AUTO_MARK)[2]
            content = pre + block + post
        else:
            content = block + "\n" + content
    else:
        content = block
    with open(status_p, "w") as f:
        f.write(content)
    print(f"[update_status] wrote summary into {status_p}")


if __name__ == "__main__":
    main()
