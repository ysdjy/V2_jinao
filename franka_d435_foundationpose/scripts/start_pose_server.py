"""Start the ZMQ pose server (FoundationPose side only).

Run in the foundationpose env:
    conda run -n foundationpose python franka_d435_foundationpose/scripts/start_pose_server.py

The IsaacLab side talks to it via franka_d435_foundationpose.server.pose_client.PoseClient,
which has NO FoundationPose dependency.
"""

import argparse
import sys

import _bootstrap  # noqa: F401

from franka_d435_foundationpose.utils.config import default_config_path


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--endpoint", default="tcp://0.0.0.0:5599", help="ZMQ bind endpoint")
    parser.add_argument("--config", default=None, help="foundationpose.yaml path")
    parser.add_argument("--mock", action="store_true", help="force mock estimator")
    args = parser.parse_args()

    # Imported here (not at top) because pose_server pulls in the estimator.
    from franka_d435_foundationpose.server.pose_server import PoseServer

    config = args.config or default_config_path("foundationpose.yaml")
    server = PoseServer(config, endpoint=args.endpoint, force_mock=args.mock)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nshutting down pose server")
    finally:
        server.close()
    return 0


if __name__ == "__main__":
    sys.exit(main())
