"""HTTP policy server (stdlib only -> runs in ANY env).

Endpoints
  GET  /health  -> {"status":"ok","backend":...,"requests":N,"uptime_s":...}
  POST /infer   -> body: Observation JSON (schemas.Observation.to_dict())
                   resp: {"action": Action JSON, "latency_ms": float, "backend": ...}

Backends
  --backend mock                       (default, no deps beyond stdlib)
  --backend openpi --config <name> --ckpt <dir>
  --backend openpi --env-default aloha_sim

Run mock (works in IsaacLab conda or anywhere):
  python adapters/policy_server/server.py --backend mock --port 8008

Run real pi0.5 (must be the OpenPI venv python):
  .venv_openpi/bin/python adapters/policy_server/server.py \
      --backend openpi --config pi05_isaaclab_franka \
      --ckpt checkpoints/pi05_isaaclab_xxx/30 --port 8008
"""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

logging.basicConfig(level=logging.INFO, format="%(asctime)s [server] %(message)s")
logger = logging.getLogger("policy_server")

_STATE = {"backend": None, "policy": None, "requests": 0, "start": time.time(), "latencies": []}


def _build_backend(args):
    if args.backend == "mock":
        from mock_policy import MockPolicy

        return MockPolicy(action_horizon=args.action_horizon)
    elif args.backend == "openpi":
        from openpi_policy import OpenPIPolicy

        return OpenPIPolicy(
            config_name=args.config,
            ckpt_dir=args.ckpt,
            env_default=args.env_default,
            default_prompt=args.default_prompt,
        )
    raise ValueError(f"unknown backend {args.backend}")


class Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence default noisy logging
        pass

    def _send(self, code: int, payload: dict):
        body = json.dumps(payload).encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):
        if self.path.rstrip("/") == "/health":
            lat = _STATE["latencies"]
            avg = sum(lat) / len(lat) if lat else 0.0
            self._send(
                200,
                {
                    "status": "ok",
                    "backend": _STATE["backend"],
                    "requests": _STATE["requests"],
                    "uptime_s": round(time.time() - _STATE["start"], 1),
                    "avg_latency_ms": round(avg, 2),
                },
            )
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        if self.path.rstrip("/") != "/infer":
            self._send(404, {"error": "not found"})
            return
        try:
            n = int(self.headers.get("Content-Length", 0))
            obs = json.loads(self.rfile.read(n).decode("utf-8")) if n else {}
        except Exception as e:  # noqa
            self._send(400, {"error": f"bad request: {e}"})
            return

        t0 = time.time()
        try:
            action = _STATE["policy"].infer(obs)
            latency_ms = (time.time() - t0) * 1000.0
            _STATE["requests"] += 1
            _STATE["latencies"].append(latency_ms)
            if len(_STATE["latencies"]) > 1000:
                _STATE["latencies"] = _STATE["latencies"][-1000:]
            self._send(200, {"action": action, "latency_ms": round(latency_ms, 3), "backend": _STATE["backend"]})
        except Exception as e:  # noqa
            logger.exception("inference failed")
            self._send(500, {"error": f"inference failed: {e}"})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--backend", choices=["mock", "openpi"], default="mock")
    p.add_argument("--host", default="0.0.0.0")
    p.add_argument("--port", type=int, default=8008)
    p.add_argument("--action-horizon", type=int, default=1)
    p.add_argument("--config", default=None, help="openpi training config name")
    p.add_argument("--ckpt", default=None, help="openpi checkpoint dir")
    p.add_argument("--env-default", default=None, help="openpi default env policy (debug)")
    p.add_argument("--default-prompt", default="Perform the manipulation task.")
    args = p.parse_args()

    logger.info("Building backend: %s", args.backend)
    _STATE["policy"] = _build_backend(args)
    _STATE["backend"] = _STATE["policy"].backend_name
    logger.info("Backend ready: %s", _STATE["backend"])

    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    logger.info("Serving on http://%s:%d  (/health, /infer)", args.host, args.port)
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        logger.info("shutting down")
        httpd.shutdown()


if __name__ == "__main__":
    main()
