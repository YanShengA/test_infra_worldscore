#!/usr/bin/env bash
set -euo pipefail

CONFIG_JSON="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python - <<'PY' "$CONFIG_JSON" "$SCRIPT_DIR"
import json
import os
import subprocess
import sys

cfg_path = sys.argv[1]
script_dir = sys.argv[2]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

compute = cfg.get("compute", {})
num_jobs = int(compute.get("eval_num_jobs", 1))

worldscore = cfg.get("worldscore", {})
wan = cfg.get("wan", {})
model_path_env = (
    worldscore.get("runs_root_base")
    or os.environ.get("MODEL_PATH")
    or wan.get("base_model_root")
    or ""
)
env = os.environ.copy()
if model_path_env:
    # Needed for ${oc.env:MODEL_PATH} in WorldScore model configs.
    env["MODEL_PATH"] = model_path_env

worldscore_root = os.path.join(cfg.get("env", {}).get("worldscore_path", ""), "WorldScore")
if not worldscore_root or not os.path.isdir(worldscore_root):
    raise RuntimeError(f"WorldScore root not found: {worldscore_root}")

cmd = [
    "python",
    os.path.join(script_dir, "evaluate_filtered.py"),
    "--config-json", cfg_path,
    "--num-jobs", str(num_jobs),
]

print("Running:", " ".join(cmd))
subprocess.check_call(cmd, env=env, cwd=worldscore_root)
PY
