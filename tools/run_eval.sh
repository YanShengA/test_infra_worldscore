#!/usr/bin/env bash
set -euo pipefail

CONFIG_JSON="$1"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

python - <<'PY' "$CONFIG_JSON" "$SCRIPT_DIR"
import json
import os
import subprocess
import sys
import time

cfg_path = sys.argv[1]
script_dir = sys.argv[2]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

compute = cfg.get("compute", {})
num_jobs = int(compute.get("eval_num_jobs", compute.get("num_gpus", 1)))
num_shards = int(compute.get("eval_num_shards", os.environ.get("NNODES", os.environ.get("MLP_WORKER_NUM", 1))))
shard_index = int(compute.get("eval_shard_index", os.environ.get("NODE_RANK", os.environ.get("MLP_ROLE_INDEX", 0))))
skip_mean = bool(compute.get("eval_skip_mean", num_shards > 1))
auto_mean = bool(compute.get("eval_auto_mean", True))

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
    "--num-shards", str(num_shards),
    "--shard-index", str(shard_index),
]

if skip_mean:
    cmd.append("--skip-mean")

print("Running:", " ".join(cmd))
subprocess.check_call(cmd, env=env, cwd=worldscore_root)

if num_shards > 1:
    run_cfg = cfg.get("run", {})
    output_root = run_cfg.get("output_root", "")
    if not output_root:
        raise RuntimeError("run.output_root is required for sharded evaluation")
    done_dir = os.path.join(output_root, "eval_shards")
    os.makedirs(done_dir, exist_ok=True)
    done_file = os.path.join(done_dir, f"shard_{shard_index}.done")
    with open(done_file, "w", encoding="utf-8") as f:
        f.write("done\n")

    if shard_index == 0 and auto_mean:
        timeout_sec = int(os.environ.get("EVAL_WAIT_TIMEOUT", "36000"))
        waited = 0
        missing = True
        while missing and waited < timeout_sec:
            missing = False
            for i in range(num_shards):
                if not os.path.exists(os.path.join(done_dir, f"shard_{i}.done")):
                    missing = True
                    break
            if missing:
                time.sleep(10)
                waited += 10

        if missing:
            raise RuntimeError("Timed out waiting for all eval shards to finish")

        mean_cmd = [
            "python",
            os.path.join(script_dir, "evaluate_filtered.py"),
            "--config-json", cfg_path,
            "--only-calc-mean",
        ]
        print("Running:", " ".join(mean_cmd))
        subprocess.check_call(mean_cmd, env=env, cwd=worldscore_root)
PY
