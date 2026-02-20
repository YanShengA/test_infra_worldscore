#!/usr/bin/env bash
set -euo pipefail

CONFIG_JSON="$1"

python - <<'PY' "$CONFIG_JSON"
import json
import os
import subprocess
import sys

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

compute = cfg.get("compute", {})
num_gpus = int(compute.get("num_gpus", 1))
use_dp = bool(compute.get("infer_use_dp", False))

wan = cfg.get("wan", {})
worldscore = cfg.get("worldscore", {})
paths = cfg.get("paths", {})

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

sampled_json = worldscore.get("sampled_json_path")
if cfg.get("filters", {}).get("enable", False):
    sampled_json = paths.get("filtered_json", sampled_json)

cmd = [
    "python",
    "/ML-vePFS/research_gen/tja/WorldScore/run_wan_cam_worldscore_dp_sampled_custom_ckpt.py",
    "--infra_config", cfg_path,
    "--worldscore_model_name", worldscore.get("model_name", "fantasy_world"),
    "--checkpoint_path", wan.get("checkpoint_path", ""),
    "--model_path", wan.get("base_model_root", ""),
    "--sampled_json_path", sampled_json,
]

if use_dp and num_gpus > 1:
    cmd = [
        "torchrun",
        f"--nproc_per_node={num_gpus}",
        "/ML-vePFS/research_gen/tja/WorldScore/run_wan_cam_worldscore_dp_sampled_custom_ckpt.py",
        "--infra_config", cfg_path,
        "--worldscore_model_name", worldscore.get("model_name", "fantasy_world"),
        "--checkpoint_path", wan.get("checkpoint_path", ""),
        "--model_path", wan.get("base_model_root", ""),
        "--sampled_json_path", sampled_json,
        "--use_dp",
    ]

print("Running:", " ".join(cmd))
subprocess.check_call(cmd, env=env)
PY
