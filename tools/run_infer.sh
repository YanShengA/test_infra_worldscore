#!/usr/bin/env bash
set -euo pipefail

CONFIG_JSON="$1"

python - <<'PY' "$CONFIG_JSON"
import json
import os
import subprocess
import sys

# 增加一个安全的获取环境变量的函数
def get_env_var(var_names, default_val):
    for name in var_names:
        val = os.environ.get(name)
        # val 必须存在且不能是空字符串
        if val and val.strip(): 
            return val
    return default_val

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)

compute = cfg.get("compute", {})
num_gpus = int(compute.get("num_gpus", 1))
use_dp = bool(compute.get("infer_use_dp", False))

# 修复优先级：火山引擎环境变量 > 普通环境变量 > 配置文件 > 默认值
nnodes = int(get_env_var(["MLP_WORKER_NUM", "NNODES"], compute.get("nnodes", 1)))
node_rank = int(get_env_var(["MLP_ROLE_INDEX", "NODE_RANK"], compute.get("node_rank", 0)))
master_addr = get_env_var(["MLP_WORKER_0_HOST", "MASTER_ADDR"], compute.get("master_addr", "localhost"))
master_port = str(get_env_var(["MLP_WORKER_0_PORT", "MASTER_PORT"], compute.get("master_port", "29500")))

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
        f"--nnodes={nnodes}",
        f"--nproc_per_node={num_gpus}",
        f"--node_rank={node_rank}",
        f"--master_addr={master_addr}",
        f"--master_port={master_port}",
        "/ML-vePFS/research_gen/tja/WorldScore/run_wan_cam_worldscore_dp_sampled_custom_ckpt.py",
        "--infra_config", cfg_path,
        "--worldscore_model_name", worldscore.get("model_name", "fantasy_world"),
        "--checkpoint_path", wan.get("checkpoint_path", ""),
        "--model_path", wan.get("base_model_root", ""),
        "--sampled_json_path", sampled_json,
        "--use_dp",
    ]

print("Running Distributed Setup:")
print(f"NNODES: {nnodes}, NODE_RANK: {node_rank}, MASTER_ADDR: {master_addr}, MASTER_PORT: {master_port}")
print("Running Command:", " ".join(cmd))
subprocess.check_call(cmd, env=env)
PY