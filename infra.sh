#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/config.yaml"
  exit 1
fi
export HF_HOME=/ML-vePFS/research_gen/tja/cache/shared_hf_cache
export HF_HUB_CACHE=/ML-vePFS/research_gen/tja/cache/shared_hf_cache/hub
export TRANSFORMERS_CACHE=/ML-vePFS/research_gen/tja/cache/shared_hf_cache/hub
export TORCH_HOME=/ML-vePFS/research_gen/tja/cache/shared_torch_cache
export CLIP_CACHE_DIR=/ML-vePFS/research_gen/tja/cache/shared_clip_cache
echo "HF_HOME=$HF_HOME"
echo "HF_HUB_CACHE=$HF_HUB_CACHE"
echo "TRANSFORMERS_CACHE=$TRANSFORMERS_CACHE"
echo "CLIP_CACHE_DIR=$CLIP_CACHE_DIR"

# Setup CLIP cache symlink to use local cached models
mkdir -p ~/.cache
rm -f ~/.cache/clip
ln -s "$CLIP_CACHE_DIR" ~/.cache/clip
echo "CLIP symlink created: ~/.cache/clip -> $CLIP_CACHE_DIR"

export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
export TORCHELASTIC_EXIT_BARRIER_TIMEOUT=3600
if [[ -z "${MASTER_ADDR:-}" && -n "${MLP_WORKER_0_HOST:-}" ]]; then
  export MASTER_ADDR="$MLP_WORKER_0_HOST"
fi
if [[ -z "${MASTER_PORT:-}" && -n "${MLP_WORKER_0_PORT:-}" ]]; then
  export MASTER_PORT="$MLP_WORKER_0_PORT"
fi
if [[ -z "${NNODES:-}" && -n "${MLP_WORKER_NUM:-}" ]]; then
  export NNODES="$MLP_WORKER_NUM"
fi
if [[ -z "${NODE_RANK:-}" && -n "${MLP_ROLE_INDEX:-}" ]]; then
  export NODE_RANK="$MLP_ROLE_INDEX"
fi
if [[ -z "${NPROC_PER_NODE:-}" && -n "${MLP_WORKER_GPU:-}" ]]; then
  export NPROC_PER_NODE="$MLP_WORKER_GPU"
fi
CONFIG_YAML="$1"
ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
RESOLVED_JSON="$ROOT_DIR/output/resolved_config.json"
ENV_FILE="$ROOT_DIR/output/run.env"

mkdir -p "$ROOT_DIR/output"

if ! command -v conda >/dev/null 2>&1; then
  echo "conda not found in PATH"
  exit 1
fi

# shellcheck disable=SC1091
source "$(conda info --base)/etc/profile.d/conda.sh"

activate_conda() {
  local env_name="$1"
  # Some env activation scripts reference unset vars; relax nounset during activation.
  set +u
  conda activate "$env_name"
  set -u
}

# Parse config in worldscore environment
activate_conda /ML-vePFS/research_gen/jmy/jmy_ws/envs_conda/worldscore6
python "$ROOT_DIR/tools/parse_config.py" \
  --config "$CONFIG_YAML" \
  --output-json "$RESOLVED_JSON" \
  --output-env "$ENV_FILE"

# shellcheck disable=SC1090
source "$ENV_FILE"

if [[ -z "${RESOLVED_CONFIG:-}" ]] || [[ ! -f "$RESOLVED_CONFIG" ]]; then
  echo "RESOLVED_CONFIG is not set or file missing: $RESOLVED_CONFIG"
  exit 1
fi

# Export env for WorldScore
export WORLDSCORE_PATH
export DATA_PATH

NODE_RANK_VALUE=${NODE_RANK:-0}
NNODES_VALUE=${NNODES:-1}
LOG_DIR_BASE="$RUN_LOGS_DIR"
LOG_DIR="$LOG_DIR_BASE/node_${NODE_RANK_VALUE}"
if [[ "$NODE_RANK_VALUE" == "0" && "$NNODES_VALUE" -le 1 ]]; then
  rm -rf "$LOG_DIR_BASE"
fi
mkdir -p "$LOG_DIR"
exec > >(tee -a "$LOG_DIR/infra.log") 2>&1

SAMPLED_JSON=$(python - "$RESOLVED_CONFIG" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
    cfg = json.load(f)
print(cfg.get("worldscore", {}).get("sampled_json_path", ""))
PY
)

RUN_MODE=$(python - "$RESOLVED_CONFIG" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
  cfg = json.load(f)
print(cfg.get("run", {}).get("mode", "full"))
PY
)

python "$ROOT_DIR/tools/filter_cases.py" \
  --config-json "$RESOLVED_CONFIG" \
  --input-json "$SAMPLED_JSON" \
  --output-json "$FILTERED_JSON"

if [[ "$RUN_MODE" != "eval-only" ]]; then
  # Inference in diffsynth environment
  activate_conda /ML-vePFS/research_gen/jmy/jmy_ws/envs_conda/diffsynth
  export WORLDSCORE_PATH
  export DATA_PATH
  bash "$ROOT_DIR/tools/run_infer.sh" "$RESOLVED_CONFIG" 2>&1 | tee -a "$LOG_DIR/infer.log"
fi

if [[ "$RUN_MODE" != "eval-only" && "$RUN_MODE" != "infer-only" && "$NNODES_VALUE" -gt 1 ]]; then
  INFER_OUTPUT_ROOT=$(python - "$RESOLVED_CONFIG" <<'PY'
import json
import sys

cfg_path = sys.argv[1]
with open(cfg_path, "r", encoding="utf-8") as f:
  cfg = json.load(f)
print(cfg.get("run", {}).get("output_root", ""))
PY
  )
  if [[ -z "$INFER_OUTPUT_ROOT" ]]; then
    echo "run.output_root is required for multi-node inference sync"
    exit 1
  fi
  INFER_DONE_DIR="$INFER_OUTPUT_ROOT/infer_nodes"
  mkdir -p "$INFER_DONE_DIR"
  echo "done" > "$INFER_DONE_DIR/node_${NODE_RANK_VALUE}.done"
  echo "Waiting for all nodes to finish inference..."
  timeout_sec=${INFER_WAIT_TIMEOUT:-36000}
  waited=0
  while [[ "$waited" -lt "$timeout_sec" ]]; do
    missing=0
    for i in $(seq 0 $((NNODES_VALUE - 1))); do
      if [[ ! -f "$INFER_DONE_DIR/node_${i}.done" ]]; then
        missing=1
        break
      fi
    done
    if [[ "$missing" -eq 0 ]]; then
      break
    fi
    sleep 10
    waited=$((waited + 10))
  done
  if [[ "$missing" -ne 0 ]]; then
    echo "Timed out waiting for all nodes to finish inference"
    exit 1
  fi
fi

if [[ "$RUN_MODE" != "infer-only" ]]; then
  # Evaluation in worldscore environment
  activate_conda /ML-vePFS/research_gen/jmy/jmy_ws/envs_conda/worldscore6
  export WORLDSCORE_PATH
  export DATA_PATH
  bash "$ROOT_DIR/tools/run_eval.sh" "$RESOLVED_CONFIG" 2>&1 | tee -a "$LOG_DIR/eval.log"
fi
