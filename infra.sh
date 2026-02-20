#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 1 ]]; then
  echo "Usage: $0 /path/to/config.yaml"
  exit 1
fi
export HF_HOME=/root/.cache/huggingface
export TRANSFORMERS_OFFLINE=1
export HF_HUB_OFFLINE=1
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

LOG_DIR="$RUN_LOGS_DIR"
rm -rf "$LOG_DIR"
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

if [[ "$RUN_MODE" != "infer-only" ]]; then
  # Evaluation in worldscore environment
  activate_conda /ML-vePFS/research_gen/jmy/jmy_ws/envs_conda/worldscore6
  export WORLDSCORE_PATH
  export DATA_PATH
  bash "$ROOT_DIR/tools/run_eval.sh" "$RESOLVED_CONFIG" 2>&1 | tee -a "$LOG_DIR/eval.log"
fi
