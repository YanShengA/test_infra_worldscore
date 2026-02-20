import argparse
import json
import os
from datetime import datetime

from omegaconf import OmegaConf


def _ensure_dir(path: str) -> None:
    os.makedirs(path, exist_ok=True)


def _default(value, fallback):
    return value if value is not None else fallback


def _get_nested(cfg, *keys, default=None):
    value = cfg
    for key in keys:
        if not isinstance(value, dict):
            return default
        value = value.get(key)
    return default if value is None else value


def _ensure_model_config_and_registry(cfg: dict) -> None:
    worldscore_path = _get_nested(cfg, "env", "worldscore_path", default="")
    model_name = _get_nested(cfg, "worldscore", "model_name", default="")
    if not worldscore_path or not model_name:
        return

    model_configs_dir = os.path.join(worldscore_path, "WorldScore", "config", "model_configs")
    model_config_path = os.path.join(model_configs_dir, f"{model_name}.yaml")

    config_overrides = _get_nested(cfg, "worldscore", "config_overrides", default={})
    wan_cfg = _get_nested(cfg, "wan", default={})
    resolution = _default(config_overrides.get("resolution"), wan_cfg.get("resolution")) or [720, 720]
    frames = _default(config_overrides.get("frames"), wan_cfg.get("frames")) or 81
    fps = _default(config_overrides.get("fps"), wan_cfg.get("fps")) or 16
    output_dir = _get_nested(cfg, "worldscore", "output_dir", default=".") or "."

    if not os.path.exists(model_config_path):
        _ensure_dir(model_configs_dir)
        model_config = [
            f"model: {model_name}",
            "",
            f"runs_root: ${{oc.env:MODEL_PATH}}/{model_name}",
            f"output_dir: \"{output_dir}\"",
            "",
            f"resolution: [{int(resolution[0])}, {int(resolution[1])}]",
            "generate_type: i2v",
            "",
            f"frames: {int(frames)}",
            f"fps: {int(fps)}",
            "",
        ]
        with open(model_config_path, "w", encoding="utf-8") as f:
            f.write("\n".join(model_config))

    modeltype_path = os.path.join(
        worldscore_path,
        "WorldScore",
        "worldscore",
        "benchmark",
        "utils",
        "modeltype.py",
    )
    if not os.path.exists(modeltype_path):
        return

    with open(modeltype_path, "r", encoding="utf-8") as f:
        text = f.read()
    if f"\"{model_name}\"" in text or f"'{model_name}'" in text:
        return

    lines = text.splitlines(keepends=True)
    out_lines = []
    in_videogen = False
    inserted = False
    for line in lines:
        if not in_videogen and ("\"videogen\"" in line or "'videogen'" in line):
            in_videogen = True
            out_lines.append(line)
            continue

        if in_videogen and line.strip().startswith("]") and not inserted:
            out_lines.append(f"        \"{model_name}\",\n")
            inserted = True
            out_lines.append(line)
            in_videogen = False
            continue

        out_lines.append(line)

    if inserted:
        with open(modeltype_path, "w", encoding="utf-8") as f:
            f.write("".join(out_lines))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True, help="Path to infra YAML config")
    parser.add_argument("--output-json", required=True, help="Path to write resolved JSON")
    parser.add_argument("--output-env", required=True, help="Path to write env file")
    args = parser.parse_args()

    cfg = OmegaConf.load(args.config)
    cfg = OmegaConf.to_container(cfg, resolve=True)

    _ensure_model_config_and_registry(cfg)

    run_cfg = cfg.get("run", {})
    run_name = run_cfg.get("name") or datetime.now().strftime("run_%Y%m%d_%H%M%S")
    output_root = run_cfg.get("output_root") or os.path.join(os.path.dirname(args.config), "output")
    run_dir = os.path.join(output_root, run_name)

    paths = {
        "run_dir": run_dir,
        "output_root": output_root,
        "resolved_config": os.path.abspath(args.output_json),
        "filtered_json": os.path.join(run_dir, "filtered_cases.json"),
        "logs_dir": os.path.join(run_dir, "logs"),
    }
    cfg["paths"] = paths

    _ensure_dir(run_dir)
    _ensure_dir(paths["logs_dir"])

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(cfg, f, indent=2, ensure_ascii=True)

    env_lines = [
        f"RUN_DIR={paths['run_dir']}",
        f"RUN_OUTPUT_ROOT={paths['output_root']}",
        f"RUN_LOGS_DIR={paths['logs_dir']}",
        f"RESOLVED_CONFIG={paths['resolved_config']}",
        f"FILTERED_JSON={paths['filtered_json']}",
        f"WORLDSCORE_PATH={cfg.get('env', {}).get('worldscore_path', '')}",
        f"DATA_PATH={cfg.get('env', {}).get('data_path', '')}",
    ]

    with open(args.output_env, "w", encoding="utf-8") as f:
        f.write("\n".join(env_lines) + "\n")


if __name__ == "__main__":
    main()
