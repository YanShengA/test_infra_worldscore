import argparse
import json
import os
import sys
import types
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch
import torch.multiprocessing as mp
from omegaconf import OmegaConf

worldscore_root = os.environ.get("WORLDSCORE_PATH", "")
if worldscore_root:
    local_ws = os.path.join(worldscore_root, "WorldScore")
    if local_ws not in sys.path:
        sys.path.insert(0, local_ws)
    sea_raft_core = os.path.join(
        worldscore_root,
        "WorldScore",
        "worldscore",
        "benchmark",
        "metrics",
        "third_party",
        "SEA-RAFT",
    )
    if sea_raft_core not in sys.path:
        sys.path.insert(0, sea_raft_core)
    raft_core = os.path.join(
        worldscore_root,
        "WorldScore",
        "worldscore",
        "benchmark",
        "metrics",
        "third_party",
        "RAFT",
    )
    if raft_core not in sys.path:
        sys.path.append(raft_core)
    droid_path = os.path.join(
        worldscore_root,
        "WorldScore",
        "worldscore",
        "benchmark",
        "metrics",
        "third_party",
        "droid_slam",
    )
    if droid_path not in sys.path:
        sys.path.insert(0, droid_path)
    vf_mamba_path = os.path.join(
        worldscore_root,
        "WorldScore",
        "worldscore",
        "benchmark",
        "metrics",
        "third_party",
        "VFIMamba",
    )
    if vf_mamba_path not in sys.path:
        sys.path.insert(0, vf_mamba_path)

try:
    import droid  # type: ignore
    _DROID_AVAILABLE = True
except Exception:
    _DROID_AVAILABLE = False
    stub = types.ModuleType("droid")

    class Droid:  # pylint: disable=too-few-public-methods
        def __init__(self, *args, **kwargs):
            raise RuntimeError("droid is not available; camera metrics cannot run.")

    stub.Droid = Droid
    sys.modules["droid"] = stub

from worldscore.benchmark.helpers.evaluator import Evaluator, process_batch
from worldscore.benchmark.utils.utils import aspect_info


def deep_update(target, source):
    for k, v in source.items():
        if isinstance(v, dict):
            if k not in target:
                target[k] = {}
            deep_update(target[k], v)
        else:
            target[k] = v
    return target


def _apply_metric_filter(selected_metrics):
    if not selected_metrics:
        return
    filtered = {}
    for aspect, info in aspect_info.items():
        metrics = {}
        for metric_name, metric_attr in info["metrics"].items():
            if metric_name in selected_metrics:
                metrics[metric_name] = metric_attr
        if metrics:
            filtered[aspect] = {
                "type": info["type"],
                "metrics": metrics,
            }
    aspect_info.clear()
    aspect_info.update(filtered)


def _matches(value, allowed):
    if not allowed:
        return True
    if value is None:
        return False
    return value in allowed


def _collect_instances(root_path: Path, visual_movement: str, filters: dict, evaluator: Evaluator):
    instances = []
    if not root_path.exists():
        return instances

    if visual_movement == "static":
        for visual_style in sorted([x.name for x in root_path.iterdir() if x.is_dir()]):
            if not _matches(visual_style, filters.get("visual_style", [])):
                continue
            visual_style_dir = root_path / visual_style
            for scene_type in sorted([x.name for x in visual_style_dir.iterdir() if x.is_dir()]):
                if not _matches(scene_type, filters.get("scene_type", [])):
                    continue
                scene_type_dir = visual_style_dir / scene_type
                for category in sorted([x.name for x in scene_type_dir.iterdir() if x.is_dir()]):
                    if not _matches(category, filters.get("category", [])):
                        continue
                    category_dir = scene_type_dir / category
                    for instance in sorted([x.name for x in category_dir.iterdir() if x.is_dir()]):
                        if not _matches(instance, filters.get("instance", [])):
                            continue
                        instance_dir = category_dir / instance
                        if not evaluator.data_exists(str(instance_dir)):
                            continue
                        instances.append([visual_style, scene_type, category, instance, instance_dir])
    else:
        for visual_style in sorted([x.name for x in root_path.iterdir() if x.is_dir()]):
            if not _matches(visual_style, filters.get("visual_style", [])):
                continue
            visual_style_dir = root_path / visual_style
            for motion_type in sorted([x.name for x in visual_style_dir.iterdir() if x.is_dir()]):
                if not _matches(motion_type, filters.get("motion_type", [])):
                    continue
                motion_type_dir = visual_style_dir / motion_type
                for instance in sorted([x.name for x in motion_type_dir.iterdir() if x.is_dir()]):
                    if not _matches(instance, filters.get("instance", [])):
                        continue
                    instance_dir = motion_type_dir / instance
                    if not evaluator.data_exists(str(instance_dir)):
                        continue
                    instances.append([visual_style, motion_type, instance, instance_dir])
    return instances


def _calculate_mean_scores(metrics_results, aspect_list, output_path):
    scores = {}
    for aspect in metrics_results:
        if aspect not in aspect_list:
            continue
        metric_score_list = []
        for metric_name, metric_scores in metrics_results[aspect].items():
            metric_score_list.append(
                np.mean(
                    np.array([metric_score["score_normalized"] for metric_score in metric_scores]),
                    axis=0,
                ).item()
            )
        if metric_score_list:
            scores[aspect] = round(np.mean(metric_score_list).item() * 100, 2)

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(scores, f, indent=2, ensure_ascii=True)
    return scores


def _worker_wrapper(gpu_id, config, instance_batch, aspect_list, visual_movement, queue):
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    mapped_gpu_id = 0
    torch.cuda.set_device(mapped_gpu_id)
    result = process_batch(config, instance_batch, aspect_list, visual_movement)
    queue.put(result)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True, help="Resolved config JSON")
    parser.add_argument("--num-jobs", type=int, default=1)
    parser.add_argument("--only-calc-mean", action="store_true")
    parser.add_argument("--delete-calculated", action="store_true")
    parser.add_argument("--num-shards", type=int, default=1)
    parser.add_argument("--shard-index", type=int, default=0)
    parser.add_argument("--skip-mean", action="store_true")
    args = parser.parse_args()

    with open(args.config_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    base_config = OmegaConf.load(os.path.join(cfg["env"]["worldscore_path"], "WorldScore/config/base_config.yaml"))
    model_name = cfg.get("worldscore", {}).get("model_name", "fantasy_world")
    model_cfg_path = os.path.join(
        cfg["env"]["worldscore_path"],
        "WorldScore/config/model_configs",
        f"{model_name}.yaml",
    )
    model_config = OmegaConf.load(model_cfg_path)
    config = OmegaConf.merge(base_config, model_config)

    config = OmegaConf.to_container(config, resolve=True)

    overrides = cfg.get("worldscore", {}).get("config_overrides", {})
    config.update(overrides)

    runs_root_base = cfg.get("worldscore", {}).get("runs_root_base")
    if runs_root_base:
        config["runs_root"] = os.path.join(runs_root_base, model_name)

    output_dir = cfg.get("worldscore", {}).get("output_dir")
    if output_dir:
        config["output_dir"] = output_dir

    visual_movements = cfg.get("worldscore", {}).get("visual_movement", ["static", "dynamic"])
    filters = cfg.get("filters", {})
    if not filters.get("enable", False):
        filters = {}

    selected_aspects = cfg.get("metrics", {}).get("aspects", [])
    selected_metrics = cfg.get("metrics", {}).get("metrics", [])

    if not _DROID_AVAILABLE:
        if "camera_control" in selected_aspects or "camera_error" in selected_metrics:
            raise RuntimeError("droid is missing; remove camera_control/camera_error from metrics to proceed.")

    _apply_metric_filter(selected_metrics)

    for visual_movement in visual_movements:
        config["visual_movement"] = visual_movement
        evaluator = Evaluator(config)
        if selected_metrics:
            available_aspects = list(aspect_info.keys())
            if selected_aspects:
                aspect_list = [a for a in selected_aspects if a in available_aspects]
            else:
                aspect_list = available_aspects
        else:
            aspect_list = selected_aspects or evaluator.build_full_aspect_list()

        if args.delete_calculated:
            _delete_existing(evaluator.root_path)

        instances = _collect_instances(evaluator.root_path, visual_movement, filters, evaluator)
        if not instances:
            print(f"No instances found for {visual_movement}")
            continue

        if args.num_shards > 1:
            if args.shard_index < 0 or args.shard_index >= args.num_shards:
                raise ValueError(f"Invalid shard-index {args.shard_index} for num-shards {args.num_shards}")
            instances = [inst for idx, inst in enumerate(instances) if idx % args.num_shards == args.shard_index]
            if not instances:
                print(f"No instances for shard {args.shard_index}/{args.num_shards} ({visual_movement})")
                continue

        if args.only_calc_mean:
            _calculate_existing_mean(evaluator.root_path, aspect_list)
            continue

        batch_size = max(len(instances) // max(args.num_jobs, 1), 1)
        instance_batches = [
            instances[start_idx : start_idx + batch_size]
            for start_idx in range(0, len(instances), batch_size)
        ]

        if args.num_jobs > 1:
            try:
                ctx = mp.get_context("spawn")
            except RuntimeError:
                ctx = mp
            queue = ctx.Queue()
            processes = []

            num_gpus = torch.cuda.device_count() if torch.cuda.is_available() else 1
            active_jobs = min(len(instance_batches), args.num_jobs)

            for i in range(active_jobs):
                batch = instance_batches[i]
                gpu_id = i % num_gpus
                p = ctx.Process(
                    target=_worker_wrapper,
                    args=(gpu_id, config, batch, aspect_list, visual_movement, queue),
                )
                processes.append(p)
                p.start()

            batch_results = []
            for _ in range(active_jobs):
                batch_results.append(queue.get())
            for p in processes:
                p.join()
        else:
            batch_results = [
                process_batch(
                    config=config,
                    instance_batch=batch,
                    aspect_list=aspect_list,
                    visual_movement=visual_movement,
                )
                for batch in instance_batches
            ]

        for result in batch_results:
            deep_update(evaluator.metrics_results, result)

        if not args.skip_mean:
            metrics_results = defaultdict(lambda: defaultdict(list))
            _collect_metrics(evaluator.root_path, metrics_results)
            output_path = os.path.join(
                config["runs_root"],
                config["output_dir"],
                f"worldscore_filtered_{visual_movement}.json",
            )
            _calculate_mean_scores(metrics_results, aspect_list, output_path)


def _collect_metrics(root_path, metrics_results):
    if not root_path.exists():
        return
    for dirpath, dirnames, filenames in os.walk(root_path):
        if "evaluation.json" not in filenames:
            continue
        eval_path = os.path.join(dirpath, "evaluation.json")
        try:
            with open(eval_path, "r", encoding="utf-8") as f:
                instance_result_dict = json.load(f)
        except Exception:
            continue
        for aspect, aspect_scores in instance_result_dict.items():
            for metric_name, metric_score in aspect_scores.items():
                if not metric_score:
                    continue
                metrics_results[aspect][metric_name].append(metric_score)


def _calculate_existing_mean(root_path, aspect_list):
    metrics_results = defaultdict(lambda: defaultdict(list))
    _collect_metrics(root_path, metrics_results)
    output_path = root_path / "worldscore_filtered_mean.json"
    _calculate_mean_scores(metrics_results, aspect_list, str(output_path))


def _delete_existing(root_path: Path):
    for dirpath, dirnames, filenames in os.walk(root_path):
        if "evaluation.json" in filenames:
            os.remove(os.path.join(dirpath, "evaluation.json"))


if __name__ == "__main__":
    main()
