import argparse
import json
import os
from typing import Any, Dict, List


def _matches(value, allowed: List[str]) -> bool:
    if not allowed:
        return True
    if value is None:
        return False
    return value in allowed


def _matches_any(values, allowed: List[str]) -> bool:
    if not allowed:
        return True
    if not values:
        return False
    return any(v in allowed for v in values)


def _matches_contains(text: str, needles: List[str]) -> bool:
    if not needles:
        return True
    if not text:
        return False
    return any(n in text for n in needles)


def _filter_item(item: Dict[str, Any], filters: Dict[str, Any]) -> bool:
    if not filters.get("enable", False):
        return True

    if not _matches(item.get("visual_movement"), filters.get("visual_movement", [])):
        return False

    if not _matches(item.get("visual_style"), filters.get("visual_style", [])):
        return False

    if not _matches(item.get("scene_type"), filters.get("scene_type", [])):
        return False

    if not _matches(item.get("motion_type"), filters.get("motion_type", [])):
        return False

    image_path = item.get("image", "")
    image_name = os.path.basename(image_path)

    if not _matches(image_name, filters.get("image_name", [])):
        return False

    if not _matches_contains(image_path, filters.get("image_contains", [])):
        return False

    camera_path = item.get("camera_path", [])
    if not _matches_any(camera_path, filters.get("camera_path_any", [])):
        return False

    if not _matches(item.get("category"), filters.get("category", [])):
        return False

    instance_name = os.path.splitext(image_name)[0]
    if not _matches(instance_name, filters.get("instance", [])):
        return False

    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config-json", required=True, help="Resolved config JSON")
    parser.add_argument("--input-json", required=True, help="Input sampled json")
    parser.add_argument("--output-json", required=True, help="Filtered output json")
    args = parser.parse_args()

    with open(args.config_json, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    filters = cfg.get("filters", {})
    with open(args.input_json, "r", encoding="utf-8") as f:
        data = json.load(f)

    filtered = [item for item in data if _filter_item(item, filters)]

    with open(args.output_json, "w", encoding="utf-8") as f:
        json.dump(filtered, f, indent=2, ensure_ascii=True)

    print(f"Filtered {len(filtered)} / {len(data)} cases")


if __name__ == "__main__":
    main()
