#!/usr/bin/env python3
"""Compare a DetZero lidar.json with a manually corrected CaliperGT export."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("baseline", type=Path)
    parser.add_argument("corrected", type=Path,
                        help="lidar.json or an extracted export directory")
    parser.add_argument("--output", type=Path)
    parser.add_argument("--max-center-distance", type=float, default=3.0)
    return parser.parse_args()


def find_lidar_json(path: Path) -> Path:
    if path.is_file():
        return path
    candidates = [item for item in path.rglob("lidar.json")
                  if item.is_file()]
    populated = []
    for item in candidates:
        document = json.loads(item.read_text())
        if document.get("annotations"):
            populated.append(item)
    if len(populated) != 1:
        raise ValueError(f"{path}: expected one populated lidar.json, "
                         f"found {len(populated)}")
    return populated[0]


def box(annotation: dict) -> np.ndarray:
    cube = annotation["cuboid_3d"]
    return np.asarray([
        cube["position"]["x"], cube["position"]["y"],
        cube["position"]["z"], cube["dimensions"]["length"],
        cube["dimensions"]["width"], cube["dimensions"]["height"],
        cube["rotation"]["yaw"],
    ], dtype=float)


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def percentile(values: list[float], quantiles=(0, .01, .5, .99, 1)) -> dict:
    if not values:
        return {}
    result = np.quantile(values, quantiles)
    return {f"p{int(q * 1000) / 10:g}": float(v)
            for q, v in zip(quantiles, result)}


def main() -> None:
    args = parse_args()
    baseline_path = find_lidar_json(args.baseline.resolve())
    corrected_path = find_lidar_json(args.corrected.resolve())
    baseline = json.loads(baseline_path.read_text())
    corrected = json.loads(corrected_path.read_text())
    if len(baseline["frames"]) != len(corrected["frames"]):
        raise ValueError("baseline and corrected frame counts differ")

    old_by_frame = defaultdict(list)
    new_by_frame = defaultdict(list)
    for annotation in baseline["annotations"]:
        old_by_frame[int(annotation["frame_id"])].append(annotation)
    for annotation in corrected["annotations"]:
        new_by_frame[int(annotation["frame_id"])].append(annotation)

    matches = []
    removed, added = [], []
    for frame_id in range(len(baseline["frames"])):
        old, new = old_by_frame[frame_id], new_by_frame[frame_id]
        if not old or not new:
            removed.extend(old)
            added.extend(new)
            continue
        old_boxes = np.stack([box(item) for item in old])
        new_boxes = np.stack([box(item) for item in new])
        distances = np.linalg.norm(
            old_boxes[:, None, :3] - new_boxes[None, :, :3], axis=2)
        used_old, used_new = set(), set()
        candidates = sorted(
            (float(distances[i, j]), i, j)
            for i in range(len(old)) for j in range(len(new)))
        for distance, old_index, new_index in candidates:
            if old_index in used_old or new_index in used_new:
                continue
            if distance > args.max_center_distance:
                break
            used_old.add(int(old_index))
            used_new.add(int(new_index))
            before, after = old[old_index], new[new_index]
            before_box, after_box = old_boxes[old_index], new_boxes[new_index]
            matches.append({
                "frame_id": frame_id,
                "before_class": str(before["class"]).lower(),
                "after_class": str(after["class"]).lower(),
                "center_distance_m": distance,
                "dimension_delta_m": (after_box[3:6] -
                                      before_box[3:6]).tolist(),
                "yaw_delta_deg": math.degrees(
                    wrap(float(after_box[6] - before_box[6]))),
                "before_track": str(before.get("track_id", "")),
                "after_track": str(after.get("track_id", "")),
                "after_box": after_box.tolist(),
            })
        removed.extend(item for index, item in enumerate(old)
                       if index not in used_old)
        added.extend(item for index, item in enumerate(new)
                     if index not in used_new)

    class_changes = Counter(
        f'{item["before_class"]}->{item["after_class"]}' for item in matches
        if item["before_class"] != item["after_class"])
    dimensions = defaultdict(lambda: defaultdict(list))
    for annotation in corrected["annotations"]:
        name = str(annotation["class"]).lower()
        values = box(annotation)[3:6]
        for axis, value in zip(("length", "width", "height"), values):
            dimensions[name][axis].append(float(value))

    yaw_changes = [abs(item["yaw_delta_deg"]) for item in matches]
    near_flips = [item for item in matches
                  if abs(abs(item["yaw_delta_deg"]) - 180) <= 20]
    same_lane = [item for item in matches
                 if -4.5 <= item["after_box"][1] <= 0.5 and
                 item["after_class"] in {"car", "van", "truck", "bus"}]
    oncoming = [item for item in matches
                if 0.5 < item["after_box"][1] <= 8 and
                item["after_class"] in {"car", "van", "truck", "bus"}]

    def heading_summary(rows: list[dict]) -> dict:
        forward = sum(math.cos(item["after_box"][6]) >= 0 for item in rows)
        return {"count": len(rows), "forward": forward,
                "backward": len(rows) - forward}

    report = {
        "baseline": str(baseline_path),
        "corrected": str(corrected_path),
        "frames": len(baseline["frames"]),
        "baseline_annotations": len(baseline["annotations"]),
        "corrected_annotations": len(corrected["annotations"]),
        "matched": len(matches),
        "removed": len(removed),
        "added": len(added),
        "baseline_classes": dict(Counter(
            str(item["class"]).lower() for item in baseline["annotations"])),
        "corrected_classes": dict(Counter(
            str(item["class"]).lower() for item in corrected["annotations"])),
        "class_changes": dict(class_changes),
        "yaw": {
            "absolute_delta_deg": percentile(yaw_changes),
            "near_180_degree_flips": len(near_flips),
            "same_lane_corrected_heading": heading_summary(same_lane),
            "oncoming_lane_corrected_heading": heading_summary(oncoming),
        },
        "corrected_dimension_quantiles_m": {
            name: {axis: percentile(values)
                   for axis, values in axes.items()}
            for name, axes in dimensions.items()
        },
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
