#!/usr/bin/env python3
"""Plan vertical cleanup from manually corrected reference frames."""

import argparse
import json
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cutoff", type=int, default=91,
                        help="first unreviewed zero-based frame index")
    args = parser.parse_args()
    task = os.environ["CALIPER_TASK_ID"]
    taxonomy = os.environ["CALIPER_TAXONOMY_ID"]
    api = os.getenv("CALIPER_API", "http://localhost:3000/api/v1")
    output = Path(os.environ["CALIPER_VERTICAL_PLAN"])
    backup = Path(os.environ["CALIPER_VERTICAL_BACKUP"])
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    response = session.get(f"{api}/tasks/{task}/frames?include_context=false", timeout=180)
    response.raise_for_status()
    frames = response.json()["frames"]
    frame_index = {row["id"]: int(row["frame_index"]) for row in frames}
    response = session.get(
        f"{api}/annotations-3d?task_id={task}&taxonomy_id={taxonomy}", timeout=180
    )
    response.raise_for_status()
    annotations = response.json()
    backup.write_text(json.dumps({"task_id": task, "annotations": annotations}, indent=2))
    by_frame = defaultdict(list)
    for annotation in annotations:
        by_frame[frame_index[annotation["frame_id"]]].append(annotation)
    samples = camera_pass.metadata_samples()
    reference = defaultdict(list)
    measurements = []
    for number, frame in enumerate(sorted(by_frame)):
        sample = samples[frame]
        points = camera_pass.load_lidar_points(sample)
        plane = camera_pass.fit_ground_plane(sample, points)
        for annotation in by_frame[frame]:
            data = annotation["data"]
            x, y, z = (float(data["center"][axis]) for axis in ("x", "y", "z"))
            height = float(data["dimensions"]["height"])
            ground = float(plane[0] * x + plane[1] * y + plane[2])
            clearance = z - height / 2 - ground
            row = {
                "annotation_id": annotation["id"], "frame_index": frame,
                "class_id": annotation["class_id"], "center_z": z,
                "height": height, "ground_z": ground,
                "bottom_clearance_m": clearance,
            }
            measurements.append(row)
            if frame < args.cutoff and clearance <= 1.0 and -2.0 <= z <= -.5:
                reference[annotation["class_id"]].append(clearance)
        if (number + 1) % 100 == 0:
            print(f"measured {number + 1}/{len(by_frame)} populated frames", flush=True)

    learned = {
        class_id: float(np.median(values)) for class_id, values in reference.items()
        if values
    }
    reference_all = [value for values in reference.values() for value in values]
    fallback = float(np.median(reference_all)) if reference_all else 0.0
    actions = []
    counts = Counter()
    for row in measurements:
        if row["frame_index"] < args.cutoff:
            counts["reference_preserved"] += 1
            continue
        if row["bottom_clearance_m"] > 1.0:
            actions.append({**row, "action": "delete",
                            "reason": "bottom_more_than_1m_above_ground"})
            counts[f"delete_{row['class_id']}"] += 1
            continue
        clearance = learned.get(row["class_id"], fallback)
        new_z = row["ground_z"] + row["height"] / 2 + clearance
        new_z = min(-.5, max(-2.0, new_z))
        if abs(new_z - row["center_z"]) < .05:
            counts["already_aligned"] += 1
            continue
        actions.append({**row, "action": "update", "new_center_z": new_z,
                        "learned_clearance_m": clearance,
                        "shift_m": new_z - row["center_z"]})
        counts[f"update_{row['class_id']}"] += 1
    def stats(selection):
        values = [row for row in measurements if selection(row)]
        if not values:
            return {
                "count": 0, "center_z_percentiles": [],
                "clearance_percentiles": [], "clearance_gt_1m": 0,
                "center_outside_minus2_minus0_5": 0,
            }
        clearances = np.asarray([row["bottom_clearance_m"] for row in values], float)
        centers = np.asarray([row["center_z"] for row in values], float)
        return {
            "count": len(values),
            "center_z_percentiles": np.percentile(centers, [0, 5, 50, 95, 100]).tolist(),
            "clearance_percentiles": np.percentile(clearances, [0, 5, 50, 95, 100]).tolist(),
            "clearance_gt_1m": int(np.count_nonzero(clearances > 1.0)),
            "center_outside_minus2_minus0_5": int(np.count_nonzero(
                (centers < -2.0) | (centers > -.5))),
        }
    result = {
        "task_id": task, "cutoff_frame_index": args.cutoff,
        "learned_clearance_by_class": learned, "fallback_clearance": fallback,
        "reference_stats": stats(lambda row: row["frame_index"] < args.cutoff),
        "target_stats": stats(lambda row: row["frame_index"] >= args.cutoff),
        "counts": dict(counts), "actions": actions,
    }
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "actions"}, indent=2))
    print(json.dumps({"actions": len(actions), "output": str(output)}, indent=2))


if __name__ == "__main__":
    main()
