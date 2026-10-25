#!/usr/bin/env python3
"""Plan yaw-only fixes using motion first and CenterPoint longitudinal axes."""

import json
import math
import os
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import requests
from scipy.optimize import linear_sum_assignment


VEHICLES = {"car", "truck", "bus"}


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def axis_difference(first, second):
    return abs(wrap(2 * (first - second))) / 2


def box(annotation):
    data = annotation["data"]
    return np.array([
        data["center"]["x"], data["center"]["y"], data["center"]["z"],
        data["dimensions"]["length"], data["dimensions"]["width"],
        data["dimensions"]["height"], data["rotation"]["yaw"],
    ], dtype=float)


def main():
    task = os.environ["CALIPER_TASK_ID"]
    taxonomy = os.environ["CALIPER_TAXONOMY_ID"]
    api = os.getenv("CALIPER_API", "http://localhost:3000/api/v1")
    output = Path(os.environ["CALIPER_AXIS_PLAN"])
    centerpoint = json.loads(Path(os.environ["CALIPER_CENTERPOINT_OUTPUT"]).read_text())["detections"]
    motion_path = Path(os.environ["CALIPER_ODOMETRY_VALIDATION"])
    motion = {
        row["annotation_id"]: row for row in json.loads(motion_path.read_text())["records"]
        if row.get("accepted")
    }

    session = requests.Session()
    if os.getenv("CALIPER_TOKEN"):
        session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    response = session.get(
        f"{api}/tasks/{task}/frames?include_context=false", timeout=180
    )
    response.raise_for_status()
    frame_index = {row["id"]: int(row["frame_index"])
                   for row in response.json()["frames"]}
    if os.getenv("CALIPER_ANNOTATION_SNAPSHOT"):
        snapshot = json.loads(Path(os.environ["CALIPER_ANNOTATION_SNAPSHOT"]).read_text())
        annotations = snapshot.get("annotations", snapshot)
        # Reconstruct the verified live state from the pre-odometry backup.
        for annotation in annotations:
            record = motion.get(annotation["id"])
            if record is not None:
                annotation["data"]["rotation"]["yaw"] = record["new_yaw"]
    else:
        response = session.get(
            f"{api}/annotations-3d?task_id={task}&taxonomy_id={taxonomy}", timeout=180
        )
        response.raise_for_status()
        annotations = response.json()
    by_frame = defaultdict(list)
    for annotation in annotations:
        if annotation["class_id"] in VEHICLES:
            by_frame[frame_index[annotation["frame_id"]]].append(annotation)

    matches = {}
    counters = Counter()
    for frame, rows in by_frame.items():
        detections = [item for item in centerpoint.get(str(frame), [])
                      if item["class_id"] in VEHICLES and item["score"] >= .25]
        if not detections:
            continue
        costs = np.empty((len(rows), len(detections)), dtype=float)
        for i, annotation in enumerate(rows):
            center = box(annotation)[:2]
            for j, detection in enumerate(detections):
                costs[i, j] = np.linalg.norm(center - np.asarray(detection["box"][:2]))
        assigned_rows, assigned_columns = linear_sum_assignment(costs)
        for i, j in zip(assigned_rows, assigned_columns):
            distance = float(costs[i, j])
            detection = detections[j]
            if distance <= 1.5 and detection["score"] >= .4:
                matches[rows[i]["id"]] = (detection, distance)
                counters["centerpoint_matches"] += 1

    candidates = []
    for frame, rows in by_frame.items():
        for annotation in rows:
            current = box(annotation)
            source = None
            proposed = None
            evidence = {}
            if annotation["id"] in motion:
                record = motion[annotation["id"]]
                proposed = float(record["new_yaw"])
                source = "bidirectional_odometry"
                evidence = {
                    "speed_mps": record["speed_mps"],
                    "fit_residual_m": record["fit_residual_m"],
                    "motion_window": record["window"],
                }
            elif annotation["id"] in matches:
                detection, distance = matches[annotation["id"]]
                proposed = float(detection["box"][6])
                source = "centerpoint_longitudinal_axis"
                evidence = {"score": detection["score"], "distance_m": distance}
            else:
                counters["no_independent_axis_evidence"] += 1
                continue
            difference = math.degrees(axis_difference(proposed, current[6]))
            if difference < 10.0:
                counters[f"already_agrees_{source}"] += 1
                continue
            candidates.append({
                "annotation_id": annotation["id"], "frame_index": frame,
                "frame_id": annotation["frame_id"], "track_id": annotation.get("track_id"),
                "class_id": annotation["class_id"], "box": current.tolist(),
                "old_yaw": float(current[6]), "new_yaw": proposed,
                "axis_change_deg": difference, "source": source, **evidence,
            })
            counters[f"candidate_{source}"] += 1

    result = {
        "task_id": task, "taxonomy_id": taxonomy,
        "annotation_count": len(annotations), "vehicle_count": sum(map(len, by_frame.values())),
        "counts": dict(counters), "candidates": candidates,
    }
    output.write_text(json.dumps(result, indent=2))
    print(json.dumps({key: value for key, value in result.items() if key != "candidates"}, indent=2))
    print(json.dumps({"candidates": len(candidates)}, indent=2))


if __name__ == "__main__":
    main()
