#!/usr/bin/env python3
"""Refit the remaining camera-containment failures without losing LiDAR support."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import mmengine
import numpy as np
import requests

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass
sys.path.insert(0, "/home/mostafa/mmdetection3d/tools/custom_data")
from pgd_fisheye_heading_pilot import API, PLAN, TASK_ID, TAXONOMY_ID, annotation_id
from refit_task_to_lidar import box_support


BASE = Path("/home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_16_09_58_56_converted/fisheye3dod_inference_2hz")
INFO = BASE / "fisheye3dod_infos_test.pkl"
INDEX = Path("/home/mostafa/bag-2026-09-16-full-manual-review.json")
CAMERA_PLAN = Path("/home/mostafa/bag-2026-09-16-camera-lidar-plan.json")
OUTPUT = Path("/home/mostafa/bag-2026-09-16-containment-fixes.json")
BACKUP = Path("/home/mostafa/task-b6da9787-before-containment-fixes.json")


def box(annotation):
    d = annotation["data"]
    return np.array([d["center"]["x"], d["center"]["y"], d["center"]["z"],
                     d["dimensions"]["length"], d["dimensions"]["width"],
                     d["dimensions"]["height"], d["rotation"]["yaw"]], float)


def evaluate(candidate, proposal, sample):
    checks, ious = [], []
    for detection in proposal["detections"]:
        target = np.asarray(detection["xyxy"], float)
        projected = camera_pass.projected_bbox(
            candidate, sample["cam_info"]["cam_fisheye"][detection["camera"]])
        checks.append(bool(projected is not None
                           and projected[0] <= target[0] + 5
                           and projected[1] <= target[1] + 5
                           and projected[2] >= target[2] - 5
                           and projected[3] >= target[3] - 5))
        ious.append(camera_pass.bbox_iou(projected, target)
                    if projected is not None else 0.0)
    return checks, ious


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    index = json.loads(INDEX.read_text())["records"]
    failures = [x for x in index if not all(x["camera_containment"])]
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    response = session.get(
        API + f"/annotations-3d?task_id={TASK_ID}&taxonomy_id={TAXONOMY_ID}", timeout=180)
    response.raise_for_status()
    annotations_list = response.json()
    annotations = {x["id"]: x for x in annotations_list}
    proposals = {annotation_id(x): x
                 for x in json.loads(PLAN.read_text())["proposals"]}
    samples = camera_pass.metadata_samples()
    metadata = {int(x["sample_idx"]): x
                for x in mmengine.load(INFO)["data_list"]}
    planes = json.loads(CAMERA_PLAN.read_text())["planes"]
    updates, records = [], []
    for row in failures:
        annotation = annotations[row["annotation_id"]]
        old = box(annotation)
        frame = row["frame_index"]
        proposal = proposals[row["annotation_id"]]
        plane = np.asarray(planes[str(frame)], float)
        group = [{**x, "class_id": annotation["class_id"]}
                 for x in proposal["detections"]]
        seed = camera_pass.initial_box(group, plane)
        path = (BASE / metadata[frame]["pc_info"]["pts_path"]).resolve()
        points = np.fromfile(path, np.float32).reshape(-1, 4)[:, :3]
        maximum = np.asarray(camera_pass.MAX_DIMS[annotation["class_id"]], float)
        candidates = []
        for fraction in np.linspace(0, 1, 9):
            center = old[:2] + fraction * (seed[:2] - old[:2])
            for scale in np.arange(1.0, 1.81, .1):
                candidate = old.copy()
                candidate[:2] = center
                candidate[3:6] = np.minimum(old[3:6] * scale, maximum)
                candidate[2] = (plane[0] * center[0] + plane[1] * center[1]
                                + plane[2] + candidate[5] / 2)
                support, nearest = box_support(points, candidate)
                checks, ious = evaluate(candidate, proposal, samples[frame])
                if nearest <= .5 and all(checks):
                    change = np.linalg.norm(candidate[:2] - old[:2]) + .5 * (scale - 1)
                    candidates.append((sum(ious) - .03 * change, candidate,
                                       support, nearest, checks))
        # Very close seam-crossing objects may need a lateral adjustment that
        # is not on the camera-contact ray. Search a bounded XY neighborhood
        # at maximum class dimensions before declaring them unresolved.
        if not candidates:
            for dx in np.arange(-3.0, 3.01, .25):
                for dy in np.arange(-3.0, 3.01, .25):
                    candidate = old.copy()
                    candidate[:2] += [dx, dy]
                    candidate[3:6] = maximum
                    candidate[2] = (plane[0] * candidate[0] + plane[1] * candidate[1]
                                    + plane[2] + candidate[5] / 2)
                    support, nearest = box_support(points, candidate)
                    checks, ious = evaluate(candidate, proposal, samples[frame])
                    if nearest <= .5 and all(checks):
                        change = math.hypot(dx, dy) + .5
                        candidates.append((sum(ious) - .03 * change, candidate,
                                           support, nearest, checks))
        if not candidates:
            records.append({"annotation_id": annotation["id"],
                            "frame_index": frame, "fixed": False})
            continue
        _, new, support, nearest, checks = max(candidates, key=lambda x: x[0])
        data = dict(annotation["data"])
        data.update({"center": {"x": float(new[0]), "y": float(new[1]),
                                "z": float(new[2])},
                     "dimensions": {"length": float(new[3]), "width": float(new[4]),
                                    "height": float(new[5])},
                     "rotation": {"yaw": float(new[6]), "pitch": 0.0, "roll": 0.0}})
        updates.append({"id": annotation["id"], "class_id": annotation["class_id"],
                        "data": data})
        records.append({"annotation_id": annotation["id"], "frame_index": frame,
                        "fixed": True,
                        "center_shift_m": float(np.linalg.norm(new[:2] - old[:2])),
                        "dimensions": new[3:6].tolist(), "support": support,
                        "nearest_m": nearest, "camera_containment": checks})
    OUTPUT.write_text(json.dumps({"applied": bool(args.apply), "records": records,
                                  "updates": updates}, indent=2))
    print(json.dumps({"failures": len(failures), "fixed": len(updates),
                      "unresolved": len(failures) - len(updates)}, indent=2))
    if not args.apply or not updates:
        return
    BACKUP.write_text(json.dumps({"task_id": TASK_ID,
                                  "annotations": annotations_list}, indent=2))
    response = session.post(API + "/annotations-3d/bulk-update",
                            json={"annotations": updates}, timeout=180)
    response.raise_for_status()
    print(f"updated {len(updates)}")


if __name__ == "__main__":
    main()
