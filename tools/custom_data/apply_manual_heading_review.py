#!/usr/bin/env python3
"""Validate/apply headings manually adjudicated from fisheye crops."""

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
REVIEW = Path("/home/mostafa/bag-2026-09-16-heading-disagreement-crops.json")
OUTPUT = Path("/home/mostafa/bag-2026-09-16-manual-heading-corrections.json")
BACKUP = Path("/home/mostafa/task-b6da9787-before-manual-heading-corrections.json")

# Visible vehicle fronts/rears were inspected in the numbered crop sheet.
# CenterPoint is used only as the yaw source after that human decision.
USE_CENTERPOINT = {2, 4, 6, 7, 8, 9, 11, 12, 14, 15, 17, 18}


def annotation_box(annotation):
    data = annotation["data"]
    return np.array([data["center"]["x"], data["center"]["y"],
                     data["center"]["z"], data["dimensions"]["length"],
                     data["dimensions"]["width"], data["dimensions"]["height"],
                     data["rotation"]["yaw"]], float)


def contains(box, proposal, sample):
    checks = []
    for detection in proposal["detections"]:
        projected = camera_pass.projected_bbox(
            box, sample["cam_info"]["cam_fisheye"][detection["camera"]])
        target = np.asarray(detection["xyxy"], float)
        checks.append(bool(projected is not None and
                           projected[0] <= target[0] + 5 and
                           projected[1] <= target[1] + 5 and
                           projected[2] >= target[2] - 5 and
                           projected[3] >= target[3] - 5))
    return checks


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    response = session.get(
        API + f"/annotations-3d?task_id={TASK_ID}&taxonomy_id={TAXONOMY_ID}",
        timeout=180)
    response.raise_for_status()
    annotations_list = response.json()
    annotations = {x["id"]: x for x in annotations_list}
    review = {x["review_number"]: x for x in json.loads(REVIEW.read_text())}
    proposals = {annotation_id(x): x
                 for x in json.loads(PLAN.read_text())["proposals"]}
    samples = camera_pass.metadata_samples()
    metadata = {int(x["sample_idx"]): x
                for x in mmengine.load(INFO)["data_list"]}

    records, updates = [], []
    for number in sorted(USE_CENTERPOINT):
        row = review[number]
        annotation = annotations[row["annotation_id"]]
        old = annotation_box(annotation)
        new = old.copy()
        axis_change = min(row["centerpoint_difference_deg"],
                          180.0 - row["centerpoint_difference_deg"])
        # A 180-degree correction changes only front/back semantics. Preserve
        # the already camera/LiDAR-fitted cuboid axis exactly in those cases.
        if row["centerpoint_difference_deg"] > 90 and axis_change <= 15:
            # Anchor the flip to the reviewed pre-correction yaw so rerunning
            # this tool is idempotent instead of flipping the object again.
            new[6] = (row["current_yaw"] + math.pi) % (2 * math.pi) - math.pi
            source = "manual_front_back_flip"
        else:
            new[6] = row["centerpoint_yaw"]
            source = "manual_centerpoint_axis"
        frame = row["frame_index"]
        path = (BASE / metadata[frame]["pc_info"]["pts_path"]).resolve()
        points = np.fromfile(path, np.float32).reshape(-1, 4)[:, :3]
        old_support, old_nearest = box_support(points, old)
        new_support, new_nearest = box_support(points, new)
        old_camera = contains(old, proposals[annotation["id"]], samples[frame])
        new_camera = contains(new, proposals[annotation["id"]], samples[frame])
        valid = new_nearest <= .5 and sum(new_camera) >= sum(old_camera)
        records.append({
            "review_number": number, "frame_index": frame,
            "annotation_id": annotation["id"], "class_id": annotation["class_id"],
            "old_yaw": float(old[6]), "new_yaw": float(new[6]),
            "source": source,
            "yaw_change_deg": row["centerpoint_difference_deg"],
            "centerpoint_score": row["centerpoint_score"],
            "old_support": old_support, "new_support": new_support,
            "old_nearest_m": old_nearest, "new_nearest_m": new_nearest,
            "old_camera_containment": old_camera,
            "new_camera_containment": new_camera, "valid": valid,
        })
        if valid:
            data = dict(annotation["data"])
            data["rotation"] = {"yaw": float(new[6]), "pitch": 0.0, "roll": 0.0}
            updates.append({"id": annotation["id"],
                            "class_id": annotation["class_id"], "data": data})

    OUTPUT.write_text(json.dumps({"applied": bool(args.apply),
                                  "records": records, "updates": updates}, indent=2))
    summary = {"reviewed": len(records), "valid": len(updates),
               "rejected": len(records) - len(updates)}
    print(json.dumps(summary, indent=2))
    if not args.apply:
        return
    BACKUP.write_text(json.dumps({"task_id": TASK_ID,
                                  "annotations": annotations_list}, indent=2))
    response = session.post(API + "/annotations-3d/bulk-update",
                            json={"annotations": updates}, timeout=180)
    response.raise_for_status()
    print(f"updated {len(updates)}")


if __name__ == "__main__":
    main()
