#!/usr/bin/env python3
"""Apply manually reviewed position, class, and removal corrections."""

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
POSITION_PLAN = Path("/home/mostafa/bag-2026-09-16-full-position-refit-plan.json")
REVIEW_INDEX = Path("/home/mostafa/bag-2026-09-16-removal-class-review/index.json")
CAMERA_PLAN = Path("/home/mostafa/bag-2026-09-16-camera-lidar-plan.json")
OUTPUT = Path("/home/mostafa/bag-2026-09-16-full-manual-cleanup.json")
BACKUP = Path("/home/mostafa/task-b6da9787-before-full-manual-cleanup.json")
CLASS_TO_CAR = {30, 43, 52, 532, 542, 544}
REMOVE = {55, 56, 252, 405, 406, 438}


def annotation_box(annotation):
    d = annotation["data"]
    return np.array([d["center"]["x"], d["center"]["y"], d["center"]["z"],
                     d["dimensions"]["length"], d["dimensions"]["width"],
                     d["dimensions"]["height"], d["rotation"]["yaw"]], float)


def containment(box, proposal, sample):
    checks, ious = [], []
    for detection in proposal["detections"]:
        target = np.asarray(detection["xyxy"], float)
        projected = camera_pass.projected_bbox(
            box, sample["cam_info"]["cam_fisheye"][detection["camera"]])
        checks.append(bool(projected is not None
                           and projected[0] <= target[0] + 5
                           and projected[1] <= target[1] + 5
                           and projected[2] >= target[2] - 5
                           and projected[3] >= target[3] - 5))
        ious.append(camera_pass.bbox_iou(projected, target)
                    if projected is not None else 0.0)
    return checks, ious


def update(annotation, new_box, class_id=None):
    data = dict(annotation["data"])
    data.update({
        "center": {"x": float(new_box[0]), "y": float(new_box[1]),
                   "z": float(new_box[2])},
        "dimensions": {"length": float(new_box[3]), "width": float(new_box[4]),
                       "height": float(new_box[5])},
        "rotation": {"yaw": float(new_box[6]), "pitch": 0.0, "roll": 0.0},
    })
    return {"id": annotation["id"],
            "class_id": class_id or annotation["class_id"], "data": data}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    a = session.get(API + f"/annotations-3d?task_id={TASK_ID}&taxonomy_id={TAXONOMY_ID}", timeout=180)
    a.raise_for_status()
    f = session.get(API + f"/tasks/{TASK_ID}/frames?include_context=false", timeout=180)
    f.raise_for_status()
    annotations_list = a.json()
    annotations = {x["id"]: x for x in annotations_list}
    frame_index = {x["id"]: int(x["frame_index"]) for x in f.json()["frames"]}
    proposals = {annotation_id(x): x
                 for x in json.loads(PLAN.read_text())["proposals"]}
    samples = camera_pass.metadata_samples()
    metadata = {int(x["sample_idx"]): x
                for x in mmengine.load(INFO)["data_list"]}
    planes = json.loads(CAMERA_PLAN.read_text())["planes"]
    records = {x["annotation_id"]: x
               for x in json.loads(INDEX.read_text())["records"]}
    review = {x["review_number"]: x
              for x in json.loads(REVIEW_INDEX.read_text())["representatives"]}

    remove_tracks = {review[x]["track_id"] for x in REMOVE}
    delete_ids = [x["id"] for x in annotations_list
                  if x.get("track_id") in remove_tracks]
    class_tracks = {review[x]["track_id"] for x in CLASS_TO_CAR}
    updates = {}
    position_records = []
    for row in json.loads(POSITION_PLAN.read_text())["candidates"]:
        if row["annotation_id"] not in annotations or row["track_id"] in remove_tracks:
            continue
        updates[row["annotation_id"]] = update(
            annotations[row["annotation_id"]], np.asarray(row["new_box"]))
        position_records.append({"annotation_id": row["annotation_id"],
                                 "frame_index": row["frame_index"],
                                 "shift_m": row["shift_m"]})

    class_records = []
    default = np.array(camera_pass.DEFAULT_DIMS["car"], float)
    maximum = np.array(camera_pass.MAX_DIMS["car"], float)
    for annotation in annotations_list:
        if annotation.get("track_id") not in class_tracks:
            continue
        aid = annotation["id"]
        row, proposal = records[aid], proposals[aid]
        frame = row["frame_index"]
        old = annotation_box(annotation)
        plane = np.asarray(planes[str(frame)], float)
        point_path = (BASE / metadata[frame]["pc_info"]["pts_path"]).resolve()
        points = np.fromfile(point_path, np.float32).reshape(-1, 4)[:, :3]
        group = [{**x, "class_id": "car"} for x in proposal["detections"]]
        seed = camera_pass.initial_box(group, plane)
        centers = [old[:2], seed[:2]]
        if "centerpoint_box" in row:
            centers.append(np.asarray(row["centerpoint_box"][:2]))
        if "parta2_center" in row:
            centers.append(np.asarray(row["parta2_center"]))
        candidates = []
        for center in centers:
            for scale in np.arange(1.0, 1.41, .1):
                candidate = old.copy()
                candidate[:2] = center
                candidate[3:6] = np.minimum(default * scale, maximum)
                candidate[2] = (plane[0] * center[0] + plane[1] * center[1]
                                + plane[2] + candidate[5] / 2)
                support, nearest = box_support(points, candidate)
                checks, ious = containment(candidate, proposal, samples[frame])
                if nearest <= .5 and all(checks):
                    score = sum(ious) - .08 * scale + .0002 * min(support, 500)
                    candidates.append((score, candidate, support, nearest, checks))
        if candidates:
            _, new_box, support, nearest, checks = max(candidates, key=lambda x: x[0])
            method = "car_geometry_refit"
        else:
            new_box = old
            support, nearest = box_support(points, old)
            checks, _ = containment(old, proposal, samples[frame])
            method = "class_only_geometry_retained"
        updates[aid] = update(annotation, new_box, "car")
        class_records.append({"annotation_id": aid, "frame_index": frame,
                              "old_class": annotation["class_id"], "new_class": "car",
                              "method": method,
                              "center_shift_m": float(np.linalg.norm(new_box[:2] - old[:2])),
                              "support": support, "nearest_m": nearest,
                              "camera_containment": checks})

    document = {"applied": bool(args.apply),
                "summary": {"position_updates": len(position_records),
                            "class_updates": len(class_records),
                            "deletions": len(delete_ids)},
                "position_records": position_records,
                "class_records": class_records,
                "delete_ids": delete_ids, "updates": list(updates.values())}
    OUTPUT.write_text(json.dumps(document, indent=2))
    print(json.dumps(document["summary"], indent=2))
    if not args.apply:
        return
    BACKUP.write_text(json.dumps({"task_id": TASK_ID,
                                  "annotations": annotations_list}, indent=2))
    values = list(updates.values())
    for start in range(0, len(values), 200):
        response = session.post(API + "/annotations-3d/bulk-update",
                                json={"annotations": values[start:start + 200]}, timeout=180)
        response.raise_for_status()
        print(f"updated {min(start + 200, len(values))}/{len(values)}", flush=True)
    for start in range(0, len(delete_ids), 200):
        response = session.post(API + "/annotations-3d/bulk-delete",
                                json={"annotation_ids": delete_ids[start:start + 200]}, timeout=180)
        response.raise_for_status()
        print(f"deleted {min(start + 200, len(delete_ids))}/{len(delete_ids)}", flush=True)


if __name__ == "__main__":
    main()
