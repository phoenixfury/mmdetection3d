#!/usr/bin/env python3
"""Build a complete per-annotation review index without changing the task."""

import csv
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import mmengine
import numpy as np
import requests
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass
sys.path.insert(0, "/home/mostafa/mmdetection3d/tools/custom_data")
from pgd_fisheye_heading_pilot import API, PLAN, TASK_ID, TAXONOMY_ID, annotation_id
from refit_task_to_lidar import box_support


BASE = Path("/home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_16_09_58_56_converted/fisheye3dod_inference_2hz")
INFO = BASE / "fisheye3dod_infos_test.pkl"
CENTERPOINT = Path("/home/mostafa/bag-2026-09-16-centerpoint-full.json")
PARTA2 = Path("/home/mostafa/bag-2026-09-16-parta2-synced/detections.csv")
OUTPUT = Path("/home/mostafa/bag-2026-09-16-full-manual-review.json")
SNAPSHOT = Path("/home/mostafa/task-b6da9787-full-review-snapshot.json")
VEHICLES = {"car", "truck", "bus"}


def box(annotation):
    d = annotation["data"]
    return np.array([d["center"]["x"], d["center"]["y"], d["center"]["z"],
                     d["dimensions"]["length"], d["dimensions"]["width"],
                     d["dimensions"]["height"], d["rotation"]["yaw"]], float)


def load_parta2():
    output = defaultdict(list)
    with PARTA2.open(newline="") as stream:
        for row in csv.DictReader(stream):
            output[int(row["frame_id"])].append({
                "x": float(row["x"]), "y": float(row["y"]),
                "yaw": float(row["yaw"]), "score": float(row["score"]),
            })
    return output


def assign(current, detections, max_distance=3.0):
    if not current or not detections:
        return {}
    cost = np.empty((len(current), len(detections)), float)
    for i, (_, b) in enumerate(current):
        for j, d in enumerate(detections):
            cost[i, j] = math.hypot(b[0] - d["box"][0], b[1] - d["box"][1])
    rows, columns = linear_sum_assignment(cost)
    return {current[i][0]: (detections[j], float(cost[i, j]))
            for i, j in zip(rows, columns) if cost[i, j] <= max_distance}


def main():
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    a = session.get(API + f"/annotations-3d?task_id={TASK_ID}&taxonomy_id={TAXONOMY_ID}", timeout=180)
    a.raise_for_status()
    f = session.get(API + f"/tasks/{TASK_ID}/frames?include_context=false", timeout=180)
    f.raise_for_status()
    annotations = a.json()
    SNAPSHOT.write_text(json.dumps({"task_id": TASK_ID, "annotations": annotations}, indent=2))
    frame_index = {x["id"]: int(x["frame_index"]) for x in f.json()["frames"]}
    proposals = {annotation_id(x): x for x in json.loads(PLAN.read_text())["proposals"]}
    samples = camera_pass.metadata_samples()
    metadata = {int(x["sample_idx"]): x for x in mmengine.load(INFO)["data_list"]}
    cp = {int(k): v for k, v in json.loads(CENTERPOINT.read_text())["detections"].items()}
    parta2_raw = load_parta2()
    by_frame = defaultdict(list)
    for annotation in annotations:
        by_frame[frame_index[annotation["frame_id"]]].append(annotation)

    records = []
    for frame, frame_annotations in sorted(by_frame.items()):
        current_vehicles = [(x["id"], box(x)) for x in frame_annotations
                            if x["class_id"] in VEHICLES]
        cp_matches = assign(current_vehicles, cp.get(frame, []))
        p2 = [{"box": [x["x"], x["y"]], **x} for x in parta2_raw.get(frame, [])]
        p2_matches = assign(current_vehicles, p2)
        point_path = (BASE / metadata[frame]["pc_info"]["pts_path"]).resolve()
        points = np.fromfile(point_path, np.float32).reshape(-1, 4)[:, :3]
        for annotation in frame_annotations:
            aid, b = annotation["id"], box(annotation)
            proposal = proposals.get(aid)
            if proposal is None:
                continue
            support, nearest = box_support(points, b)
            containment, ious = [], []
            sample = samples[frame]
            for detection in proposal["detections"]:
                target = np.asarray(detection["xyxy"], float)
                projected = camera_pass.projected_bbox(
                    b, sample["cam_info"]["cam_fisheye"][detection["camera"]])
                ok = bool(projected is not None and projected[0] <= target[0] + 5
                          and projected[1] <= target[1] + 5
                          and projected[2] >= target[2] - 5
                          and projected[3] >= target[3] - 5)
                containment.append(ok)
                ious.append(float(camera_pass.bbox_iou(projected, target))
                            if projected is not None else 0.0)
            cp_match = cp_matches.get(aid)
            p2_match = p2_matches.get(aid)
            record = {
                "annotation_id": aid, "frame_index": frame,
                "track_id": annotation.get("track_id"),
                "class_id": annotation["class_id"], "box": b.tolist(),
                "support": support, "nearest_m": nearest,
                "camera_containment": containment, "camera_ious": ious,
                "yolo_confidence": max(x["confidence"] for x in proposal["detections"]),
                "camera_count": len({x["camera"] for x in proposal["detections"]}),
                "image_track_length": max(x["track_length"] for x in proposal["detections"]),
                "candidate_component": proposal["candidate_component"],
            }
            if cp_match:
                d, distance = cp_match
                record.update(centerpoint_box=d["box"], centerpoint_score=d["score"],
                              centerpoint_class=d["class_id"], centerpoint_distance_m=distance)
            if p2_match:
                d, distance = p2_match
                record.update(parta2_yaw=d["yaw"], parta2_score=d["score"],
                              parta2_distance_m=distance,
                              parta2_center=[d["x"], d["y"]])
            records.append(record)
        if frame % 100 == 0:
            print(f"indexed frame {frame}", flush=True)

    tracks = defaultdict(list)
    for record in records:
        tracks[record["track_id"]].append(record)
    summary = {
        "annotations": len(records), "tracks": len(tracks),
        "classes": dict(Counter(x["class_id"] for x in records)),
        "camera_containment_failures": sum(not all(x["camera_containment"]) for x in records),
        "centerpoint_matches": sum("centerpoint_box" in x for x in records),
        "parta2_matches": sum("parta2_center" in x for x in records),
        "short_tracks": sum(len(x) <= 2 for x in tracks.values()),
    }
    OUTPUT.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
