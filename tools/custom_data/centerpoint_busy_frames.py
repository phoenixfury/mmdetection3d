#!/usr/bin/env python3
"""Run official nuScenes CenterPoint on the ten PGD review frames (read-only)."""

import json
import os
from pathlib import Path

import mmengine
import numpy as np
import requests
from mmengine.config import Config
from scipy.optimize import linear_sum_assignment

from mmdet3d.apis import inference_detector, init_model


ROOT = Path("/home/mostafa/mmdetection3d")
BASE = ROOT / "data/Custom/BAG_2026_09_16_09_58_56_converted/fisheye3dod_inference_2hz"
INFO = BASE / "fisheye3dod_infos_test.pkl"
CONFIG = ROOT / "configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
CHECKPOINT = ROOT / "checkpoints/centerpoint_nus_voxel01.pth"
PGD = Path("/home/mostafa/bag-2026-09-16-pgd-busy-frame-preview.json")
OUTPUT = Path("/home/mostafa/bag-2026-09-16-centerpoint-busy-frames.json")
VEHICLES = {0: "car", 1: "truck", 3: "bus"}
API = "http://localhost:3000/api/v1"
TASK_ID = "b6da9787-6615-4a13-8747-5466ff1a6aed"
TAXONOMY_ID = "981f59a2-75c6-41f1-a204-8d177ada63ce"


def angle_difference(a, b):
    return abs((a - b + np.pi) % (2 * np.pi) - np.pi)


def main():
    frames = json.loads(PGD.read_text())["selected_frames"]
    samples = {int(x["sample_idx"]): x for x in mmengine.load(INFO)["data_list"]}

    # CenterPoint expects x/y/z/intensity/time. This sensor has x/y/z plus an
    # empty reflectance channel, so provide a zero time channel explicitly.
    clouds = []
    for frame in frames:
        path = (BASE / samples[frame]["pc_info"]["pts_path"]).resolve()
        points = np.fromfile(path, np.float32).reshape(-1, 4)
        clouds.append(np.column_stack((points, np.zeros(len(points), np.float32))))

    cfg = Config.fromfile(CONFIG)
    # Use the keyframe only: repeating it as nine fake nuScenes sweeps adds no
    # information and distorts point density.
    pipeline = cfg.test_dataloader.dataset.pipeline
    cfg.test_dataloader.dataset.pipeline = [
        step for step in pipeline if step["type"] != "LoadPointsFromMultiSweeps"
    ]
    model = init_model(cfg, str(CHECKPOINT), device="cuda:0")

    detections = {}
    for start in range(0, len(clouds), 4):
        batch = clouds[start:start + 4]
        results, _ = inference_detector(model, batch)
        for frame, result in zip(frames[start:start + 4], results):
            pred = result.pred_instances_3d
            boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
            scores = pred.scores_3d.detach().cpu().numpy()
            labels = pred.labels_3d.detach().cpu().numpy()
            detections[frame] = [
                {"class_id": VEHICLES[int(label)], "score": float(score),
                 "box": box[:7].astype(float).tolist()}
                for box, score, label in zip(boxes, scores, labels)
                if int(label) in VEHICLES and score >= 0.1
            ]
        print(f"processed {min(start + len(batch), len(clouds))}/{len(clouds)}", flush=True)

    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    annotations_response = session.get(
        API + f"/annotations-3d?task_id={TASK_ID}&taxonomy_id={TAXONOMY_ID}",
        timeout=180)
    annotations_response.raise_for_status()
    frames_response = session.get(
        API + f"/tasks/{TASK_ID}/frames?include_context=false", timeout=180)
    frames_response.raise_for_status()
    frame_index = {x["id"]: int(x["frame_index"])
                   for x in frames_response.json()["frames"]}
    annotations = {x["id"]: x for x in annotations_response.json()
                   if frame_index[x["frame_id"]] in frames}

    # PGD rows identify the same current annotations. Associate CenterPoint
    # only for comparison; this script never updates the task.
    pgd_records = json.loads(PGD.read_text())["records"]
    current = {}
    for row in pgd_records:
        current.setdefault(int(row["frame_index"]), []).append(row)
    comparisons = []
    for frame in frames:
        rows = [r for r in current.get(frame, []) if r["annotation_id"] in annotations]
        preds = detections.get(frame, [])
        if not rows or not preds:
            continue
        boxes = []
        for row in rows:
            d = annotations[row["annotation_id"]]["data"]
            boxes.append([d["center"]["x"], d["center"]["y"], row["current_yaw"]])
        cost = np.full((len(rows), len(preds)), 1e3, dtype=float)
        for i, (x, y, _) in enumerate(boxes):
            for j, pred in enumerate(preds):
                if rows[i]["class_id"] == pred["class_id"] or {rows[i]["class_id"], pred["class_id"]} <= {"car", "truck", "bus"}:
                    cost[i, j] = np.hypot(x - pred["box"][0], y - pred["box"][1])
        ii, jj = linear_sum_assignment(cost)
        for i, j in zip(ii, jj):
            if cost[i, j] > 3.0:
                continue
            comparisons.append({
                "frame_index": frame,
                "annotation_id": rows[i]["annotation_id"],
                "class_id": rows[i]["class_id"],
                "centerpoint_class": preds[j]["class_id"],
                "center_distance_m": float(cost[i, j]),
                "current_yaw": float(rows[i]["current_yaw"]),
                "pgd_yaw": rows[i].get("pgd_yaw"),
                "pgd_score": rows[i].get("pgd_score"),
                "centerpoint_yaw": float(preds[j]["box"][6]),
                "centerpoint_score": float(preds[j]["score"]),
                "centerpoint_difference_deg": float(np.degrees(angle_difference(preds[j]["box"][6], rows[i]["current_yaw"]))),
            })

    OUTPUT.write_text(json.dumps({"frames": frames, "detections": detections,
                                  "comparisons": comparisons}, indent=2))
    print(json.dumps({"output": str(OUTPUT),
                      "detections": sum(map(len, detections.values())),
                      "matched": len(comparisons)}, indent=2))


if __name__ == "__main__":
    main()
