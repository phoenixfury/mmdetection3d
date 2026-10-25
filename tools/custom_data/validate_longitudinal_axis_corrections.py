#!/usr/bin/env python3
"""Validate yaw-axis corrections against YOLO cameras and raw LiDAR."""

import json
import math
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass


WEIGHTS = Path("/home/mostafa/perception-pipeline-cyclonedds/onnx_export_pkg/yolo11m-seg.pt")
VEHICLE_LABELS = {2, 5, 7}


def box_support(points, box, tolerance=.5):
    delta = points - box[:3]
    cosine, sine = math.cos(box[6]), math.sin(box[6])
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    outside = np.column_stack((
        np.abs(local_x) - box[3] / 2,
        np.abs(local_y) - box[4] / 2,
        np.abs(delta[:, 2]) - box[5] / 2,
    ))
    distance = np.linalg.norm(np.maximum(outside, 0), axis=1)
    return int(np.count_nonzero(distance <= tolerance)), float(distance.min())


def overlap_fraction(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / area


def main():
    plan = json.loads(Path(os.environ["CALIPER_AXIS_PLAN"]).read_text())
    output = Path(os.environ["CALIPER_AXIS_VALIDATION"])
    rows = plan["candidates"]
    samples = camera_pass.metadata_samples()
    jobs = {}
    projections = {}
    for row in rows:
        frame = row["frame_index"]
        old = np.asarray(row["box"], dtype=float)
        new = old.copy()
        new[6] = row["new_yaw"]
        for name, camera in samples[frame]["cam_info"]["cam_fisheye"].items():
            old_bbox = camera_pass.projected_bbox(old, camera)
            new_bbox = camera_pass.projected_bbox(new, camera)
            if old_bbox is None or new_bbox is None:
                continue
            area = max(0, old_bbox[2] - old_bbox[0]) * max(0, old_bbox[3] - old_bbox[1])
            if area < 100:
                continue
            projections[(row["annotation_id"], name)] = (old_bbox, new_bbox)
            jobs[(frame, name)] = str(camera_pass.image_path(samples[frame], name))

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    detections = defaultdict(list)
    job_list = list(jobs.items())
    for start in range(0, len(job_list), 4):
        batch = job_list[start:start + 4]
        results = model.predict(
            [path for _, path in batch], imgsz=768, conf=.12, iou=.65,
            classes=sorted(VEHICLE_LABELS), device=0, batch=4,
            half=True, verbose=False,
        )
        for (key, _), result in zip(batch, results):
            if result.boxes is None:
                continue
            for xyxy, confidence, label in zip(
                    result.boxes.xyxy.cpu().numpy(), result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy()):
                detections[key].append({
                    "xyxy": list(map(float, xyxy)), "confidence": float(confidence),
                    "label": int(label),
                })
        if start % 80 == 0 or start + 4 >= len(job_list):
            print(f"YOLO {min(start + len(batch), len(job_list))}/{len(job_list)}", flush=True)

    point_cache = {}
    validated = []
    for row in rows:
        frame = row["frame_index"]
        old = np.asarray(row["box"], dtype=float)
        new = old.copy()
        new[6] = row["new_yaw"]
        if frame not in point_cache:
            point_cache[frame] = camera_pass.load_lidar_points(samples[frame])
        old_support, old_nearest = box_support(point_cache[frame], old)
        new_support, new_nearest = box_support(point_cache[frame], new)
        matches = []
        for name in samples[frame]["cam_info"]["cam_fisheye"]:
            pair = projections.get((row["annotation_id"], name))
            if pair is None:
                continue
            old_bbox, new_bbox = pair
            choices = []
            expanded = old_bbox + [-10, -10, 10, 10]
            for detection in detections[(frame, name)]:
                overlap = overlap_fraction(expanded, detection["xyxy"])
                if overlap >= .55:
                    choices.append((overlap * detection["confidence"], detection))
            if not choices:
                continue
            _, detection = max(choices, key=lambda item: item[0])
            target = np.asarray(detection["xyxy"], dtype=float)
            contained = bool(
                new_bbox[0] <= target[0] + 5 and new_bbox[1] <= target[1] + 5
                and new_bbox[2] >= target[2] - 5 and new_bbox[3] >= target[3] - 5
            )
            matches.append({"camera": name, "detection": detection,
                            "contained": contained,
                            "new_overlap": overlap_fraction(new_bbox, target)})
        camera_ok = bool(matches) and all(match["contained"] for match in matches)
        lidar_ok = bool(new_support > 0 and new_nearest <= .5)
        validated.append({
            **row, "old_support": old_support, "new_support": new_support,
            "old_nearest_m": old_nearest, "new_nearest_m": new_nearest,
            "camera_matches": matches, "camera_ok": camera_ok,
            "lidar_ok": lidar_ok, "accepted": bool(camera_ok and lidar_ok),
        })
    summary = {
        "candidates": len(rows), "yolo_jobs": len(job_list),
        "camera_ok": sum(row["camera_ok"] for row in validated),
        "lidar_ok": sum(row["lidar_ok"] for row in validated),
        "accepted": sum(row["accepted"] for row in validated),
    }
    output.write_text(json.dumps({"summary": summary, "records": validated}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
