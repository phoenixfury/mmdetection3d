#!/usr/bin/env python3
"""Validate odometry heading changes against YOLO images and raw LiDAR."""

import json
import os
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass
sys.path.insert(0, "/home/mostafa/mmdetection3d/tools/custom_data")
from refit_task_to_lidar import box_support


PLAN = Path(os.getenv("CALIPER_ODOMETRY_PLAN", "/home/mostafa/bag-2026-09-16-odometry-heading-plan.json"))
OUTPUT = Path(os.getenv("CALIPER_ODOMETRY_VALIDATION", "/home/mostafa/bag-2026-09-16-odometry-heading-validation.json"))
WEIGHTS = Path("/home/mostafa/perception-pipeline-cyclonedds/onnx_export_pkg/yolo11m-seg.pt")
VEHICLE_LABELS = {2, 5, 7}


def overlap_fraction(first, second):
    x1, y1 = max(first[0], second[0]), max(first[1], second[1])
    x2, y2 = min(first[2], second[2]), min(first[3], second[3])
    intersection = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    area = max(1.0, (second[2] - second[0]) * (second[3] - second[1]))
    return intersection / area


def expanded(box, pixels):
    return np.asarray(box, float) + [-pixels, -pixels, pixels, pixels]


def main():
    rows = [row for row in json.loads(PLAN.read_text())["candidates"]
            if row["axis_change_deg"] >= 10]
    samples = camera_pass.metadata_samples()
    projections = {}
    jobs = {}
    for row in rows:
        sample = samples[row["frame_index"]]
        old = np.asarray(row["box"], float)
        new = old.copy()
        new[6] = row["new_yaw"]
        for name, camera in sample["cam_info"]["cam_fisheye"].items():
            old_bbox = camera_pass.projected_bbox(old, camera)
            new_bbox = camera_pass.projected_bbox(new, camera)
            if old_bbox is None or new_bbox is None:
                continue
            area = (old_bbox[2] - old_bbox[0]) * (old_bbox[3] - old_bbox[1])
            if area < 100:
                continue
            key = (row["frame_index"], name)
            projections[(row["annotation_id"], name)] = (old_bbox, new_bbox)
            jobs[key] = str(camera_pass.image_path(sample, name))

    from ultralytics import YOLO
    model = YOLO(str(WEIGHTS))
    detections = defaultdict(list)
    job_list = list(jobs.items())
    for start in range(0, len(job_list), 4):
        batch = job_list[start:start + 4]
        results = model.predict([path for _, path in batch], imgsz=768, conf=.12,
                                iou=.65, classes=sorted(VEHICLE_LABELS),
                                device=0, batch=4, half=True, verbose=False)
        for (key, _), result in zip(batch, results):
            if result.boxes is None:
                continue
            for xyxy, confidence, label in zip(
                    result.boxes.xyxy.cpu().numpy(),
                    result.boxes.conf.cpu().numpy(),
                    result.boxes.cls.cpu().numpy()):
                detections[key].append({"xyxy": list(map(float, xyxy)),
                                        "confidence": float(confidence),
                                        "label": int(label)})
        if start % 80 == 0:
            print(f"YOLO {min(start + len(batch), len(job_list))}/{len(job_list)}",
                  flush=True)

    point_cache = {}
    validated = []
    for row in rows:
        frame = row["frame_index"]
        old = np.asarray(row["box"], float)
        new = old.copy()
        new[6] = row["new_yaw"]
        if frame not in point_cache:
            point_cache[frame] = camera_pass.load_lidar_points(samples[frame])
        old_support, old_nearest = box_support(point_cache[frame], old)
        new_support, new_nearest = box_support(point_cache[frame], new)
        matches = []
        for name in samples[frame]["cam_info"]["cam_fisheye"]:
            projection = projections.get((row["annotation_id"], name))
            if projection is None:
                continue
            old_bbox, new_bbox = projection
            choices = []
            for detection in detections[(frame, name)]:
                old_overlap = overlap_fraction(expanded(old_bbox, 8), detection["xyxy"])
                if old_overlap >= .65:
                    choices.append((old_overlap * detection["confidence"], detection))
            if not choices:
                continue
            _, detection = max(choices, key=lambda item: item[0])
            new_overlap = overlap_fraction(expanded(new_bbox, 8), detection["xyxy"])
            matches.append({"camera": name, "detection": detection,
                            "old_overlap": overlap_fraction(old_bbox, detection["xyxy"]),
                            "new_overlap": new_overlap})
        camera_ok = bool(matches) and all(match["new_overlap"] >= .90 for match in matches)
        lidar_ok = (new_nearest <= .5 and new_support >= max(3, old_support * .5))
        motion_ok = (row["bidirectional_heading_agreement_deg"] <= 15 and
                     row["fit_residual_m"] <= .8 and
                     row["consensus_track_length"] >= 7)
        if row["pose_source"] == "aligned_odometry_tail":
            motion_ok &= row["speed_mps"] >= 4 and row["fit_residual_m"] <= .65
        validated.append({**row, "old_support": old_support,
                          "new_support": new_support,
                          "old_nearest_m": old_nearest,
                          "new_nearest_m": new_nearest,
                          "camera_matches": matches, "camera_ok": camera_ok,
                          "lidar_ok": lidar_ok, "motion_ok": bool(motion_ok),
                          "accepted": bool(camera_ok and lidar_ok and motion_ok)})

    summary = {
        "candidates": len(validated), "yolo_jobs": len(job_list),
        "camera_ok": sum(row["camera_ok"] for row in validated),
        "lidar_ok": sum(row["lidar_ok"] for row in validated),
        "motion_ok": sum(row["motion_ok"] for row in validated),
        "accepted": sum(row["accepted"] for row in validated),
    }
    OUTPUT.write_text(json.dumps({"summary": summary, "records": validated}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
