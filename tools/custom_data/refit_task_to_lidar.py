#!/usr/bin/env python3
"""Refit a CaliperGT task to raw foreground LiDAR with a 0.5 m hard gate."""

import argparse
import csv
import json
import math
import os
import sys
import uuid
from collections import Counter, defaultdict
from datetime import datetime, timezone
from pathlib import Path

import mmengine
import numpy as np
import requests
from scipy.optimize import linear_sum_assignment

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass


TASK_ID = "b6da9787-6615-4a13-8747-5466ff1a6aed"
TAXONOMY_ID = "981f59a2-75c6-41f1-a204-8d177ada63ce"
API = "http://localhost:3000/api/v1"
BASE = Path("/home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_16_09_58_56_converted/fisheye3dod_inference_2hz")
INFO = BASE / "fisheye3dod_infos_test.pkl"
SOURCE_PLAN = Path("/home/mostafa/bag-2026-09-16-camera-lidar-plan.json")
PARTA2 = Path("/home/mostafa/bag-2026-09-16-parta2-synced/detections.csv")
OUTPUT = Path("/home/mostafa/bag-2026-09-16-point-anchored-comparison.json")
BACKUP = Path("/home/mostafa/task-b6da9787-before-point-anchor-refit.json")
VEHICLES = {"car", "truck", "bus"}
TOLERANCE = 0.5


def api_json(session, method, path, **kwargs):
    response = session.request(method, API + path, timeout=180, **kwargs)
    response.raise_for_status()
    return response.json() if response.content else None


def box_support(points, box, tolerance=TOLERANCE):
    x, y, z, length, width, height, yaw = box
    delta = points - np.array([x, y, z])
    cosine, sine = math.cos(yaw), math.sin(yaw)
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    outside = np.column_stack((
        np.abs(local_x) - length / 2,
        np.abs(local_y) - width / 2,
        np.abs(delta[:, 2]) - height / 2,
    ))
    distance = np.linalg.norm(np.maximum(outside, 0), axis=1)
    return int(np.count_nonzero(distance <= tolerance)), float(distance.min())


def annotation_box(annotation):
    data = annotation["data"]
    return np.array([
        data["center"]["x"], data["center"]["y"], data["center"]["z"],
        data["dimensions"]["length"], data["dimensions"]["width"],
        data["dimensions"]["height"], data["rotation"]["yaw"],
    ], dtype=float)


def load_detections():
    output = defaultdict(list)
    with PARTA2.open(newline="") as stream:
        for row in csv.DictReader(stream):
            output[int(row["frame_id"])].append({
                "x": float(row["x"]), "y": float(row["y"]),
                "z": float(row["z"]), "length": float(row["length"]),
                "width": float(row["width"]), "height": float(row["height"]),
                "yaw": float(row["yaw"]), "score": float(row["score"]),
            })
    return output


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()

    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    frames = api_json(session, "GET", f"/tasks/{TASK_ID}/frames?include_context=false")["frames"]
    annotations = api_json(
        session, "GET",
        f"/annotations-3d?task_id={TASK_ID}&taxonomy_id={TAXONOMY_ID}",
    )
    frame_index = {item["id"]: int(item["frame_index"]) for item in frames}
    samples = {
        int(item["sample_idx"]): item
        for item in mmengine.load(INFO)["data_list"]
    }
    planes = json.loads(SOURCE_PLAN.read_text())["planes"]
    detections = load_detections()
    by_frame = defaultdict(list)
    for annotation in annotations:
        by_frame[frame_index[annotation["frame_id"]]].append(annotation)

    updates, deletes = [], []
    comparison = []
    counters = Counter()
    for index, frame_annotations in sorted(by_frame.items()):
        path = (BASE / samples[index]["pc_info"]["pts_path"]).resolve()
        points = np.fromfile(path, np.float32).reshape(-1, 4)[:, :3]
        plane = np.asarray(planes.get(str(index), [0.0, 0.0, -2.0]), dtype=float)
        above = points[:, 2] - (plane[0] * points[:, 0] + plane[1] * points[:, 1] + plane[2])
        foreground = points[(above > 0.15) & (above < 5.5)]

        vehicle_indices = [
            i for i, annotation in enumerate(frame_annotations)
            if annotation["class_id"] in VEHICLES
        ]
        detector_rows = detections.get(index, [])
        assignments = {}
        if vehicle_indices and detector_rows:
            cost = np.empty((len(vehicle_indices), len(detector_rows)), dtype=float)
            for row_index, annotation_index in enumerate(vehicle_indices):
                box = annotation_box(frame_annotations[annotation_index])
                for column_index, detection in enumerate(detector_rows):
                    cost[row_index, column_index] = math.hypot(
                        box[0] - detection["x"], box[1] - detection["y"]
                    )
            rows, columns = linear_sum_assignment(cost)
            for row, column in zip(rows, columns):
                distance = float(cost[row, column])
                if distance <= 3.0:
                    assignments[vehicle_indices[row]] = (detector_rows[column], distance)

        for annotation_index, annotation in enumerate(frame_annotations):
            old_box = annotation_box(annotation)
            old_count, old_distance = box_support(foreground, old_box)
            new_box = old_box.copy()
            method = "existing_point_supported"
            detector_distance = None

            assigned = assignments.get(annotation_index)
            if assigned is not None:
                detection, detector_distance = assigned
                detector_box = np.array([
                    detection["x"], detection["y"],
                    plane[0] * detection["x"] + plane[1] * detection["y"] + plane[2] + old_box[5] / 2,
                    old_box[3], old_box[4], old_box[5],
                    (detection["yaw"] + math.pi) % (2 * math.pi) - math.pi,
                ])
                detector_count, detector_support_distance = box_support(foreground, detector_box)
                if detector_count > 0 and old_count == 0:
                    new_box = detector_box
                    method = "parta2_point_anchored"
                elif detector_count > 0 and detector_distance <= 0.75:
                    # Preserve camera-derived position but use the detector's
                    # full 360-degree heading for the associated object.
                    new_box[6] = detector_box[6]
                    method = "parta2_yaw_point_confirmed"

            new_count, new_distance = box_support(foreground, new_box)
            # Never replace valid stored geometry with a detector alternative
            # that loses the required 0.5 m foreground-point support.
            if old_count > 0 and (new_count == 0 or new_distance > TOLERANCE):
                new_box = old_box.copy()
                new_count, new_distance = old_count, old_distance
                method = "kept_existing_point_geometry"
            record = {
                "id": annotation["id"], "frame_index": index,
                "class_id": annotation["class_id"], "method": method,
                "old_support": old_count, "new_support": new_count,
                "old_nearest_m": old_distance, "new_nearest_m": new_distance,
                "center_shift_m": float(np.linalg.norm(new_box[:2] - old_box[:2])),
                "yaw_change_deg": math.degrees(abs(
                    (new_box[6] - old_box[6] + math.pi) % (2 * math.pi) - math.pi
                )),
                "detector_distance_m": detector_distance,
            }
            comparison.append(record)

            if new_count == 0 or new_distance > TOLERANCE:
                deletes.append(annotation["id"])
                counters[f"deleted_{annotation['class_id']}"] += 1
                continue

            counters[method] += 1
            if np.max(np.abs(new_box - old_box)) > 1e-7:
                data = dict(annotation["data"])
                data.update({
                    "center": {"x": new_box[0], "y": new_box[1], "z": new_box[2]},
                    "dimensions": {
                        "length": new_box[3], "width": new_box[4], "height": new_box[5]
                    },
                    "rotation": {"yaw": new_box[6], "pitch": 0.0, "roll": 0.0},
                })
                updates.append({
                    "id": annotation["id"],
                    "class_id": annotation["class_id"],
                    "data": data,
                })

    retained = len(annotations) - len(deletes)
    # A LiDAR refit must also retain every supporting camera detection. Use
    # the exact deterministic IDs from the original camera plan.
    namespace = uuid.UUID("67f0a17c-dfc4-4ea4-b882-e731ff51bd66")
    original = json.loads(camera_pass.PRE_ROBUST_PLAN.read_text())["proposals"]
    proposal_by_id = {}
    for proposal in original:
        track_key = min(item["image_track"] for item in proposal["detections"])
        x, y = proposal["box"][:2]
        identity = (
            f"{TASK_ID}:2d-annotation:{proposal['frame_index']}:{track_key}:"
            f"{proposal['class_id']}:{x:.2f}:{y:.2f}"
        )
        proposal_by_id[str(uuid.uuid5(namespace, identity))] = proposal
    native_samples = camera_pass.metadata_samples()
    record_by_id = {item["id"]: item for item in comparison}
    camera_safe_updates = []
    camera_rejected = 0
    delete_set = set(deletes)
    for update in updates:
        proposal = proposal_by_id.get(update["id"])
        if proposal is None:
            camera_safe_updates.append(update)
            continue
        data = update["data"]
        box = np.array([
            data["center"]["x"], data["center"]["y"], data["center"]["z"],
            data["dimensions"]["length"], data["dimensions"]["width"],
            data["dimensions"]["height"], data["rotation"]["yaw"],
        ])
        sample = native_samples[proposal["frame_index"]]
        contained = True
        for detection in proposal["detections"]:
            projected = camera_pass.projected_bbox(
                box,
                sample["cam_info"]["cam_fisheye"][detection["camera"]],
            )
            target = np.asarray(detection["xyxy"], dtype=float)
            contained &= bool(
                projected is not None
                and projected[0] <= target[0] + 5
                and projected[1] <= target[1] + 5
                and projected[2] >= target[2] - 5
                and projected[3] >= target[3] - 5
            )
        if contained:
            camera_safe_updates.append(update)
            continue
        camera_rejected += 1
        record = record_by_id[update["id"]]
        counters[record["method"]] -= 1
        if record["old_support"] > 0:
            counters["kept_existing_camera_geometry"] += 1
            record.update({
                "method": "kept_existing_camera_geometry",
                "new_support": record["old_support"],
                "new_nearest_m": record["old_nearest_m"],
                "center_shift_m": 0.0, "yaw_change_deg": 0.0,
            })
        else:
            delete_set.add(update["id"])
            counters[f"deleted_{record['class_id']}"] += 1
            record["method"] = "deleted_refit_failed_camera"
    updates = camera_safe_updates
    deletes = sorted(delete_set)
    retained = len(annotations) - len(deletes)
    shifts = [item["center_shift_m"] for item in comparison if item["id"] not in set(deletes)]
    yaw_changes = [item["yaw_change_deg"] for item in comparison if item["id"] not in set(deletes)]
    summary = {
        "old_strategy_annotations": len(annotations),
        "old_strategy_point_supported": sum(item["old_support"] > 0 for item in comparison),
        "new_strategy_retained": retained,
        "new_strategy_deleted": len(deletes),
        "new_strategy_updated": len(updates),
        "camera_rejected_refits": camera_rejected,
        "new_strategy_all_within_0_5m": all(
            item["new_nearest_m"] <= TOLERANCE
            for item in comparison if item["id"] not in set(deletes)
        ),
        "center_shift_median": float(np.median(shifts)) if shifts else 0.0,
        "center_shift_p95": float(np.quantile(shifts, 0.95)) if shifts else 0.0,
        "yaw_change_deg_median": float(np.median(yaw_changes)) if yaw_changes else 0.0,
        "yaw_change_deg_p95": float(np.quantile(yaw_changes, 0.95)) if yaw_changes else 0.0,
        "methods": dict(counters),
    }
    OUTPUT.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary, "updates": updates,
        "delete_ids": deletes, "comparison": comparison,
    }, indent=2))
    print(json.dumps(summary, indent=2))
    if not args.apply:
        return

    BACKUP.write_text(json.dumps({
        "created_at": datetime.now(timezone.utc).isoformat(),
        "task_id": TASK_ID, "annotations": annotations,
    }, indent=2))
    for start in range(0, len(updates), 200):
        batch = updates[start:start + 200]
        api_json(session, "POST", "/annotations-3d/bulk-update", json={"annotations": batch})
        print(f"updated {start + len(batch)}/{len(updates)}", flush=True)
    for start in range(0, len(deletes), 200):
        batch = deletes[start:start + 200]
        api_json(session, "POST", "/annotations-3d/bulk-delete", json={"annotation_ids": batch})
        print(f"deleted {start + len(batch)}/{len(deletes)}", flush=True)


if __name__ == "__main__":
    main()
