#!/usr/bin/env python3
"""Plan motion-derived vehicle headings using bidirectional world tracking."""

import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import mmengine
import requests
from scipy.optimize import linear_sum_assignment

ROOT = Path("/home/mostafa/mmdetection3d")
TASK_ID = os.getenv("CALIPER_TASK_ID", "b6da9787-6615-4a13-8747-5466ff1a6aed")
TAXONOMY_ID = os.getenv("CALIPER_TAXONOMY_ID", "981f59a2-75c6-41f1-a204-8d177ada63ce")
API = os.getenv("CALIPER_API", "http://localhost:3000/api/v1")
BASE = Path(os.getenv("CALIPER_BAG_BASE", str(
    ROOT / "data/Custom/BAG_2026_09_16_09_58_56_converted/fisheye3dod_inference_2hz"
)))
CALIBRATION = Path(os.getenv("CALIPER_CALIBRATION", str(BASE.parent / "calibration.json")))
ODOMETRY = Path(os.getenv("CALIPER_ODOMETRY", "/home/mostafa/bag-2026-09-16-odometry.json"))
OUTPUT = Path(os.getenv("CALIPER_ODOMETRY_PLAN", "/home/mostafa/bag-2026-09-16-odometry-heading-plan.json"))
VEHICLES = {"car", "truck", "bus"}


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def axis_difference(a, b):
    return abs(wrap(2 * (a - b))) / 2


def interpolation(rows, timestamps_ms):
    rows = np.asarray(rows, dtype=float)
    source_t = rows[:, 0] / 1e6  # wall-clock timestamps are monotonic
    yaw = np.unwrap(rows[:, 4])
    return np.column_stack([
        np.interp(timestamps_ms, source_t, rows[:, 2]),
        np.interp(timestamps_ms, source_t, rows[:, 3]),
        np.interp(timestamps_ms, source_t, yaw),
    ])


def poses():
    raw = json.loads(ODOMETRY.read_text())
    fused = np.asarray(raw["fused"], dtype=float)
    odom = np.asarray(raw["odometry"], dtype=float)
    samples = sorted(mmengine.load(BASE / "fisheye3dod_infos_test.pkl")["data_list"],
                     key=lambda row: int(row["sample_idx"]))
    frame_ns = np.asarray([int(row["token"]) for row in samples], dtype=np.int64)
    frame_ms = frame_ns / 1e6
    fused_pose = interpolation(fused, frame_ms)
    odom_pose = interpolation(odom, frame_ms)
    fused_start, fused_end = fused[0, 0] / 1e6, fused[-1, 0] / 1e6

    # Odometry is a drifting local solution rather than a fixed transform of
    # FusedOdometry. Align it at each boundary, using it only outside fused
    # coverage. This guarantees continuity without pretending the map frames
    # have a global rigid relationship.
    first = int(np.searchsorted(frame_ms, fused_start))
    last = int(np.searchsorted(frame_ms, fused_end, side="right") - 1)
    result = fused_pose.copy()
    sources = np.full(len(frame_ms), "fused", dtype=object)
    for boundary, selection, label in (
        (first, np.arange(0, first), "aligned_odometry_prefix"),
        (last, np.arange(last + 1, len(frame_ms)), "aligned_odometry_tail"),
    ):
        if not len(selection):
            continue
        delta = wrap(fused_pose[boundary, 2] - odom_pose[boundary, 2])
        cosine, sine = math.cos(delta), math.sin(delta)
        rotation = np.array([[cosine, -sine], [sine, cosine]])
        translation = fused_pose[boundary, :2] - odom_pose[boundary, :2] @ rotation.T
        result[selection, :2] = odom_pose[selection, :2] @ rotation.T + translation
        result[selection, 2] = odom_pose[selection, 2] + delta
        sources[selection] = label
    return frame_ns, result, sources, first, last


def annotation_box(annotation):
    data = annotation["data"]
    return np.array([data["center"]["x"], data["center"]["y"],
                     data["center"]["z"], data["dimensions"]["length"],
                     data["dimensions"]["width"], data["dimensions"]["height"],
                     data["rotation"]["yaw"]], dtype=float)


def world_detections(annotations, frame_index, pose, lidar_to_base):
    yaw_offset = math.atan2(lidar_to_base[1, 0], lidar_to_base[0, 0])
    output = defaultdict(list)
    for annotation in annotations:
        if annotation["class_id"] not in VEHICLES:
            continue
        frame = frame_index[annotation["frame_id"]]
        box = annotation_box(annotation)
        base = lidar_to_base @ np.array([box[0], box[1], box[2], 1.0])
        ego_yaw = pose[frame, 2]
        cosine, sine = math.cos(ego_yaw), math.sin(ego_yaw)
        world_xy = pose[frame, :2] + np.array([
            cosine * base[0] - sine * base[1],
            sine * base[0] + cosine * base[1],
        ])
        output[frame].append({
            "id": annotation["id"], "frame": frame,
            "class_id": annotation["class_id"],
            "original_track_id": annotation.get("track_id"),
            "box": box, "world": world_xy,
            "world_yaw": wrap(box[6] + yaw_offset + ego_yaw),
        })
    return output, yaw_offset


def track(frames, frame_ns, reverse=False):
    order = sorted(frames, reverse=reverse)
    active, next_track, assignments = {}, 0, {}
    for frame in order:
        detections = frames[frame]
        now = frame_ns[frame] / 1e9
        candidates = [(key, state) for key, state in active.items()
                      if abs(now - state["time"]) <= 2.1]
        cost = np.full((len(candidates), len(detections)), 1e6, dtype=float)
        distances = np.full_like(cost, np.inf)
        for i, (_, state) in enumerate(candidates):
            dt = now - state["time"]
            prediction = state["world"] + state["velocity"] * dt
            gate = min(14.0, 3.0 + 20.0 * abs(dt))
            for j, detection in enumerate(detections):
                distance = float(np.linalg.norm(prediction - detection["world"]))
                distances[i, j] = distance
                if distance > gate:
                    continue
                size = np.linalg.norm(
                    np.log(np.maximum(detection["box"][3:5], .2) /
                           np.maximum(state["box"][3:5], .2)))
                axis = axis_difference(detection["world_yaw"], state["world_yaw"])
                same = (detection["original_track_id"] is not None and
                        detection["original_track_id"] == state["original_track_id"])
                cost[i, j] = distance + .8 * size + .35 * axis - (2.0 if same else 0.0)
        matched_detections = set()
        if cost.size:
            rows, columns = linear_sum_assignment(cost)
            for i, j in zip(rows, columns):
                if cost[i, j] >= 1e5:
                    continue
                key, state = candidates[i]
                detection = detections[j]
                dt = now - state["time"]
                measured = ((detection["world"] - state["world"]) / dt
                            if abs(dt) > 1e-6 else state["velocity"])
                speed = np.linalg.norm(measured)
                if speed > 40:
                    measured *= 40 / speed
                velocity = .55 * measured + .45 * state["velocity"]
                active[key] = {**detection, "time": now, "velocity": velocity}
                assignments[detection["id"]] = key
                matched_detections.add(j)
        for j, detection in enumerate(detections):
            if j in matched_detections:
                continue
            key = next_track
            next_track += 1
            active[key] = {**detection, "time": now, "velocity": np.zeros(2)}
            assignments[detection["id"]] = key
    return assignments


def local_velocity(rows, index, frame_ns):
    center_time = frame_ns[rows[index]["frame"]] / 1e9
    chosen = [row for row in rows
              if abs(frame_ns[row["frame"]] / 1e9 - center_time) <= 2.6]
    if len(chosen) < 5:
        return None
    times = np.array([frame_ns[row["frame"]] / 1e9 - center_time for row in chosen])
    points = np.array([row["world"] for row in chosen])
    design = np.column_stack([times, np.ones(len(times))])
    fit = np.linalg.lstsq(design, points, rcond=None)[0]
    velocity = fit[0]
    residual = np.linalg.norm(points - design @ fit, axis=1)
    side_velocities = []
    for mask in (times <= 0, times >= 0):
        if mask.sum() < 3 or np.ptp(times[mask]) < .7:
            return None
        side_velocities.append(np.linalg.lstsq(design[mask], points[mask], rcond=None)[0][0])
    if min(np.linalg.norm(value) for value in side_velocities) < 2.0:
        return None
    direction_agreement = abs(wrap(
        math.atan2(side_velocities[0][1], side_velocities[0][0]) -
        math.atan2(side_velocities[1][1], side_velocities[1][0])))
    return velocity, float(np.median(residual)), len(chosen), direction_agreement


def main():
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    response = session.get(
        API + f"/annotations-3d?task_id={TASK_ID}&taxonomy_id={TAXONOMY_ID}",
        timeout=180)
    response.raise_for_status()
    annotations = response.json()
    response = session.get(API + f"/tasks/{TASK_ID}/frames?include_context=false",
                           timeout=180)
    response.raise_for_status()
    frame_index = {row["id"]: int(row["frame_index"])
                   for row in response.json()["frames"]}
    response = session.get(
        API + f"/annotations-2d?task_id={TASK_ID}&taxonomy_id={TAXONOMY_ID}",
        timeout=180)
    response.raise_for_status()
    annotations_2d = response.json()

    frame_ns, pose, pose_source, first, last = poses()
    calibration = json.loads(CALIBRATION.read_text())
    lidar_to_base = np.asarray(calibration["extrinsics"]["CLIDAR"]["matrix"], float)
    frames, yaw_offset = world_detections(
        annotations, frame_index, pose, lidar_to_base)
    forward = track(frames, frame_ns)
    backward = track(frames, frame_ns, reverse=True)
    consensus = defaultdict(list)
    for detections in frames.values():
        for detection in detections:
            key = (forward[detection["id"]], backward[detection["id"]])
            consensus[key].append(detection)

    candidates = []
    for key, rows in consensus.items():
        rows.sort(key=lambda row: row["frame"])
        if len(rows) < 5:
            continue
        for index, row in enumerate(rows):
            estimate = local_velocity(rows, index, frame_ns)
            if estimate is None:
                continue
            velocity, residual, window, direction_agreement = estimate
            speed = float(np.linalg.norm(velocity))
            if speed < 2.5 or residual > 1.0 or direction_agreement > math.radians(20):
                continue
            world_yaw = math.atan2(velocity[1], velocity[0])
            new_yaw = wrap(world_yaw - pose[row["frame"], 2] - yaw_offset)
            change = abs(wrap(new_yaw - row["box"][6]))
            candidates.append({
                "annotation_id": row["id"], "frame_index": row["frame"],
                "class_id": row["class_id"], "consensus_track": list(key),
                "box": row["box"].tolist(),
                "consensus_track_length": len(rows), "window": window,
                "speed_mps": speed, "fit_residual_m": residual,
                "bidirectional_heading_agreement_deg": math.degrees(direction_agreement),
                "old_yaw": float(row["box"][6]), "new_yaw": new_yaw,
                "signed_change_deg": math.degrees(wrap(new_yaw - row["box"][6])),
                "axis_change_deg": math.degrees(axis_difference(new_yaw, row["box"][6])),
                "pose_source": pose_source[row["frame"]],
            })

    changes = [row for row in candidates if row["axis_change_deg"] >= 10.0]
    result = {
        "summary": {
            "frames": len(frame_ns), "annotations": len(annotations),
            "annotations_2d": len(annotations_2d),
            "annotation_2d_sources": dict(Counter(
                row.get("source", "unknown") for row in annotations_2d)),
            "vehicle_annotations": sum(len(rows) for rows in frames.values()),
            "forward_tracks": len(set(forward.values())),
            "backward_tracks": len(set(backward.values())),
            "consensus_segments": len(consensus),
            "stable_motion_candidates": len(candidates),
            "axis_changes_ge_10deg": len(changes),
            "fused_frame_range": [first, last],
            "pose_sources": dict(Counter(pose_source)),
        },
        "annotation_2d_sample": annotations_2d[:3],
        "candidates": candidates,
    }
    OUTPUT.write_text(json.dumps(result, indent=2))
    print(json.dumps(result["summary"], indent=2))
    if changes:
        print("largest axis changes")
        for row in sorted(changes, key=lambda item: item["axis_change_deg"], reverse=True)[:20]:
            print(row["frame_index"], row["class_id"],
                  f"axis={row['axis_change_deg']:.1f}",
                  f"signed={row['signed_change_deg']:.1f}",
                  f"speed={row['speed_mps']:.1f}",
                  f"residual={row['fit_residual_m']:.2f}")


if __name__ == "__main__":
    main()
