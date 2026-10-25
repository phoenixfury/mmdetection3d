#!/usr/bin/env python3
"""Attach conservative CenterPoint yaw evidence to camera-first proposals."""

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


def axis_difference(first, second):
    delta = (first - second + math.pi / 2) % math.pi - math.pi / 2
    return abs(delta)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", type=Path, required=True)
    parser.add_argument("--centerpoint", type=Path, required=True)
    args = parser.parse_args()
    document = json.loads(args.plan.read_text())
    backup = args.plan.with_name(args.plan.stem + "-before-centerpoint.json")
    backup.write_text(json.dumps(document, indent=2))
    detections = json.loads(args.centerpoint.read_text())["detections"]
    by_frame = defaultdict(list)
    for proposal in document["proposals"]:
        if proposal["class_id"] in {"car", "truck", "bus"}:
            by_frame[int(proposal["frame_index"])].append(proposal)
    summary = Counter()
    for frame, proposals in by_frame.items():
        lidar = detections.get(str(frame), [])
        if not lidar:
            continue
        cost = np.empty((len(proposals), len(lidar)), dtype=float)
        for row, proposal in enumerate(proposals):
            for column, detection in enumerate(lidar):
                cost[row, column] = np.linalg.norm(
                    np.asarray(proposal["box"][:2]) - np.asarray(detection["box"][:2]))
        rows, columns = linear_sum_assignment(cost)
        for row, column in zip(rows, columns):
            distance = float(cost[row, column])
            detection = lidar[column]
            if distance > 2.5 or detection["score"] < .25:
                continue
            proposal = proposals[row]
            lidar_yaw = proposal.get("lidar_heading", {}).get("yaw")
            difference = (math.degrees(axis_difference(lidar_yaw, detection["box"][6]))
                          if lidar_yaw is not None else None)
            proposal["centerpoint_heading"] = {
                "yaw": float(detection["box"][6]),
                "score": float(detection["score"]),
                "distance_m": distance,
                "class_id": detection["class_id"],
                "lidar_axis_difference_deg": difference,
            }
            summary["matched"] += 1
            # Use CenterPoint only when it agrees with the local point-cloud
            # axis, or when that local fit is unavailable. Large disagreements
            # are deferred to the later world-coordinate tracking pass.
            if detection["score"] >= .4 and distance <= 1.5 and (
                    lidar_yaw is None or difference <= 25):
                proposal["robust_yaw"] = {
                    "yaw": float(detection["box"][6]),
                    "source": "centerpoint_confirmed_by_raw_lidar",
                    "score": float(detection["score"]),
                    "distance_m": distance,
                }
                summary["yaw_selected"] += 1
            elif difference is not None and difference > 45:
                summary["large_disagreement_deferred"] += 1
    document["centerpoint_evidence"] = {
        "source": str(args.centerpoint),
        "policy": "yaw-only; no object creation/class override; large disagreements deferred",
        **dict(summary),
    }
    args.plan.write_text(json.dumps(document, indent=2))
    print(json.dumps(document["centerpoint_evidence"], indent=2))


if __name__ == "__main__":
    main()
