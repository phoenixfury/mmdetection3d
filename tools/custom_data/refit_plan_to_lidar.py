#!/usr/bin/env python3
"""Restore raw-LiDAR support to a camera-derived annotation plan."""

import argparse
import json
import math
import os
import sys
from collections import Counter, defaultdict
from pathlib import Path

import mmengine
import numpy as np

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass


def box_support(points, box, tolerance=0.5):
    delta = points - box[:3]
    cosine, sine = math.cos(box[6]), math.sin(box[6])
    local_x = cosine * delta[:, 0] + sine * delta[:, 1]
    local_y = -sine * delta[:, 0] + cosine * delta[:, 1]
    outside = np.column_stack((
        np.abs(local_x) - box[3] / 2,
        np.abs(local_y) - box[4] / 2,
        np.abs(delta[:, 2]) - box[5] / 2,
    ))
    distances = np.linalg.norm(np.maximum(outside, 0), axis=1)
    index = int(np.argmin(distances))
    return int(np.count_nonzero(distances <= tolerance)), float(distances[index]), index


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--backup", required=True, type=Path)
    parser.add_argument("--report", required=True, type=Path)
    args = parser.parse_args()

    document = json.loads(args.plan.read_text())
    args.backup.write_text(json.dumps(document, indent=2))
    samples = camera_pass.metadata_samples()
    info = {
        int(item["sample_idx"]): item
        for item in mmengine.load(camera_pass.METADATA)["data_list"]
    }
    prior_path = Path(os.getenv(
        "CALIPER_PRIOR_PLAN", str(camera_pass.PRE_ROBUST_PLAN)
    ))
    prior = defaultdict(list)
    if prior_path.exists():
        for item in json.loads(prior_path.read_text()).get("proposals", []):
            prior[(int(item["frame_index"]), int(item["candidate_component"]))].append(item)

    def contained(proposal, box):
        sample = samples[int(proposal["frame_index"])]
        results = []
        ious = []
        for detection in proposal["detections"]:
            projected = camera_pass.projected_bbox(
                box, sample["cam_info"]["cam_fisheye"][detection["camera"]]
            )
            target = np.asarray(detection["xyxy"], dtype=float)
            ok = bool(
                projected is not None
                and projected[0] <= target[0] + 5
                and projected[1] <= target[1] + 5
                and projected[2] >= target[2] - 5
                and projected[3] >= target[3] - 5
            )
            results.append(ok)
            ious.append(0.0 if projected is None else camera_pass.bbox_iou(projected, target))
        return all(results), results, ious

    kept = []
    report = []
    counters = Counter()
    frame_points = {}
    for proposal in document["proposals"]:
        frame = int(proposal["frame_index"])
        if frame not in frame_points:
            path = (camera_pass.BASE / info[frame]["pc_info"]["pts_path"]).resolve()
            points = np.fromfile(path, np.float32).reshape(-1, 4)[:, :3]
            # The task requirement is distance to the raw cloud. Road returns
            # below a distant or partially occluded object are legitimate
            # evidence that its cuboid occupies the measured scene.
            frame_points[frame] = points
        points = frame_points[frame]
        box = np.asarray(proposal["box"], dtype=float)
        support, nearest, nearest_index = box_support(points, box)
        if support and nearest <= 0.5:
            kept.append(proposal)
            counters["already_valid"] += 1
            continue

        plane = np.asarray(document["planes"][str(frame)], dtype=float)
        xy_targets = []
        seed = np.asarray(proposal.get("seed_box", box), dtype=float)
        xy_targets.append(("seed", seed[:2]))
        key = (frame, int(proposal["candidate_component"]))
        for old in prior.get(key, []):
            xy_targets.append(("prior", np.asarray(old["box"], dtype=float)[:2]))
        nearest_point = points[nearest_index]
        # Camera optimization is usually only slightly beyond the object. Walk
        # toward the nearest foreground return and stop at the first valid pose.
        xy_targets.append(("nearest_foreground", nearest_point[:2]))

        candidates = []
        for method, target in xy_targets:
            delta = target - box[:2]
            distance = float(np.linalg.norm(delta))
            if distance > 8.0:
                continue
            steps = max(2, int(math.ceil(distance / 0.10)) + 1)
            for alpha in np.linspace(0.0, 1.0, steps)[1:]:
                candidate = box.copy()
                candidate[:2] = box[:2] + alpha * delta
                candidate[2] = (
                    plane[0] * candidate[0] + plane[1] * candidate[1]
                    + plane[2] + candidate[5] / 2
                )
                count, candidate_nearest, _ = box_support(points, candidate)
                if not count or candidate_nearest > 0.5:
                    continue
                ok, containments, ious = contained(proposal, candidate)
                if ok:
                    shift = float(np.linalg.norm(candidate[:2] - box[:2]))
                    score = shift - 0.01 * min(count, 200) - 0.02 * sum(ious)
                    candidates.append((score, method, candidate, count, candidate_nearest,
                                       containments, ious, shift))
                    break

        if not candidates:
            counters[f"dropped_{proposal['class_id']}"] += 1
            report.append({
                "frame": frame, "component": proposal["candidate_component"],
                "class_id": proposal["class_id"], "action": "drop",
                "old_nearest_m": nearest,
            })
            continue

        _, method, candidate, count, candidate_nearest, containments, ious, shift = min(
            candidates, key=lambda item: item[0]
        )
        proposal["box"] = candidate.tolist()
        proposal["containment"] = containments
        proposal["ious"] = ious
        proposal["contains_all"] = True
        proposal["lidar_refit"] = {
            "method": method, "support": count,
            "nearest_m": candidate_nearest, "center_shift_m": shift,
        }
        kept.append(proposal)
        counters[f"refit_{method}"] += 1
        report.append({
            "frame": frame, "component": proposal["candidate_component"],
            "class_id": proposal["class_id"], "action": "refit", "method": method,
            "old_nearest_m": nearest, "new_nearest_m": candidate_nearest,
            "support": count, "center_shift_m": shift,
        })

    document["proposals"] = kept
    document["lidar_refit"] = {
        "hard_gate_m": 0.5, "counts": dict(counters),
        "backup": str(args.backup), "report": str(args.report),
    }
    args.plan.write_text(json.dumps(document, indent=2))
    args.report.write_text(json.dumps({"counts": dict(counters), "rows": report}, indent=2))
    print(json.dumps({"kept": len(kept), "counts": dict(counters)}, indent=2))


if __name__ == "__main__":
    main()
