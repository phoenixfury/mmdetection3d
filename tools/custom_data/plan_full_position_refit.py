#!/usr/bin/env python3
"""Plan conservative full-bag center corrections from two LiDAR detectors."""

import json
import math
import sys
from pathlib import Path

import mmengine
import numpy as np

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass
sys.path.insert(0, "/home/mostafa/mmdetection3d/tools/custom_data")
from pgd_fisheye_heading_pilot import PLAN, annotation_id
from refit_task_to_lidar import box_support


BASE = Path("/home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_16_09_58_56_converted/fisheye3dod_inference_2hz")
INFO = BASE / "fisheye3dod_infos_test.pkl"
INDEX = Path("/home/mostafa/bag-2026-09-16-full-manual-review.json")
CAMERA_PLAN = Path("/home/mostafa/bag-2026-09-16-camera-lidar-plan.json")
OUTPUT = Path("/home/mostafa/bag-2026-09-16-full-position-refit-plan.json")


def contains(box, proposal, sample):
    output = []
    for detection in proposal["detections"]:
        projected = camera_pass.projected_bbox(
            box, sample["cam_info"]["cam_fisheye"][detection["camera"]])
        target = np.asarray(detection["xyxy"], float)
        output.append(bool(projected is not None
                           and projected[0] <= target[0] + 5
                           and projected[1] <= target[1] + 5
                           and projected[2] >= target[2] - 5
                           and projected[3] >= target[3] - 5))
    return output


def main():
    records = json.loads(INDEX.read_text())["records"]
    proposals = {annotation_id(x): x
                 for x in json.loads(PLAN.read_text())["proposals"]}
    planes = json.loads(CAMERA_PLAN.read_text())["planes"]
    samples = camera_pass.metadata_samples()
    metadata = {int(x["sample_idx"]): x
                for x in mmengine.load(INFO)["data_list"]}
    selected = []
    for r in records:
        if r.get("centerpoint_score", 0) < .6 or r.get("parta2_score", 0) < 3:
            continue
        detector_agreement = math.dist(r["centerpoint_box"][:2], r["parta2_center"])
        if detector_agreement > .75 or r["centerpoint_distance_m"] <= .75:
            continue
        old = np.asarray(r["box"], float)
        target = (np.asarray(r["centerpoint_box"][:2]) * .65
                  + np.asarray(r["parta2_center"]) * .35)
        frame = r["frame_index"]
        path = (BASE / metadata[frame]["pc_info"]["pts_path"]).resolve()
        points = np.fromfile(path, np.float32).reshape(-1, 4)[:, :3]
        proposal = proposals[r["annotation_id"]]
        old_camera = contains(old, proposal, samples[frame])
        best = None
        # Move as far toward the independent consensus as camera containment
        # permits. Keep dimensions and reviewed heading unchanged.
        for fraction in (1.0, .75, .5, .25):
            candidate = old.copy()
            candidate[:2] = old[:2] + fraction * (target - old[:2])
            plane = planes.get(str(frame), [0.0, 0.0, -2.0])
            candidate[2] = (plane[0] * candidate[0] + plane[1] * candidate[1]
                            + plane[2] + candidate[5] / 2)
            support, nearest = box_support(points, candidate)
            camera = contains(candidate, proposal, samples[frame])
            if nearest <= .5 and sum(camera) >= sum(old_camera):
                best = (candidate, fraction, support, nearest, camera)
                break
        if best is None:
            continue
        candidate, fraction, support, nearest, camera = best
        shift = float(np.linalg.norm(candidate[:2] - old[:2]))
        if shift < .25:
            continue
        selected.append({**r, "detector_agreement_m": detector_agreement,
                         "new_box": candidate.tolist(), "fraction": fraction,
                         "shift_m": shift, "new_support": support,
                         "new_nearest_m": nearest,
                         "new_camera_containment": camera})
    OUTPUT.write_text(json.dumps({"candidates": selected}, indent=2))
    print(json.dumps({"candidates": len(selected),
                      "tracks": len({x['track_id'] for x in selected}),
                      "full_consensus_moves": sum(x['fraction'] == 1 for x in selected)},
                     indent=2))


if __name__ == "__main__":
    main()
