#!/usr/bin/env python3
"""Audit accepted odometry yaw changes against every source camera box."""

import json
import os
import sys
import uuid
from pathlib import Path

import numpy as np

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass


def main():
    validation_path = Path(os.environ["CALIPER_ODOMETRY_VALIDATION"])
    validation = json.loads(validation_path.read_text())
    samples = camera_pass.metadata_samples()
    namespace = uuid.UUID("67f0a17c-dfc4-4ea4-b882-e731ff51bd66")
    proposals = {}
    for proposal in json.loads(camera_pass.PLAN.read_text())["proposals"]:
        track_key = min(item["image_track"] for item in proposal["detections"])
        x, y = proposal["box"][:2]
        identity = (
            f"{camera_pass.TASK_ID}:2d-annotation:{proposal['frame_index']}:{track_key}:"
            f"{proposal['class_id']}:{x:.2f}:{y:.2f}"
        )
        proposals[str(uuid.uuid5(namespace, identity))] = proposal

    failed = []
    checked = 0
    for row in validation["records"]:
        if not row["accepted"]:
            continue
        proposal = proposals.get(row["annotation_id"])
        if proposal is None:
            row["accepted"] = False
            row["containment_audit"] = "proposal_not_found"
            failed.append(row["annotation_id"])
            continue
        checked += 1
        box = np.asarray(row["box"], dtype=float)
        box[6] = row["new_yaw"]
        sample = samples[int(row["frame_index"])]
        results = []
        for detection in proposal["detections"]:
            projected = camera_pass.projected_bbox(
                box, sample["cam_info"]["cam_fisheye"][detection["camera"]]
            )
            target = np.asarray(detection["xyxy"], dtype=float)
            results.append(bool(
                projected is not None
                and projected[0] <= target[0] + 5
                and projected[1] <= target[1] + 5
                and projected[2] >= target[2] - 5
                and projected[3] >= target[3] - 5
            ))
        row["source_detection_containment"] = results
        row["containment_audit"] = "passed" if all(results) else "failed"
        if not all(results):
            row["accepted"] = False
            failed.append(row["annotation_id"])

    validation["summary"]["accepted_before_source_containment"] = checked
    validation["summary"]["source_containment_failed"] = len(failed)
    validation["summary"]["accepted"] = checked - len(failed)
    validation_path.write_text(json.dumps(validation, indent=2))
    print(json.dumps({"checked": checked, "failed": len(failed),
                      "accepted": checked - len(failed)}, indent=2))


if __name__ == "__main__":
    main()
