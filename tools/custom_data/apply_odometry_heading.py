#!/usr/bin/env python3
"""Apply fully validated odometry-derived headings to the Caliper task."""

import argparse
import json
import math
import os
import sys
from pathlib import Path

import requests

TASK_ID = os.getenv("CALIPER_TASK_ID", "b6da9787-6615-4a13-8747-5466ff1a6aed")
TAXONOMY_ID = os.getenv("CALIPER_TAXONOMY_ID", "981f59a2-75c6-41f1-a204-8d177ada63ce")
API = os.getenv("CALIPER_API", "http://localhost:3000/api/v1")
VALIDATION = Path(os.getenv("CALIPER_ODOMETRY_VALIDATION", "/home/mostafa/bag-2026-09-16-odometry-heading-validation.json"))
BACKUP = Path(os.getenv("CALIPER_ODOMETRY_BACKUP", "/home/mostafa/task-b6da9787-before-odometry-heading.json"))
AUDIT = Path(os.getenv("CALIPER_ODOMETRY_AUDIT", "/home/mostafa/bag-2026-09-16-odometry-heading-apply.json"))


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    accepted = [row for row in json.loads(VALIDATION.read_text())["records"]
                if row["accepted"]]
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    response = session.get(
        API + f"/annotations-3d?task_id={TASK_ID}&taxonomy_id={TAXONOMY_ID}",
        timeout=180)
    response.raise_for_status()
    annotation_list = response.json()
    annotations = {row["id"]: row for row in annotation_list}
    updates, skipped, conflicts = [], [], []
    for row in accepted:
        annotation = annotations.get(row["annotation_id"])
        if annotation is None:
            conflicts.append({"annotation_id": row["annotation_id"],
                              "reason": "missing"})
            continue
        current = float(annotation["data"]["rotation"]["yaw"])
        if abs(wrap(current - row["new_yaw"])) < 1e-5:
            skipped.append(row["annotation_id"])
            continue
        if abs(wrap(current - row["old_yaw"])) > 1e-5:
            conflicts.append({"annotation_id": row["annotation_id"],
                              "reason": "yaw_changed_since_validation",
                              "expected": row["old_yaw"], "current": current})
            continue
        data = dict(annotation["data"])
        data["rotation"] = {**data["rotation"], "yaw": row["new_yaw"]}
        updates.append({"id": annotation["id"],
                        "class_id": annotation["class_id"], "data": data})
    audit = {"applied": bool(args.apply), "accepted": len(accepted),
             "updates": len(updates), "already_applied": len(skipped),
             "conflicts": conflicts,
             "annotation_ids": [row["id"] for row in updates]}
    AUDIT.write_text(json.dumps(audit, indent=2))
    print(json.dumps({key: value for key, value in audit.items()
                      if key not in {"annotation_ids", "conflicts"}}, indent=2))
    if conflicts:
        raise RuntimeError(f"refusing to apply with {len(conflicts)} conflicts")
    if not args.apply or not updates:
        return
    BACKUP.write_text(json.dumps({"task_id": TASK_ID,
                                  "annotations": annotation_list}, indent=2))
    response = session.post(API + "/annotations-3d/bulk-update",
                            json={"annotations": updates}, timeout=180)
    response.raise_for_status()
    response = session.get(
        API + f"/annotations-3d?task_id={TASK_ID}&taxonomy_id={TAXONOMY_ID}",
        timeout=180)
    response.raise_for_status()
    final = {row["id"]: row for row in response.json()}
    failures = [row["id"] for row in updates
                if abs(wrap(float(final[row["id"]]["data"]["rotation"]["yaw"]) -
                            float(row["data"]["rotation"]["yaw"]))) > 1e-5]
    if failures:
        raise RuntimeError(f"post-update verification failed for {len(failures)}")
    audit["verified"] = len(updates)
    AUDIT.write_text(json.dumps(audit, indent=2))
    print(f"updated and verified {len(updates)} headings")


if __name__ == "__main__":
    main()
