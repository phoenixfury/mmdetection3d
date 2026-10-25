#!/usr/bin/env python3
"""Revert applied yaw changes rejected by the source-camera containment audit."""

import json
import math
import os
from collections import Counter
from pathlib import Path

import requests


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    task = os.environ["CALIPER_TASK_ID"]
    taxonomy = os.environ["CALIPER_TAXONOMY_ID"]
    api = os.getenv("CALIPER_API", "http://localhost:3000/api/v1")
    validation = json.loads(Path(os.environ["CALIPER_ODOMETRY_VALIDATION"]).read_text())
    rejected = {
        row["annotation_id"]: row for row in validation["records"]
        if row.get("containment_audit") in {"failed", "proposal_not_found"}
    }
    backup_rows = json.loads(Path(os.environ["CALIPER_ODOMETRY_BACKUP"]).read_text())["annotations"]
    backup = {row["id"]: row for row in backup_rows}
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    response = session.get(
        f"{api}/annotations-3d?task_id={task}&taxonomy_id={taxonomy}", timeout=180
    )
    response.raise_for_status()
    current = {row["id"]: row for row in response.json()}
    updates = []
    already_reverted = 0
    for annotation_id, row in rejected.items():
        annotation = current[annotation_id]
        current_yaw = annotation["data"]["rotation"]["yaw"]
        old_yaw = backup[annotation_id]["data"]["rotation"]["yaw"]
        if abs(wrap(current_yaw - old_yaw)) <= 1e-5:
            already_reverted += 1
            continue
        if abs(wrap(current_yaw - row["new_yaw"])) > 1e-5:
            raise RuntimeError(f"unexpected current yaw for {annotation_id}")
        original = backup[annotation_id]
        updates.append({"id": annotation_id, "class_id": original["class_id"],
                        "data": original["data"]})
    if updates:
        response = session.post(f"{api}/annotations-3d/bulk-update",
                                json={"annotations": updates}, timeout=180)
        response.raise_for_status()
    response = session.get(
        f"{api}/annotations-3d?task_id={task}&taxonomy_id={taxonomy}", timeout=180
    )
    response.raise_for_status()
    final = response.json()
    print(json.dumps({"reverted": len(updates), "already_reverted": already_reverted,
                      "retained_yaw_corrections": validation["summary"]["accepted"],
                      "annotations": len(final),
                      "classes": dict(Counter(row["class_id"] for row in final))}, indent=2))


if __name__ == "__main__":
    main()
