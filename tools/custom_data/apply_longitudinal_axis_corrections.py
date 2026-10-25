#!/usr/bin/env python3
"""Apply validated longitudinal-axis yaw corrections with conflict checks."""

import argparse
import json
import math
import os
from pathlib import Path

import requests


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    task = os.environ["CALIPER_TASK_ID"]
    taxonomy = os.environ["CALIPER_TAXONOMY_ID"]
    api = os.getenv("CALIPER_API", "http://localhost:3000/api/v1")
    validation = json.loads(Path(os.environ["CALIPER_AXIS_VALIDATION"]).read_text())
    accepted = [row for row in validation["records"] if row["accepted"]]
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    response = session.get(
        f"{api}/annotations-3d?task_id={task}&taxonomy_id={taxonomy}", timeout=180
    )
    response.raise_for_status()
    annotation_list = response.json()
    annotations = {row["id"]: row for row in annotation_list}
    updates, conflicts, unchanged = [], [], 0
    for row in accepted:
        annotation = annotations.get(row["annotation_id"])
        if annotation is None:
            conflicts.append({"id": row["annotation_id"], "reason": "missing"})
            continue
        current_yaw = float(annotation["data"]["rotation"]["yaw"])
        if abs(wrap(current_yaw - row["new_yaw"])) <= 1e-5:
            unchanged += 1
            continue
        if abs(wrap(current_yaw - row["old_yaw"])) > 1e-5:
            conflicts.append({"id": row["annotation_id"], "reason": "yaw_changed",
                              "expected": row["old_yaw"], "current": current_yaw})
            continue
        data = dict(annotation["data"])
        data["rotation"] = {**data["rotation"], "yaw": row["new_yaw"]}
        updates.append({"id": annotation["id"], "class_id": annotation["class_id"],
                        "data": data})
    print(json.dumps({"accepted": len(accepted), "updates": len(updates),
                      "already_applied": unchanged, "conflicts": len(conflicts),
                      "apply": args.apply}, indent=2))
    if conflicts:
        raise RuntimeError(f"refusing to apply with {len(conflicts)} conflicts")
    if not args.apply:
        return
    Path(os.environ["CALIPER_AXIS_BACKUP"]).write_text(json.dumps({
        "task_id": task, "annotations": annotation_list,
    }, indent=2))
    for start in range(0, len(updates), 200):
        batch = updates[start:start + 200]
        response = session.post(f"{api}/annotations-3d/bulk-update",
                                json={"annotations": batch}, timeout=180)
        response.raise_for_status()
        print(f"updated {start + len(batch)}/{len(updates)}", flush=True)
    response = session.get(
        f"{api}/annotations-3d?task_id={task}&taxonomy_id={taxonomy}", timeout=180
    )
    response.raise_for_status()
    final = {row["id"]: row for row in response.json()}
    failures = [row["id"] for row in updates if abs(wrap(
        final[row["id"]]["data"]["rotation"]["yaw"]
        - row["data"]["rotation"]["yaw"])) > 1e-5]
    if failures:
        raise RuntimeError(f"verification failed for {len(failures)} updates")
    print(f"verified {len(updates)} yaw updates")


if __name__ == "__main__":
    main()
