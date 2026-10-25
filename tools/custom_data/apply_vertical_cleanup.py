#!/usr/bin/env python3
"""Apply a conflict-checked ground-contact cleanup plan."""

import argparse
import json
import os
from pathlib import Path

import requests


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--apply", action="store_true")
    args = parser.parse_args()
    task = os.environ["CALIPER_TASK_ID"]
    taxonomy = os.environ["CALIPER_TAXONOMY_ID"]
    api = os.getenv("CALIPER_API", "http://localhost:3000/api/v1")
    plan = json.loads(Path(os.environ["CALIPER_VERTICAL_PLAN"]).read_text())
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    response = session.get(
        f"{api}/annotations-3d?task_id={task}&taxonomy_id={taxonomy}", timeout=180
    )
    response.raise_for_status()
    annotation_list = response.json()
    annotations = {row["id"]: row for row in annotation_list}
    updates, deletes, conflicts = [], [], []
    for action in plan["actions"]:
        annotation = annotations.get(action["annotation_id"])
        if annotation is None:
            conflicts.append({"id": action["annotation_id"], "reason": "missing"})
            continue
        current_z = float(annotation["data"]["center"]["z"])
        if abs(current_z - float(action["center_z"])) > 1e-5:
            conflicts.append({"id": action["annotation_id"], "reason": "center_z_changed",
                              "expected": action["center_z"], "current": current_z})
            continue
        if action["action"] == "delete":
            deletes.append(annotation["id"])
            continue
        data = dict(annotation["data"])
        data["center"] = {**data["center"], "z": action["new_center_z"]}
        updates.append({"id": annotation["id"], "class_id": annotation["class_id"],
                        "data": data})
    print(json.dumps({"updates": len(updates), "deletes": len(deletes),
                      "conflicts": len(conflicts), "apply": args.apply}, indent=2))
    if conflicts:
        raise RuntimeError(f"refusing to apply with {len(conflicts)} conflicts")
    if not args.apply:
        return
    Path(os.environ["CALIPER_VERTICAL_APPLY_BACKUP"]).write_text(json.dumps({
        "task_id": task, "annotations": annotation_list,
    }, indent=2))
    for start in range(0, len(updates), 200):
        batch = updates[start:start + 200]
        response = session.post(f"{api}/annotations-3d/bulk-update",
                                json={"annotations": batch}, timeout=180)
        response.raise_for_status()
        print(f"updated {start + len(batch)}/{len(updates)}", flush=True)
    for start in range(0, len(deletes), 200):
        batch = deletes[start:start + 200]
        response = session.post(f"{api}/annotations-3d/bulk-delete",
                                json={"annotation_ids": batch}, timeout=180)
        response.raise_for_status()
        print(f"deleted {start + len(batch)}/{len(deletes)}", flush=True)
    response = session.get(
        f"{api}/annotations-3d?task_id={task}&taxonomy_id={taxonomy}", timeout=180
    )
    response.raise_for_status()
    final = {row["id"]: row for row in response.json()}
    failures = []
    for update in updates:
        if update["id"] not in final or abs(
                final[update["id"]]["data"]["center"]["z"]
                - update["data"]["center"]["z"]) > 1e-5:
            failures.append(update["id"])
    failures.extend(annotation_id for annotation_id in deletes if annotation_id in final)
    if failures:
        raise RuntimeError(f"verification failed for {len(failures)} actions")
    print(json.dumps({"verified_updates": len(updates), "verified_deletes": len(deletes),
                      "final_annotations": len(final)}, indent=2))


if __name__ == "__main__":
    main()
