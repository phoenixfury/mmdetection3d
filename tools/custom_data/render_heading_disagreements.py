#!/usr/bin/env python3
"""Render fisheye crops for manual current/CenterPoint/PGD heading review."""

import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import requests

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass
sys.path.insert(0, "/home/mostafa/mmdetection3d/tools/custom_data")
from pgd_fisheye_heading_pilot import API, PLAN, TASK_ID, TAXONOMY_ID, annotation_id


INPUT = Path("/home/mostafa/bag-2026-09-16-centerpoint-busy-frames.json")
OUTPUT = Path("/home/mostafa/bag-2026-09-16-heading-disagreement-crops.jpg")
DETAIL = Path("/home/mostafa/bag-2026-09-16-heading-disagreement-crops.json")


def project(point, camera):
    xyz = (np.asarray(camera["lidar2cam"]) @ np.r_[point, 1.0])[:3, None]
    return camera_pass.native_world2cam(xyz, camera["native_intrinsic"])[0]


def arrow(image, box, yaw, camera, color):
    center = np.asarray(box[:3], float)
    front = center + [math.cos(yaw) * box[3] / 2,
                      math.sin(yaw) * box[3] / 2, 0]
    p0, p1 = project(center, camera), project(front, camera)
    if np.isfinite(np.r_[p0, p1]).all():
        cv2.arrowedLine(image, tuple(np.rint(p0).astype(int)),
                        tuple(np.rint(p1).astype(int)), color, 8,
                        cv2.LINE_AA, tipLength=.25)


def main():
    result = json.loads(INPUT.read_text())
    rows = [x for x in result["comparisons"]
            if x["centerpoint_score"] >= .4
            and x["center_distance_m"] <= 1.5
            and x["centerpoint_difference_deg"] >= 45]
    proposals = {annotation_id(x): x
                 for x in json.loads(PLAN.read_text())["proposals"]}
    samples = camera_pass.metadata_samples()

    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    response = session.get(
        API + f"/annotations-3d?task_id={TASK_ID}&taxonomy_id={TAXONOMY_ID}",
        timeout=180)
    response.raise_for_status()
    annotations = {x["id"]: x for x in response.json()}

    tiles, details = [], []
    for number, row in enumerate(rows, 1):
        proposal = proposals[row["annotation_id"]]
        detection = max(proposal["detections"], key=lambda x:
                        (x["xyxy"][2] - x["xyxy"][0]) *
                        (x["xyxy"][3] - x["xyxy"][1]))
        sample = samples[row["frame_index"]]
        camera = sample["cam_info"]["cam_fisheye"][detection["camera"]]
        image = cv2.imread(str(camera_pass.image_path(sample, detection["camera"])))
        data = annotations[row["annotation_id"]]["data"]
        box = np.array([data["center"]["x"], data["center"]["y"],
                        data["center"]["z"], data["dimensions"]["length"],
                        data["dimensions"]["width"], data["dimensions"]["height"],
                        data["rotation"]["yaw"]], float)
        toward_camera = np.asarray(camera["cam2lidar"], float)[:2, 3] - box[:2]
        toward_camera /= max(np.linalg.norm(toward_camera), 1e-9)
        facing = lambda yaw: float(np.dot([math.cos(yaw), math.sin(yaw)],
                                         toward_camera))
        details.append({**row, "review_number": number,
                        "current_faces_camera": facing(row["current_yaw"]),
                        "centerpoint_faces_camera": facing(row["centerpoint_yaw"]),
                        "pgd_faces_camera": (facing(row["pgd_yaw"])
                                             if row.get("pgd_yaw") is not None else None),
                        "camera": detection["camera"]})
        x1, y1, x2, y2 = map(int, detection["xyxy"])
        pad_x = max(120, x2 - x1)
        pad_y = max(100, y2 - y1)
        left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
        right, bottom = min(image.shape[1], x2 + pad_x), min(image.shape[0], y2 + pad_y)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 230, 230), 5)
        arrow(image, box, row["current_yaw"], camera, (0, 220, 0))
        arrow(image, box, row["centerpoint_yaw"], camera, (255, 180, 0))
        if row.get("pgd_yaw") is not None and row.get("pgd_score", 0) >= 8:
            arrow(image, box, row["pgd_yaw"], camera, (220, 0, 220))
        crop = image[top:bottom, left:right]
        crop = cv2.resize(crop, (640, 430), interpolation=cv2.INTER_AREA)
        canvas = np.full((500, 640, 3), 245, np.uint8)
        canvas[70:] = crop
        title = (f"{number} frame {row['frame_index']} {row['class_id']} "
                 f"CP diff {row['centerpoint_difference_deg']:.0f} score {row['centerpoint_score']:.2f}")
        cv2.putText(canvas, title, (10, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    .58, (20, 20, 20), 2, cv2.LINE_AA)
        cv2.putText(canvas, "GREEN current | CYAN CenterPoint | MAGENTA PGD",
                    (10, 55), cv2.FONT_HERSHEY_SIMPLEX, .48,
                    (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(canvas)

    blank = np.full((500, 640, 3), 245, np.uint8)
    while len(tiles) % 4:
        tiles.append(blank.copy())
    montage = np.vstack([np.hstack(tiles[i:i + 4])
                         for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(OUTPUT), montage, [cv2.IMWRITE_JPEG_QUALITY, 96])
    DETAIL.write_text(json.dumps(details, indent=2))
    print(json.dumps({"output": str(OUTPUT), "candidates": len(rows)}))


if __name__ == "__main__":
    main()
