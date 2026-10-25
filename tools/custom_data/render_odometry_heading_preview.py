#!/usr/bin/env python3
"""Render camera/LiDAR previews for odometry-derived heading candidates."""

import json
import math
import os
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass


PLAN = Path(os.getenv("CALIPER_ODOMETRY_PLAN", "/home/mostafa/bag-2026-09-16-odometry-heading-plan.json"))
VALIDATION = Path(os.getenv("CALIPER_ODOMETRY_VALIDATION", "/home/mostafa/bag-2026-09-16-odometry-heading-validation.json"))
OUTPUT = Path(os.getenv("CALIPER_ODOMETRY_PREVIEW", "/home/mostafa/bag-2026-09-16-odometry-heading-preview.jpg"))
DETAIL = Path(os.getenv("CALIPER_ODOMETRY_PREVIEW_DETAIL", "/home/mostafa/bag-2026-09-16-odometry-heading-preview.json"))


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


def rectangle(image, box, color, thickness=3):
    x, y, _, length, width, _, yaw = box
    local = np.array([[length, width], [length, -width],
                      [-length, -width], [-length, width]]) / 2
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])
    corners = local @ rotation.T + [x, y]
    scale = 18.0
    pixels = np.rint((corners - [x, y]) * [scale, -scale] + [170, 170]).astype(int)
    cv2.polylines(image, [pixels], True, color, thickness, cv2.LINE_AA)
    front = np.rint((np.array([math.cos(yaw), math.sin(yaw)]) * length / 2) *
                    [scale, -scale] + [170, 170]).astype(int)
    cv2.arrowedLine(image, (170, 170), tuple(front), color, thickness,
                    cv2.LINE_AA, tipLength=.25)


def main():
    if VALIDATION.exists():
        rows = [row for row in json.loads(VALIDATION.read_text())["records"]
                if row["accepted"]]
    else:
        rows = [row for row in json.loads(PLAN.read_text())["candidates"]
                if row["axis_change_deg"] >= 20]
    rows = sorted(rows, key=lambda row: row["axis_change_deg"], reverse=True)
    samples = camera_pass.metadata_samples()
    tiles, details = [], []
    for number, row in enumerate(rows, 1):
        sample = samples[row["frame_index"]]
        old = np.asarray(row["box"], float)
        new = old.copy()
        new[6] = row["new_yaw"]
        choices = []
        matched = {match["camera"]: match for match in row.get("camera_matches", [])}
        for name, camera in sample["cam_info"]["cam_fisheye"].items():
            bbox = camera_pass.projected_bbox(old, camera)
            if bbox is None:
                continue
            area = max(0, bbox[2] - bbox[0]) * max(0, bbox[3] - bbox[1])
            match = matched.get(name)
            if match:
                detection = match["detection"]
                target = detection["xyxy"]
                area = ((target[2] - target[0]) * (target[3] - target[1]) *
                        detection["confidence"])
            choices.append((area, name, camera, bbox, match))
        if not choices:
            continue
        _, name, camera, bbox, match = max(choices)
        image = cv2.imread(str(camera_pass.image_path(sample, name)))
        new_bbox = camera_pass.projected_bbox(new, camera)
        union = bbox if new_bbox is None else np.array([
            min(bbox[0], new_bbox[0]), min(bbox[1], new_bbox[1]),
            max(bbox[2], new_bbox[2]), max(bbox[3], new_bbox[3])])
        if match:
            target = np.asarray(match["detection"]["xyxy"], float)
            union = np.array([min(union[0], target[0]), min(union[1], target[1]),
                              max(union[2], target[2]), max(union[3], target[3])])
        x1, y1, x2, y2 = union
        pad_x, pad_y = max(100, x2 - x1), max(90, y2 - y1)
        left, top = max(0, int(x1 - pad_x)), max(0, int(y1 - pad_y))
        right = min(image.shape[1], int(x2 + pad_x))
        bottom = min(image.shape[0], int(y2 + pad_y))
        arrow(image, old, old[6], camera, (0, 220, 0))
        arrow(image, new, new[6], camera, (0, 0, 240))
        if match:
            tx1, ty1, tx2, ty2 = np.rint(match["detection"]["xyxy"]).astype(int)
            cv2.rectangle(image, (tx1, ty1), (tx2, ty2), (230, 200, 0), 5)
        crop = image[top:bottom, left:right]
        if not crop.size:
            continue
        crop = cv2.resize(crop, (640, 400), interpolation=cv2.INTER_AREA)
        canvas = np.full((470, 980, 3), 245, np.uint8)
        canvas[70:, :640] = crop
        bev = np.full((340, 340, 3), 250, np.uint8)
        cv2.line(bev, (170, 0), (170, 339), (210, 210, 210), 1)
        cv2.line(bev, (0, 170), (339, 170), (210, 210, 210), 1)
        rectangle(bev, old, (0, 180, 0))
        rectangle(bev, new, (0, 0, 230))
        canvas[100:440, 640:980] = bev
        cv2.putText(canvas,
                    f"{number} frame {row['frame_index']} {row['class_id']}  axis {row['axis_change_deg']:.0f} deg",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .62, (20, 20, 20), 2, cv2.LINE_AA)
        evidence = (f"speed {row['speed_mps']:.1f} m/s  fit {row['fit_residual_m']:.2f} m"
                    if "speed_mps" in row else
                    f"{row.get('source', 'axis')} score {row.get('score', 0):.2f} dist {row.get('distance_m', 0):.2f} m")
        cv2.putText(canvas,
                    f"GREEN current | RED proposed | CYAN YOLO  {evidence}",
                    (10, 56), cv2.FONT_HERSHEY_SIMPLEX, .48, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(canvas)
        details.append({**row, "review_number": number, "camera": name})
    blank = np.full((470, 980, 3), 245, np.uint8)
    outputs = []
    for page, start in enumerate(range(0, len(tiles), 24), 1):
        page_tiles = tiles[start:start + 24]
        while len(page_tiles) % 2:
            page_tiles.append(blank.copy())
        montage = np.vstack([np.hstack(page_tiles[index:index + 2])
                             for index in range(0, len(page_tiles), 2)])
        destination = OUTPUT.with_name(f"{OUTPUT.stem}-{page:02d}{OUTPUT.suffix}")
        cv2.imwrite(str(destination), montage, [cv2.IMWRITE_JPEG_QUALITY, 95])
        outputs.append(str(destination))
    DETAIL.write_text(json.dumps(details, indent=2))
    print(json.dumps({"outputs": outputs, "tiles": len(details)}, indent=2))


if __name__ == "__main__":
    main()
