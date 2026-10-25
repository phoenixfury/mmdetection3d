#!/usr/bin/env python3
"""Preview current versus PGD heading on ten random busy frames."""

import json
import math
import os
import random
import sys
import tempfile
from collections import defaultdict
from pathlib import Path

import cv2
import mmengine
import numpy as np
import requests

from mmdet3d.apis import inference_mono_3d_detector, init_model

sys.path.insert(0, "/home/mostafa/mmdetection3d/tools/custom_data")
from pgd_fisheye_heading_pilot import (  # noqa: E402
    API, CHECKPOINT, CONFIG, FOCAL, HEIGHT, PLAN, TASK_ID, TAXONOMY_ID,
    WIDTH, annotation_id, choose_prediction, tangent_image, wrap)

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass  # noqa: E402


BASE = Path("/home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_16_09_58_56_converted/fisheye3dod_inference_2hz")
INFO = BASE / "fisheye3dod_infos_test.pkl"
OUTPUT = Path("/home/mostafa/bag-2026-09-16-pgd-busy-frame-preview.jpg")
DETAIL = Path("/home/mostafa/bag-2026-09-16-pgd-busy-frame-preview.json")
VEHICLES = {"car", "truck", "bus"}


def annotation_box(annotation):
    data = annotation["data"]
    return np.array([
        data["center"]["x"], data["center"]["y"], data["center"]["z"],
        data["dimensions"]["length"], data["dimensions"]["width"],
        data["dimensions"]["height"], data["rotation"]["yaw"],
    ], dtype=float)


def draw_box(panel, box, color, scale, origin):
    x, y, _, length, width, _, yaw = box
    local = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                      [-length / 2, -width / 2], [-length / 2, width / 2]])
    rotation = np.array([[math.cos(yaw), -math.sin(yaw)],
                         [math.sin(yaw), math.cos(yaw)]])
    corners = local @ rotation.T + [x, y]
    pixels = np.column_stack((origin[0] - corners[:, 1] * scale,
                              origin[1] - corners[:, 0] * scale)).astype(int)
    cv2.polylines(panel, [pixels], True, color, 2, cv2.LINE_AA)
    center = (int(origin[0] - y * scale), int(origin[1] - x * scale))
    front = (int(origin[0] - (y + math.sin(yaw) * length / 2) * scale),
             int(origin[1] - (x + math.cos(yaw) * length / 2) * scale))
    cv2.arrowedLine(panel, center, front, color, 3, cv2.LINE_AA, tipLength=0.28)
    return center


def render_panel(frame_index, entries, points):
    size, scale = 650, 8.0
    origin = np.array([size / 2, size * 0.72])
    panel = np.full((size, size, 3), 250, dtype=np.uint8)
    mask = ((np.abs(points[:, 0]) < 40) & (np.abs(points[:, 1]) < 40))
    cloud = points[mask][::3]
    pixels = np.column_stack((origin[0] - cloud[:, 1] * scale,
                              origin[1] - cloud[:, 0] * scale)).astype(int)
    valid = ((pixels[:, 0] >= 0) & (pixels[:, 0] < size) &
             (pixels[:, 1] >= 35) & (pixels[:, 1] < size))
    panel[pixels[valid, 1], pixels[valid, 0]] = (195, 195, 195)
    cv2.putText(panel, f"frame {frame_index} | {len(entries)} vehicles",
                (12, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (15, 15, 15), 2)
    for number, entry in enumerate(entries, 1):
        old_box = entry["box"].copy()
        center = draw_box(panel, old_box, (0, 90, 255), scale, origin)
        if entry.get("pgd_yaw") is not None:
            pgd_box = old_box.copy()
            pgd_box[6] = entry["pgd_yaw"]
            draw_box(panel, pgd_box, (255, 170, 0), scale, origin)
            label = f"{number}: {entry['difference_deg']:.0f}d s{entry['pgd_score']:.0f}"
        else:
            label = f"{number}: no PGD"
        cv2.putText(panel, label, (center[0] + 4, center[1] - 4),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.42, (30, 30, 30), 1,
                    cv2.LINE_AA)
    cv2.putText(panel, "ORANGE=current LiDAR yaw", (12, size - 32),
                cv2.FONT_HERSHEY_SIMPLEX, 0.48, (0, 90, 255), 1)
    cv2.putText(panel, "BLUE=PGD image yaw | label: difference, score",
                (12, size - 12), cv2.FONT_HERSHEY_SIMPLEX, 0.43,
                (255, 120, 0), 1)
    return panel


def main():
    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    annotations_response = session.get(
        API + f"/annotations-3d?task_id={TASK_ID}&taxonomy_id={TAXONOMY_ID}",
        timeout=180)
    annotations_response.raise_for_status()
    frames_response = session.get(
        API + f"/tasks/{TASK_ID}/frames?include_context=false", timeout=180)
    frames_response.raise_for_status()
    frame_index = {item["id"]: int(item["frame_index"])
                   for item in frames_response.json()["frames"]}
    annotations = {item["id"]: item for item in annotations_response.json()
                   if item["class_id"] in VEHICLES}
    proposals = {annotation_id(item): item
                 for item in json.loads(PLAN.read_text())["proposals"]}
    by_frame = defaultdict(list)
    for aid, annotation in annotations.items():
        if aid in proposals:
            by_frame[frame_index[annotation["frame_id"]]].append(
                (annotation, proposals[aid]))
    ranked = sorted(by_frame, key=lambda index: len(by_frame[index]), reverse=True)
    busy_pool = ranked[:min(100, len(ranked))]
    selected_frames = sorted(random.Random(20261019).sample(busy_pool, 10))

    samples = camera_pass.metadata_samples()
    prepared = []
    with tempfile.TemporaryDirectory(prefix="pgd_busy_") as directory:
        directory = Path(directory)
        for frame in selected_frames:
            for annotation, proposal in by_frame[frame]:
                detection = max(
                    proposal["detections"],
                    key=lambda item: ((item["xyxy"][2] - item["xyxy"][0]) *
                                      (item["xyxy"][3] - item["xyxy"][1])))
                camera = samples[frame]["cam_info"]["cam_fisheye"][detection["camera"]]
                image = directory / f"crop_{len(prepared):04d}.jpg"
                rotation = tangent_image(detection, camera, image)
                prepared.append({
                    "frame": frame, "annotation": annotation,
                    "proposal": proposal, "detection": detection,
                    "camera": camera, "rotation": rotation, "image": image,
                    "box": annotation_box(annotation),
                })

        model = init_model(CONFIG, CHECKPOINT, device="cuda:0")
        for start in range(0, len(prepared), 4):
            batch = prepared[start:start + 4]
            info_path = directory / "batch.pkl"
            mmengine.dump({"metainfo": {"classes": ["Pedestrian", "Cyclist", "Car"]},
                           "data_list": [{"images": {"CAM2": {
                               "img_path": str(item["image"]),
                               "cam2img": [[FOCAL, 0, WIDTH / 2, 0],
                                           [0, FOCAL, HEIGHT / 2, 0],
                                           [0, 0, 1, 0], [0, 0, 0, 1]],
                           }}} for item in batch]}, info_path)
            results = inference_mono_3d_detector(
                model, [str(item["image"]) for item in batch], str(info_path), "CAM2")
            for item, result in zip(batch, results):
                prediction = choose_prediction(result)
                item["pgd_yaw"] = None
                if prediction is None:
                    continue
                box, score, _, _ = prediction
                heading_virtual = np.array([math.cos(box[6]), 0.0,
                                            -math.sin(box[6])])
                heading_lidar = (np.asarray(item["camera"]["cam2lidar"])[:3, :3]
                                 @ item["rotation"] @ heading_virtual)
                pgd_yaw = math.atan2(heading_lidar[1], heading_lidar[0])
                difference = abs(wrap(pgd_yaw - item["box"][6]))
                item.update(pgd_yaw=pgd_yaw, pgd_score=float(score),
                            difference_deg=math.degrees(difference))
            print(f"processed {min(start + len(batch), len(prepared))}/{len(prepared)}",
                  flush=True)

    metadata = {int(item["sample_idx"]): item
                for item in mmengine.load(INFO)["data_list"]}
    frame_entries = defaultdict(list)
    serializable = []
    for item in prepared:
        frame_entries[item["frame"]].append(item)
        serializable.append({
            "frame_index": item["frame"],
            "annotation_id": item["annotation"]["id"],
            "class_id": item["annotation"]["class_id"],
            "current_yaw": float(item["box"][6]),
            "pgd_yaw": item.get("pgd_yaw"),
            "pgd_score": item.get("pgd_score"),
            "difference_deg": item.get("difference_deg"),
        })
    panels = []
    for frame in selected_frames:
        path = (BASE / metadata[frame]["pc_info"]["pts_path"]).resolve()
        points = np.fromfile(path, np.float32).reshape(-1, 4)[:, :3]
        panels.append(render_panel(frame, frame_entries[frame], points))
    montage = np.vstack((np.hstack(panels[:5]), np.hstack(panels[5:])))
    cv2.imwrite(str(OUTPUT), montage, [cv2.IMWRITE_JPEG_QUALITY, 94])
    DETAIL.write_text(json.dumps({
        "seed": 20261019, "selected_frames": selected_frames,
        "vehicle_counts": {str(frame): len(frame_entries[frame])
                           for frame in selected_frames},
        "records": serializable,
    }, indent=2))
    print(json.dumps({"preview": str(OUTPUT), "frames": selected_frames,
                      "objects": len(prepared)}, indent=2))


if __name__ == "__main__":
    main()
