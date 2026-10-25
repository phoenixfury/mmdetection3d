#!/usr/bin/env python3
"""Compare PGD image heading with stored LiDAR-supported cuboid yaw."""

import argparse
import json
import math
import os
import sys
import tempfile
import uuid
from pathlib import Path

import cv2
import mmengine
import numpy as np
import requests

from mmdet3d.apis import inference_mono_3d_detector, init_model

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass


TASK_ID = "b6da9787-6615-4a13-8747-5466ff1a6aed"
TAXONOMY_ID = "981f59a2-75c6-41f1-a204-8d177ada63ce"
API = "http://localhost:3000/api/v1"
CONFIG = "configs/pgd/pgd_r101-caffe_fpn_head-gn_4xb3-4x_kitti-mono3d.py"
CHECKPOINT = "checkpoints/pgd_kitti.pth"
PLAN = Path("/home/mostafa/bag-2026-09-16-pre-robust-yaw-plan.json")
OUTPUT = Path("/home/mostafa/bag-2026-09-16-pgd-heading-pilot.json")
WIDTH, HEIGHT, FOCAL = 1242, 375, 721.0


def wrap(angle):
    return (angle + math.pi) % (2 * math.pi) - math.pi


def annotation_id(proposal):
    namespace = uuid.UUID("67f0a17c-dfc4-4ea4-b882-e731ff51bd66")
    track_key = min(item["image_track"] for item in proposal["detections"])
    x, y = proposal["box"][:2]
    identity = (
        f"{TASK_ID}:2d-annotation:{proposal['frame_index']}:{track_key}:"
        f"{proposal['class_id']}:{x:.2f}:{y:.2f}"
    )
    return str(uuid.uuid5(namespace, identity))


def tangent_image(detection, camera, destination):
    source = cv2.imread(detection["image"], cv2.IMREAD_COLOR)
    x1, y1, x2, y2 = detection["xyxy"]
    center = ((x1 + x2) / 2, (y1 + y2) / 2)
    z_axis = camera_pass.native_cam2world(center, camera["native_intrinsic"])
    z_axis /= np.linalg.norm(z_axis)
    x_axis = np.cross(np.array([0.0, 1.0, 0.0]), z_axis)
    x_axis /= np.linalg.norm(x_axis)
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.column_stack((x_axis, y_axis, z_axis))

    u, v = np.meshgrid(np.arange(WIDTH), np.arange(HEIGHT))
    rays = np.stack(((u - WIDTH / 2) / FOCAL,
                     (v - HEIGHT / 2) / FOCAL,
                     np.ones_like(u)), axis=0).reshape(3, -1)
    pixels = camera_pass.native_world2cam(rotation @ rays,
                                           camera["native_intrinsic"])
    rectified = cv2.remap(
        source, pixels[:, 0].reshape(HEIGHT, WIDTH).astype(np.float32),
        pixels[:, 1].reshape(HEIGHT, WIDTH).astype(np.float32),
        interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)
    cv2.imwrite(str(destination), rectified, [cv2.IMWRITE_JPEG_QUALITY, 90])
    return rotation


def choose_prediction(result):
    instances = result.pred_instances_3d
    boxes = instances.bboxes_3d.tensor.detach().cpu().numpy()
    scores = instances.scores_3d.detach().cpu().numpy()
    labels = instances.labels_3d.detach().cpu().numpy()
    candidates = []
    for box, score, label in zip(boxes, scores, labels):
        if label != 2 or score < 0.03 or box[2] <= 0:
            continue
        u = FOCAL * box[0] / box[2] + WIDTH / 2
        v = FOCAL * box[1] / box[2] + HEIGHT / 2
        center_error = math.hypot((u - WIDTH / 2) / WIDTH,
                                  (v - HEIGHT / 2) / HEIGHT)
        candidates.append((center_error - 0.15 * float(score), box, score, u, v))
    if not candidates:
        return None
    _, box, score, u, v = min(candidates, key=lambda item: item[0])
    if abs(u - WIDTH / 2) > WIDTH * 0.30 or abs(v - HEIGHT / 2) > HEIGHT * 0.45:
        return None
    return box, float(score), float(u), float(v)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="cuda:0")
    args = parser.parse_args()
    if args.batch_size != 4:
        raise ValueError("This pilot is fixed to batch size 4 for GPU safety")

    session = requests.Session()
    session.headers["Authorization"] = "Bearer " + os.environ["CALIPER_TOKEN"]
    response = session.get(
        API + f"/annotations-3d?task_id={TASK_ID}&taxonomy_id={TAXONOMY_ID}",
        timeout=180)
    response.raise_for_status()
    annotations = {item["id"]: item for item in response.json()}
    samples = camera_pass.metadata_samples()
    proposals = json.loads(PLAN.read_text())["proposals"]
    candidates = []
    used_frames = set()
    for proposal in proposals:
        aid = annotation_id(proposal)
        if (aid not in annotations or proposal["class_id"] not in {"car", "truck", "bus"}
                or proposal["frame_index"] in used_frames):
            continue
        detection = max(
            proposal["detections"],
            key=lambda item: ((item["xyxy"][2] - item["xyxy"][0]) *
                              (item["xyxy"][3] - item["xyxy"][1])))
        if detection["confidence"] < 0.25:
            continue
        candidates.append((proposal, detection, annotations[aid]))
        used_frames.add(proposal["frame_index"])
    indices = np.linspace(0, len(candidates) - 1,
                          min(args.samples, len(candidates)), dtype=int)
    selected = [candidates[index] for index in indices]

    model = init_model(CONFIG, CHECKPOINT, device=args.device)
    records = []
    with tempfile.TemporaryDirectory(prefix="pgd_fisheye_") as directory:
        directory = Path(directory)
        prepared = []
        for index, (proposal, detection, annotation) in enumerate(selected):
            camera = samples[proposal["frame_index"]]["cam_info"]["cam_fisheye"][detection["camera"]]
            image = directory / f"crop_{index:04d}.jpg"
            rotation = tangent_image(detection, camera, image)
            prepared.append((proposal, detection, annotation, camera, rotation, image))

        for start in range(0, len(prepared), args.batch_size):
            batch = prepared[start:start + args.batch_size]
            info = {
                "metainfo": {"classes": ["Pedestrian", "Cyclist", "Car"]},
                "data_list": [{"images": {"CAM2": {
                    "img_path": str(item[5]),
                    "cam2img": [[FOCAL, 0, WIDTH / 2, 0],
                                [0, FOCAL, HEIGHT / 2, 0],
                                [0, 0, 1, 0], [0, 0, 0, 1]],
                }}} for item in batch],
            }
            info_path = directory / "batch.pkl"
            mmengine.dump(info, info_path)
            results = inference_mono_3d_detector(
                model, [str(item[5]) for item in batch], str(info_path), "CAM2")
            for item, result in zip(batch, results):
                proposal, detection, annotation, camera, rotation, _ = item
                prediction = choose_prediction(result)
                if prediction is None:
                    continue
                box, score, image_u, image_v = prediction
                # Camera yaw is zero on +x and decreases toward +z.
                heading_virtual = np.array([math.cos(box[6]), 0.0, -math.sin(box[6])])
                heading_lidar = np.asarray(camera["cam2lidar"])[:3, :3] @ rotation @ heading_virtual
                image_yaw = math.atan2(heading_lidar[1], heading_lidar[0])
                stored_yaw = float(annotation["data"]["rotation"]["yaw"])
                difference = abs(wrap(image_yaw - stored_yaw))
                axis_difference = min(difference, abs(math.pi - difference))
                records.append({
                    "annotation_id": annotation["id"],
                    "frame_index": proposal["frame_index"],
                    "camera": detection["camera"],
                    "class_id": annotation["class_id"],
                    "pgd_score": score,
                    "stored_yaw": stored_yaw,
                    "image_yaw": image_yaw,
                    "difference_deg": math.degrees(difference),
                    "axis_difference_deg": math.degrees(axis_difference),
                    "image_suggests_flip": difference > math.pi / 2,
                    "pgd_center_pixel": [image_u, image_v],
                })
            print(f"processed {min(start + len(batch), len(prepared))}/{len(prepared)}", flush=True)

    # PGD uses a depth-uncertainty score that is not probability-normalized;
    # the official KITTI demo uses 8 as its display threshold.
    strong = [item for item in records if item["pgd_score"] >= 8.0]
    summary = {
        "requested": len(selected),
        "matched": len(records),
        "strong": len(strong),
        "batch_size": args.batch_size,
        "median_axis_difference_deg": float(np.median(
            [item["axis_difference_deg"] for item in strong])) if strong else None,
        "p90_axis_difference_deg": float(np.quantile(
            [item["axis_difference_deg"] for item in strong], 0.9)) if strong else None,
        "image_suggested_flips": sum(item["image_suggests_flip"] for item in strong),
    }
    OUTPUT.write_text(json.dumps({"summary": summary, "records": records}, indent=2))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
