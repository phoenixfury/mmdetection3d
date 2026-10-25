#!/usr/bin/env python3
"""Run official nuScenes CenterPoint over every bag frame, batch size four."""

import json
import os
from pathlib import Path

import mmengine
import numpy as np
from mmengine.config import Config

from mmdet3d.apis import inference_detector, init_model


ROOT = Path("/home/mostafa/mmdetection3d")
BASE = Path(os.getenv(
    "CALIPER_BAG_BASE",
    str(ROOT / "data/Custom/BAG_2026_09_16_09_58_56_converted/fisheye3dod_inference_2hz"),
))
INFO = BASE / "fisheye3dod_infos_test.pkl"
CONFIG = ROOT / "configs/centerpoint/centerpoint_voxel01_second_secfpn_head-circlenms_8xb4-cyclic-20e_nus-3d.py"
CHECKPOINT = ROOT / "checkpoints/centerpoint_nus_voxel01.pth"
OUTPUT = Path(os.getenv(
    "CALIPER_CENTERPOINT_OUTPUT",
    "/home/mostafa/bag-2026-09-16-centerpoint-full.json",
))
VEHICLES = {0: "car", 1: "truck", 3: "bus"}


def main():
    samples = {int(x["sample_idx"]): x for x in mmengine.load(INFO)["data_list"]}
    frames = sorted(samples)
    cfg = Config.fromfile(CONFIG)
    cfg.test_dataloader.dataset.pipeline = [
        step for step in cfg.test_dataloader.dataset.pipeline
        if step["type"] != "LoadPointsFromMultiSweeps"
    ]
    model = init_model(cfg, str(CHECKPOINT), device="cuda:0")
    detections = {}
    for start in range(0, len(frames), 4):
        batch_frames = frames[start:start + 4]
        clouds = []
        for frame in batch_frames:
            path = (BASE / samples[frame]["pc_info"]["pts_path"]).resolve()
            points = np.fromfile(path, np.float32).reshape(-1, 4)
            clouds.append(np.column_stack((points, np.zeros(len(points), np.float32))))
        results, _ = inference_detector(model, clouds)
        for frame, result in zip(batch_frames, results):
            pred = result.pred_instances_3d
            boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
            scores = pred.scores_3d.detach().cpu().numpy()
            labels = pred.labels_3d.detach().cpu().numpy()
            detections[str(frame)] = [
                {"class_id": VEHICLES[int(label)], "score": float(score),
                 "box": box[:7].astype(float).tolist()}
                for box, score, label in zip(boxes, scores, labels)
                if int(label) in VEHICLES and score >= .2
            ]
        if start % 40 == 0 or start + 4 >= len(frames):
            OUTPUT.write_text(json.dumps({"completed": min(start + 4, len(frames)),
                                          "detections": detections}))
            print(f"processed {min(start + 4, len(frames))}/{len(frames)}", flush=True)
    OUTPUT.write_text(json.dumps({"completed": len(frames),
                                  "detections": detections}, indent=2))


if __name__ == "__main__":
    main()
