#!/usr/bin/env python3
"""Render the proposed remaining-containment fixes for manual inspection."""

import json
import sys
from pathlib import Path

import cv2
import mmengine
import numpy as np

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass
sys.path.insert(0, "/home/mostafa/mmdetection3d/tools/custom_data")
from pgd_fisheye_heading_pilot import PLAN, annotation_id
from render_full_position_review import draw_box


BASE = Path("/home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_16_09_58_56_converted/fisheye3dod_inference_2hz")
INFO = BASE / "fisheye3dod_infos_test.pkl"
FIXES = Path("/home/mostafa/bag-2026-09-16-containment-fixes.json")
INDEX = Path("/home/mostafa/bag-2026-09-16-full-manual-review.json")
OUTPUT = Path("/home/mostafa/bag-2026-09-16-containment-fixes-preview.jpg")


def main():
    fixes = json.loads(FIXES.read_text())
    updates = {x["id"]: x for x in fixes["updates"]}
    old = {x["annotation_id"]: x for x in json.loads(INDEX.read_text())["records"]}
    proposals = {annotation_id(x): x for x in json.loads(PLAN.read_text())["proposals"]}
    samples = camera_pass.metadata_samples()
    metadata = {int(x["sample_idx"]): x
                for x in mmengine.load(INFO)["data_list"]}
    tiles = []
    for record in fixes["records"]:
        if not record["fixed"]:
            continue
        aid, frame = record["annotation_id"], record["frame_index"]
        row, update, proposal = old[aid], updates[aid], proposals[aid]
        detection = max(proposal["detections"], key=lambda x:
                        (x["xyxy"][2] - x["xyxy"][0]) *
                        (x["xyxy"][3] - x["xyxy"][1]))
        sample = samples[frame]
        camera = sample["cam_info"]["cam_fisheye"][detection["camera"]]
        image = cv2.imread(str(camera_pass.image_path(sample, detection["camera"])))
        x1, y1, x2, y2 = map(int, detection["xyxy"])
        pad_x, pad_y = max(150, x2 - x1), max(120, y2 - y1)
        left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
        right, bottom = min(image.shape[1], x2 + pad_x), min(image.shape[0], y2 + pad_y)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 220), 6)
        new = update["data"]
        new_box = np.array([new["center"]["x"], new["center"]["y"], new["center"]["z"],
                            new["dimensions"]["length"], new["dimensions"]["width"],
                            new["dimensions"]["height"], new["rotation"]["yaw"]])
        old_box = np.asarray(row["box"], float)
        for b, color in ((old_box, (0, 90, 255)), (new_box, (0, 190, 0))):
            projected = camera_pass.projected_bbox(b, camera)
            if projected is not None:
                q = np.rint(projected).astype(int)
                cv2.rectangle(image, tuple(q[:2]), tuple(q[2:]), color, 5)
        crop = cv2.resize(image[top:bottom, left:right], (700, 420), interpolation=cv2.INTER_AREA)
        path = (BASE / metadata[frame]["pc_info"]["pts_path"]).resolve()
        points = np.fromfile(path, np.float32).reshape(-1, 4)[:, :3]
        center = new_box[:2]
        mask = ((np.abs(points[:, 0] - center[0]) < 12)
                & (np.abs(points[:, 1] - center[1]) < 12))
        cloud = points[mask]
        bev = np.full((280, 700, 3), 248, np.uint8)
        scale, origin = 10.0, np.array([350., 140.])
        pix = np.column_stack((origin[0] - (cloud[:, 1] - center[1]) * scale,
                               origin[1] - (cloud[:, 0] - center[0]) * scale)).astype(int)
        valid = ((pix[:, 0] >= 0) & (pix[:, 0] < 700)
                 & (pix[:, 1] >= 0) & (pix[:, 1] < 280))
        bev[pix[valid, 1], pix[valid, 0]] = 170
        for b, color in ((old_box.copy(), (0, 90, 255)), (new_box.copy(), (0, 190, 0))):
            b[:2] -= center
            draw_box(bev, b, color, scale, origin)
        tile = np.vstack((crop, bev))
        cv2.rectangle(tile, (0, 0), (699, 38), (245, 245, 245), -1)
        cv2.putText(tile, f"frame {frame} shift {record['center_shift_m']:.1f}m",
                    (10, 27), cv2.FONT_HERSHEY_SIMPLEX, .65, (20, 20, 20), 2)
        tiles.append(tile)
    blank = np.full_like(tiles[0], 245)
    while len(tiles) % 4:
        tiles.append(blank.copy())
    montage = np.vstack([np.hstack(tiles[i:i + 4])
                         for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(OUTPUT), montage, [cv2.IMWRITE_JPEG_QUALITY, 96])
    print(OUTPUT)


if __name__ == "__main__":
    main()
