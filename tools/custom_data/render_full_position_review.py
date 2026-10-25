#!/usr/bin/env python3
"""Render one camera/BEV tile per track in the position-refit plan."""

import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import mmengine
import numpy as np

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass
sys.path.insert(0, "/home/mostafa/mmdetection3d/tools/custom_data")
from pgd_fisheye_heading_pilot import PLAN, annotation_id


BASE = Path("/home/mostafa/mmdetection3d/data/Custom/BAG_2026_09_16_09_58_56_converted/fisheye3dod_inference_2hz")
INFO = BASE / "fisheye3dod_infos_test.pkl"
INPUT = Path("/home/mostafa/bag-2026-09-16-full-position-refit-plan.json")
OUTDIR = Path("/home/mostafa/bag-2026-09-16-full-position-review")


def draw_box(panel, box, color, scale, origin):
    x, y, _, length, width, _, yaw = box
    local = np.array([[length / 2, width / 2], [length / 2, -width / 2],
                      [-length / 2, -width / 2], [-length / 2, width / 2]])
    rot = np.array([[math.cos(yaw), -math.sin(yaw)],
                    [math.sin(yaw), math.cos(yaw)]])
    corners = local @ rot.T + [x, y]
    pixels = np.column_stack((origin[0] - corners[:, 1] * scale,
                              origin[1] - corners[:, 0] * scale)).astype(int)
    cv2.polylines(panel, [pixels], True, color, 3, cv2.LINE_AA)


def main():
    candidates = json.loads(INPUT.read_text())["candidates"]
    tracks = defaultdict(list)
    for row in candidates:
        tracks[row["track_id"]].append(row)
    representatives = sorted(
        (max(rows, key=lambda x: x["shift_m"]) for rows in tracks.values()),
        key=lambda x: (x["frame_index"], x["annotation_id"]))
    proposals = {annotation_id(x): x
                 for x in json.loads(PLAN.read_text())["proposals"]}
    samples = camera_pass.metadata_samples()
    metadata = {int(x["sample_idx"]): x
                for x in mmengine.load(INFO)["data_list"]}
    OUTDIR.mkdir(exist_ok=True)
    tiles = []
    for number, row in enumerate(representatives, 1):
        proposal = proposals[row["annotation_id"]]
        detection = max(proposal["detections"], key=lambda x:
                        (x["xyxy"][2] - x["xyxy"][0]) *
                        (x["xyxy"][3] - x["xyxy"][1]))
        frame = row["frame_index"]
        sample = samples[frame]
        camera = sample["cam_info"]["cam_fisheye"][detection["camera"]]
        image = cv2.imread(str(camera_pass.image_path(sample, detection["camera"])))
        x1, y1, x2, y2 = map(int, detection["xyxy"])
        pad_x, pad_y = max(100, x2 - x1), max(80, y2 - y1)
        left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
        right, bottom = min(image.shape[1], x2 + pad_x), min(image.shape[0], y2 + pad_y)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 220), 5)
        for b, color in ((row["box"], (0, 90, 255)),
                         (row["new_box"], (0, 190, 0))):
            projected = camera_pass.projected_bbox(np.asarray(b), camera)
            if projected is not None:
                q = np.rint(projected).astype(int)
                cv2.rectangle(image, tuple(q[:2]), tuple(q[2:]), color, 4)
        crop = cv2.resize(image[top:bottom, left:right], (600, 300),
                          interpolation=cv2.INTER_AREA)

        point_path = (BASE / metadata[frame]["pc_info"]["pts_path"]).resolve()
        points = np.fromfile(point_path, np.float32).reshape(-1, 4)[:, :3]
        center = np.asarray(row["new_box"][:2])
        mask = ((np.abs(points[:, 0] - center[0]) < 8)
                & (np.abs(points[:, 1] - center[1]) < 8))
        cloud = points[mask]
        bev = np.full((200, 600, 3), 248, np.uint8)
        scale, origin = 12.0, np.array([300.0, 100.0])
        pix = np.column_stack((origin[0] - (cloud[:, 1] - center[1]) * scale,
                               origin[1] - (cloud[:, 0] - center[0]) * scale)).astype(int)
        valid = ((pix[:, 0] >= 0) & (pix[:, 0] < 600)
                 & (pix[:, 1] >= 0) & (pix[:, 1] < 200))
        bev[pix[valid, 1], pix[valid, 0]] = (175, 175, 175)
        local_old = np.asarray(row["box"], float).copy()
        local_new = np.asarray(row["new_box"], float).copy()
        local_old[:2] -= center
        local_new[:2] -= center
        draw_box(bev, local_old, (0, 90, 255), scale, origin)
        draw_box(bev, local_new, (0, 190, 0), scale, origin)
        tile = np.vstack((crop, bev))
        cv2.rectangle(tile, (0, 0), (599, 39), (245, 245, 245), -1)
        label = (f"{number} f{frame} shift {row['shift_m']:.1f}m "
                 f"CP {row['centerpoint_score']:.2f} P2 {row['parta2_score']:.1f}")
        cv2.putText(tile, label, (8, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    .58, (20, 20, 20), 2, cv2.LINE_AA)
        tiles.append(tile)
    blank = np.full((500, 600, 3), 245, np.uint8)
    outputs = []
    for page, start in enumerate(range(0, len(tiles), 40), 1):
        page_tiles = tiles[start:start + 40]
        while len(page_tiles) < 40:
            page_tiles.append(blank.copy())
        montage = np.vstack([np.hstack(page_tiles[i:i + 4])
                             for i in range(0, 40, 4)])
        output = OUTDIR / f"position-review-{page:02d}.jpg"
        cv2.imwrite(str(output), montage, [cv2.IMWRITE_JPEG_QUALITY, 94])
        outputs.append(str(output))
    (OUTDIR / "index.json").write_text(json.dumps({
        "representatives": [{"review_number": i + 1, **x}
                            for i, x in enumerate(representatives)],
        "sheets": outputs}, indent=2))
    print(json.dumps({"tracks": len(representatives), "sheets": outputs}, indent=2))


if __name__ == "__main__":
    main()
