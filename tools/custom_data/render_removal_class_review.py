#!/usr/bin/env python3
"""Render track representatives needing manual existence/class review."""

import json
import sys
from collections import defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass
sys.path.insert(0, "/home/mostafa/mmdetection3d/tools/custom_data")
from pgd_fisheye_heading_pilot import PLAN, annotation_id


INDEX = Path("/home/mostafa/bag-2026-09-16-full-manual-review.json")
OUTDIR = Path("/home/mostafa/bag-2026-09-16-removal-class-review")


def main():
    records = json.loads(INDEX.read_text())["records"]
    proposals = {annotation_id(x): x
                 for x in json.loads(PLAN.read_text())["proposals"]}
    samples = camera_pass.metadata_samples()
    tracks = defaultdict(list)
    for row in records:
        tracks[row["track_id"]].append(row)
    review_tracks = []
    for track_id, rows in tracks.items():
        low_short = (len(rows) <= 2
                     and max(x["yolo_confidence"] for x in rows) < .35)
        noncar = any(x["class_id"] != "car" for x in rows)
        containment = any(not all(x["camera_containment"]) for x in rows)
        if low_short or noncar or containment:
            review_tracks.append((track_id, rows))

    representatives = []
    for track_id, rows in review_tracks:
        choices = []
        for row in rows:
            proposal = proposals[row["annotation_id"]]
            detection = max(proposal["detections"], key=lambda x:
                            (x["xyxy"][2] - x["xyxy"][0]) *
                            (x["xyxy"][3] - x["xyxy"][1]))
            area = ((detection["xyxy"][2] - detection["xyxy"][0])
                    * (detection["xyxy"][3] - detection["xyxy"][1]))
            choices.append((area, row, detection))
        _, row, detection = max(choices, key=lambda x: x[0])
        representatives.append({**row, "track_length": len(rows),
                                "detection": detection})
    representatives.sort(key=lambda x: (x["class_id"], x["frame_index"]))

    OUTDIR.mkdir(exist_ok=True)
    tiles = []
    for number, row in enumerate(representatives, 1):
        detection = row["detection"]
        sample = samples[row["frame_index"]]
        image = cv2.imread(str(camera_pass.image_path(sample, detection["camera"])))
        x1, y1, x2, y2 = map(int, detection["xyxy"])
        pad_x, pad_y = max(50, (x2 - x1) // 2), max(40, (y2 - y1) // 2)
        left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
        right, bottom = min(image.shape[1], x2 + pad_x), min(image.shape[0], y2 + pad_y)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 220), 5)
        crop = cv2.resize(image[top:bottom, left:right], (300, 165),
                          interpolation=cv2.INTER_AREA)
        tile = np.full((200, 300, 3), 245, np.uint8)
        tile[35:] = crop
        text = (f"{number} f{row['frame_index']} {row['class_id']} "
                f"c{row['yolo_confidence']:.2f} t{row['track_length']}")
        cv2.putText(tile, text, (5, 23), cv2.FONT_HERSHEY_SIMPLEX,
                    .47, (20, 20, 20), 1, cv2.LINE_AA)
        tiles.append(tile)
    blank = np.full((200, 300, 3), 245, np.uint8)
    outputs = []
    for page, start in enumerate(range(0, len(tiles), 80), 1):
        page_tiles = tiles[start:start + 80]
        while len(page_tiles) < 80:
            page_tiles.append(blank.copy())
        montage = np.vstack([np.hstack(page_tiles[i:i + 8])
                             for i in range(0, 80, 8)])
        output = OUTDIR / f"removal-class-review-{page:02d}.jpg"
        cv2.imwrite(str(output), montage, [cv2.IMWRITE_JPEG_QUALITY, 94])
        outputs.append(str(output))
    serializable = []
    for number, row in enumerate(representatives, 1):
        item = {k: v for k, v in row.items() if k != "detection"}
        item["review_number"] = number
        serializable.append(item)
    (OUTDIR / "index.json").write_text(json.dumps({
        "representatives": serializable, "sheets": outputs}, indent=2))
    print(json.dumps({"tracks": len(representatives), "sheets": len(outputs)}, indent=2))


if __name__ == "__main__":
    main()
