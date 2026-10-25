#!/usr/bin/env python3
"""Render high-resolution crops for ambiguous removal/class decisions."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/mostafa")
import camera_second_pass as camera_pass
sys.path.insert(0, "/home/mostafa/mmdetection3d/tools/custom_data")
from pgd_fisheye_heading_pilot import PLAN, annotation_id


NUMBERS = {21, 30, 43, 52, 55, 56, 214, 231, 233, 234, 252, 269,
           290, 302, 314, 405, 406, 438, 440, 503, 511, 532, 542, 544}
INDEX = Path("/home/mostafa/bag-2026-09-16-removal-class-review/index.json")
OUTPUT = Path("/home/mostafa/bag-2026-09-16-ambiguous-review.jpg")


def main():
    rows = [x for x in json.loads(INDEX.read_text())["representatives"]
            if x["review_number"] in NUMBERS]
    proposals = {annotation_id(x): x
                 for x in json.loads(PLAN.read_text())["proposals"]}
    samples = camera_pass.metadata_samples()
    tiles = []
    for row in rows:
        proposal = proposals[row["annotation_id"]]
        detection = max(proposal["detections"], key=lambda x:
                        (x["xyxy"][2] - x["xyxy"][0]) *
                        (x["xyxy"][3] - x["xyxy"][1]))
        sample = samples[row["frame_index"]]
        image = cv2.imread(str(camera_pass.image_path(sample, detection["camera"])))
        x1, y1, x2, y2 = map(int, detection["xyxy"])
        pad_x, pad_y = max(160, x2 - x1), max(120, y2 - y1)
        left, top = max(0, x1 - pad_x), max(0, y1 - pad_y)
        right, bottom = min(image.shape[1], x2 + pad_x), min(image.shape[0], y2 + pad_y)
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 220, 220), 6)
        crop = cv2.resize(image[top:bottom, left:right], (600, 360),
                          interpolation=cv2.INTER_AREA)
        tile = np.full((400, 600, 3), 245, np.uint8)
        tile[40:] = crop
        text = (f"{row['review_number']} frame {row['frame_index']} "
                f"{row['class_id']} conf {row['yolo_confidence']:.2f} "
                f"track {row['track_length']}")
        cv2.putText(tile, text, (8, 27), cv2.FONT_HERSHEY_SIMPLEX,
                    .62, (20, 20, 20), 2, cv2.LINE_AA)
        tiles.append(tile)
    blank = np.full((400, 600, 3), 245, np.uint8)
    while len(tiles) % 4:
        tiles.append(blank.copy())
    montage = np.vstack([np.hstack(tiles[i:i + 4])
                         for i in range(0, len(tiles), 4)])
    cv2.imwrite(str(OUTPUT), montage, [cv2.IMWRITE_JPEG_QUALITY, 96])
    print(json.dumps({"output": str(OUTPUT), "items": len(rows)}))


if __name__ == "__main__":
    main()
