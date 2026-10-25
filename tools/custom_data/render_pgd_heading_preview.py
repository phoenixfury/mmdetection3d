#!/usr/bin/env python3
"""Render representative trusted PGD front/back corrections."""

import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, "/home/mostafa/mmdetection3d/tools/custom_data")
from pgd_fisheye_heading_pilot import (  # noqa: E402
    HEIGHT, PLAN, WIDTH, annotation_id, tangent_image)

RESULT = Path("/home/mostafa/bag-2026-09-16-pgd-heading-pilot.json")
OUTPUT = Path("/home/mostafa/bag-2026-09-16-pgd-heading-preview.jpg")


def main():
    sys.path.insert(0, "/home/mostafa")
    import camera_second_pass as camera_pass

    records = json.loads(RESULT.read_text())["records"]
    trusted = sorted((item for item in records
                      if item["pgd_score"] >= 8
                      and item["axis_difference_deg"] <= 20
                      and item["image_suggests_flip"]),
                     key=lambda item: item["frame_index"])
    proposals = {annotation_id(item): item
                 for item in json.loads(PLAN.read_text())["proposals"]}
    samples = camera_pass.metadata_samples()
    indices = np.linspace(0, len(trusted) - 1, min(12, len(trusted)), dtype=int)
    tiles = []
    for number, index in enumerate(indices):
        record = trusted[index]
        proposal = proposals[record["annotation_id"]]
        detection = next(item for item in proposal["detections"]
                         if item["camera"] == record["camera"])
        camera = samples[record["frame_index"]]["cam_info"]["cam_fisheye"][record["camera"]]
        path = Path(f"/tmp/pgd_preview_{number}.jpg")
        tangent_image(detection, camera, path)
        tile = cv2.imread(str(path))
        tile = cv2.resize(tile, (621, 188))
        text = (f"frame {record['frame_index']}  PGD score {record['pgd_score']:.1f}  "
                f"axis error {record['axis_difference_deg']:.1f} deg  FLIP 180")
        cv2.rectangle(tile, (0, 0), (621, 28), (0, 0, 0), -1)
        cv2.putText(tile, text, (8, 19), cv2.FONT_HERSHEY_SIMPLEX,
                    0.48, (0, 255, 255), 1, cv2.LINE_AA)
        tiles.append(tile)
    while len(tiles) % 3:
        tiles.append(np.zeros_like(tiles[0]))
    rows = [np.hstack(tiles[start:start + 3])
            for start in range(0, len(tiles), 3)]
    cv2.imwrite(str(OUTPUT), np.vstack(rows), [cv2.IMWRITE_JPEG_QUALITY, 92])
    print(OUTPUT)


if __name__ == "__main__":
    main()
