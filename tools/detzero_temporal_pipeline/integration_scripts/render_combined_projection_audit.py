#!/usr/bin/env python3
"""Render one deterministic 4-camera + LiDAR-BEV audit video for all BAGs."""
from __future__ import annotations

import argparse
import csv
import json
import random
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from camera_motion import CameraMotion
from render_improved_detzero import (
    CAMERAS, CLASS_COLORS, draw_box_camera, get_calibration, letterbox,
    make_writer, read_xyzi_pcd, render_bev, safe_child,
)
from temporal_annotations import load_ego_poses


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('combined', type=Path)
    parser.add_argument('--calibration', type=Path, required=True)
    parser.add_argument('--fraction', type=float, default=.1)
    parser.add_argument('--seed', type=int, default=3407)
    parser.add_argument('--fps', type=float, default=2.)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--camera-width', type=int, default=640)
    parser.add_argument('--camera-height', type=int, default=480)
    parser.add_argument('--bev-range', type=float, default=60.)
    parser.add_argument('--max-bev-points', type=int, default=160000)
    args = parser.parse_args()
    if not 0 < args.fraction <= 1 or args.fps <= 0:
        parser.error('fraction must be in (0,1] and fps must be positive')
    return args


def main() -> None:
    args = arguments()
    combined = args.combined.expanduser().resolve()
    calibration = args.calibration.expanduser().resolve()
    summary = json.loads((combined / 'summary.json').read_text())
    sequences = summary.get('sequences', [])
    if not sequences:
        raise ValueError('combined summary contains no sequences')
    output = (args.output.expanduser().resolve() if args.output else
              combined / 'combined_projection_audit_10pct.mp4')
    output.parent.mkdir(parents=True, exist_ok=True)
    models, transforms = get_calibration(calibration)
    cw, ch = args.camera_width, args.camera_height
    bev_size = ch * 2
    writer = make_writer(output, args.fps, (cw * 2 + bev_size, ch * 2))
    report_sequences = []
    class_counts = Counter()
    total = 0
    try:
        for entry in sequences:
            name = str(entry['name'])
            dataset = Path(entry['dataset']).expanduser().resolve()
            with (dataset / 'manifest.csv').open(newline='') as stream:
                rows = list(csv.DictReader(stream))
            frames = json.loads((dataset / 'detzero_output/improved_results.json').read_text())
            if len(rows) != len(frames) or not rows:
                raise ValueError(f'{name}: manifest/annotation frame mismatch')
            count = max(1, round(len(rows) * args.fraction))
            rng = random.Random(f'{args.seed}:{name}')
            indices = sorted(rng.sample(range(len(rows)), count))
            _, poses, _ = load_ego_poses(dataset, rows, calibration)
            motion = CameraMotion(rows, poses)
            for idx in indices:
                row, frame = rows[idx], frames[idx]
                if int(frame['frame_id']) != idx or int(frame['timestamp_ns']) != int(row['timestamp_ns']):
                    raise ValueError(f'{name} frame {idx}: stale annotations')
                points = read_xyzi_pcd(safe_child(dataset, row['lidar_pcd']))
                if len(points) > args.max_bev_points:
                    select = np.linspace(0, len(points) - 1,
                                         args.max_bev_points, dtype=int)
                    points = points[select]
                objects = []
                for obj in frame['objects']:
                    box = np.asarray(obj['box'], dtype=float)
                    if box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
                        raise ValueError(f'{name} frame {idx}: invalid box')
                    objects.append(obj)
                    class_counts[str(obj['name'])] += 1
                bev = render_bev(points, objects, bev_size, args.bev_range, idx)
                tiles = {}
                for camera, column in CAMERAS.items():
                    image = cv2.imread(str(safe_child(dataset, row[column])),
                                       cv2.IMREAD_COLOR)
                    if image is None:
                        raise ValueError(f'{name} frame {idx}: unreadable {camera}')
                    expected = models[camera]
                    if image.shape[1::-1] != (expected['width'], expected['height']):
                        raise ValueError(f'{name} frame {idx}: {camera} dimensions disagree with calibration')
                    for obj in objects:
                        projected = motion.box(np.asarray(obj['box'], float), idx,
                                               camera, obj.get('velocity_world'))
                        label = f"{obj['name']} #{obj['track_id']}"
                        if obj.get('source') == 'camera_track_hold':
                            label += ' [hold]'
                        draw_box_camera(image, projected, transforms[camera],
                                        expected, CLASS_COLORS.get(
                                            obj['name'], (240, 240, 240)), label)
                    tiles[camera] = letterbox(image, cw, ch, camera)
                cameras = np.vstack((np.hstack((tiles['ARC_F'], tiles['ARC_SL'])),
                                     np.hstack((tiles['ARC_R'], tiles['ARC_SR']))))
                combined_frame = np.hstack((cameras, bev))
                cv2.rectangle(combined_frame, (0, 0), (980, 42), (0, 0, 0), -1)
                cv2.putText(combined_frame,
                            f'{name} | random 10% audit | source frame {idx:05d}',
                            (12, 29), cv2.FONT_HERSHEY_SIMPLEX, .75,
                            (255, 255, 255), 2, cv2.LINE_AA)
                writer.write(combined_frame)
                total += 1
                if total % 50 == 0:
                    print(f'audit frames: {total}', flush=True)
            report_sequences.append({
                'name': name, 'dataset': str(dataset), 'source_frames': len(rows),
                'sampled_frames': len(indices), 'indices': indices,
                'camera_motion_status': motion.status_counts,
            })
    finally:
        writer.release()
    report = {
        'video': str(output), 'fraction_per_sequence': args.fraction,
        'seed': args.seed, 'fps': args.fps, 'frames': total,
        'sampling': 'deterministic random without replacement; chronological within each BAG',
        'sequences': report_sequences, 'class_counts': dict(class_counts),
        'calibration': str(calibration),
    }
    report_path = output.with_suffix('.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'video': str(output), 'report': str(report_path),
                      'frames': total, 'sequences': len(sequences)}, indent=2))


if __name__ == '__main__':
    main()
