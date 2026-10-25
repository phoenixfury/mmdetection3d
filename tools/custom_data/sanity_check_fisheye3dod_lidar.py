#!/usr/bin/env python3
"""Measure and visualize LiDAR support for Fisheye3DOD predictions."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from typing import Dict, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmengine
import numpy as np


COLORS = {
    'Car': '#00d8ff', 'Van': '#2979ff', 'Truck': '#ff9100',
    'Bus': '#ff1744', 'Pedestrian': '#76ff03', 'Cyclist': '#d500f9',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('converted', type=Path)
    parser.add_argument('predictions', type=Path,
                        help='SaveDetectionHook epoch_0.pkl')
    parser.add_argument('--sync', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--visualizations', type=int, default=12)
    return parser.parse_args()


def local_coordinates(points: np.ndarray, box: np.ndarray
                      ) -> Tuple[np.ndarray, np.ndarray]:
    dx = points[:, 0] - box[0]
    dy = points[:, 1] - box[1]
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def local_ground(points: np.ndarray, box: np.ndarray) -> Optional[float]:
    local_x, local_y = local_coordinates(points, box)
    half_x, half_y = box[3] / 2, box[4] / 2
    outer = (np.abs(local_x) <= half_x + 2.0) & (
        np.abs(local_y) <= half_y + 2.0)
    inner = (np.abs(local_x) <= half_x + 0.3) & (
        np.abs(local_y) <= half_y + 0.3)
    z = points[outer & ~inner, 2]
    z = z[(z > -4.0) & (z < 0.5)]
    if len(z) < 20:
        return None
    # Nearby ground normally dominates the lower part of the vertical returns;
    # using the lower-half median resists cars, poles, and vegetation.
    cutoff = np.median(z)
    lower = z[z <= cutoff]
    return float(np.median(lower)) if len(lower) else None


def box_support(points: np.ndarray, box: np.ndarray) -> Dict[str, object]:
    local_x, local_y = local_coordinates(points, box)
    half_x, half_y, height = box[3] / 2, box[4] / 2, box[5]
    inside_xy = (np.abs(local_x) <= half_x) & (np.abs(local_y) <= half_y)
    inside_z = (points[:, 2] >= box[2] - 0.15) & (
        points[:, 2] <= box[2] + height + 0.15)
    # Ignore the lowest 25 cm when looking for object rather than road returns.
    object_z = (points[:, 2] >= box[2] + 0.25) & (
        points[:, 2] <= box[2] + height + 0.15)
    expanded_xy = (np.abs(local_x) <= half_x + 0.5) & (
        np.abs(local_y) <= half_y + 0.5)
    expanded_z = (points[:, 2] >= box[2] - 0.3) & (
        points[:, 2] <= box[2] + height + 0.3)
    ground = local_ground(points, box)
    return {
        'points_in_box': int(np.count_nonzero(inside_xy & inside_z)),
        'object_points_in_box': int(np.count_nonzero(inside_xy & object_z)),
        'points_in_expanded_box': int(np.count_nonzero(expanded_xy & expanded_z)),
        'local_ground_z': ground,
        'bottom_minus_ground_m': None if ground is None else float(box[2] - ground),
    }


def box_corners(box: np.ndarray) -> np.ndarray:
    half_x, half_y = box[3] / 2, box[4] / 2
    local = np.array([[half_x, half_y], [half_x, -half_y],
                      [-half_x, -half_y], [-half_x, half_y],
                      [half_x, half_y]])
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + box[:2]


def summarize(records: list) -> dict:
    summary = {'detections': len(records), 'thresholds': {}}
    for threshold in (0.1, 0.25, 0.5):
        selected = [row for row in records if row['score'] >= threshold]
        supported = [row for row in selected if row['object_points_in_box'] >= 3]
        strong = [row for row in selected if row['object_points_in_box'] >= 10]
        ground = [abs(row['bottom_minus_ground_m']) for row in selected
                  if row['bottom_minus_ground_m'] is not None]
        summary['thresholds'][str(threshold)] = {
            'boxes': len(selected),
            'with_at_least_3_object_points': len(supported),
            'support_rate_percent': round(100 * len(supported) / len(selected), 2)
            if selected else None,
            'with_at_least_10_object_points': len(strong),
            'strong_support_rate_percent': round(100 * len(strong) / len(selected), 2)
            if selected else None,
            'median_object_points': float(np.median(
                [row['object_points_in_box'] for row in selected]))
            if selected else None,
            'median_abs_bottom_ground_error_m': float(np.median(ground))
            if ground else None,
        }
    summary['classes'] = dict(Counter(row['class'] for row in records))
    return summary


def main() -> None:
    args = parse_args()
    sync_path = args.sync or args.converted / 'synchronization.csv'
    output = args.output or args.predictions.parent / 'lidar_sanity'
    output.mkdir(parents=True, exist_ok=True)
    with sync_path.open(newline='') as stream:
        sync = {row['timestamp_ns']: row for row in csv.DictReader(stream)}
    prediction_data = mmengine.load(args.predictions)['data_list']

    records = []
    frame_cache = {}
    for prediction in mmengine.track_iter_progress(prediction_data):
        token = str(prediction['token'])
        if token not in sync:
            raise KeyError(f'prediction token {token} is absent from synchronization.csv')
        lidar_path = args.converted / sync[token]['lidar_path']
        points = np.fromfile(lidar_path, dtype=np.float32).reshape(-1, 4)[:, :3]
        valid = np.isfinite(points).all(axis=1)
        valid &= np.linalg.norm(points, axis=1) > 0.1
        valid &= np.linalg.norm(points[:, :2], axis=1) < 80.0
        points = points[valid]
        annos = prediction['annos']
        frame_records = []
        for name, box, score in zip(
                annos['pred_names'], annos['pred_boxes'], annos['pred_scores']):
            box = np.asarray(box, dtype=np.float64)
            row = {
                'token': token,
                'class': str(name),
                'score': float(score),
                'x': float(box[0]), 'y': float(box[1]), 'z_bottom': float(box[2]),
                'length': float(box[3]), 'width': float(box[4]),
                'height': float(box[5]), 'yaw': float(box[6]),
                'range_m': float(np.linalg.norm(box[:2])),
            }
            row.update(box_support(points, box))
            records.append(row)
            frame_records.append((box, row))
        if frame_records:
            frame_cache[token] = (points, frame_records)

    fieldnames = list(records[0]) if records else []
    with (output / 'box_lidar_support.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(records)
    summary = summarize(records)
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')

    ranked_tokens = []
    for token, (_, frame_records) in frame_cache.items():
        ranked_tokens.append((max(row['score'] for _, row in frame_records), token))
    selected_tokens = [token for _, token in sorted(ranked_tokens, reverse=True)
                       [:args.visualizations]]
    if selected_tokens:
        columns = 3
        rows = int(np.ceil(len(selected_tokens) / columns))
        figure, axes = plt.subplots(rows, columns, figsize=(7 * columns, 7 * rows),
                                    squeeze=False)
        for axis, token in zip(axes.ravel(), selected_tokens):
            points, frame_records = frame_cache[token]
            visible = (np.abs(points[:, 0]) <= 52) & (np.abs(points[:, 1]) <= 52)
            draw = points[visible][::3]
            axis.scatter(draw[:, 0], draw[:, 1], s=0.15, c=draw[:, 2],
                         cmap='gray', vmin=-3, vmax=2, alpha=0.65)
            for box, row in frame_records:
                corners = box_corners(box)
                color = COLORS.get(row['class'], 'red')
                axis.plot(corners[:, 0], corners[:, 1], color=color, linewidth=2)
                axis.text(box[0], box[1],
                          f"{row['class']} {row['score']:.2f}\n"
                          f"obj pts={row['object_points_in_box']}",
                          color=color, fontsize=7)
            axis.set_title(token)
            axis.set_xlim(-52, 52); axis.set_ylim(-52, 52)
            axis.set_aspect('equal'); axis.grid(alpha=0.15)
            axis.set_xlabel('LiDAR x [m]'); axis.set_ylabel('LiDAR y [m]')
        for axis in axes.ravel()[len(selected_tokens):]:
            axis.axis('off')
        figure.tight_layout()
        figure.savefig(output / 'top_score_bev.png', dpi=160)
        plt.close(figure)

    print(json.dumps(summary, indent=2))
    print(f'outputs: {output}')


if __name__ == '__main__':
    main()
