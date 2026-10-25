#!/usr/bin/env python3
"""Optimize per-camera pitch from LiDAR depth edges and image edges."""

from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image

from project_lidar import CAMERAS, project


def rotation_x(angle: float) -> np.ndarray:
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1.0, 0.0, 0.0], [0.0, c, -s], [0.0, s, c]])


def edge_distance(image_path: Path, scale: int) -> np.ndarray:
    image = Image.open(image_path).convert('L')
    image = image.resize((image.width // scale, image.height // scale),
                         Image.Resampling.BILINEAR)
    gray = np.asarray(image, dtype=np.float32) / 255.0
    gx = np.zeros_like(gray)
    gy = np.zeros_like(gray)
    gx[:, 1:-1] = gray[:, 2:] - gray[:, :-2]
    gy[1:-1, :] = gray[2:, :] - gray[:-2, :]
    magnitude = np.abs(gy)
    # Pitch is constrained by horizontal contours (vertical image gradients).
    # Vertical poles and facade edges are deliberately excluded: translating a
    # vertical line vertically does not provide a meaningful pitch constraint.
    horizontal = np.abs(gy) >= np.abs(gx)
    threshold = np.quantile(magnitude[8:-8, 8:-8], 0.88)
    edges = horizontal & (magnitude >= max(float(threshold), 0.05))
    return chamfer_distance(edges)


def chamfer_distance(edges: np.ndarray) -> np.ndarray:
    """Approximate Euclidean distance transform without an OpenCV dependency."""
    height, width = edges.shape
    distance = np.where(edges, 0.0, 1e6).astype(np.float32)
    diagonal = np.float32(2**0.5)
    for y in range(height):
        for x in range(width):
            value = distance[y, x]
            if y:
                value = min(value, distance[y-1, x] + 1)
                if x:
                    value = min(value, distance[y-1, x-1] + diagonal)
                if x + 1 < width:
                    value = min(value, distance[y-1, x+1] + diagonal)
            if x:
                value = min(value, distance[y, x-1] + 1)
            distance[y, x] = value
    for y in range(height - 1, -1, -1):
        for x in range(width - 1, -1, -1):
            value = distance[y, x]
            if y + 1 < height:
                value = min(value, distance[y+1, x] + 1)
                if x:
                    value = min(value, distance[y+1, x-1] + diagonal)
                if x + 1 < width:
                    value = min(value, distance[y+1, x+1] + diagonal)
            if x + 1 < width:
                value = min(value, distance[y, x+1] + 1)
            distance[y, x] = value
    return distance


def depth_edges(pixels: np.ndarray, depth: np.ndarray, width: int, height: int,
                scale: int) -> tuple[np.ndarray, np.ndarray]:
    width //= scale
    height //= scale
    xy = np.rint(pixels / scale).astype(np.int32)
    keep = ((xy[:, 0] >= 1) & (xy[:, 0] < width - 1)
            & (xy[:, 1] >= 1) & (xy[:, 1] < height - 1))
    xy, depth = xy[keep], depth[keep]
    flat = xy[:, 1] * width + xy[:, 0]
    zbuffer = np.full(height * width, np.inf, dtype=np.float32)
    np.minimum.at(zbuffer, flat, depth.astype(np.float32))
    zbuffer = zbuffer.reshape(height, width)

    # Compare occupied neighbours in a 3x3 window. Large depth jumps correspond
    # to object silhouettes and are substantially more useful than road texture.
    selected = np.zeros((height, width), dtype=bool)
    finite = np.isfinite(zbuffer)
    # Only depth changes between vertical neighbours constrain image pitch.
    for dy, dx in ((1, 0),):
        a = zbuffer[max(0, dy):height + min(0, dy),
                    max(0, dx):width + min(0, dx)]
        b = zbuffer[max(0, -dy):height - max(0, dy),
                    max(0, -dx):width - max(0, dx)]
        valid = np.isfinite(a) & np.isfinite(b)
        difference = np.zeros_like(a)
        threshold = np.zeros_like(a)
        difference[valid] = np.abs(a[valid] - b[valid])
        threshold[valid] = np.maximum(0.45, 0.025*np.minimum(a[valid], b[valid]))
        jump = valid & (difference > threshold)
        view = selected[max(0, dy):height + min(0, dy),
                        max(0, dx):width + min(0, dx)]
        view |= jump
    y, x = np.nonzero(selected & finite)
    return x, y


def transform_with_pitch(calibration: dict, camera: str,
                         degrees: float) -> np.ndarray:
    lidar_to_base = np.asarray(
        calibration['extrinsics']['CLIDAR']['matrix'], dtype=np.float64)
    camera_to_base = np.asarray(
        calibration['extrinsics'][camera]['matrix'], dtype=np.float64).copy()
    camera_to_base[:3, :3] = (
        camera_to_base[:3, :3] @ rotation_x(np.deg2rad(degrees)))
    return np.linalg.inv(camera_to_base) @ lidar_to_base


def evaluate(camera: str, degrees: float, records: list[dict], calibration: dict,
             converted: Path, scale: int, max_range: float) -> tuple[float, int]:
    transform = transform_with_pitch(calibration, camera, degrees)
    intrinsic = calibration['intrinsics'][camera]
    distances = []
    edge_count = 0
    for record in records:
        points, edge_map, size = record['points'], record['edges'][camera], record['size'][camera]
        pixels, depth = project(points, intrinsic, transform, 'fisheye', max_range)
        x, y = depth_edges(pixels, depth, size[0], size[1], scale)
        if len(x):
            values = edge_map[y, x]
            # Remove the per-row background distance. Without this correction a
            # pitch search can drift toward the horizon merely because that row
            # contains many unrelated image edges.
            values = values - np.mean(edge_map, axis=1)[y]
            # Clipping prevents a few unmatched/occluded silhouettes dominating.
            distances.append(np.clip(values, -12.0, 12.0))
            edge_count += len(values)
    if not distances:
        return float('inf'), 0
    return float(np.mean(np.concatenate(distances))), edge_count


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('converted', type=Path)
    parser.add_argument('calibration', type=Path)
    parser.add_argument('--sync', type=Path)
    parser.add_argument('--output', type=Path,
                        help='write calibration with optimized camera extrinsics')
    parser.add_argument('--samples', type=int, default=10)
    parser.add_argument('--start-sample', type=int, default=0)
    parser.add_argument('--cameras', nargs='+', choices=CAMERAS,
                        default=list(CAMERAS))
    parser.add_argument('--scale', type=int, default=4)
    parser.add_argument('--max-range', type=float, default=60.0)
    parser.add_argument('--min-pitch', type=float, default=-3.0)
    parser.add_argument('--max-pitch', type=float, default=3.0)
    parser.add_argument('--step', type=float, default=0.1)
    args = parser.parse_args()

    calibration = json.loads(args.calibration.read_text())
    sync_path = args.sync or args.converted / 'synchronization.csv'
    with sync_path.open(newline='') as stream:
        all_samples = list(csv.DictReader(stream))
    samples = all_samples[args.start_sample:args.start_sample + args.samples]
    records = []
    for sample in samples:
        points = np.fromfile(args.converted / sample['lidar_path'], '<f4').reshape(-1, 4)
        lidar_to_base = np.asarray(
            calibration['extrinsics']['CLIDAR']['matrix'], dtype=np.float64)
        base = points[:, :3] @ lidar_to_base[:3, :3].T + lidar_to_base[:3, 3]
        # Road points dominate a scan but carry few trustworthy cross-modal
        # edges. Optimize on objects above the road (cars, poles, barriers and
        # buildings), where depth and appearance boundaries should coincide.
        points = points[base[:, 2] > -0.10]
        record = {'points': points, 'edges': {}, 'size': {}}
        for camera in args.cameras:
            image_path = args.converted / sample[f'{camera.lower()}_path']
            with Image.open(image_path) as image:
                record['size'][camera] = image.size
            record['edges'][camera] = edge_distance(image_path, args.scale)
        records.append(record)

    candidates = np.arange(args.min_pitch, args.max_pitch + args.step/2, args.step)
    results = {}
    for camera in args.cameras:
        scores = []
        for degrees in candidates:
            score, count = evaluate(camera, float(degrees), records, calibration,
                                    args.converted, args.scale, args.max_range)
            scores.append(score)
        best_index = int(np.argmin(scores))
        best = float(candidates[best_index])
        baseline, baseline_count = evaluate(
            camera, 0.0, records, calibration, args.converted,
            args.scale, args.max_range)
        best_score, best_count = evaluate(
            camera, best, records, calibration, args.converted,
            args.scale, args.max_range)
        results[camera] = {'pitch_correction_degrees': best,
                           'baseline_score_pixels': baseline,
                           'optimized_score_pixels': best_score,
                           'score_improvement_pixels': baseline-best_score,
                           'baseline_edges': baseline_count,
                           'optimized_edges': best_count}
        print(f'{camera}: pitch={best:+.3f} deg, score {baseline:.4f} -> '
              f'{best_score:.4f} (improvement '
              f'{results[camera]["score_improvement_pixels"]:+.4f} px)')

    if args.output:
        for camera, result in results.items():
            old = np.asarray(calibration['extrinsics'][camera]['matrix'], dtype=float)
            new = old.copy()
            new[:3, :3] = old[:3, :3] @ rotation_x(
                np.deg2rad(result['pitch_correction_degrees']))
            calibration['extrinsics'][camera]['matrix'] = new.tolist()
            calibration['lidar_to_camera'][camera] = (
                np.linalg.inv(new) @ np.asarray(
                    calibration['extrinsics']['CLIDAR']['matrix'])).tolist()
        calibration['camera_pitch_optimization'] = {
            'method': 'LiDAR z-buffer depth edges to image-gradient chamfer distance',
            'samples': len(records), 'start_sample': args.start_sample,
            'downsample_scale': args.scale,
            'search_degrees': [args.min_pitch, args.max_pitch, args.step],
            'results': results,
        }
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(json.dumps(calibration, indent=2) + '\n')
        print(f'wrote {args.output}')


if __name__ == '__main__':
    main()
