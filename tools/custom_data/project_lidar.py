#!/usr/bin/env python3
"""Project synchronized CLIDAR scans onto the four camera images."""

from __future__ import annotations

import argparse
import colorsys
import csv
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


CAMERAS = ('ARC_F', 'ARC_R', 'ARC_SL', 'ARC_SR')


def project(points: np.ndarray, intrinsic: dict, transform: np.ndarray,
            model: str, max_range: float) -> tuple[np.ndarray, np.ndarray]:
    xyz = points[:, :3].astype(np.float64)
    homogeneous = np.column_stack((xyz, np.ones(len(xyz))))
    camera = (transform @ homogeneous.T).T[:, :3]
    distance = np.linalg.norm(xyz, axis=1)
    keep = (np.all(np.isfinite(camera), axis=1) & (camera[:, 2] > 0.05)
            & (distance <= max_range))
    camera = camera[keep]
    distance = distance[keep]
    matrix = np.asarray(intrinsic['camera_matrix'], dtype=np.float64)
    distortion = np.asarray(
        intrinsic['distortion_coefficients'], dtype=np.float64)
    if model == 'fisheye':
        if distortion.size != 4:
            raise ValueError('OpenCV fisheye projection requires four coefficients')
        normalized = camera[:, :2] / camera[:, 2, None]
        radius = np.linalg.norm(normalized, axis=1)
        theta = np.arctan(radius)
        theta2 = theta * theta
        theta_distorted = theta * (1 + distortion[0] * theta2
                                    + distortion[1] * theta2**2
                                    + distortion[2] * theta2**3
                                    + distortion[3] * theta2**4)
        scale = np.divide(theta_distorted, radius, out=np.ones_like(radius),
                          where=radius > 1e-12)
        distorted = normalized * scale[:, None]
    else:
        distorted = camera[:, :2] / camera[:, 2, None]
        x, y = distorted[:, 0], distorted[:, 1]
        r2 = x*x + y*y
        k1, k2, p1, p2 = distortion[:4]
        k3 = distortion[4] if distortion.size > 4 else 0.0
        radial = 1 + k1*r2 + k2*r2*r2 + k3*r2*r2*r2
        distorted = np.column_stack((
            x*radial + 2*p1*x*y + p2*(r2 + 2*x*x),
            y*radial + p1*(r2 + 2*y*y) + 2*p2*x*y))
    pixels = np.column_stack((
        matrix[0, 0] * distorted[:, 0] + matrix[0, 1] * distorted[:, 1]
        + matrix[0, 2],
        matrix[1, 1] * distorted[:, 1] + matrix[1, 2]))
    return pixels, distance


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('converted', type=Path)
    parser.add_argument('calibration', type=Path)
    parser.add_argument('--sync', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--samples', type=int, default=1)
    parser.add_argument('--start-sample', type=int, default=0)
    parser.add_argument('--projection', choices=('calibration', 'fisheye', 'pinhole'),
                        default='calibration')
    parser.add_argument('--max-range', type=float, default=80.0)
    parser.add_argument('--point-radius', type=int, default=1)
    args = parser.parse_args()
    sync_path = args.sync or args.converted / 'synchronization.csv'
    output = args.output or args.converted / 'projection_check'
    output.mkdir(parents=True, exist_ok=True)
    calibration = json.loads(args.calibration.read_text())

    with sync_path.open(newline='') as stream:
        all_samples = list(csv.DictReader(stream))
    samples = all_samples[args.start_sample:args.start_sample + args.samples]
    stats = []
    for sample in samples:
        timestamp = sample['timestamp_ns']
        points = np.fromfile(args.converted / sample['lidar_path'], dtype='<f4')
        if points.size % 4:
            raise ValueError(f"{sample['lidar_path']} is not float32 XYZI")
        points = points.reshape(-1, 4)
        for camera in CAMERAS:
            image_path = args.converted / sample[f'{camera.lower()}_path']
            image = Image.open(image_path).convert('RGB')
            intrinsic = calibration['intrinsics'][camera]
            model = args.projection
            if model == 'calibration':
                model = ('fisheye' if intrinsic['projection_for_validation'] ==
                         'opencv_fisheye' else 'pinhole')
            pixels, depth = project(
                points, intrinsic,
                np.asarray(calibration['lidar_to_camera'][camera]),
                model, args.max_range)
            width, height = image.size
            visible = (np.all(np.isfinite(pixels), axis=1)
                       & (pixels[:, 0] >= 0) & (pixels[:, 0] < width)
                       & (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
            pixels = np.rint(pixels[visible]).astype(np.int32)
            visible_depth = depth[visible]
            normalized = np.clip(visible_depth / args.max_range, 0, 1)
            colors = [tuple(round(channel * 255) for channel in
                            colorsys.hsv_to_rgb((2/3) * value, 1.0, 1.0))
                      for value in normalized]
            order = np.argsort(visible_depth)[::-1]  # draw near points last
            draw = ImageDraw.Draw(image)
            for index in order:
                x, y = pixels[index]
                radius = args.point_radius
                draw.ellipse((x-radius, y-radius, x+radius, y+radius),
                             fill=colors[index])
            draw.text((24, 24), f'{camera}  {model}  points={len(pixels)}',
                      fill=(255, 255, 255), stroke_width=1, stroke_fill=(0, 0, 0))
            destination = output / f'{timestamp}_{camera.lower()}_{model}.jpg'
            image.save(destination, quality=95)
            try:
                relative_output = destination.relative_to(args.converted).as_posix()
            except ValueError:
                relative_output = str(destination)
            stats.append({
                'timestamp_ns': timestamp,
                'camera': camera,
                'projection': model,
                'front_of_camera': len(depth),
                'inside_image': len(pixels),
                'inside_percent': (100.0 * len(pixels) / len(depth)) if len(depth) else 0,
                'output': relative_output,
            })
            print(f'{camera}: {len(pixels)} / {len(depth)} front-facing points in image')

    with (output / 'projection_stats.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(stats[0]))
        writer.writeheader()
        writer.writerows(stats)
    print(f'overlays: {output}')


if __name__ == '__main__':
    main()
