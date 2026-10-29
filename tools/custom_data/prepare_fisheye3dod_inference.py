#!/usr/bin/env python3
"""Prepare synchronized custom cameras for Fisheye3DOD inference.

The released model was trained with a single 800x800 OCamCalib lens model.
This tool resamples each calibrated OpenCV-fisheye camera into that canonical
lens geometry and writes an annotation-free MMDetection3D info file.
"""

from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import cv2
import mmengine
import numpy as np
from ocamcamera import OcamCamera


CAMERAS = {
    'fisheye_camera_front': 'ARC_F',
    'fisheye_camera_left': 'ARC_SL',
    'fisheye_camera_right': 'ARC_SR',
    'fisheye_camera_rear': 'ARC_R',
}

CATEGORIES = {
    'Car': 0,
    'Van': 1,
    'Truck': 2,
    'Bus': 3,
    'Pedestrian': 4,
    'Cyclist': 5,
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('converted', type=Path)
    parser.add_argument('calibration', type=Path)
    parser.add_argument('--sync', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--template-info', type=Path,
        help='preserve the exact token order from an existing info PKL')
    parser.add_argument(
        '--target-ocam', type=Path,
        default=Path('data/Fisheye3DODdataset/calib_results.txt'))
    parser.add_argument('--target-fov', type=float, default=220.0)
    parser.add_argument('--sample-rate', type=float, default=2.0)
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument(
        '--image-mode', choices=('canonical-ocam', 'native'),
        default='canonical-ocam',
        help=('canonical-ocam remaps to the released model lens; native '
              'preserves the synchronized source fisheye images unchanged'))
    parser.add_argument('--limit', type=int)
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument(
        '--prune', action='store_true',
        help='remove output camera JPGs not selected for this info file')
    return parser.parse_args()


def link_or_copy(source: Path, destination: Path, overwrite: bool) -> None:
    """Materialize an unchanged native frame, preferring a zero-copy hard link."""
    if destination.exists():
        if not overwrite:
            return
        destination.unlink()
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def select_rows(rows: List[Dict[str, str]], rate: Optional[float]
                ) -> List[Dict[str, str]]:
    if not rate:
        return rows
    period = round(1_000_000_000 / rate)
    selected = []
    last = None
    for row in rows:
        timestamp = int(row['timestamp_ns'])
        if last is None or timestamp - last >= period:
            selected.append(row)
            last = timestamp
    return selected


def fisheye_project(rays: np.ndarray, intrinsic: dict
                    ) -> Tuple[np.ndarray, np.ndarray]:
    """Project 3xN OpenCV-camera rays with the OpenCV fisheye equation."""
    x, y, z = rays
    radius = np.hypot(x, y)
    theta = np.arctan2(radius, z)
    theta2 = theta * theta
    k1, k2, k3, k4 = intrinsic['distortion_coefficients'][:4]
    theta_d = theta * (
        1 + theta2 * (k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))))
    scale = np.divide(theta_d, radius, out=np.ones_like(radius),
                      where=radius > 1e-12)
    matrix = np.asarray(intrinsic['camera_matrix'], dtype=np.float64)
    map_x = matrix[0, 0] * x * scale + matrix[0, 2]
    map_y = matrix[1, 1] * y * scale + matrix[1, 2]
    return map_x.astype(np.float32), map_y.astype(np.float32)


def make_remap(ocam: OcamCamera, intrinsic: dict
               ) -> Tuple[np.ndarray, np.ndarray]:
    height, width = ocam.height, ocam.width
    u, v = np.meshgrid(np.arange(width, dtype=np.float64),
                       np.arange(height, dtype=np.float64))
    rays = ocam.cam2world(np.stack((u.ravel(), v.ravel())))
    map_x, map_y = fisheye_project(rays, intrinsic)
    map_x = map_x.reshape(height, width)
    map_y = map_y.reshape(height, width)
    valid = ocam.valid_area() > 0
    valid &= (map_x >= 0) & (map_x < intrinsic['width'] - 1)
    valid &= (map_y >= 0) & (map_y < intrinsic['height'] - 1)
    map_x[~valid] = -1
    map_y[~valid] = -1
    return map_x, map_y


def main() -> None:
    args = parse_args()
    sync_path = args.sync or args.converted / 'synchronization.csv'
    output = args.output or args.converted / 'fisheye3dod_inference_2hz'
    with sync_path.open(newline='') as stream:
        rows = sorted(csv.DictReader(stream),
                      key=lambda row: int(row['timestamp_ns']))
    rows = select_rows(list(rows), args.sample_rate)
    if args.template_info is not None:
        template = mmengine.load(args.template_info)
        by_token = {row['timestamp_ns']: row for row in rows}
        tokens = [str(item['token']) for item in template['data_list']]
        missing = [token for token in tokens if token not in by_token]
        if missing:
            raise RuntimeError(
                f'{len(missing)} template tokens are absent from {sync_path}')
        rows = [by_token[token] for token in tokens]
    if args.limit is not None:
        rows = rows[:args.limit]
    if not rows:
        raise RuntimeError('no synchronized samples selected')

    if args.prune:
        selected_tokens = {row['timestamp_ns'] for row in rows}
        for model_name in CAMERAS:
            camera_dir = output / 'images' / model_name
            if camera_dir.is_dir():
                for path in camera_dir.glob('*.jpg'):
                    if path.stem not in selected_tokens:
                        path.unlink()

    calibration = json.loads(args.calibration.read_text())
    ocam = None
    remaps = None
    if args.image_mode == 'canonical-ocam':
        ocam = OcamCamera(filename=str(args.target_ocam), fov=args.target_fov)
        remaps = {
            frame: make_remap(ocam, calibration['intrinsics'][frame])
            for frame in CAMERAS.values()
        }

    output.mkdir(parents=True, exist_ok=True)
    data_list = []
    for sample_index, row in enumerate(mmengine.track_iter_progress(rows)):
        timestamp = row['timestamp_ns']
        camera_info = {}
        for model_name, frame in CAMERAS.items():
            source = args.converted / row[frame.lower() + '_path']
            relative = Path('images') / model_name / f'{timestamp}.jpg'
            destination = output / relative
            destination.parent.mkdir(parents=True, exist_ok=True)
            if args.image_mode == 'native':
                link_or_copy(source, destination, args.overwrite)
            elif args.overwrite or not destination.exists():
                image = cv2.imread(str(source), cv2.IMREAD_COLOR)
                if image is None:
                    raise RuntimeError(f'failed to read {source}')
                map_x, map_y = remaps[frame]
                warped = cv2.remap(
                    image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)
                if not cv2.imwrite(
                        str(destination), warped,
                        [cv2.IMWRITE_JPEG_QUALITY, args.jpeg_quality]):
                    raise RuntimeError(f'failed to write {destination}')

            lidar2cam = np.asarray(
                calibration['lidar_to_camera'][frame], dtype=np.float32)
            intrinsic = calibration['intrinsics'][frame]
            cam2img = np.eye(4, dtype=np.float32)
            if args.image_mode == 'native':
                cam2img[:3, :3] = np.asarray(
                    intrinsic['camera_matrix'], dtype=np.float32)
                image_size = (int(intrinsic['width']),
                              int(intrinsic['height']))
            else:
                image_size = (ocam.width, ocam.height)
            camera_info[model_name] = {
                'cam_path': relative.as_posix(),
                'image_size': image_size,
                'cam2img': cam2img,
                'lidar2img': np.eye(4, dtype=np.float32),
                'lidar2cam': lidar2cam,
                'cam2lidar': np.linalg.inv(lidar2cam).astype(np.float32),
            }

        # Recorder-pipeline dataset manifests call the exported cloud
        # ``lidar_pcd``; legacy synchronization manifests call the source
        # binary ``lidar_path``.  Both are relative to ``converted`` here.
        lidar_path = row.get('lidar_pcd') or row.get('lidar_path')
        if not lidar_path:
            raise KeyError('manifest requires lidar_pcd or lidar_path')
        lidar_relative = Path('..') / lidar_path
        data_list.append({
            'sample_idx': sample_index,
            'token': timestamp,
            'metainfo': {
                'scene_name': args.converted.name,
                'vehicle_name': 'ego',
                'weather': 'unknown',
                'frame_id': timestamp,
                'prev_id': rows[sample_index - 1]['timestamp_ns']
                if sample_index else '',
                'next_id': rows[sample_index + 1]['timestamp_ns']
                if sample_index + 1 < len(rows) else '',
            },
            'pc_info': {
                'num_features': 4,
                'pts_path': lidar_relative.as_posix(),
                'semantic_pts_path': lidar_relative.as_posix(),
            },
            'cam_info': {'cam_fisheye': camera_info},
        })

    info = {
        'metainfo': {
            'categories': CATEGORIES,
            'classes': tuple(CATEGORIES),
            'dataset': 'custom_fisheye3dod_inference',
            'version': 'custom-2hz' if args.sample_rate == 2 else 'custom',
        },
        'data_list': data_list,
    }
    info_path = output / 'fisheye3dod_infos_test.pkl'
    mmengine.dump(info, info_path)
    metadata = {
        'source_sync': str(sync_path.resolve()),
        'source_calibration': str(args.calibration.resolve()),
        'template_info': (str(args.template_info.resolve())
                          if args.template_info is not None else None),
        'image_mode': args.image_mode,
        'target_ocam': (str(args.target_ocam.resolve())
                        if args.image_mode == 'canonical-ocam' else None),
        'target_fov_degrees': (args.target_fov
                               if args.image_mode == 'canonical-ocam' else None),
        'sample_rate_hz': args.sample_rate,
        'samples': len(data_list),
        'camera_mapping': CAMERAS,
    }
    (output / 'preparation.json').write_text(json.dumps(metadata, indent=2) + '\n')
    print(f'wrote {len(data_list)} samples to {info_path}')


if __name__ == '__main__':
    main()
