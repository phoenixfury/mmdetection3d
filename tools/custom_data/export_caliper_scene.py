#!/usr/bin/env python3
"""Export synchronized custom data as a CaliperGT scene import bundle."""

from __future__ import annotations

import argparse
import csv
import json
import os
import re
import shutil
import struct
import zipfile
from pathlib import Path

import mmengine
import numpy as np
from PIL import Image


CAMERAS = {
    'front': ('ARC_F', 'arc_f_path'),
    'left': ('ARC_SL', 'arc_sl_path'),
    'right': ('ARC_SR', 'arc_sr_path'),
    'rear': ('ARC_R', 'arc_r_path'),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path,
                        help='prepared Fisheye3DOD dataset directory')
    parser.add_argument('calibration', type=Path,
                        help='custom calibration JSON with manual extrinsics')
    parser.add_argument('--converted', type=Path,
                        help='source converted directory; defaults to DATASET parent')
    parser.add_argument('--info', type=Path,
                        help='defaults to DATASET/fisheye3dod_infos_test.pkl')
    parser.add_argument('--output', type=Path,
                        help='defaults to DATASET/caliperai_scene')
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int,
                        help='inclusive; defaults to the final prepared frame')
    parser.add_argument('--copy-images', action='store_true',
                        help='copy instead of hard-linking source JPEGs')
    parser.add_argument('--zip', action='store_true', dest='make_zip',
                        help='also create an upload-ready ZIP next to OUTPUT')
    return parser.parse_args()


def pcd_header(point_count: int) -> bytes:
    text = (
        '# .PCD v0.7 - Point Cloud Data file format\n'
        'VERSION 0.7\n'
        'FIELDS x y z intensity\n'
        'SIZE 4 4 4 4\n'
        'TYPE F F F F\n'
        'COUNT 1 1 1 1\n'
        f'WIDTH {point_count}\n'
        'HEIGHT 1\n'
        'VIEWPOINT 0 0 0 1 0 0 0\n'
        f'POINTS {point_count}\n'
        'DATA binary\n'
    )
    return text.encode('ascii')


def convert_bin_to_pcd(source: Path, destination: Path) -> int:
    size = source.stat().st_size
    if size % 16:
        raise ValueError(f'{source}: size {size} is not float32 XYZI')
    point_count = size // 16
    temporary = destination.with_suffix('.pcd.tmp')
    with temporary.open('wb') as stream:
        stream.write(pcd_header(point_count))
        with source.open('rb') as raw:
            shutil.copyfileobj(raw, stream, length=8 * 1024 * 1024)
    temporary.replace(destination)
    return point_count


def link_or_copy(source: Path, destination: Path, copy: bool) -> None:
    if destination.exists():
        if destination.stat().st_size != source.stat().st_size:
            raise FileExistsError(f'existing file differs: {destination}')
        return
    if copy:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def caliper_calibration(document: dict) -> dict:
    cameras = {}
    for camera_name, (sensor_name, _) in CAMERAS.items():
        intrinsic = document['intrinsics'][sensor_name]
        matrix = np.asarray(document['lidar_to_camera'][sensor_name], dtype=float)
        camera_matrix = np.asarray(intrinsic['camera_matrix'], dtype=float)
        distortion = [float(value) for value in
                      intrinsic['distortion_coefficients']]
        if matrix.shape != (4, 4):
            raise ValueError(f'{sensor_name}: lidar_to_camera must be 4x4')
        if len(distortion) != 4:
            raise ValueError(f'{sensor_name}: expected four fisheye coefficients')
        cameras[camera_name] = {
            'extrinsic': {
                'rotation': matrix[:3, :3].tolist(),
                'translation': matrix[:3, 3].tolist(),
            },
            'intrinsic': {
                'fx': float(camera_matrix[0, 0]),
                'fy': float(camera_matrix[1, 1]),
                'cx': float(camera_matrix[0, 2]),
                'cy': float(camera_matrix[1, 2]),
                'distortion': distortion,
                'resolution': [int(intrinsic['width']),
                               int(intrinsic['height'])],
                'camera_model': 'kannala_brandt',
            },
        }
    return {'lidar_to_cameras': cameras}


def verify_pcd(path: Path, expected_points: int) -> None:
    with path.open('rb') as stream:
        header = b''
        while not header.endswith(b'DATA binary\n'):
            line = stream.readline()
            if not line:
                raise ValueError(f'{path}: incomplete PCD header')
            header += line
        payload_bytes = path.stat().st_size - len(header)
    if f'POINTS {expected_points}\n'.encode() not in header:
        raise ValueError(f'{path}: incorrect POINTS header')
    if payload_bytes != expected_points * 16:
        raise ValueError(f'{path}: incorrect binary payload size')


def write_zip(scene: Path) -> Path:
    archive = scene.with_suffix('.zip')
    temporary = archive.with_suffix('.zip.tmp')
    if temporary.exists():
        temporary.unlink()
    with zipfile.ZipFile(temporary, 'w', compression=zipfile.ZIP_STORED,
                         allowZip64=True) as bundle:
        for path in sorted(scene.rglob('*')):
            if path.is_file():
                bundle.write(path, Path(scene.name) / path.relative_to(scene))
    temporary.replace(archive)
    return archive


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    converted = (args.converted or dataset.parent).resolve()
    info_path = (args.info or dataset / 'fisheye3dod_infos_test.pkl').resolve()
    output = (args.output or dataset / 'caliperai_scene').resolve()
    calibration_path = args.calibration.resolve()

    info = mmengine.load(info_path)
    samples = info['data_list']
    end = len(samples) - 1 if args.end_frame is None else args.end_frame
    if args.start_frame < 0 or end < args.start_frame or end >= len(samples):
        raise ValueError(
            f'require 0 <= start-frame <= end-frame < {len(samples)}')
    selected = samples[args.start_frame:end + 1]

    with (converted / 'synchronization.csv').open(newline='') as stream:
        sync = {row['timestamp_ns']: row for row in csv.DictReader(stream)}
    missing = [str(sample['token']) for sample in selected
               if str(sample['token']) not in sync]
    if missing:
        raise KeyError(f'{len(missing)} prepared tokens absent from synchronization.csv')

    lidar_dir = output / 'lidar'
    camera_dirs = {name: output / 'cameras' / name for name in CAMERAS}
    lidar_dir.mkdir(parents=True, exist_ok=True)
    for directory in camera_dirs.values():
        directory.mkdir(parents=True, exist_ok=True)

    calibration_source = json.loads(calibration_path.read_text())
    calibration = caliper_calibration(calibration_source)
    (output / 'calibration.json').write_text(
        json.dumps(calibration, indent=2) + '\n')
    date_match = re.search(r'BAG_(\d{4})_(\d{2})_(\d{2})', converted.name)
    recording_date = ('-'.join(date_match.groups())
                      if date_match else 'unknown')
    metadata = {
        'description': 'Synchronized four-fisheye-camera and LiDAR scene',
        'fps': 2.0,
        'location': 'Moscow',
        'weather': 'clear',
        'recording_date': recording_date,
    }
    (output / 'scene_metadata.json').write_text(
        json.dumps(metadata, indent=2) + '\n')

    mapping_path = output / 'frame_mapping.csv'
    total_points = 0
    rows = []
    for export_index, sample in enumerate(mmengine.track_iter_progress(selected)):
        source_index = args.start_frame + export_index
        token = str(sample['token'])
        sync_row = sync[token]
        stem = f'frame_{export_index:06d}'
        lidar_source = (dataset / sample['pc_info']['pts_path']).resolve()
        lidar_destination = lidar_dir / f'{stem}.pcd'
        point_count = convert_bin_to_pcd(lidar_source, lidar_destination)
        verify_pcd(lidar_destination, point_count)
        total_points += point_count

        row = {
            'frame_index': export_index, 'source_frame_index': source_index,
            'token': token, 'lidar_source': str(lidar_source),
            'point_count': point_count,
        }
        for camera_name, (sensor_name, sync_column) in CAMERAS.items():
            source = converted / sync_row[sync_column]
            destination = camera_dirs[camera_name] / f'{stem}.jpg'
            expected = calibration['lidar_to_cameras'][camera_name][
                'intrinsic']['resolution']
            with Image.open(source) as image:
                if list(image.size) != expected:
                    raise ValueError(
                        f'{source}: image size {image.size} != {tuple(expected)}')
            link_or_copy(source, destination, args.copy_images)
            row[f'{camera_name}_sensor'] = sensor_name
            row[f'{camera_name}_source'] = str(source)
        rows.append(row)

    with mapping_path.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)

    expected_names = [f'frame_{index:06d}' for index in range(len(selected))]
    for directory, suffix in [(lidar_dir, '.pcd')] + [
            (directory, '.jpg') for directory in camera_dirs.values()]:
        actual = [path.stem for path in sorted(directory.glob(f'*{suffix}'))]
        if actual != expected_names:
            raise RuntimeError(f'frame ordering/count mismatch in {directory}')

    summary = {
        'format': 'CaliperGT scene import bundle',
        'source_info': str(info_path),
        'source_calibration': str(calibration_path),
        'source_frame_start': args.start_frame,
        'source_frame_end': end,
        'frames': len(selected), 'cameras': list(CAMERAS),
        'lidar_format': 'PCD v0.7 binary float32 x y z intensity',
        'camera_format': 'JPEG 1920x1536 synchronized raw fisheye',
        'camera_model': 'kannala_brandt',
        'total_points': total_points,
        'image_storage': 'copy' if args.copy_images else 'hard-link-or-copy',
    }
    (output / 'export_summary.json').write_text(
        json.dumps(summary, indent=2) + '\n')
    archive = write_zip(output) if args.make_zip else None
    if archive is not None:
        summary['archive'] = str(archive)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
