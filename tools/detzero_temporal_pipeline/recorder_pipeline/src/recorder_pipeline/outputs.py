"""Dataset writer: synchronized native JPEG, binary PCD and fused poses."""
from __future__ import annotations

import csv
import json
import os
import shutil
import warnings
from collections import Counter
from pathlib import Path

import numpy as np
from PIL import Image

from .config import CAMERAS
from .odometry import FusedOdometrySeries, AlignedOdometry, load_fused_odometry, pose_matrix
from .raw_odometry import load_raw_odometry
from .storage import convert_bin_to_pcd, reuse_binary_pcd


def _inside(base: Path, relative: str) -> Path:
    """Manifest entries are relative paths; never follow paths outside dataset."""
    p = Path(relative)
    if not relative or p.is_absolute():
        raise ValueError(f'unsafe relative manifest path: {relative!r}')
    base_resolved = base.resolve()
    candidate = (base_resolved / p).resolve()
    try:
        candidate.relative_to(base_resolved)
    except ValueError as error:
        raise ValueError(f'unsafe relative manifest path: {relative!r}') from error
    return candidate


def _atomic_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f'cannot write empty CSV: {path}')
    temp = path.with_suffix('.tmp.csv')
    try:
        with temp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        os.replace(temp, path)
    finally:
        temp.unlink(missing_ok=True)


def _copy_jpeg(source: Path, destination: Path, overwrite: bool,
               link_images: bool = False) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.exists() and not overwrite:
        raise FileExistsError(f'{destination} already exists; use --overwrite')
    temp = destination.with_suffix('.tmp.jpg')
    try:
        if source.suffix.lower() in ('.jpg', '.jpeg'):
            if link_images:
                try:
                    os.link(source, temp)
                except OSError:
                    # Cross-filesystem or unsupported link: exact bytewise copy.
                    shutil.copy2(source, temp)
            else:
                shutil.copy2(source, temp)
        else:
            with Image.open(source) as image:
                image.convert('RGB').save(temp, format='JPEG', quality=95)
        os.replace(temp, destination)
    finally:
        temp.unlink(missing_ok=True)


def export(converted: Path, output: Path, odometry_dir: Path | None = None,
           max_delta_ms: float = 200., overwrite: bool = False, limit: int = 0,
           pose_timestamp_unit: str = 'ms', link_images: bool = False,
           raw_odometry_dir: Path | None = None, proto_dir: Path | None = None,
           missing_odometry_policy: str = 'error') -> int:
    if max_delta_ms <= 0 or not np.isfinite(max_delta_ms) or limit < 0:
        raise ValueError('max_delta_ms must be finite positive and limit nonnegative')
    if missing_odometry_policy not in ('error', 'skip'):
        raise ValueError('missing_odometry_policy must be error or skip')
    with (converted / 'synchronization.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if limit:
        rows = rows[:limit]
    if not rows:
        raise ValueError('no synchronized rows to export')
    samples = None
    if raw_odometry_dir is not None and odometry_dir is None:
        raise ValueError('--raw-odometry-dir requires --odometry-dir')
    if odometry_dir is not None:
        max_ns = round(max_delta_ms * 1_000_000)
        fused = FusedOdometrySeries(load_fused_odometry(odometry_dir, pose_timestamp_unit), max_ns)
        samples = fused
        if raw_odometry_dir is not None:
            raw = FusedOdometrySeries(load_raw_odometry(raw_odometry_dir, proto_dir, pose_timestamp_unit), max_ns)
            samples = AlignedOdometry(fused, raw)
    output.mkdir(parents=True, exist_ok=True)
    if not overwrite:
        for filename in ('manifest.csv','odometry.csv','odometry_poses.npy','metadata.json'):
            if (output / filename).exists():
                raise FileExistsError(f'{output/filename} already exists; use --overwrite')
    manifest, poses, odometry, skipped_odometry = [], [], [], []
    tokens: set[int] = set()
    for row in rows:
        timestamp = int(row['timestamp_ns'])
        if timestamp in tokens:
            raise ValueError(f'duplicate LiDAR timestamp in synchronization manifest: {timestamp}')
        tokens.add(timestamp)
        sensor_ns = int(row.get('lidar_sensor_timestamp_ns') or timestamp)
        pose = None
        if samples is not None:
            try:
                pose = samples.at(sensor_ns)
            except ValueError as error:
                if missing_odometry_policy == 'error':
                    raise
                skipped_odometry.append({
                    'timestamp_ns': timestamp,
                    'lidar_sensor_timestamp_ns': sensor_ns,
                    'reason': str(error),
                })
                continue
        lidar_source = _inside(converted, row['lidar_path'])
        lidar_relative = Path('lidar_pcd') / f'{timestamp}.pcd'
        lidar_dest = output / lidar_relative
        lidar_dest.parent.mkdir(parents=True, exist_ok=True)
        if lidar_source.is_file():
            point_count = convert_bin_to_pcd(lidar_source, lidar_dest, overwrite)
        else:
            legacy_pcd = converted / 'dataset' / lidar_relative
            if not legacy_pcd.is_file():
                raise FileNotFoundError(
                    f'no LiDAR BIN or legacy PCD for {timestamp}')
            point_count = reuse_binary_pcd(legacy_pcd, lidar_dest, overwrite)
        item = dict(timestamp_ns=timestamp, lidar_sensor_timestamp_ns=sensor_ns,
                    lidar_transport_delay_ns=timestamp - sensor_ns,
                    camera_sync_clock=row.get('camera_sync_clock', ''),
                    camera_sync_target_ns=row.get('camera_sync_target_ns', ''),
                    lidar_pcd=lidar_relative.as_posix(), point_count=point_count)
        for camera in CAMERAS:
            prefix = camera.lower()
            source = _inside(converted, row[f'{prefix}_path'])
            relative = Path('images') / prefix / f'{timestamp}.jpg'
            if not source.is_file():
                legacy_jpeg = converted / 'dataset' / relative
                if not legacy_jpeg.is_file():
                    raise FileNotFoundError(
                        f'no decoded or legacy synchronized {prefix} image for {timestamp}')
                source = legacy_jpeg
            _copy_jpeg(source, output / relative, overwrite, link_images)
            item[f'{prefix}_path'] = relative.as_posix()
            item[f'{prefix}_delta_ns'] = int(row[f'{prefix}_delta_ns'])
            # Preserve the original camera packet timestamp, not the LiDAR
            # timestamp used to name the synchronized exported JPG.
            item[f'{prefix}_timestamp_ns'] = row.get(f'{prefix}_timestamp_ns', '')
        if pose is not None:
            item['odometry_index'] = len(odometry)
            item['odometry_delta_ns'] = pose['delta_ns']
            item['odometry_status'] = pose['status']
            odometry.append(pose)
            poses.append(pose_matrix(pose))
        manifest.append(item)
    if not manifest:
        raise ValueError('no exportable synchronized rows remain')
    # Manifests are published only after all payload files have been written.
    _atomic_csv(output / 'manifest.csv', manifest)
    if odometry:
        _atomic_csv(output / 'odometry.csv', odometry)
        temp = output / 'odometry_poses.tmp.npy'
        try:
            np.save(temp, np.stack(poses))
            os.replace(temp, output / 'odometry_poses.npy')
        finally:
            temp.unlink(missing_ok=True)
    skipped_path = output / 'skipped_odometry.csv'
    if skipped_odometry:
        _atomic_csv(skipped_path, skipped_odometry)
        warnings.warn(
            f'skipped {len(skipped_odometry)} synchronized samples without '
            f'bounded odometry; details: {skipped_path}', RuntimeWarning,
            stacklevel=2)
    elif overwrite:
        skipped_path.unlink(missing_ok=True)
    metadata = dict(samples=len(manifest),
                    synchronized_input_samples=len(rows),
                    skipped_missing_odometry=len(skipped_odometry),
                    missing_odometry_policy=missing_odometry_policy,
                    odometry_sources=dict(Counter(
                        row['source'] for row in odometry)),
                    cameras=list(CAMERAS),
                    camera_format='native JPG',
                    image_storage=('hardlink-or-copy' if link_images else 'copy'),
                    cloud_format='PCD binary XYZI float32',
                    cloud_coordinate_frame='CLIDAR',
                    synchronization='camera nearest LiDAR sensor timestamp (unless read explicitly selected)',
                    timestamp_provenance={
                        'lidar_sensor_timestamp_ns': 'embedded PointCloud acquisition timestamp',
                        'timestamp_ns': 'LiDAR recorder index timestamp',
                        'camera_timestamp_ns': 'camera recorder index packet timestamp (not verified exposure time)',
                        'camera_delta_ns': 'camera recorder timestamp minus selected LiDAR synchronization clock',
                    },
                    odometry=(dict(source_dir=str(odometry_dir),
                                   timestamp_unit=pose_timestamp_unit,
                                   clock='EgoPoseFused.timestamp',
                                   frame='pose frame in source message; user must verify',
                                   matrices='T_world_base assuming Rz(yaw)Ry(pitch)Rx(roll)',
                                   method='bracketed interpolation, no extrapolation',
                                   fallback=('boundary-aligned Localization/Odometry' if raw_odometry_dir else 'disabled'),
                                   raw_source_dir=str(raw_odometry_dir) if raw_odometry_dir else None)
                              if samples is not None else None))
    temp = output / 'metadata.tmp.json'
    try:
        temp.write_text(json.dumps(metadata, indent=2) + '\n')
        os.replace(temp, output / 'metadata.json')
    finally:
        temp.unlink(missing_ok=True)
    return len(manifest)
