"""Synchronize camera frames to LiDAR acquisition timestamps."""
from __future__ import annotations

import bisect
import csv
import math
import os
from pathlib import Path

from .config import CAMERAS


def read_manifest(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def nearest(rows: list[dict], timestamps: list[int], target: int) -> tuple[dict, int]:
    if not rows:
        raise ValueError('cannot search an empty camera stream')
    where = bisect.bisect_left(timestamps, target)
    candidate = min((i for i in (where-1, where) if 0 <= i < len(rows)),
                    key=lambda i: (abs(timestamps[i]-target), timestamps[i]))
    return rows[candidate], timestamps[candidate]-target


def synchronize(converted: Path, output: Path, tolerance_ms: float = 20.,
                sample_rate: float = 2., camera_sync_clock: str = 'sensor',
                overwrite: bool = False) -> int:
    if not (math.isfinite(tolerance_ms) and math.isfinite(sample_rate)) or tolerance_ms < 0 or sample_rate < 0:
        raise ValueError('tolerance and sample rate must be finite and nonnegative')
    if output.exists() and not overwrite:
        raise FileExistsError(f'{output} exists; use --overwrite')
    if camera_sync_clock not in ('sensor', 'read'):
        raise ValueError('camera_sync_clock must be sensor or read')
    cams = {camera: [] for camera in CAMERAS}
    for row in read_manifest(converted / 'camera_manifest.csv'):
        if row['sensor'] in cams:
            cams[row['sensor']].append(row)
    times = {}
    for camera, rows in cams.items():
        rows.sort(key=lambda r: int(r['timestamp_ns']))
        if not rows:
            raise ValueError(f'camera manifest has no {camera} images')
        times[camera] = [int(row['timestamp_ns']) for row in rows]
    lidar_rows = read_manifest(converted / 'lidar_manifest.csv')
    if not lidar_rows:
        raise ValueError('LiDAR manifest is empty')
    lidar_rows.sort(key=lambda r: int(r.get('sensor_timestamp_ns') or r['timestamp_ns'])
                    if camera_sync_clock == 'sensor' else int(r['timestamp_ns']))
    tolerance_ns = round(tolerance_ms * 1_000_000)
    period_ns = round(1_000_000_000 / sample_rate) if sample_rate else 0
    last_accepted = None
    result = []
    for lidar in lidar_rows:
        recorder_ns = int(lidar['timestamp_ns'])
        sensor_ns = int(lidar.get('sensor_timestamp_ns') or recorder_ns)
        sync_ns = sensor_ns if camera_sync_clock == 'sensor' else recorder_ns
        if last_accepted is not None and period_ns and sync_ns-last_accepted < period_ns:
            continue
        matches = {camera: nearest(cams[camera], times[camera], sync_ns)
                   for camera in CAMERAS}
        if any(abs(delta) > tolerance_ns for _, delta in matches.values()):
            continue
        item = dict(timestamp_ns=recorder_ns, lidar_path=lidar['path'],
                    lidar_sensor_timestamp_ns=sensor_ns,
                    camera_sync_clock=camera_sync_clock, camera_sync_target_ns=sync_ns)
        for camera, (row, delta) in matches.items():
            prefix = camera.lower()
            item[f'{prefix}_path'] = row['path']
            item[f'{prefix}_timestamp_ns'] = row['timestamp_ns']
            item[f'{prefix}_delta_ns'] = delta
        result.append(item)
        last_accepted = sync_ns
    if not result:
        raise ValueError(f'no samples synchronized within {tolerance_ms} ms')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix('.tmp.csv')
    try:
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(result[0]))
            writer.writeheader()
            writer.writerows(result)
        os.replace(temporary, output)
    finally:
        temporary.unlink(missing_ok=True)
    return len(result)
