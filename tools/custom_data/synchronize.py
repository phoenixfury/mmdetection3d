#!/usr/bin/env python3
"""Synchronize four timestamped camera streams to Ouster scans."""

from __future__ import annotations

import argparse
import bisect
import csv
from collections import defaultdict
from pathlib import Path


CAMERAS = ('ARC_F', 'ARC_R', 'ARC_SL', 'ARC_SR')


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def nearest(rows: list[dict[str, str]], times: list[int], target: int
            ) -> tuple[dict[str, str], int]:
    position = bisect.bisect_left(times, target)
    candidates = [index for index in (position - 1, position) if 0 <= index < len(rows)]
    index = min(candidates, key=lambda item: abs(times[item] - target))
    return rows[index], times[index] - target


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('converted', type=Path,
                        help='directory containing camera_manifest.csv and lidar_manifest.csv')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--tolerance-ms', type=float, default=20.0)
    parser.add_argument('--sample-rate', type=float, default=2.0,
                        help='optional maximum output rate in Hz (for example 2)')
    parser.add_argument(
        '--camera-sync-clock', choices=('sensor', 'read'), default='sensor',
        help=('timestamp used to select camera frames: the embedded LiDAR '
              'acquisition timestamp (default) or the delayed bag/read '
              'timestamp'))
    args = parser.parse_args()
    output = args.output or args.converted / 'synchronization.csv'

    camera_rows: dict[str, list[dict[str, str]]] = defaultdict(list)
    for row in read_csv(args.converted / 'camera_manifest.csv'):
        if row['sensor'] in CAMERAS:
            camera_rows[row['sensor']].append(row)
    for camera in CAMERAS:
        camera_rows[camera].sort(key=lambda row: int(row['timestamp_ns']))
        if not camera_rows[camera]:
            raise RuntimeError(f'camera manifest has no {camera} frames')
    lidar_rows = sorted(read_csv(args.converted / 'lidar_manifest.csv'),
                        key=lambda row: int(row['timestamp_ns']))
    camera_times = {
        camera: [int(row['timestamp_ns']) for row in rows]
        for camera, rows in camera_rows.items()
    }

    tolerance_ns = round(args.tolerance_ms * 1_000_000)
    period_ns = round(1_000_000_000 / args.sample_rate) if args.sample_rate else 0
    accepted = []
    rejected = 0
    last_timestamp = -period_ns
    for lidar in lidar_rows:
        timestamp = int(lidar['timestamp_ns'])
        sensor_timestamp = int(lidar.get('sensor_timestamp_ns') or timestamp)
        camera_target = (sensor_timestamp if args.camera_sync_clock == 'sensor'
                         else timestamp)
        if period_ns and timestamp - last_timestamp < period_ns:
            continue
        matches = {}
        valid = True
        for camera in CAMERAS:
            row, delta = nearest(
                camera_rows[camera], camera_times[camera], camera_target)
            matches[camera] = (row, delta)
            valid &= abs(delta) <= tolerance_ns
        if not valid:
            rejected += 1
            continue
        result = {
            'timestamp_ns': timestamp,
            'lidar_path': lidar['path'],
            'lidar_sensor_timestamp_ns': sensor_timestamp,
            'camera_sync_clock': args.camera_sync_clock,
            'camera_sync_target_ns': camera_target,
        }
        for camera in CAMERAS:
            row, delta = matches[camera]
            prefix = camera.lower()
            result[f'{prefix}_path'] = row['path']
            result[f'{prefix}_timestamp_ns'] = row['timestamp_ns']
            result[f'{prefix}_delta_ns'] = delta
        accepted.append(result)
        last_timestamp = timestamp

    if not accepted:
        raise RuntimeError(
            f'no synchronized samples within {args.tolerance_ms:g} ms; '
            'inspect timestamps or increase --tolerance-ms')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(accepted[0]))
        writer.writeheader()
        writer.writerows(accepted)
    worst = max(abs(int(row[f'{camera.lower()}_delta_ns']))
                for row in accepted for camera in CAMERAS) / 1e6
    print(f'wrote {len(accepted)} synchronized samples to {output}; '
          f'rejected {rejected}; worst accepted delta {worst:.3f} ms')


if __name__ == '__main__':
    main()
