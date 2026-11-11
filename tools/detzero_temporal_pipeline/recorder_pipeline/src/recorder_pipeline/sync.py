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
                sample_rate: float = 2., camera_sync_clock: str = 'auto',
                overwrite: bool = False) -> int:
    if not (math.isfinite(tolerance_ms) and math.isfinite(sample_rate)) or tolerance_ms < 0 or sample_rate < 0:
        raise ValueError('tolerance and sample rate must be finite and nonnegative')
    if output.exists() and not overwrite:
        raise FileExistsError(f'{output} exists; use --overwrite')
    if camera_sync_clock not in ('auto', 'sensor', 'read'):
        raise ValueError('camera_sync_clock must be auto, sensor or read')
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
    tolerance_ns = round(tolerance_ms * 1_000_000)
    period_ns = round(1_000_000_000 / sample_rate) if sample_rate else 0
    sensor_order = [int(row.get('sensor_timestamp_ns') or row['timestamp_ns'])
                    for row in lidar_rows]
    recorder_order = [int(row['timestamp_ns']) for row in lidar_rows]
    sensor_reversals = sum(b <= a for a, b in zip(sensor_order, sensor_order[1:]))
    align_sensor_to_recorder = False
    sensor_to_recorder_ns = 0

    def match_count(clock: str) -> int:
        count = 0
        for lidar in lidar_rows:
            recorder_ns = int(lidar['timestamp_ns'])
            target = (int(lidar.get('sensor_timestamp_ns') or recorder_ns)
                      if clock == 'sensor' else recorder_ns)
            matches = (nearest(cams[camera], times[camera], target)[1]
                       for camera in CAMERAS)
            count += int(all(abs(delta) <= tolerance_ns for delta in matches))
        return count

    if camera_sync_clock == 'auto':
        scores = {clock: match_count(clock) for clock in ('sensor', 'read')}
        # Acquisition time determines the physical order of point clouds.
        # Recorder time is packet-arrival time and may contain a growing queue
        # delay; choosing it merely because it has a few more matches can make
        # a moving scene rock backwards and forwards. Use sensor time whenever
        # it has substantial overlap with the camera clock, and reserve read
        # time for rigs whose acquisition clock is genuinely incomparable.
        # A queued topic can make nearly every recorder-arrival timestamp fall
        # inside the camera recording even though the payloads were acquired
        # minutes earlier.  Do not compare overlap *ratios* in that case: a
        # sustained set of exact sensor/camera matches proves that the two
        # acquisition clocks are comparable.  Prefer sensor time either when
        # it wins outright or when at least ten all-camera matches establish
        # a real overlap.  Non-epoch/device clocks normally produce zero such
        # matches and still use the fixed-offset fallback below.
        substantial_sensor_overlap = (
            scores['sensor'] > scores['read'] or scores['sensor'] >= 10)
        camera_sync_clock = 'sensor' if substantial_sensor_overlap else 'read'
        # A recorder backlog can deliver valid scans out of acquisition order.
        # Direct sensor/camera comparison may still fail when the LiDAR clock
        # has a fixed offset from the recorder clock.  In that case, the
        # minimum observed arrival-minus-acquisition delay estimates the fixed
        # clock offset (all additional queueing delay is non-negative).  Keep
        # the recorder timescale for camera/odometry lookup, but order scans by
        # acquisition time so a moving scene cannot rock backwards and forwards.
        transport_delays = [recorded - sensor for recorded, sensor
                            in zip(recorder_order, sensor_order)]
        if (camera_sync_clock == 'read' and sensor_reversals and
                transport_delays and min(transport_delays) > 0):
            align_sensor_to_recorder = True
            sensor_to_recorder_ns = min(transport_delays)
        detail = ('; acquisition-order backlog alignment offset_ns={}'
                  .format(sensor_to_recorder_ns) if align_sensor_to_recorder else '')
        print('Camera sync auto-selection: {} (sensor matches={}, read matches={}){}'.format(
            camera_sync_clock, scores['sensor'], scores['read'], detail))

    lidar_rows.sort(key=lambda r: int(r.get('sensor_timestamp_ns') or r['timestamp_ns'])
                    if camera_sync_clock == 'sensor' or align_sensor_to_recorder
                    else int(r['timestamp_ns']))
    last_accepted = None
    result = []
    for lidar in lidar_rows:
        recorder_ns = int(lidar['timestamp_ns'])
        sensor_ns = int(lidar.get('sensor_timestamp_ns') or recorder_ns)
        sync_ns = (sensor_ns if camera_sync_clock == 'sensor' else
                   sensor_ns + sensor_to_recorder_ns if align_sensor_to_recorder else
                   recorder_ns)
        if last_accepted is not None and period_ns and sync_ns-last_accepted < period_ns:
            continue
        matches = {camera: nearest(cams[camera], times[camera], sync_ns)
                   for camera in CAMERAS}
        if any(abs(delta) > tolerance_ns for _, delta in matches.values()):
            continue
        item = dict(timestamp_ns=recorder_ns, lidar_path=lidar['path'],
                    lidar_sensor_timestamp_ns=sensor_ns,
                    camera_sync_clock=camera_sync_clock, camera_sync_target_ns=sync_ns,
                    lidar_order_clock=('sensor' if camera_sync_clock == 'sensor' or
                                       align_sensor_to_recorder else 'read'),
                    sensor_to_sync_offset_ns=(sensor_to_recorder_ns
                                              if align_sensor_to_recorder else 0))
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
