#!/usr/bin/env python3
"""Remap an old read-clock CaliperGT review onto acquisition-clock frames.

The nearest reviewed LiDAR frame is selected by acquisition time, transformed
through the global frame, and expressed in the corresponding new CLIDAR frame.
This avoids comparing frame numbers from two different synchronizers.
"""
from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import sys
from collections import Counter, defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'tools/recorder_pipeline/src'))

from recorder_pipeline.odometry import (  # noqa: E402
    AlignedOdometry, FusedOdometrySeries, load_fused_odometry, pose_matrix)
from recorder_pipeline.raw_odometry import load_raw_odometry  # noqa: E402


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('manual_json', type=Path)
    parser.add_argument('old_sync_csv', type=Path)
    parser.add_argument('new_dataset', type=Path)
    parser.add_argument('bag', type=Path,
                        help='unconverted recorder bag containing odometry')
    parser.add_argument('calibration', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--endpoint-tolerance', type=float, default=.31,
                        help='maximum extrapolation to a track endpoint [s]')
    parser.add_argument('--odometry-tolerance-ms', type=float, default=200.)
    return parser.parse_args()


def angle(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def read_csv(path: Path) -> list[dict]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def lidar_poses(args: argparse.Namespace, timestamps: list[int]) -> dict[int, np.ndarray]:
    fused_dir = args.bag / '__engineering__adcu__debug__ZMQ__Localization__FusedOdometry'
    raw_dir = args.bag / '__engineering__adcu__debug__ZMQ__Localization__Odometry'
    maximum = round(args.odometry_tolerance_ms * 1_000_000)
    fused = FusedOdometrySeries(load_fused_odometry(fused_dir), maximum)
    raw = FusedOdometrySeries(load_raw_odometry(raw_dir), maximum)
    odometry = AlignedOdometry(fused, raw)
    calibration = json.loads(args.calibration.read_text())
    base_from_lidar = np.asarray(
        calibration['extrinsics']['CLIDAR']['matrix'], dtype=float)
    poses = {}
    for timestamp in timestamps:
        try:
            poses[timestamp] = pose_matrix(odometry.at(timestamp)) @ base_from_lidar
        except ValueError:
            # The Sep-16 recorder has a real localization outage.  New export
            # frames inside it are already omitted, so unusable old frames are
            # intentionally excluded from the reference as well.
            continue
    return poses


def cuboid_array(annotation: dict) -> np.ndarray:
    cuboid = annotation['cuboid_3d']
    p, d, r = cuboid['position'], cuboid['dimensions'], cuboid['rotation']
    return np.array([p['x'], p['y'], p['z'], d['length'], d['width'],
                     d['height'], r['yaw']], dtype=float)


def to_global(box: np.ndarray, pose: np.ndarray) -> np.ndarray:
    result = box.copy()
    result[:3] = (pose @ np.r_[box[:3], 1.])[:3]
    result[6] = angle(box[6] + math.atan2(pose[1, 0], pose[0, 0]))
    return result


def to_local(box: np.ndarray, pose: np.ndarray) -> np.ndarray:
    result = box.copy()
    inverse = np.linalg.inv(pose)
    result[:3] = (inverse @ np.r_[box[:3], 1.])[:3]
    result[6] = angle(box[6] - math.atan2(pose[1, 0], pose[0, 0]))
    return result


def interpolate(a: np.ndarray, b: np.ndarray, fraction: float) -> np.ndarray:
    result = a + fraction * (b - a)
    result[6] = angle(a[6] + fraction * angle(b[6] - a[6]))
    return result


def write_cuboid(annotation: dict, box: np.ndarray) -> None:
    cuboid = annotation['cuboid_3d']
    cuboid['position'] = dict(zip(('x', 'y', 'z'), map(float, box[:3])))
    cuboid['dimensions'] = dict(zip(
        ('length', 'width', 'height'), map(float, box[3:6])))
    cuboid['rotation']['yaw'] = float(box[6])


def main() -> None:
    args = arguments()
    manual = json.loads(args.manual_json.read_text())
    old_rows, new_rows = read_csv(args.old_sync_csv), read_csv(
        args.new_dataset / 'manifest.csv')
    old_times = [int(row['lidar_sensor_timestamp_ns']) for row in old_rows]
    new_times = [int(row['lidar_sensor_timestamp_ns']) for row in new_rows]
    old_poses = lidar_poses(args, old_times)
    new_base_poses = np.load(args.new_dataset / 'odometry_poses.npy')
    calibration = json.loads(args.calibration.read_text())
    base_from_lidar = np.asarray(
        calibration['extrinsics']['CLIDAR']['matrix'], dtype=float)
    new_poses = [new_base_poses[int(row['odometry_index'])] @ base_from_lidar
                 for row in new_rows]

    by_frame: dict[int, list[dict]] = defaultdict(list)
    invalid_frames = set()
    for source in manual['annotations']:
        frame = int(source['frame_id'])
        if not 0 <= frame < len(old_times) or old_times[frame] not in old_poses:
            invalid_frames.add(frame)
            continue
        by_frame[frame].append(source)

    # The historical CaliperGT import assigned a distinct ID to most boxes
    # (8,505 singleton tracks in the reviewed export), so track interpolation
    # would silently discard much of the usable ground truth.  Select the
    # closest *reviewed frame* and compensate the ego transform instead.
    valid_frames = sorted(by_frame)
    valid_times = [old_times[frame] for frame in valid_frames]
    endpoint = round(args.endpoint_tolerance * 1e9)
    annotations = []
    temporal_offsets = []
    matched_frames = 0
    for frame, (timestamp, pose) in enumerate(zip(new_times, new_poses)):
        index = bisect.bisect_left(valid_times, timestamp)
        nearest = min((j for j in (index - 1, index)
                       if 0 <= j < len(valid_times)),
                      key=lambda j: abs(valid_times[j] - timestamp),
                      default=None)
        if nearest is None:
            continue
        old_frame = valid_frames[nearest]
        old_timestamp = valid_times[nearest]
        offset = abs(old_timestamp - timestamp)
        if offset > endpoint:
            continue
        matched_frames += 1
        temporal_offsets.append(offset / 1e6)
        for template in by_frame[old_frame]:
            global_box = to_global(
                cuboid_array(template), old_poses[old_timestamp])
            annotation = deepcopy(template)
            annotation['id'] = len(annotations) + 1
            annotation['frame_id'] = frame
            annotation['source'] = 'manual_acquisition_remap'
            write_cuboid(annotation, to_local(global_box, pose))
            annotations.append(annotation)

    frames = []
    for index, (row, timestamp) in enumerate(zip(new_rows, new_times)):
        frames.append({'id': index, 'file_name': Path(row['lidar_pcd']).name,
                       'timestamp': timestamp / 1e9})
    result = deepcopy(manual)
    result['annotations'] = annotations
    result['frames'] = frames
    result.setdefault('info', {})['remapping'] = (
        'old read-clock review -> nearest acquisition time -> global ego '
        'motion compensation -> new '
        'acquisition-clock CLIDAR frames')
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(result, indent=2) + '\n')
    offsets = sorted(temporal_offsets)
    percentile = lambda p: offsets[round((len(offsets) - 1) * p)] if offsets else None
    report = {
        'old_frames': len(old_rows), 'new_frames': len(new_rows),
        'manual_annotations': len(manual['annotations']),
        'remapped_annotations': len(annotations),
        'matched_new_frames': matched_frames,
        'manual_track_ids': len({str(item['track_id'])
                                 for item in manual['annotations']}),
        'manual_singleton_track_ids': sum(
            count == 1 for count in Counter(
                str(item['track_id']) for item in manual['annotations']).values()),
        'old_frames_without_valid_odometry': len(invalid_frames),
        'temporal_support_ms': {
            'median': percentile(.5), 'p90': percentile(.9),
            'p99': percentile(.99), 'maximum': percentile(1.)},
        'endpoint_tolerance_seconds': args.endpoint_tolerance,
    }
    report_path = args.output.with_suffix('.report.json')
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))
    print(args.output)


if __name__ == '__main__':
    main()
