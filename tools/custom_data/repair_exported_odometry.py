#!/usr/bin/env python3
"""Rebuild an exported sequence's poses using embedded LiDAR timestamps."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import shutil
import struct
from functools import lru_cache
from pathlib import Path

import numpy as np
from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

from export_synchronized_dual_images import (
    align_raw_odometry,
    load_fused_odometry,
    pose_matrix,
    synchronize_odometry,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', type=Path)
    parser.add_argument('--lidar-topic', type=Path, required=True)
    parser.add_argument('--odometry-dir', type=Path, required=True)
    parser.add_argument('--proto-dir', type=Path, required=True)
    parser.add_argument('--max-odometry-delta-ms', type=float, default=200.0)
    return parser.parse_args()


def timestamp_ns(timestamp: object) -> int:
    return int(timestamp.seconds) * 1_000_000_000 + int(timestamp.nanos)


@lru_cache(maxsize=1)
def topic_data_class() -> type:
    descriptor = descriptor_pb2.FileDescriptorProto(
        name='runtime_bag.proto', package='bag', syntax='proto3')
    timestamp = descriptor.message_type.add()
    timestamp.name = 'Timestamp'
    field = timestamp.field.add()
    field.name, field.number, field.label, field.type = 'seconds', 1, 1, 3
    field = timestamp.field.add()
    field.name, field.number, field.label, field.type = 'nanos', 2, 1, 5
    sample = descriptor.message_type.add()
    sample.name = 'Sample'
    field = sample.field.add()
    field.name, field.number, field.label, field.type = 'timestamp', 1, 1, 11
    field.type_name = '.bag.Timestamp'
    field = sample.field.add()
    field.name, field.number, field.label, field.type = 'data', 2, 1, 12
    field = sample.field.add()
    field.name, field.number, field.label, field.type = 'read_timestamp', 3, 1, 11
    field.type_name = '.bag.Timestamp'
    topic = descriptor.message_type.add()
    topic.name = 'TopicData'
    field = topic.field.add()
    field.name, field.number, field.label, field.type = 'timestamp', 1, 1, 11
    field.type_name = '.bag.Timestamp'
    field = topic.field.add()
    field.name, field.number, field.label, field.type = 'samples', 2, 3, 11
    field.type_name = '.bag.Sample'
    pool = descriptor_pool.DescriptorPool()
    pool.Add(descriptor)
    return message_factory.GetMessageClass(
        pool.FindMessageTypeByName('bag.TopicData'))


def read_varint(data: bytes, offset: int) -> tuple[int, int]:
    value = shift = 0
    while offset < len(data):
        byte = data[offset]
        offset += 1
        value |= (byte & 0x7f) << shift
        if byte < 0x80:
            return value, offset
        shift += 7
    raise ValueError('truncated protobuf varint')


def first_varint(data: bytes, wanted_field: int) -> int:
    offset = 0
    while offset < len(data):
        tag, offset = read_varint(data, offset)
        field, wire = tag >> 3, tag & 7
        if wire == 0:
            value, offset = read_varint(data, offset)
            if field == wanted_field:
                return value
        elif wire == 1:
            offset += 8
        elif wire == 2:
            length, offset = read_varint(data, offset)
            offset += length
        elif wire == 5:
            offset += 4
        else:
            raise ValueError(f'unsupported protobuf wire type {wire}')
    raise ValueError(f'payload has no varint field {wanted_field}')


def lidar_sensor_times(topic_dir: Path,
                       wanted: set[int]) -> dict[int, int]:
    result = {}
    topic_type = topic_data_class()
    for index_path in sorted(topic_dir.glob('*.idx')):
        raw_index = index_path.read_bytes()
        read_times = struct.unpack(
            f'<{len(raw_index) // 8}Q', raw_index)
        selected = [
            index for index, value in enumerate(read_times)
            if value in wanted
        ]
        if not selected:
            continue
        topic = topic_type.FromString(
            index_path.with_suffix('.bin').read_bytes())
        if len(topic.samples) != len(read_times):
            raise ValueError(f'{index_path}: sample/index length mismatch')
        for index in selected:
            sample = topic.samples[index]
            read_time = timestamp_ns(sample.read_timestamp)
            if read_time != read_times[index]:
                raise ValueError(f'{index_path}: read timestamp mismatch')
            result[read_time] = first_varint(bytes(sample.data), 1)
    missing = wanted - result.keys()
    if missing:
        raise ValueError(f'{len(missing)} exported LiDAR timestamps not found')
    return result


def synchronize_pose(timestamp: int, fused: list[dict], fused_times: list[int],
                     raw: list[dict], raw_times: list[int], boundaries: dict,
                     max_delta_ns: int) -> dict:
    try:
        pose = synchronize_odometry(
            timestamp, fused, fused_times, max_delta_ns)
        pose['source'] = 'fused'
        return pose
    except ValueError as fused_error:
        try:
            pose = synchronize_odometry(
                timestamp, raw, raw_times, max_delta_ns)
            if timestamp < fused_times[0]:
                return align_raw_odometry(
                    pose, *boundaries['prefix'], 'aligned_odometry_prefix')
            if timestamp > fused_times[-1]:
                return align_raw_odometry(
                    pose, *boundaries['tail'], 'aligned_odometry_tail')
            insertion = bisect.bisect_left(fused_times, timestamp)
            candidates = {
                max(0, insertion - 1), min(insertion, len(fused_times) - 1)
            }
            boundary_index = min(
                candidates,
                key=lambda index: abs(fused_times[index] - timestamp))
            boundary_time = fused_times[boundary_index]
            fused_boundary = synchronize_odometry(
                boundary_time, fused, fused_times, max_delta_ns)
            raw_boundary = synchronize_odometry(
                boundary_time, raw, raw_times, max_delta_ns)
        except ValueError as raw_error:
            # A few recordings continue publishing LiDAR while both decoded
            # localization streams contain a genuine acquisition hole.  In
            # the September 16 bag this is the stationary interval between
            # the fused tail and the raw stream resuming.  There is no motion
            # measurement to interpolate, so explicitly hold the closest
            # fused endpoint instead of silently aligning to transport time.
            if timestamp > fused_times[-1]:
                held = synchronize_odometry(
                    fused_times[-1], fused, fused_times, max_delta_ns)
                held.update({
                    'timestamp_ns': timestamp,
                    'delta_ns': timestamp - fused_times[-1],
                    'source': 'held_fused_tail_no_odometry',
                })
                return held
            if timestamp < fused_times[0]:
                held = synchronize_odometry(
                    fused_times[0], fused, fused_times, max_delta_ns)
                held.update({
                    'timestamp_ns': timestamp,
                    'delta_ns': timestamp - fused_times[0],
                    'source': 'held_fused_prefix_no_odometry',
                })
                return held
            raise ValueError(
                f'{fused_error}; raw fallback failed: {raw_error}') from raw_error
        return align_raw_odometry(
            pose, fused_boundary, raw_boundary, 'aligned_odometry_gap')


def overlap_boundary(fused: list[dict], fused_times: list[int],
                     raw: list[dict], raw_times: list[int],
                     max_delta_ns: int, reverse: bool) -> tuple[dict, dict]:
    """Find an endpoint-near timestamp covered by both odometry streams.

    Some bags have a long hole in raw Odometry at the exact final
    FusedOdometry timestamp.  The rigid raw-to-fused frame alignment only
    needs a genuine overlap sample; requiring overlap at the endpoint makes
    otherwise usable tail data impossible to synchronize.
    """
    indices = range(len(fused_times) - 1, -1, -1) if reverse else range(
        len(fused_times))
    for index in indices:
        timestamp = fused_times[index]
        try:
            raw_pose = synchronize_odometry(
                timestamp, raw, raw_times, max_delta_ns)
        except ValueError:
            continue
        fused_pose = synchronize_odometry(
            timestamp, fused, fused_times, max_delta_ns)
        return fused_pose, raw_pose
    raise ValueError('fused and raw odometry streams have no valid overlap')


def write_csv_atomic(path: Path, rows: list[dict], fields: list[str]) -> None:
    temporary = path.with_suffix('.tmp.csv')
    with temporary.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    sequence = args.sequence.resolve()
    manifest_path = sequence / 'manifest.csv'
    with manifest_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        manifest_fields = list(reader.fieldnames or [])
        rows = list(reader)
    read_times = {int(row['timestamp_ns']) for row in rows}
    sensor_times = lidar_sensor_times(args.lidar_topic.resolve(), read_times)

    fused, raw = load_fused_odometry(
        args.odometry_dir.resolve(), args.proto_dir.resolve())
    fused_times = [row['message_ms'] * 1_000_000 for row in fused]
    raw_times = [row['message_ms'] * 1_000_000 for row in raw]
    max_delta_ns = round(args.max_odometry_delta_ms * 1_000_000)
    boundaries = {
        'prefix': overlap_boundary(
            fused, fused_times, raw, raw_times, max_delta_ns, reverse=False),
        'tail': overlap_boundary(
            fused, fused_times, raw, raw_times, max_delta_ns, reverse=True),
    }

    synchronized = []
    poses = []
    delays = []
    for index, row in enumerate(rows):
        read_time = int(row['timestamp_ns'])
        sensor_time = sensor_times[read_time]
        delay = read_time - sensor_time
        row['lidar_sensor_timestamp_ns'] = sensor_time
        row['lidar_transport_delay_ns'] = delay
        row['odometry_index'] = index
        pose = synchronize_pose(
            sensor_time, fused, fused_times, raw, raw_times, boundaries,
            max_delta_ns)
        synchronized.append(pose)
        poses.append(pose_matrix(pose))
        delays.append(delay)

    if 'lidar_sensor_timestamp_ns' not in manifest_fields:
        manifest_fields.insert(1, 'lidar_sensor_timestamp_ns')
    if 'lidar_transport_delay_ns' not in manifest_fields:
        manifest_fields.insert(2, 'lidar_transport_delay_ns')

    outputs = [
        manifest_path, sequence / 'odometry.csv',
        sequence / 'odometry_poses.npy',
    ]
    for path in outputs:
        backup = path.with_name(
            f'{path.stem}.read_timestamp_backup{path.suffix}')
        if path.exists() and not backup.exists():
            shutil.copy2(path, backup)

    write_csv_atomic(manifest_path, rows, manifest_fields)
    write_csv_atomic(
        sequence / 'odometry.csv', synchronized,
        list(synchronized[0].keys()))
    temporary_poses = sequence / 'odometry_poses.tmp.npy'
    np.save(temporary_poses, np.stack(poses))
    temporary_poses.replace(sequence / 'odometry_poses.npy')

    metadata_path = sequence / 'metadata.json'
    if metadata_path.exists():
        metadata = json.loads(metadata_path.read_text())
        metadata.setdefault('odometry', {})['synchronization_clock'] = (
            'LiDAR embedded sensor timestamp matched to protobuf '
            'localization timestamp')
        metadata['odometry']['timestamp_repair'] = (
            'reconstructed from raw Ouster payload field 1')
        metadata['odometry']['max_delta_ms'] = args.max_odometry_delta_ms
        metadata['odometry']['lidar_read_minus_sensor_delay_ms'] = {
            'minimum': float(np.min(delays)) / 1e6,
            'median': float(np.median(delays)) / 1e6,
            'p90': float(np.percentile(delays, 90)) / 1e6,
            'maximum': float(np.max(delays)) / 1e6,
        }
        temporary = metadata_path.with_suffix('.tmp.json')
        temporary.write_text(json.dumps(metadata, indent=2) + '\n')
        temporary.replace(metadata_path)

    delay_ms = np.asarray(delays, dtype=float) / 1e6
    print(f'rebuilt {len(rows)} poses')
    print('LiDAR read-minus-sensor delay ms '
          f'min/median/p90/max={delay_ms.min():.3f}/'
          f'{np.median(delay_ms):.3f}/'
          f'{np.percentile(delay_ms, 90):.3f}/{delay_ms.max():.3f}')
    print(f'pose sources: fused={sum(p["source"] == "fused" for p in synchronized)}, '
          f'fallback={sum(p["source"] != "fused" for p in synchronized)}')


if __name__ == '__main__':
    main()
