#!/usr/bin/env python3
"""Export synchronized native/OCam images and binary XYZI PCD files."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
import os
import shutil
import struct
import subprocess
import tempfile
from pathlib import Path

import cv2
import mmengine
import numpy as np
from ocamcamera import OcamCamera
from tqdm import tqdm

from prepare_fisheye3dod_inference import CAMERAS, make_remap


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('converted', type=Path,
                        help='converted bag containing synchronization.csv')
    parser.add_argument('calibration', type=Path,
                        help='camera calibration JSON')
    parser.add_argument('--sync', type=Path,
                        help='defaults to CONVERTED/synchronization.csv')
    parser.add_argument('--output', type=Path,
                        help='defaults to CONVERTED/synchronized_dual_images')
    parser.add_argument(
        '--target-ocam', type=Path,
        default=Path('data/Fisheye3DODdataset/calib_results.txt'))
    parser.add_argument('--target-fov', type=float, default=220.0)
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--copy-native', action='store_true',
                        help='copy native JPEGs instead of using hard links')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--sample-rate', type=float, default=2.0,
                        help='maximum output rate in Hz; use 0 for every row')
    parser.add_argument(
        '--template-info', type=Path,
        help='preserve the exact token order from an MMDetection3D info PKL')
    parser.add_argument(
        '--odometry-dir', type=Path,
        help='folder-bag FusedOdometry topic directory (*.bin + *.idx)')
    parser.add_argument(
        '--proto-dir', type=Path,
        help='directory containing bag.proto and localization.proto')
    parser.add_argument(
        '--max-odometry-delta-ms', type=float, default=200.0,
        help='reject frames farther than this from odometry (default: 200)')
    parser.add_argument('--limit', type=int,
                        help='export only the first N rows (useful for testing)')
    return parser.parse_args()


def load_fused_odometry(folder: Path, proto_dir: Path) -> tuple[list[dict], list[dict]]:
    """Decode fused and sibling raw odometry using the supplied protos."""
    from google.protobuf import descriptor_pb2, descriptor_pool, message_factory

    bag_proto = proto_dir / 'bag.proto'
    localization_proto = proto_dir / 'localization.proto'
    common_proto = proto_dir / 'common.proto'
    for path in (bag_proto, localization_proto, common_proto):
        if not path.is_file():
            raise FileNotFoundError(path)

    with tempfile.TemporaryDirectory(prefix='fused_odometry_proto_') as tmp:
        descriptor_path = Path(tmp) / 'messages.pb'
        subprocess.run([
            'protoc', f'-I{proto_dir}', '--include_imports',
            f'--descriptor_set_out={descriptor_path}',
            str(bag_proto), str(localization_proto), str(common_proto),
        ], check=True)
        descriptor_set = descriptor_pb2.FileDescriptorSet.FromString(
            descriptor_path.read_bytes())

    pool = descriptor_pool.DescriptorPool()
    pending = list(descriptor_set.file)
    while pending:
        remaining = []
        for descriptor in pending:
            try:
                pool.AddSerializedFile(descriptor.SerializeToString())
            except TypeError:
                remaining.append(descriptor)
        if len(remaining) == len(pending):
            raise RuntimeError('could not resolve protobuf descriptor dependencies')
        pending = remaining
    topic_type = message_factory.GetMessageClass(
        pool.FindMessageTypeByName('bag.TopicData'))
    pose_type = message_factory.GetMessageClass(
        pool.FindMessageTypeByName('adas.localization.EgoPoseFused'))
    raw_pose_type = message_factory.GetMessageClass(
        pool.FindMessageTypeByName('adas.common.EgoPose'))

    def topic_samples(topic_folder: Path, message_type, fused: bool) -> list[dict]:
        result = []
        bin_paths = sorted(topic_folder.glob('*.bin'))
        if not bin_paths:
            raise FileNotFoundError(f'no .bin files in {topic_folder}')
        for bin_path in bin_paths:
            index_path = bin_path.with_suffix('.idx')
            if not index_path.is_file():
                raise FileNotFoundError(index_path)
            topic = topic_type.FromString(bin_path.read_bytes())
            index = index_path.read_bytes()
            if len(index) != 8 * len(topic.samples):
                raise ValueError(
                    f'{index_path}: {len(index) // 8} timestamps for '
                    f'{len(topic.samples)} samples')
            for number, sample in enumerate(topic.samples):
                pose = message_type.FromString(sample.data)
                if fused:
                    position, yaw = pose.position, pose.orientation.yaw
                    roll, pitch = pose.orientation.roll, pose.orientation.pitch
                    velocity = pose.velocity
                    yaw_rate = pose.yaw_rate
                else:
                    position, yaw = pose.pose.position, pose.pose.orientation
                    roll = pitch = yaw_rate = 0.0
                    velocity = None
                result.append({
                    'wall_ns': struct.unpack_from('<Q', index, number * 8)[0],
                    'message_ms': int(pose.timestamp),
                    'x': float(position.x), 'y': float(position.y),
                    'z': float(getattr(position, 'z', 0.0)),
                    'roll': float(roll), 'pitch': float(pitch),
                    'yaw': float(yaw),
                    'vx': float(velocity.x) if velocity is not None else 0.0,
                    'vy': float(velocity.y) if velocity is not None else 0.0,
                    'vz': float(velocity.z) if velocity is not None else 0.0,
                    'yaw_rate': float(yaw_rate), 'status': int(pose.status),
                })
        result.sort(key=lambda row: row['wall_ns'])
        return result

    raw_folder = folder.with_name(folder.name.replace('FusedOdometry', 'Odometry'))
    return (topic_samples(folder, pose_type, True),
            topic_samples(raw_folder, raw_pose_type, False))


def interpolate_angle(a: float, b: float, fraction: float) -> float:
    delta = (b - a + math.pi) % (2 * math.pi) - math.pi
    return a + fraction * delta


def synchronize_odometry(timestamp_ns: int, samples: list[dict],
                         sample_times: list[int], max_delta_ns: int) -> dict:
    after = bisect.bisect_left(sample_times, timestamp_ns)
    before = max(0, after - 1)
    after = min(after, len(samples) - 1)
    nearest = min((before, after),
                  key=lambda index: abs(sample_times[index] - timestamp_ns))
    delta_ns = sample_times[nearest] - timestamp_ns
    if abs(delta_ns) > max_delta_ns:
        raise ValueError(
            f'no FusedOdometry within {max_delta_ns / 1e6:.1f} ms of '
            f'LiDAR timestamp {timestamp_ns}; nearest delta is '
            f'{delta_ns / 1e6:.1f} ms')

    lo, hi = samples[before], samples[after]
    span = sample_times[after] - sample_times[before]
    fraction = (0.0 if span == 0 else
                (timestamp_ns - sample_times[before]) / span)
    fraction = min(1.0, max(0.0, fraction))
    linear = ('x', 'y', 'z', 'roll', 'pitch', 'vx', 'vy', 'vz', 'yaw_rate')
    synced = {
        key: lo[key] + fraction * (hi[key] - lo[key]) for key in linear
    }
    synced.update({
        'timestamp_ns': timestamp_ns,
        'source_wall_ns': samples[nearest]['wall_ns'],
        'source_message_ms': samples[nearest]['message_ms'],
        'delta_ns': delta_ns,
        'yaw': interpolate_angle(lo['yaw'], hi['yaw'], fraction),
        'status': samples[nearest]['status'],
    })
    return synced


def align_raw_odometry(raw: dict, fused_boundary: dict,
                       raw_boundary: dict, source: str) -> dict:
    # This is a frame-to-frame rotation, not an interpolated absolute yaw.
    # Using interpolate_angle(..., 1.0) returned fused_boundary['yaw'] and
    # caused a large orientation jump at the raw/fused source hand-off.
    delta_yaw = (
        fused_boundary['yaw'] - raw_boundary['yaw'] + math.pi
    ) % (2 * math.pi) - math.pi
    cosine, sine = math.cos(delta_yaw), math.sin(delta_yaw)
    boundary_rotated = np.array([
        cosine * raw_boundary['x'] - sine * raw_boundary['y'],
        sine * raw_boundary['x'] + cosine * raw_boundary['y'],
    ])
    translation = np.array([fused_boundary['x'], fused_boundary['y']]) - \
        boundary_rotated
    xy = np.array([
        cosine * raw['x'] - sine * raw['y'],
        sine * raw['x'] + cosine * raw['y'],
    ]) + translation
    result = dict(raw)
    result.update({
        'x': float(xy[0]), 'y': float(xy[1]),
        'z': fused_boundary['z'],
        'roll': fused_boundary['roll'], 'pitch': fused_boundary['pitch'],
        'yaw': raw['yaw'] + delta_yaw,
        'source': source,
    })
    return result


def pose_matrix(row: dict) -> np.ndarray:
    cr, sr = math.cos(row['roll']), math.sin(row['roll'])
    cp, sp = math.cos(row['pitch']), math.sin(row['pitch'])
    cy, sy = math.cos(row['yaw']), math.sin(row['yaw'])
    rotation = np.array([
        [cy * cp, cy * sp * sr - sy * cr, cy * sp * cr + sy * sr],
        [sy * cp, sy * sp * sr + cy * cr, sy * sp * cr - cy * sr],
        [-sp, cp * sr, cp * cr],
    ], dtype=np.float64)
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = rotation
    pose[:3, 3] = (row['x'], row['y'], row['z'])
    return pose


def pcd_header(point_count: int) -> bytes:
    return (
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
    ).encode('ascii')


def convert_bin_to_pcd(source: Path, destination: Path,
                       overwrite: bool) -> int:
    size = source.stat().st_size
    if size % 16:
        raise ValueError(f'{source}: size {size} is not float32 XYZI')
    point_count = size // 16
    if destination.exists() and not overwrite:
        return point_count
    temporary = destination.with_suffix('.tmp.pcd')
    with temporary.open('wb') as stream:
        stream.write(pcd_header(point_count))
        with source.open('rb') as raw:
            shutil.copyfileobj(raw, stream, length=8 * 1024 * 1024)
    temporary.replace(destination)
    return point_count


def link_or_copy(source: Path, destination: Path, copy: bool,
                 overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            return
        destination.unlink()
    if copy:
        shutil.copy2(source, destination)
        return
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def write_jpeg(destination: Path, image, quality: int) -> None:
    temporary = destination.with_suffix('.tmp.jpg')
    if not cv2.imwrite(str(temporary), image,
                       [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f'failed to write {destination}')
    temporary.replace(destination)


def main() -> None:
    args = parse_args()
    converted = args.converted.resolve()
    calibration_path = args.calibration.resolve()
    sync_path = (args.sync or converted / 'synchronization.csv').resolve()
    output = (args.output or converted / 'synchronized_dual_images').resolve()

    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError('--jpeg-quality must be between 1 and 100')
    if (args.odometry_dir is None) != (args.proto_dir is None):
        raise ValueError('--odometry-dir and --proto-dir must be used together')
    if args.max_odometry_delta_ms <= 0:
        raise ValueError('--max-odometry-delta-ms must be positive')
    with sync_path.open(newline='') as stream:
        rows = sorted(csv.DictReader(stream),
                      key=lambda row: int(row['timestamp_ns']))
    if args.sample_rate < 0:
        raise ValueError('--sample-rate must be nonnegative')
    if args.sample_rate:
        period_ns = round(1_000_000_000 / args.sample_rate)
        sampled_rows = []
        last_timestamp = None
        for row in rows:
            timestamp = int(row['timestamp_ns'])
            if (last_timestamp is None or
                    timestamp - last_timestamp >= period_ns):
                sampled_rows.append(row)
                last_timestamp = timestamp
        rows = sampled_rows
    if args.template_info is not None:
        template = mmengine.load(args.template_info.resolve())
        by_token = {row['timestamp_ns']: row for row in rows}
        tokens = [str(item['token']) for item in template['data_list']]
        missing = [token for token in tokens if token not in by_token]
        if missing:
            raise RuntimeError(
                f'{len(missing)} template tokens are absent after sampling; '
                'check --sample-rate and --sync')
        rows = [by_token[token] for token in tokens]
    if args.limit is not None:
        if args.limit < 1:
            raise ValueError('--limit must be positive')
        rows = rows[:args.limit]
    if not rows:
        raise RuntimeError(f'no synchronized rows in {sync_path}')
    print(
        f'Selected {len(rows)} synchronized frames '
        f'(sample_rate={args.sample_rate or "all"})')

    calibration = json.loads(calibration_path.read_text())
    ocam = OcamCamera(filename=str(args.target_ocam), fov=args.target_fov)
    if (ocam.width, ocam.height) != (800, 800):
        raise ValueError(
            f'target OCam size is {ocam.width}x{ocam.height}, expected 800x800')
    remaps = {
        sensor: make_remap(ocam, calibration['intrinsics'][sensor])
        for sensor in CAMERAS.values()
    }
    odometry_samples = raw_odometry_samples = None
    odometry_times = raw_odometry_times = None
    odometry_boundaries = None
    if args.odometry_dir is not None:
        odometry_samples, raw_odometry_samples = load_fused_odometry(
            args.odometry_dir.resolve(), args.proto_dir.resolve())
        # The .idx timestamp records when the bag writer flushed a sample and
        # can jump during recorder stalls.  The protobuf timestamp is the
        # localization measurement clock and matches the LiDAR epoch.
        odometry_times = [
            row['message_ms'] * 1_000_000 for row in odometry_samples]
        raw_odometry_times = [
            row['message_ms'] * 1_000_000 for row in raw_odometry_samples]
        max_delta_ns = int(args.max_odometry_delta_ms * 1e6)
        odometry_boundaries = {}
        for label, index in (('prefix', 0), ('tail', -1)):
            timestamp_ns = odometry_times[index]
            odometry_boundaries[label] = (
                synchronize_odometry(timestamp_ns, odometry_samples,
                                     odometry_times, max_delta_ns),
                synchronize_odometry(timestamp_ns, raw_odometry_samples,
                                     raw_odometry_times, max_delta_ns))

    native_root = output / 'images_native'
    ocam_root = output / 'images_ocam_800x800'
    lidar_root = output / 'lidar_pcd'
    for model_name in CAMERAS:
        (native_root / model_name).mkdir(parents=True, exist_ok=True)
        (ocam_root / model_name).mkdir(parents=True, exist_ok=True)
    lidar_root.mkdir(parents=True, exist_ok=True)

    manifest_rows = []
    synchronized_odometry = []
    odometry_poses = []
    for row in tqdm(rows, desc='Exporting synchronized frames', unit='frame'):
        token = row['timestamp_ns']
        lidar_sensor_timestamp_ns = int(
            row.get('lidar_sensor_timestamp_ns') or token)
        exported = {
            'timestamp_ns': token,
            'lidar_sensor_timestamp_ns': lidar_sensor_timestamp_ns,
            'lidar_transport_delay_ns': (
                int(token) - lidar_sensor_timestamp_ns),
            'lidar_source': row['lidar_path'],
        }
        lidar_source = converted / row['lidar_path']
        lidar_relative = Path('lidar_pcd') / f'{token}.pcd'
        exported['lidar_pcd'] = lidar_relative.as_posix()
        exported['point_count'] = convert_bin_to_pcd(
            lidar_source, output / lidar_relative, args.overwrite)

        if odometry_samples is not None:
            # The bag/read timestamp can trail acquisition by hundreds of
            # milliseconds (or seconds during recorder backlog).  Transform
            # the cloud with the pose at its embedded sensor timestamp.
            timestamp_ns = lidar_sensor_timestamp_ns
            max_delta_ns = int(args.max_odometry_delta_ms * 1e6)
            if timestamp_ns < odometry_times[0]:
                raw = synchronize_odometry(
                    timestamp_ns, raw_odometry_samples,
                    raw_odometry_times, max_delta_ns)
                synchronized = align_raw_odometry(
                    raw, *odometry_boundaries['prefix'],
                    'aligned_odometry_prefix')
            elif timestamp_ns > odometry_times[-1]:
                raw = synchronize_odometry(
                    timestamp_ns, raw_odometry_samples,
                    raw_odometry_times, max_delta_ns)
                synchronized = align_raw_odometry(
                    raw, *odometry_boundaries['tail'],
                    'aligned_odometry_tail')
            else:
                try:
                    synchronized = synchronize_odometry(
                        timestamp_ns, odometry_samples, odometry_times,
                        max_delta_ns)
                    synchronized['source'] = 'fused'
                except ValueError as fused_error:
                    # FusedOdometry can have short recorder gaps even though
                    # the sibling raw Odometry topic remains continuous. Align
                    # raw poses at the nearest edge of the fused gap, just as
                    # is done for frames outside fused topic coverage.
                    try:
                        raw = synchronize_odometry(
                            timestamp_ns, raw_odometry_samples,
                            raw_odometry_times, max_delta_ns)
                        insertion = bisect.bisect_left(
                            odometry_times, timestamp_ns)
                        boundary_candidates = {
                            max(0, insertion - 1),
                            min(insertion, len(odometry_times) - 1),
                        }
                        boundary_index = min(
                            boundary_candidates,
                            key=lambda index: abs(
                                odometry_times[index] - timestamp_ns))
                        boundary_timestamp = odometry_times[boundary_index]
                        fused_boundary = synchronize_odometry(
                            boundary_timestamp, odometry_samples,
                            odometry_times, max_delta_ns)
                        raw_boundary = synchronize_odometry(
                            boundary_timestamp, raw_odometry_samples,
                            raw_odometry_times, max_delta_ns)
                    except ValueError as raw_error:
                        raise ValueError(
                            f'{fused_error}; raw Odometry fallback also '
                            f'failed: {raw_error}') from raw_error
                    synchronized = align_raw_odometry(
                        raw, fused_boundary, raw_boundary,
                        'aligned_odometry_gap')
            odometry_index = len(synchronized_odometry)
            synchronized_odometry.append(synchronized)
            odometry_poses.append(pose_matrix(synchronized))
            exported.update({
                'odometry_index': odometry_index,
                'odometry_delta_ns': synchronized['delta_ns'],
                'odometry_x': synchronized['x'],
                'odometry_y': synchronized['y'],
                'odometry_z': synchronized['z'],
                'odometry_roll': synchronized['roll'],
                'odometry_pitch': synchronized['pitch'],
                'odometry_yaw': synchronized['yaw'],
                'odometry_status': synchronized['status'],
            })

        for model_name, sensor in CAMERAS.items():
            key = sensor.lower()
            source = converted / row[f'{key}_path']
            native_relative = Path('images_native') / model_name / f'{token}.jpg'
            ocam_relative = (Path('images_ocam_800x800') / model_name /
                             f'{token}.jpg')
            native_destination = output / native_relative
            ocam_destination = output / ocam_relative

            image = cv2.imread(str(source), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'failed to read {source}')
            expected = calibration['intrinsics'][sensor]
            expected_size = (int(expected['width']), int(expected['height']))
            actual_size = (image.shape[1], image.shape[0])
            if actual_size != expected_size:
                raise ValueError(
                    f'{source}: image is {actual_size}, calibration expects '
                    f'{expected_size}')

            link_or_copy(source, native_destination, args.copy_native,
                         args.overwrite)
            if args.overwrite or not ocam_destination.exists():
                map_x, map_y = remaps[sensor]
                warped = cv2.remap(
                    image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                    borderMode=cv2.BORDER_CONSTANT)
                write_jpeg(ocam_destination, warped, args.jpeg_quality)

            exported[f'{key}_source'] = row[f'{key}_path']
            exported[f'{key}_delta_ns'] = row[f'{key}_delta_ns']
            exported[f'{key}_native'] = native_relative.as_posix()
            exported[f'{key}_ocam_800x800'] = ocam_relative.as_posix()
        manifest_rows.append(exported)

    manifest_path = output / 'manifest.csv'
    temporary_manifest = manifest_path.with_suffix('.tmp.csv')
    with temporary_manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=manifest_rows[0].keys())
        writer.writeheader()
        writer.writerows(manifest_rows)
    temporary_manifest.replace(manifest_path)

    if synchronized_odometry:
        odometry_path = output / 'odometry.csv'
        temporary_odometry = odometry_path.with_suffix('.tmp.csv')
        with temporary_odometry.open('w', newline='') as stream:
            writer = csv.DictWriter(
                stream, fieldnames=synchronized_odometry[0].keys())
            writer.writeheader()
            writer.writerows(synchronized_odometry)
        temporary_odometry.replace(odometry_path)
        poses_path = output / 'odometry_poses.npy'
        temporary_poses = output / 'odometry_poses.tmp.npy'
        np.save(temporary_poses, np.stack(odometry_poses))
        temporary_poses.replace(poses_path)

    metadata = {
        'source_converted': str(converted),
        'source_synchronization': str(sync_path),
        'source_calibration': str(calibration_path),
        'target_ocam': str(args.target_ocam.resolve()),
        'target_ocam_fov_degrees': args.target_fov,
        'target_ocam_size': [ocam.width, ocam.height],
        'native_images_preserved': True,
        'native_storage': 'copy' if args.copy_native else 'hard-link-or-copy',
        'point_cloud_format': 'binary PCD v0.7, float32 XYZI',
        'samples': len(manifest_rows),
        'sample_rate_hz': args.sample_rate or None,
        'template_info': (str(args.template_info.resolve())
                          if args.template_info else None),
        'camera_mapping': CAMERAS,
    }
    if odometry_samples is not None:
        metadata['odometry'] = {
            'source': str(args.odometry_dir.resolve()),
            'proto_source': str(args.proto_dir.resolve()),
            'synchronization_clock': (
                'LiDAR embedded sensor timestamp matched to protobuf '
                'localization timestamp'),
            'method': 'linear position/velocity and shortest-path Euler interpolation',
            'fallback': ('boundary-aligned sibling Localization/Odometry '
                         'outside fused coverage and across internal gaps'),
            'max_delta_ms': args.max_odometry_delta_ms,
            'csv': 'odometry.csv',
            'poses': 'odometry_poses.npy',
            'pose_convention': '4x4 local-to-global, Rz(yaw) Ry(pitch) Rx(roll)',
        }
    metadata_path = output / 'metadata.json'
    temporary_metadata = metadata_path.with_suffix('.tmp.json')
    temporary_metadata.write_text(json.dumps(metadata, indent=2) + '\n')
    temporary_metadata.replace(metadata_path)
    print(f'Exported {len(manifest_rows)} synchronized samples to {output}')
    print(f'Manifest: {manifest_path}')


if __name__ == '__main__':
    main()
