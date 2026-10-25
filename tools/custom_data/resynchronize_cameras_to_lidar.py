#!/usr/bin/env python3
"""Repair exported camera frames using the LiDAR acquisition timestamp.

The folder-bag index timestamp for an Ouster scan is a recorder/read time and
can trail the timestamp embedded in the scan by hundreds of milliseconds.
Camera frames must be selected against ``lidar_sensor_timestamp_ns`` rather
than ``timestamp_ns``.  This tool decodes only the camera chunks needed by an
existing synchronized sequence and atomically replaces its native and OCam
image directories while retaining hard-link backups of the old images.
"""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import os
import re
import shutil
import subprocess
import tempfile
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np
from ocamcamera import OcamCamera

from bag_io import read_index
from convert_cameras import packet_timing
from prepare_fisheye3dod_inference import make_remap


CAMERAS = {
    'ARC_F': ('arc_f', 'fisheye_camera_front'),
    'ARC_SL': ('arc_sl', 'fisheye_camera_left'),
    'ARC_SR': ('arc_sr', 'fisheye_camera_right'),
    'ARC_R': ('arc_r', 'fisheye_camera_rear'),
}


@dataclass(frozen=True)
class CameraFrame:
    timestamp_ns: int
    chunk: Path
    frame_index: int


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path, help='raw folder bag')
    parser.add_argument('sequence', type=Path,
                        help='existing synchronized_dual_images directory')
    parser.add_argument('--calibration', type=Path,
                        help='defaults to SEQUENCE.parent/calibration.json')
    parser.add_argument(
        '--target-ocam', type=Path,
        default=Path('data/Fisheye3DODdataset/calib_results.txt'))
    parser.add_argument('--target-fov', type=float, default=220.0)
    parser.add_argument('--max-delta-ms', type=float, default=150.0,
                        help='fail if a nearest camera is farther away')
    parser.add_argument('--warning-delta-ms', type=float, default=40.0)
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--apply', action='store_true',
                        help='decode and replace images; otherwise only plan')
    return parser.parse_args()


def camera_frames(bag: Path, folder_name: str) -> list[CameraFrame]:
    folder = bag / f'__engineering__adcu__debug__ZMQ__camera__{folder_name}'
    frames = []
    for index_path in sorted(folder.glob('*.idx')):
        chunk = index_path.with_suffix('.h264')
        if not chunk.is_file():
            raise FileNotFoundError(chunk)
        frames.extend(
            CameraFrame(int(timestamp), chunk, frame_index)
            for frame_index, timestamp in enumerate(read_index(index_path)))
    if not frames:
        raise FileNotFoundError(f'no indexed camera chunks in {folder}')
    frames.sort(key=lambda item: item.timestamp_ns)
    return frames


def nearest(frames: list[CameraFrame], times: list[int], target: int
            ) -> CameraFrame:
    position = bisect.bisect_left(times, target)
    candidates = [index for index in (position - 1, position)
                  if 0 <= index < len(frames)]
    return frames[min(candidates, key=lambda index: abs(times[index] - target))]


def write_jpeg(path: Path, image: np.ndarray, quality: int) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix('.tmp.jpg')
    if not cv2.imwrite(str(temporary), image,
                       [cv2.IMWRITE_JPEG_QUALITY, quality]):
        raise RuntimeError(f'failed to write {temporary}')
    temporary.replace(path)


def decode_selected_chunk(chunk: Path, wanted: dict[int, list[tuple]],
                          native_root: Path, ocam_root: Path,
                          model_name: str, remap: tuple[np.ndarray, np.ndarray],
                          quality: int) -> int:
    packet_pts, time_base = packet_timing(chunk)
    pts_to_index = {pts: index for index, pts in enumerate(packet_pts)}
    if len(pts_to_index) != len(packet_pts):
        raise RuntimeError(f'{chunk}: duplicate packet PTS')
    with tempfile.TemporaryDirectory(prefix='camera-resync-') as temporary:
        pattern = str(Path(temporary) / '%08d.jpg')
        qscale = max(2, min(31, round((100 - quality) * .29 + 2)))
        command = [
            'ffmpeg', '-v', 'info', '-copyts', '-i', str(chunk),
            '-vf', 'showinfo,setpts=N', '-fps_mode', 'passthrough',
            '-enc_time_base', time_base, '-q:v', str(qscale), pattern,
        ]
        result = subprocess.run(command, capture_output=True, text=True)
        if result.returncode:
            raise RuntimeError(f'{chunk}: FFmpeg failed:\n{result.stderr[-4000:]}')
        decoded_pts = [
            int(match.group(1)) for match in re.finditer(
                r'Parsed_showinfo_.+?\bpts:\s*(-?\d+)', result.stderr)
        ]
        decoded_files = sorted(Path(temporary).glob('*.jpg'),
                               key=lambda path: int(path.stem))
        if len(decoded_pts) < len(decoded_files):
            raise RuntimeError(
                f'{chunk}: decoded {len(decoded_files)} frames but captured '
                f'only {len(decoded_pts)} PTS values')
        written = 0
        decoded_by_source = {}
        for decoded_file, pts in zip(decoded_files, decoded_pts):
            source_index = pts_to_index.get(pts)
            if source_index is not None:
                decoded_by_source[source_index] = decoded_file
            if source_index not in wanted:
                continue
            image = cv2.imread(str(decoded_file), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'failed to read {decoded_file}')
            map_x, map_y = remap
            warped = cv2.remap(
                image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT)
            for token, _manifest_index in wanted[source_index]:
                native = native_root / model_name / f'{token}.jpg'
                native.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(decoded_file, native)
                write_jpeg(ocam_root / model_name / f'{token}.jpg',
                           warped, quality)
                written += 1
        # A chunk can start with one non-decodable inter-frame packet.  Keep
        # synchronization deterministic by substituting the closest decoded
        # frame in that same chunk for only the missing selection.  This is
        # preferable to aborting an otherwise complete sequence, and limits
        # the additional timing error to the adjacent camera frame interval.
        missing = sorted(set(wanted) - decoded_by_source.keys())
        for source_index in missing:
            if not decoded_by_source:
                raise RuntimeError(f'{chunk}: no decodable camera frames')
            replacement_index = min(
                decoded_by_source, key=lambda index: abs(index - source_index))
            decoded_file = decoded_by_source[replacement_index]
            image = cv2.imread(str(decoded_file), cv2.IMREAD_COLOR)
            if image is None:
                raise RuntimeError(f'failed to read {decoded_file}')
            map_x, map_y = remap
            warped = cv2.remap(
                image, map_x, map_y, interpolation=cv2.INTER_LINEAR,
                borderMode=cv2.BORDER_CONSTANT)
            for token, _manifest_index in wanted[source_index]:
                native = native_root / model_name / f'{token}.jpg'
                native.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(decoded_file, native)
                write_jpeg(ocam_root / model_name / f'{token}.jpg',
                           warped, quality)
                written += 1
            print(f'{chunk}: substituted decoded frame {replacement_index} '
                  f'for unavailable selected frame {source_index}')
        return written


def hardlink_backup(source: Path, backup: Path) -> None:
    if backup.exists():
        raise FileExistsError(f'backup already exists: {backup}')
    shutil.copytree(source, backup, copy_function=os.link)


def main() -> None:
    args = parse_args()
    bag = args.bag.resolve()
    sequence = args.sequence.resolve()
    calibration_path = (args.calibration or
                        sequence.parent / 'calibration.json').resolve()
    if not 1 <= args.jpeg_quality <= 100:
        raise ValueError('--jpeg-quality must be between 1 and 100')
    rows = list(csv.DictReader((sequence / 'manifest.csv').open(newline='')))
    if not rows or any(not row.get('lidar_sensor_timestamp_ns') for row in rows):
        raise ValueError('manifest has no LiDAR sensor timestamps')

    calibration = json.loads(calibration_path.read_text())
    ocam = OcamCamera(filename=str(args.target_ocam.resolve()),
                      fov=args.target_fov)
    remaps = {
        sensor: make_remap(ocam, calibration['intrinsics'][sensor])
        for sensor in CAMERAS
    }
    selections = {}
    summary = {}
    maximum_ns = round(args.max_delta_ms * 1e6)
    warning_ns = round(args.warning_delta_ms * 1e6)
    for sensor, (prefix, model_name) in CAMERAS.items():
        frames = camera_frames(bag, prefix)
        times = [frame.timestamp_ns for frame in frames]
        chosen = []
        deltas = []
        by_chunk = defaultdict(lambda: defaultdict(list))
        for manifest_index, row in enumerate(rows):
            target = int(row['lidar_sensor_timestamp_ns'])
            frame = nearest(frames, times, target)
            delta = frame.timestamp_ns - target
            if abs(delta) > maximum_ns:
                raise RuntimeError(
                    f'{sensor} frame {manifest_index}: nearest camera delta '
                    f'{delta / 1e6:.3f} ms exceeds {args.max_delta_ms:g} ms')
            token = row['timestamp_ns']
            chosen.append(frame)
            deltas.append(delta)
            by_chunk[frame.chunk][frame.frame_index].append(
                (token, manifest_index))
            row[f'{prefix}_source'] = str(frame.chunk.relative_to(bag))
            row[f'{prefix}_timestamp_ns'] = str(frame.timestamp_ns)
            row[f'{prefix}_delta_ns'] = str(delta)
            row[f'{prefix}_native'] = (
                f'images_native/{model_name}/{token}.jpg')
            row[f'{prefix}_ocam_800x800'] = (
                f'images_ocam_800x800/{model_name}/{token}.jpg')
        absolute = sorted(abs(delta) for delta in deltas)
        summary[sensor] = {
            'frames': len(chosen),
            'median_abs_delta_ms': float(np.median(absolute) / 1e6),
            'p90_abs_delta_ms': float(
                absolute[round(.9 * (len(absolute) - 1))] / 1e6),
            'max_abs_delta_ms': max(absolute) / 1e6,
            'over_warning': sum(abs(delta) > warning_ns for delta in deltas),
            'chunks': len(by_chunk),
        }
        selections[sensor] = (model_name, by_chunk)

    report = {
        'clock': 'lidar_sensor_timestamp_ns',
        'frames': len(rows),
        'cameras': summary,
        'applied': bool(args.apply),
    }
    print(json.dumps(report, indent=2))
    if not args.apply:
        return

    staging = sequence / '.camera_resync_staging'
    if staging.exists():
        raise FileExistsError(staging)
    native_stage = staging / 'images_native'
    ocam_stage = staging / 'images_ocam_800x800'
    try:
        for sensor, (model_name, by_chunk) in selections.items():
            written = 0
            for chunk, wanted in sorted(by_chunk.items(), key=lambda item: str(item[0])):
                written += decode_selected_chunk(
                    chunk, wanted, native_stage, ocam_stage, model_name,
                    remaps[sensor], args.jpeg_quality)
            if written != len(rows):
                raise RuntimeError(
                    f'{sensor}: wrote {written}/{len(rows)} selected frames')

        native = sequence / 'images_native'
        canonical = sequence / 'images_ocam_800x800'
        hardlink_backup(native, sequence / 'images_native_read_sync_backup')
        hardlink_backup(
            canonical, sequence / 'images_ocam_800x800_read_sync_backup')
        old_manifest = sequence / 'manifest.csv'
        shutil.copy2(old_manifest,
                     sequence / 'manifest.camera_read_sync_backup.csv')
        shutil.rmtree(native)
        shutil.rmtree(canonical)
        (staging / 'images_native').replace(native)
        (staging / 'images_ocam_800x800').replace(canonical)

        fields = list(rows[0])
        temporary = old_manifest.with_suffix('.camera-sync.tmp.csv')
        with temporary.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields)
            writer.writeheader()
            writer.writerows(rows)
        temporary.replace(old_manifest)

        converted = sequence.parent
        sync_rows = []
        for row in rows:
            item = {
                'timestamp_ns': row['timestamp_ns'],
                'lidar_path': str(Path(sequence.name) / row['lidar_pcd']),
                'lidar_sensor_timestamp_ns': row['lidar_sensor_timestamp_ns'],
                'camera_sync_clock': 'sensor',
                'camera_sync_target_ns': row['lidar_sensor_timestamp_ns'],
            }
            for _sensor, (prefix, model_name) in CAMERAS.items():
                item[f'{prefix}_path'] = str(
                    Path(sequence.name) / 'images_native' / model_name /
                    f"{row['timestamp_ns']}.jpg")
                item[f'{prefix}_timestamp_ns'] = row[f'{prefix}_timestamp_ns']
                item[f'{prefix}_delta_ns'] = row[f'{prefix}_delta_ns']
            sync_rows.append(item)
        sync_path = converted / 'synchronization.csv'
        sync_tmp = sync_path.with_suffix('.tmp.csv')
        with sync_tmp.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=list(sync_rows[0]))
            writer.writeheader()
            writer.writerows(sync_rows)
        sync_tmp.replace(sync_path)

        report_path = sequence / 'camera_resynchronization_report.json'
        report_path.write_text(json.dumps(report, indent=2) + '\n')
        metadata_path = sequence / 'metadata.json'
        if metadata_path.is_file():
            metadata = json.loads(metadata_path.read_text())
            metadata['camera_synchronization'] = {
                'clock': 'LiDAR embedded sensor timestamp',
                'method': 'nearest raw camera frame by folder-bag index',
                'report': report_path.name,
                'read_clock_backup': 'images_native_read_sync_backup',
                'cameras': summary,
            }
            metadata_tmp = metadata_path.with_suffix('.tmp.json')
            metadata_tmp.write_text(json.dumps(metadata, indent=2) + '\n')
            metadata_tmp.replace(metadata_path)
    finally:
        if staging.exists():
            shutil.rmtree(staging)


if __name__ == '__main__':
    main()
