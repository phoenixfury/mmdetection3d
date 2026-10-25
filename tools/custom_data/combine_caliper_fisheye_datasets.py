#!/usr/bin/env python3
"""Merge the two custom bags and CaliperGT labels for Fisheye3DOD training."""

from __future__ import annotations

import argparse
import copy
import json
import os
import shutil
import struct
import zipfile
from collections import Counter, defaultdict
from pathlib import Path

import mmengine
import numpy as np


ROOT = Path(__file__).resolve().parents[2]
CUSTOM = ROOT / 'data' / 'Custom'
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
CLASS_MAP = {
    'car': 'Car',
    'van': 'Van',
    'truck': 'Truck',
    'bus': 'Bus',
    'pedestrian': 'Pedestrian',
    'cyclist': 'Cyclist',
    'bicycle': 'Cyclist',
    'motorcycle': 'Cyclist',
    'motorcycle_copy': 'Cyclist',
}
DEFAULT_SEQUENCES = (
    (
        '16_09',
        CUSTOM / 'BAG_2026_09_16_09_58_56_converted'
        / 'synchronized_dual_images',
        CUSTOM / '16_09_almost_ready.zip',
        CUSTOM / 'BAG_2026_09_16_09_58_56_converted'
        / 'fisheye3dod_inference_2hz' / 'fisheye3dod_infos_test.pkl',
    ),
    (
        '30_09',
        CUSTOM / 'BAG_2026_09_30_10_10_54_converted'
        / 'synchronized_dual_images',
        CUSTOM / '30_09_almost_ready.zip',
        CUSTOM / 'BAG_2026_09_30_10_10_54_converted'
        / 'fisheye3dod_native_2hz' / 'fisheye3dod_infos_test.pkl',
    ),
)
DEFAULT_CALIBRATION = (
    CUSTOM / 'BAG_2026_09_30_10_10_54_converted' / 'calibration.json')
DEFAULT_OCAM_CALIBRATION = ROOT / 'data' / 'Fisheye3DODdataset' / 'calib_results.txt'
DEFAULT_OUTPUT = CUSTOM / 'combined_caliper_fisheye3dod'


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        '--sequence', action='append', nargs=4,
        metavar=('NAME', 'SYNC_DIR', 'CALIPER_ZIP', 'TEMPLATE_INFO'),
        help=('sequence specification; repeatable. The two requested bags are '
              'used when this option is omitted'))
    parser.add_argument('--calibration', type=Path,
                        default=DEFAULT_CALIBRATION,
                        help='one shared calibration applied to every bag')
    parser.add_argument('--ocam-calibration', type=Path,
                        default=DEFAULT_OCAM_CALIBRATION,
                        help='shared calibration for the remapped 800x800 images')
    parser.add_argument('--output', type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument('--storage', choices=('hardlink', 'copy', 'symlink'),
                        default='hardlink',
                        help='how images and LiDAR BIN files are materialized')
    parser.add_argument('--val-fraction', type=float, default=0.20)
    parser.add_argument('--split-block-seconds', type=float, default=40.0)
    parser.add_argument('--overwrite', action='store_true',
                        help='replace conflicting output files')
    parser.add_argument('--dry-run', action='store_true',
                        help='validate and report without writing output')
    return parser.parse_args()


def load_caliper_export(path: Path) -> tuple[dict, dict, str]:
    with zipfile.ZipFile(path) as archive:
        names = archive.namelist()
        metadata_names = [name for name in names
                          if Path(name).name == 'metadata.json']
        if len(metadata_names) != 1:
            raise ValueError(f'{path}: expected one metadata.json')
        metadata = json.loads(archive.read(metadata_names[0]))

        populated = []
        for name in names:
            if Path(name).name != 'lidar.json':
                continue
            document = json.loads(archive.read(name))
            if document.get('annotations'):
                populated.append((name, document))
        if len(populated) != 1:
            raise ValueError(
                f'{path}: expected one populated lidar.json, '
                f'found {len(populated)}')
        annotation_name, document = populated[0]
    return document, metadata, annotation_name


def validate_alignment(name: str, document: dict, metadata: dict,
                       samples: list[dict]) -> None:
    label_frames = document.get('frames', [])
    export_frames = metadata.get('frames', [])
    counts = (len(label_frames), len(export_frames), len(samples))
    if len(set(counts)) != 1:
        raise ValueError(
            f'{name}: frame count mismatch: labels={counts[0]}, '
            f'metadata={counts[1]}, template={counts[2]}')
    for index, (label_frame, export_frame) in enumerate(
            zip(label_frames, export_frames)):
        if int(label_frame['id']) != index:
            raise ValueError(
                f'{name}: label frame {index} has id {label_frame["id"]}')
        if int(export_frame['frame_index']) != index:
            raise ValueError(
                f'{name}: metadata frame {index} has index '
                f'{export_frame["frame_index"]}')
        label_file = Path(label_frame['file_name']).name
        export_file = Path(export_frame['file_paths']['lidar']).name
        if label_file != export_file:
            raise ValueError(
                f'{name}: frame {index} filename mismatch: '
                f'{label_file} != {export_file}')


def shared_camera_info(calibration: dict) -> dict[str, dict]:
    result = {}
    for model_name, frame in CAMERAS.items():
        intrinsic = calibration['intrinsics'][frame]
        lidar2cam = np.asarray(
            calibration['lidar_to_camera'][frame], dtype=np.float32)
        cam2img = np.eye(4, dtype=np.float32)
        cam2img[:3, :3] = np.asarray(
            intrinsic['camera_matrix'], dtype=np.float32)
        result[model_name] = {
            'image_size': (int(intrinsic['width']),
                           int(intrinsic['height'])),
            'cam2img': cam2img,
            'lidar2img': np.eye(4, dtype=np.float32),
            'lidar2cam': lidar2cam,
            'cam2lidar': np.linalg.inv(lidar2cam).astype(np.float32),
        }
    return result


def annotations_by_frame(document: dict, sequence: str) -> tuple[dict, dict]:
    by_frame = defaultdict(list)
    classes = Counter()
    skipped = Counter()
    for annotation in document['annotations']:
        frame_id = int(annotation['frame_id'])
        raw_class = str(annotation.get('class', '')).strip().lower()
        class_name = CLASS_MAP.get(raw_class)
        if class_name is None:
            skipped[raw_class or '<missing>'] += 1
            continue
        cuboid = annotation['cuboid_3d']
        position = cuboid['position']
        dimensions = cuboid['dimensions']
        rotation = cuboid['rotation']
        box = np.asarray([
            position['x'], position['y'], position['z'],
            dimensions['length'], dimensions['width'], dimensions['height'],
            rotation['yaw'],
        ], dtype=np.float32)
        if not np.isfinite(box).all() or np.any(box[3:6] <= 0):
            raise ValueError(
                f'{sequence}: annotation {annotation.get("id")} '
                'has an invalid cuboid')
        if (abs(float(rotation.get('pitch', 0))) > 1e-6 or
                abs(float(rotation.get('roll', 0))) > 1e-6):
            raise ValueError(
                f'{sequence}: annotation {annotation.get("id")} '
                'has nonzero pitch/roll')
        by_frame[frame_id].append((class_name, box, annotation.get('id')))
        classes[class_name] += 1
    return by_frame, {'classes': classes, 'skipped': skipped}


def materialize(source: Path, destination: Path, mode: str,
                overwrite: bool) -> None:
    if destination.exists() or destination.is_symlink():
        if (not destination.is_symlink() and destination.stat().st_size ==
                source.stat().st_size):
            return
        if not overwrite:
            raise FileExistsError(f'conflicting output: {destination}')
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == 'copy':
        shutil.copy2(source, destination)
    elif mode == 'symlink':
        destination.symlink_to(source.resolve())
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def pcd_to_bin(source: Path, destination: Path, overwrite: bool) -> None:
    if destination.exists():
        if not overwrite:
            return
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix(destination.suffix + '.tmp')
    with source.open('rb') as src, temporary.open('wb') as dst:
        header = {}
        while True:
            line = src.readline()
            if not line:
                raise ValueError(f'{source}: missing PCD DATA header')
            text = line.decode('ascii').strip()
            if text and not text.startswith('#'):
                key, *values = text.split()
                header[key.upper()] = values
            if text.upper() == 'DATA BINARY':
                break
        if (header.get('FIELDS') != ['x', 'y', 'z', 'intensity'] or
                header.get('SIZE') != ['4', '4', '4', '4'] or
                header.get('TYPE') != ['F', 'F', 'F', 'F']):
            raise ValueError(f'{source}: expected float32 XYZI binary PCD')
        points = int(header['POINTS'][0])
        payload = src.read()
        expected = points * 4 * struct.calcsize('f')
        if len(payload) != expected:
            raise ValueError(
                f'{source}: payload is {len(payload)} bytes, expected '
                f'{expected}')
        dst.write(payload)
    temporary.replace(destination)


def temporal_split(items: list[dict], val_fraction: float,
                   block_seconds: float) -> tuple[list[dict], list[dict]]:
    cycle = max(2, round(1.0 / val_fraction))
    block_ns = max(1, round(block_seconds * 1e9))
    by_sequence = defaultdict(list)
    for item in items:
        if len(item['annos']['gt_names']):
            by_sequence[item['metainfo']['sequence_id']].append(item)
    train, val = [], []
    for sequence_items in by_sequence.values():
        sequence_items.sort(key=lambda item: int(item['token']))
        first_token = int(sequence_items[0]['token'])
        for item in sequence_items:
            block = (int(item['token']) - first_token) // block_ns
            (val if block % cycle == cycle - 1 else train).append(item)
    if not train or not val:
        raise RuntimeError('temporal block split produced an empty split')
    return train, val


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError('--val-fraction must be between zero and one')
    sequences = DEFAULT_SEQUENCES if args.sequence is None else tuple(
        (row[0], Path(row[1]), Path(row[2]), Path(row[3]))
        for row in args.sequence)
    calibration_path = args.calibration.resolve()
    ocam_calibration_path = args.ocam_calibration.resolve()
    output = args.output.resolve()
    calibration = json.loads(calibration_path.read_text())
    if not ocam_calibration_path.is_file():
        raise FileNotFoundError(ocam_calibration_path)
    camera_template = shared_camera_info(calibration)

    all_native_samples = []
    all_ocam_samples = []
    tokens = set()
    total_classes = Counter()
    total_skipped = Counter()
    sequence_summaries = []
    global_index = 0
    global_annotation_id = 0

    for sequence_name, sync_dir, archive_path, template_path in sequences:
        sync_dir = sync_dir.resolve()
        archive_path = archive_path.resolve()
        template_path = template_path.resolve()
        document, metadata, annotation_member = load_caliper_export(
            archive_path)
        source_info = mmengine.load(template_path)
        source_samples = source_info['data_list']
        validate_alignment(
            sequence_name, document, metadata, source_samples)
        by_frame, counts = annotations_by_frame(document, sequence_name)
        total_classes.update(counts['classes'])
        total_skipped.update(counts['skipped'])
        sequence_samples = []

        for frame_index, source_sample in enumerate(source_samples):
            token = str(source_sample['token'])
            if token in tokens:
                raise ValueError(f'duplicate token across bags: {token}')
            tokens.add(token)
            sample = copy.deepcopy(source_sample)
            sample['sample_idx'] = global_index
            sample['token'] = token
            sample.setdefault('metainfo', {})
            sample['metainfo'].update({
                'sequence_id': sequence_name,
                'scene_name': sequence_name,
                'frame_id': token,
                'prev_id': (str(source_samples[frame_index - 1]['token'])
                            if frame_index else ''),
                'next_id': (str(source_samples[frame_index + 1]['token'])
                            if frame_index + 1 < len(source_samples) else ''),
            })

            sample['cam_info'] = {'cam_fisheye': {}}
            for camera_name, fixed_info in camera_template.items():
                image_source = (sync_dir / 'images_native' / camera_name /
                                f'{token}.jpg')
                if not image_source.is_file():
                    raise FileNotFoundError(image_source)
                image_relative = (Path('images_native') / camera_name /
                                  f'{sequence_name}_{token}.jpg')
                camera_info = copy.deepcopy(fixed_info)
                camera_info['cam_path'] = image_relative.as_posix()
                sample['cam_info']['cam_fisheye'][camera_name] = camera_info
                if not args.dry_run:
                    materialize(image_source, output / image_relative,
                                args.storage, args.overwrite)

            ocam_sample = copy.deepcopy(sample)
            for camera_name, camera_info in (
                    ocam_sample['cam_info']['cam_fisheye'].items()):
                ocam_source = (sync_dir / 'images_ocam_800x800' /
                               camera_name / f'{token}.jpg')
                if not ocam_source.is_file():
                    raise FileNotFoundError(ocam_source)
                ocam_relative = (Path('images_ocam_800x800') / camera_name /
                                 f'{sequence_name}_{token}.jpg')
                camera_info['cam_path'] = ocam_relative.as_posix()
                camera_info['image_size'] = (800, 800)
                camera_info['cam2img'] = np.eye(4, dtype=np.float32)
                if not args.dry_run:
                    materialize(ocam_source, output / ocam_relative,
                                args.storage, args.overwrite)

            lidar_relative = Path('lidar') / f'{sequence_name}_{token}.bin'
            sample['pc_info'] = {
                'num_features': 4,
                'pts_path': lidar_relative.as_posix(),
                'semantic_pts_path': lidar_relative.as_posix(),
            }
            lidar_bin = sync_dir.parent / 'lidar' / f'{token}.bin'
            lidar_pcd = sync_dir / 'lidar_pcd' / f'{token}.pcd'
            if not args.dry_run:
                if lidar_bin.is_file():
                    materialize(lidar_bin, output / lidar_relative,
                                args.storage, args.overwrite)
                elif lidar_pcd.is_file():
                    pcd_to_bin(lidar_pcd, output / lidar_relative,
                               args.overwrite)
                else:
                    raise FileNotFoundError(
                        f'no LiDAR BIN or PCD for token {token}')
            elif not lidar_bin.is_file() and not lidar_pcd.is_file():
                raise FileNotFoundError(f'no LiDAR data for token {token}')

            records = by_frame[frame_index]
            ids = []
            for _ in records:
                ids.append(global_annotation_id)
                global_annotation_id += 1
            sample['annos'] = {
                'gt_boxes': np.asarray(
                    [record[1] for record in records], dtype=np.float32
                ).reshape(-1, 7),
                'gt_names': np.asarray([record[0] for record in records]),
                'gt_mesh': np.asarray([
                    f'caliper.{record[0].lower()}' for record in records]),
                'gt_ids': np.asarray(ids, dtype=np.int64),
            }
            ocam_sample['pc_info'] = copy.deepcopy(sample['pc_info'])
            ocam_sample['annos'] = copy.deepcopy(sample['annos'])
            sequence_samples.append(sample)
            all_native_samples.append(sample)
            all_ocam_samples.append(ocam_sample)
            global_index += 1

        sequence_summaries.append({
            'name': sequence_name,
            'synchronized_dataset': str(sync_dir),
            'caliper_export': str(archive_path),
            'annotation_member': annotation_member,
            'template_info': str(template_path),
            'samples': len(sequence_samples),
            'labeled_samples': sum(
                bool(len(item['annos']['gt_names']))
                for item in sequence_samples),
            'boxes': sum(counts['classes'].values()),
            'class_counts': dict(counts['classes']),
            'skipped_classes': dict(counts['skipped']),
        })

    native_train, native_val = temporal_split(
        all_native_samples, args.val_fraction, args.split_block_seconds)
    train_keys = {
        (item['metainfo']['sequence_id'], item['token'])
        for item in native_train
    }
    val_keys = {
        (item['metainfo']['sequence_id'], item['token'])
        for item in native_val
    }
    ocam_train = [
        item for item in all_ocam_samples
        if (item['metainfo']['sequence_id'], item['token']) in train_keys
    ]
    ocam_val = [
        item for item in all_ocam_samples
        if (item['metainfo']['sequence_id'], item['token']) in val_keys
    ]
    common_metainfo = {
        'categories': CATEGORIES,
        'classes': tuple(CATEGORIES),
        'dataset': 'combined_custom_caliper_fisheye3dod',
        'shared_calibration': str(calibration_path),
    }
    summary = {
        'output': str(output),
        'dry_run': args.dry_run,
        'storage': args.storage,
        'shared_calibration': str(calibration_path),
        'ocam_calibration': str(ocam_calibration_path),
        'samples': len(all_native_samples),
        'labeled_samples': sum(
            bool(len(item['annos']['gt_names']))
            for item in all_native_samples),
        'train_samples': len(native_train),
        'val_samples': len(native_val),
        'boxes': sum(total_classes.values()),
        'class_counts': dict(total_classes),
        'skipped_classes': dict(total_skipped),
        'sequences': sequence_summaries,
        'settings': {
            'val_fraction': args.val_fraction,
            'split_block_seconds': args.split_block_seconds,
        },
    }

    if not args.dry_run:
        variants = {
            'native': (all_native_samples, native_train, native_val),
            'ocam_800x800': (all_ocam_samples, ocam_train, ocam_val),
        }
        for variant, (all_data, train_data, val_data) in variants.items():
            annotation_dir = output / f'caliper_annotations_{variant}'
            annotation_dir.mkdir(parents=True, exist_ok=True)
            metainfo = dict(common_metainfo)
            metainfo['version'] = f'combined-{variant}-v1'
            if variant == 'ocam_800x800':
                metainfo['ocam_calibration'] = str(ocam_calibration_path)
            for split_name, data_list in (
                    ('all', all_data), ('train', train_data),
                    ('val', val_data)):
                mmengine.dump(
                    {'metainfo': metainfo, 'data_list': data_list},
                    annotation_dir / f'fisheye3dod_infos_{split_name}.pkl')
        shutil.copy2(calibration_path, output / 'calibration.json')
        shutil.copy2(ocam_calibration_path, output / 'calib_results.txt')
        (output / 'summary.json').write_text(
            json.dumps(summary, indent=2) + '\n')

    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
