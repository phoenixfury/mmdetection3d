#!/usr/bin/env python3
"""Convert a CaliperGT LiDAR export to Fisheye3DOD MMDetection3D infos."""

from __future__ import annotations

import argparse
import copy
import json
from collections import Counter, defaultdict
from pathlib import Path

import mmengine
import numpy as np


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
    'construction_vehicle': 'Van',
    'bus': 'Bus',
    'pedestrian': 'Pedestrian',
    'cyclist': 'Cyclist',
    'bicycle': 'Cyclist',
    'motorcycle': 'Cyclist',
    'motorcycle_copy': 'Cyclist',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('scene', type=Path,
                        help='downloaded CaliperGT scene export directory')
    parser.add_argument('--dataset', type=Path,
                        help='prepared Fisheye3DOD custom dataset root')
    parser.add_argument('--input-info', type=Path,
                        help='annotation-free fisheye3dod_infos_test.pkl')
    parser.add_argument('--output', type=Path,
                        help='output directory (default: DATASET/caliper_annotations)')
    parser.add_argument('--val-fraction', type=float, default=0.20)
    parser.add_argument('--split-block-seconds', type=float, default=40.0)
    parser.add_argument(
        '--include-empty', action='store_true',
        help='include reviewed frames without supported boxes in train/val')
    parser.add_argument(
        '--flatten-pitch-roll', action='store_true',
        help='discard pitch/roll because Fisheye3DOD boxes encode yaw only')
    return parser.parse_args()


def find_lidar_json(scene: Path) -> Path:
    candidates = sorted((scene / 'annotations' / 'tasks').glob('*/lidar.json'))
    populated = []
    for path in candidates:
        document = json.loads(path.read_text())
        if document.get('annotations'):
            populated.append((path, document))
    if len(populated) != 1:
        raise RuntimeError(
            f'expected exactly one populated lidar.json below {scene}, '
            f'found {len(populated)}')
    return populated[0][0]


def infer_dataset(scene: Path) -> Path:
    # Caliper downloads are normally placed directly below the prepared root.
    candidate = scene.parent
    if (candidate / 'fisheye3dod_infos_test.pkl').is_file():
        return candidate
    raise RuntimeError('could not infer dataset root; pass --dataset')


def validate_alignment(document: dict, metadata: dict, samples: list) -> None:
    frames = document.get('frames', [])
    exported_frames = metadata.get('frames', [])
    if len(frames) != len(samples) or len(exported_frames) != len(samples):
        raise ValueError(
            'frame count mismatch: labels=%d metadata=%d input-info=%d' %
            (len(frames), len(exported_frames), len(samples)))
    for index, (label_frame, exported_frame) in enumerate(
            zip(frames, exported_frames)):
        if int(label_frame['id']) != index:
            raise ValueError(f'label frame {index} has id {label_frame["id"]}')
        if int(exported_frame['frame_index']) != index:
            raise ValueError(
                f'metadata frame {index} has index '
                f'{exported_frame["frame_index"]}')
        label_name = Path(label_frame['file_name']).name
        metadata_name = Path(exported_frame['file_paths']['lidar']).name
        if label_name != metadata_name:
            raise ValueError(
                f'frame {index} filename mismatch: {label_name} != '
                f'{metadata_name}')


def main() -> None:
    args = parse_args()
    scene = args.scene.resolve()
    dataset = (args.dataset.resolve() if args.dataset else infer_dataset(scene))
    input_info = (args.input_info.resolve() if args.input_info else
                  dataset / 'fisheye3dod_infos_test.pkl')
    output = (args.output.resolve() if args.output else
              dataset / 'caliper_annotations')
    if not 0 < args.val_fraction < 1:
        raise ValueError('--val-fraction must be between zero and one')

    lidar_json = find_lidar_json(scene)
    document = json.loads(lidar_json.read_text())
    metadata = json.loads((scene / 'metadata.json').read_text())
    source = mmengine.load(input_info)
    samples = source['data_list']
    validate_alignment(document, metadata, samples)

    by_frame = defaultdict(list)
    skipped_classes = Counter()
    class_counts = Counter()
    source_counts = Counter()
    flattened_boxes = 0
    for annotation in document['annotations']:
        frame_id = int(annotation['frame_id'])
        if not 0 <= frame_id < len(samples):
            raise ValueError(f'annotation frame_id {frame_id} is out of range')
        raw_class = str(annotation.get('class', '')).strip().lower()
        class_name = CLASS_MAP.get(raw_class)
        if class_name is None:
            skipped_classes[raw_class or '<missing>'] += 1
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
                f'annotation {annotation.get("id")} has an invalid cuboid')
        if abs(float(rotation.get('pitch', 0))) > 1e-6 or abs(
                float(rotation.get('roll', 0))) > 1e-6:
            if not args.flatten_pitch_roll:
                raise ValueError(
                    f'annotation {annotation.get("id")} has nonzero '
                    'pitch/roll; Fisheye3DOD boxes only support yaw')
            flattened_boxes += 1
        by_frame[frame_id].append((annotation, class_name, box))
        class_counts[class_name] += 1
        source_counts[str(annotation.get('source', 'unknown'))] += 1

    annotated = []
    for frame_id, sample in enumerate(samples):
        records = by_frame[frame_id]
        item = copy.deepcopy(sample)
        item['annos'] = {
            'gt_boxes': np.asarray(
                [record[2] for record in records], dtype=np.float32
            ).reshape(-1, 7),
            'gt_names': np.asarray([record[1] for record in records]),
            'gt_mesh': np.asarray([
                f'caliper.{record[1].lower()}' for record in records
            ]),
            'gt_ids': np.asarray([
                int(record[0]['id']) for record in records
            ], dtype=np.int64),
        }
        annotated.append(item)

    labeled = [item for item in annotated if len(item['annos']['gt_names'])]
    if not labeled:
        raise RuntimeError('the export contains no supported labeled samples')
    split_samples = annotated if args.include_empty else labeled
    cycle = max(2, round(1.0 / args.val_fraction))
    block_ns = max(1, round(args.split_block_seconds * 1e9))
    first_token = int(split_samples[0]['token'])
    train, val = [], []
    for item in split_samples:
        block = (int(item['token']) - first_token) // block_ns
        (val if block % cycle == cycle - 1 else train).append(item)
    if not train or not val:
        raise RuntimeError('temporal block split produced an empty train or val set')

    metainfo = dict(source.get('metainfo', {}))
    metainfo.update({
        'categories': CATEGORIES,
        'classes': tuple(CATEGORIES),
        'dataset': 'custom_caliper_human_annotations',
    })
    output.mkdir(parents=True, exist_ok=True)
    for name, data_list in [('all', annotated), ('train', train), ('val', val)]:
        mmengine.dump(
            {'metainfo': metainfo, 'data_list': data_list},
            output / f'fisheye3dod_infos_{name}.pkl')

    summary = {
        'label_kind': 'CaliperGT human-reviewed LiDAR annotations',
        'source_scene': str(scene),
        'source_annotations': str(lidar_json),
        'source_info': str(input_info),
        'samples': len(annotated),
        'labeled_samples': len(labeled),
        'excluded_empty_samples': len(annotated) - len(labeled),
        'train_samples': len(train),
        'val_samples': len(val),
        'boxes': int(sum(class_counts.values())),
        'class_counts': dict(class_counts),
        'annotation_source_counts': dict(source_counts),
        'skipped_unsupported_classes': dict(skipped_classes),
        'flattened_pitch_roll_boxes': flattened_boxes,
        'settings': {
            'val_fraction': args.val_fraction,
            'split_block_seconds': args.split_block_seconds,
            'include_empty': args.include_empty,
        },
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))
    print(f'outputs: {output}')


if __name__ == '__main__':
    main()
