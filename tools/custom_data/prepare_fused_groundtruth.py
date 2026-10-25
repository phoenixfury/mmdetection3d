#!/usr/bin/env python3
"""Create LiDAR-verified Fisheye3DOD pseudo ground truth from fused objects.

The decoder follows ``adas.dynamic_object_fusion.FusedObjectList`` from
object_fusion.proto.  Fused objects are expressed in BASELINK (kEgo).  Their
2D boxes are transformed to CLIDAR, checked against the synchronized point
cloud, and augmented with a LiDAR-derived vertical center and height.

These labels are suitable for bootstrapping/manual correction.  They are not
human ground truth: an onboard fusion output can contain both omissions and
systematic perception errors.
"""

from __future__ import annotations

import argparse
import copy
import csv
import json
import struct
import sys
from collections import Counter
from pathlib import Path
from typing import Dict, Iterable, Optional, Tuple

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmengine
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from bag_io import field_map, integer, iter_record_payloads, read_index, repeated_bytes


FUSED_TOPIC = '__engineering__adcu__debug__ZMQ__object_fusion__FusedObjectList'
CLASS_INFO = {
    1: {'name': 'Car', 'height': 1.60, 'height_range': (1.20, 2.20),
        'length_range': (2.2, 6.8), 'width_range': (1.0, 2.8), 'min_points': 3},
    2: {'name': 'Pedestrian', 'height': 1.70, 'height_range': (1.15, 2.20),
        'length_range': (0.20, 1.30), 'width_range': (0.20, 1.30), 'min_points': 2},
    3: {'name': 'Truck', 'height': 3.00, 'height_range': (1.70, 4.20),
        'length_range': (3.5, 16.0), 'width_range': (1.3, 3.5), 'min_points': 4},
    4: {'name': 'Cyclist', 'height': 1.70, 'height_range': (1.15, 2.30),
        'length_range': (0.45, 3.2), 'width_range': (0.25, 1.5), 'min_points': 2},
}
CATEGORIES = {'Car': 0, 'Van': 1, 'Truck': 2, 'Bus': 3,
              'Pedestrian': 4, 'Cyclist': 5}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('dataset', type=Path,
                        help='prepared Fisheye3DOD directory containing the test info')
    parser.add_argument('calibration', type=Path)
    parser.add_argument('--input-info', type=Path)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--confidence', type=int, default=50)
    parser.add_argument('--min-age-ms', type=float, default=200.0,
                        help='reject newly-created tracks below this age')
    parser.add_argument('--max-sync-ms', type=float, default=80.0)
    parser.add_argument('--max-range', type=float, default=48.0)
    parser.add_argument('--val-fraction', type=float, default=0.20)
    parser.add_argument('--split-block-seconds', type=float, default=40.0,
                        help='duration of temporal blocks used for train/val splitting')
    parser.add_argument('--visualizations', type=int, default=16)
    parser.add_argument('--limit', type=int)
    return parser.parse_args()


def float32(fields: dict, number: int, default: float = 0.0) -> float:
    value = fields.get(number)
    if value is None:
        return default
    if not isinstance(value, bytes) or len(value) != 4:
        raise ValueError(f'protobuf field {number} is not float32')
    return struct.unpack('<f', value)[0]


def load_fused_messages(topic: Path) -> Tuple[np.ndarray, list[bytes]]:
    records = []
    for chunk in sorted(topic.glob('*.bin')):
        timestamps = read_index(chunk.with_suffix('.idx'))
        payloads = list(iter_record_payloads(chunk))
        if len(timestamps) != len(payloads):
            raise RuntimeError(
                f'{chunk}: {len(payloads)} messages but {len(timestamps)} timestamps')
        records.extend(zip(timestamps, payloads))
    if not records:
        raise RuntimeError(f'no fused-object messages found in {topic}')
    records.sort(key=lambda item: item[0])
    return np.asarray([item[0] for item in records], dtype=np.int64), [
        item[1] for item in records]


def nearest_index(timestamps: np.ndarray, target: int) -> int:
    right = int(np.searchsorted(timestamps, target))
    candidates = [max(0, right - 1), min(len(timestamps) - 1, right)]
    return min(candidates, key=lambda index: abs(int(timestamps[index]) - target))


def decode_objects(payload: bytes) -> Tuple[int, int, Iterable[dict]]:
    top = field_map(payload)
    availability = integer(top, 1, 0)
    coordinate_system = integer(top, 5, 0)
    objects = []
    for message in repeated_bytes(payload, 6):
        if not message:
            continue
        fields = field_map(message)
        objects.append({
            'id': integer(fields, 1),
            'x': float32(fields, 2), 'y': float32(fields, 4),
            'yaw': float32(fields, 10),
            'length': float32(fields, 12), 'width': float32(fields, 13),
            'class_id': integer(fields, 15),
            'sensors': integer(fields, 16),
            'confidence': integer(fields, 17),
            'measurement_status': integer(fields, 19),
            'age': float32(fields, 21),
            'reference_point': integer(fields, 23),
        })
    return availability, coordinate_system, objects


def known_track_classes(payloads: Iterable[bytes]) -> Dict[int, int]:
    """Return the modal non-unknown class for every track in the bag."""
    votes: Dict[int, Counter] = {}
    for payload in payloads:
        _, _, objects = decode_objects(payload)
        for obj in objects:
            if obj['class_id'] in CLASS_INFO:
                votes.setdefault(obj['id'], Counter())[obj['class_id']] += 1
    return {track_id: counts.most_common(1)[0][0]
            for track_id, counts in votes.items()}


def infer_class(obj: dict, track_classes: Dict[int, int]) -> dict:
    """Fill kUnknownClassification using track history, then dimensions."""
    obj = dict(obj)
    obj['original_class_id'] = obj['class_id']
    if obj['class_id'] in CLASS_INFO:
        obj['class_source'] = 'fused_classification'
        return obj
    if obj['id'] in track_classes:
        obj['class_id'] = track_classes[obj['id']]
        obj['class_source'] = 'same_track_majority'
        return obj
    length, width = obj['length'], obj['width']
    # The rear sensors frequently provide geometry while leaving class unknown.
    # These disjoint rules intentionally prefer rejection over an ambiguous label.
    if 5.5 <= length <= 16.0 and 1.3 <= width <= 3.5:
        obj['class_id'] = 3
    elif 2.2 <= length <= 6.8 and 1.0 <= width <= 2.8:
        obj['class_id'] = 1
    elif (0.7 <= length <= 3.2 and 0.25 <= width <= 1.5 and
          length / max(width, 1e-6) >= 1.7):
        obj['class_id'] = 4
    elif (0.2 <= length <= 1.3 and 0.2 <= width <= 1.3 and
          length / max(width, 1e-6) < 1.7):
        obj['class_id'] = 2
    obj['class_source'] = ('dimensions_rear' if obj['x'] < 0
                           else 'dimensions_front')
    return obj


def load_points(path: Path) -> np.ndarray:
    points = np.fromfile(path, dtype=np.float32)
    if len(points) % 4:
        raise RuntimeError(f'{path}: float count is not divisible by four')
    points = points.reshape(-1, 4)[:, :3]
    valid = np.isfinite(points).all(axis=1)
    valid &= np.linalg.norm(points, axis=1) > 0.1
    valid &= np.linalg.norm(points[:, :2], axis=1) < 100.0
    return points[valid]


def local_xy(points: np.ndarray, center: np.ndarray, yaw: float
             ) -> Tuple[np.ndarray, np.ndarray]:
    dx, dy = points[:, 0] - center[0], points[:, 1] - center[1]
    cosine, sine = np.cos(yaw), np.sin(yaw)
    return cosine * dx + sine * dy, -sine * dx + cosine * dy


def estimate_ground(points: np.ndarray, local_x: np.ndarray, local_y: np.ndarray,
                    length: float, width: float) -> Optional[float]:
    inner = (np.abs(local_x) <= length / 2 + 0.20) & (
        np.abs(local_y) <= width / 2 + 0.20)
    outer = (np.abs(local_x) <= length / 2 + 1.50) & (
        np.abs(local_y) <= width / 2 + 1.50)
    nearby = points[outer & ~inner, 2]
    nearby = nearby[(nearby > -4.5) & (nearby < 0.5)]
    if len(nearby) < 12:
        nearby = points[outer, 2]
        nearby = nearby[(nearby > -4.5) & (nearby < 0.5)]
    if len(nearby) < 6:
        return None
    # The road dominates the lower half; this resists vehicle and vegetation returns.
    cutoff = np.median(nearby)
    lower = nearby[nearby <= cutoff]
    return float(np.median(lower))


def lidar_box(obj: dict, points: np.ndarray, base_to_lidar: np.ndarray,
              confidence: int, min_age_ms: float,
              max_range: float) -> Tuple[Optional[np.ndarray], dict]:
    result = dict(obj)
    class_info = CLASS_INFO.get(obj['class_id'])
    if class_info is None:
        result['reason'] = 'unknown_class'
        return None, result
    result['class'] = class_info['name']
    if obj['reference_point'] != 0:
        result['reason'] = 'non_center_reference'
        return None, result
    if obj['confidence'] < confidence:
        result['reason'] = 'low_confidence'
        return None, result
    if obj['age'] < min_age_ms:
        result['reason'] = 'immature_track'
        return None, result
    if not (class_info['length_range'][0] <= obj['length'] <=
            class_info['length_range'][1] and
            class_info['width_range'][0] <= obj['width'] <=
            class_info['width_range'][1]):
        result['reason'] = 'invalid_dimensions'
        return None, result
    if np.hypot(obj['x'], obj['y']) > max_range:
        result['reason'] = 'outside_training_range'
        return None, result

    center = base_to_lidar @ np.array([obj['x'], obj['y'], 0.0, 1.0])
    heading = base_to_lidar[:3, :3] @ np.array(
        [np.cos(obj['yaw']), np.sin(obj['yaw']), 0.0])
    yaw = float(np.arctan2(heading[1], heading[0]))
    lx, ly = local_xy(points, center, yaw)
    inside = (np.abs(lx) <= obj['length'] / 2 + 0.10) & (
        np.abs(ly) <= obj['width'] / 2 + 0.10)
    ground = estimate_ground(points, lx, ly, obj['length'], obj['width'])
    result['ground_z'] = ground
    if ground is None:
        result['reason'] = 'no_local_ground'
        return None, result

    minimum, maximum = class_info['height_range']
    vertical = inside & (points[:, 2] >= ground + 0.25) & (
        points[:, 2] <= ground + maximum + 0.45)
    object_z = points[vertical, 2]
    result['lidar_points'] = int(len(object_z))
    required = int(class_info['min_points'])
    if str(obj.get('class_source', '')).startswith('dimensions_'):
        required = max(required, 4)
    if len(object_z) < required:
        result['reason'] = 'insufficient_lidar_support'
        return None, result

    if len(object_z) >= 6:
        measured_height = float(np.percentile(object_z, 95) - ground + 0.10)
        height = float(np.clip(measured_height, minimum, maximum))
    else:
        height = float(class_info['height'])
    z_center = ground + height / 2
    box = np.array([center[0], center[1], z_center, obj['length'],
                    obj['width'], height, yaw], dtype=np.float32)
    result.update({'reason': 'accepted', 'lidar_x': float(center[0]),
                   'lidar_y': float(center[1]), 'z_center': z_center,
                   'height': height, 'lidar_yaw': yaw})
    return box, result


def corners(box: np.ndarray) -> np.ndarray:
    half_l, half_w = box[3] / 2, box[4] / 2
    local = np.array([[half_l, half_w], [half_l, -half_w],
                      [-half_l, -half_w], [-half_l, half_w],
                      [half_l, half_w]])
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + box[:2]


def save_visualizations(frames: list, output: Path, count: int) -> None:
    if not frames or count <= 0:
        return
    choices = np.linspace(0, len(frames) - 1, min(count, len(frames)), dtype=int)
    columns = 4
    rows = int(np.ceil(len(choices) / columns))
    figure, axes = plt.subplots(rows, columns, figsize=(6 * columns, 6 * rows),
                                squeeze=False)
    for axis, choice in zip(axes.ravel(), choices):
        token, points, accepted, rejected = frames[choice]
        visible = (np.abs(points[:, 0]) < 50) & (np.abs(points[:, 1]) < 50)
        draw = points[visible][::3]
        axis.scatter(draw[:, 0], draw[:, 1], c=draw[:, 2], s=0.15,
                     cmap='gray', vmin=-3, vmax=2, alpha=0.65)
        for box, label, support in accepted:
            polygon = corners(box)
            axis.plot(polygon[:, 0], polygon[:, 1], linewidth=1.5)
            axis.text(box[0], box[1], f'{label} n={support}', fontsize=6)
        for x, y in rejected:
            axis.plot(x, y, 'rx', markersize=4)
        axis.set(title=f'{token}\naccepted={len(accepted)}, rejected={len(rejected)}',
                 xlim=(-50, 50), ylim=(-50, 50), aspect='equal')
        axis.grid(alpha=0.15)
    for axis in axes.ravel()[len(choices):]:
        axis.axis('off')
    figure.tight_layout()
    figure.savefig(output / 'lidar_verified_annotations_bev.png', dpi=160)
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_info = args.input_info or args.dataset / 'fisheye3dod_infos_test.pkl'
    output = args.output or args.dataset / 'groundtruth'
    output.mkdir(parents=True, exist_ok=True)
    source = mmengine.load(input_info)
    data_list = source['data_list'][:args.limit]
    calibration = json.loads(args.calibration.read_text())
    lidar_to_base = np.asarray(
        calibration['extrinsics']['CLIDAR']['matrix'], dtype=np.float64)
    base_to_lidar = np.linalg.inv(lidar_to_base)
    timestamps, payloads = load_fused_messages(args.bag / FUSED_TOPIC)
    track_classes = known_track_classes(payloads)

    annotated, audit, visual_frames = [], [], []
    frame_stats = Counter()
    for sample in mmengine.track_iter_progress(data_list):
        token = int(sample['token'])
        message_index = nearest_index(timestamps, token)
        delta_ms = abs(int(timestamps[message_index]) - token) / 1e6
        points_path = args.dataset / sample['pc_info']['pts_path']
        points = load_points(points_path)
        availability, coordinate_system, objects = decode_objects(
            payloads[message_index])
        boxes, names, ids = [], [], []
        accepted_draw, rejected_draw = [], []
        if delta_ms > args.max_sync_ms:
            frame_stats['sync_too_far'] += 1
            objects = []
        elif availability == 0:
            frame_stats['fusion_unavailable'] += 1
            objects = []
        elif coordinate_system != 1:
            raise RuntimeError(
                f'token {token}: expected kEgo coordinate system, got {coordinate_system}')
        for raw_obj in objects:
            obj = infer_class(raw_obj, track_classes)
            box, row = lidar_box(obj, points, base_to_lidar,
                                 args.confidence, args.min_age_ms,
                                 args.max_range)
            row.update({'token': str(token), 'fused_timestamp_ns':
                        str(int(timestamps[message_index])), 'sync_delta_ms': delta_ms})
            audit.append(row)
            frame_stats[row['reason']] += 1
            if box is not None:
                boxes.append(box)
                names.append(row['class'])
                ids.append(obj['id'])
                accepted_draw.append((box, row['class'], row['lidar_points']))
            elif 'lidar_x' in row:
                rejected_draw.append((row['lidar_x'], row['lidar_y']))
            else:
                point = base_to_lidar @ np.array([obj['x'], obj['y'], 0.0, 1.0])
                rejected_draw.append((point[0], point[1]))

        item = copy.deepcopy(sample)
        item['annos'] = {
            'gt_boxes': np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
            'gt_names': np.asarray(names),
            'gt_mesh': np.asarray([f'fused.{name.lower()}' for name in names]),
            'gt_ids': np.asarray(ids, dtype=np.int64),
        }
        annotated.append(item)
        visual_frames.append((str(token), points, accepted_draw, rejected_draw))

    metainfo = dict(source.get('metainfo', {}))
    metainfo.update({'categories': CATEGORIES, 'classes': tuple(CATEGORIES),
                     'dataset': 'custom_fused_lidar_verified_pseudo_groundtruth'})
    # Do not train empty pseudo-label frames as background: fusion may simply
    # have missed real objects (and is explicitly unavailable early in this bag).
    labeled = [item for item in annotated if len(item['annos']['gt_names'])]
    # Alternate contiguous time blocks instead of individual frames. This
    # distributes rare classes while limiting adjacent-frame leakage.
    if not 0 < args.val_fraction < 1:
        raise ValueError('--val-fraction must be between zero and one')
    cycle = max(2, round(1.0 / args.val_fraction))
    block_ns = max(1, round(args.split_block_seconds * 1e9))
    first_token = int(labeled[0]['token'])
    train, val = [], []
    for item in labeled:
        block = (int(item['token']) - first_token) // block_ns
        (val if block % cycle == cycle - 1 else train).append(item)
    if not train or not val:
        raise RuntimeError('temporal block split produced an empty train or val set')
    mmengine.dump({'metainfo': metainfo, 'data_list': annotated},
                  output / 'fisheye3dod_infos_all.pkl')
    mmengine.dump({'metainfo': metainfo, 'data_list': train},
                  output / 'fisheye3dod_infos_train.pkl')
    mmengine.dump({'metainfo': metainfo, 'data_list': val},
                  output / 'fisheye3dod_infos_val.pkl')

    fieldnames = sorted({key for row in audit for key in row})
    with (output / 'annotation_audit.csv').open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(audit)
    class_counts = Counter(name for item in annotated for name in item['annos']['gt_names'])
    class_sources = Counter(row.get('class_source', 'unknown') for row in audit
                            if row['reason'] == 'accepted')
    summary = {
        'label_kind': 'LiDAR-verified pseudo-ground-truth from onboard object fusion',
        'proto': 'adas.dynamic_object_fusion.FusedObjectList',
        'source_info': str(input_info.resolve()),
        'source_calibration': str(args.calibration.resolve()),
        'samples': len(annotated), 'labeled_samples': len(labeled),
        'excluded_empty_samples': len(annotated) - len(labeled),
        'train_samples': len(train),
        'val_samples': len(val), 'accepted_boxes': int(sum(class_counts.values())),
        'class_counts': dict(class_counts), 'filter_counts': dict(frame_stats),
        'accepted_class_sources': dict(class_sources),
        'settings': {'minimum_confidence': args.confidence,
                     'minimum_track_age_ms': args.min_age_ms,
                     'max_sync_ms': args.max_sync_ms,
                     'max_range_m': args.max_range,
                     'val_fraction': args.val_fraction,
                     'split_block_seconds': args.split_block_seconds},
        'warning': ('These are automatically generated pseudo-labels, not human '
                    'ground truth. Review the audit CSV and BEV visualization.'),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    save_visualizations(visual_frames, output, args.visualizations)
    print(json.dumps(summary, indent=2))
    print(f'outputs: {output}')


if __name__ == '__main__':
    main()
