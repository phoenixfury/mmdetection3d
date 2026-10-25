#!/usr/bin/env python3
"""Convert DetZero results to CaliperGT LiDAR cuboid annotations."""

from __future__ import annotations

import argparse
import csv
import json
import math
import uuid
import zipfile
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path

import numpy as np


CLASSES = ('Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist')
CLASS_ALIASES = {
    'vehicle': 'Car',
    'car': 'Car',
    'van': 'Van',
    'truck': 'Truck',
    'bus': 'Bus',
    'pedestrian': 'Pedestrian',
    'cyclist': 'Cyclist',
    'bicycle': 'Cyclist',
    'motorcycle': 'Cyclist',
}
NAMESPACE = uuid.UUID('8fe1f5a6-3187-478f-9a42-843d41a89a98')


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', type=Path,
                        help='synchronized_dual_images directory')
    parser.add_argument('--results', type=Path,
                        help='default: SEQUENCE/detzero_output/results.json')
    parser.add_argument('--format', choices=('lidar-json', 'api'),
                        default='lidar-json')
    parser.add_argument('--metadata', type=Path,
                        help='CaliperGT scene metadata.json; required for api')
    parser.add_argument('--frame-mapping', type=Path,
                        help='frame_mapping.csv from export_caliper_scene.py')
    parser.add_argument('--task-id', help='CaliperGT task UUID override')
    parser.add_argument('--taxonomy-id', help='CaliperGT taxonomy UUID override')
    parser.add_argument('--min-score', type=float, default=0.0)
    parser.add_argument(
        '--track-class-policy', choices=('majority', 'first', 'strict'),
        default='majority',
        help='resolve one class for every source track; majority counts frames '
             'and uses confidence only to break ties (default: majority)')
    parser.add_argument('--output', type=Path)
    parser.add_argument(
        '--track-map', type=Path,
        help='source-to-CaliperGT track mapping report (default: beside output)')
    parser.add_argument('--zip', action='store_true', dest='make_zip',
                        help='also create an upload-ready ZIP containing lidar.json')
    return parser.parse_args()


def normalize_class(value: object) -> str:
    key = str(value).strip().lower().replace(' ', '_')
    if key not in CLASS_ALIASES:
        raise ValueError(f'unsupported result class: {value!r}')
    return CLASS_ALIASES[key]


def load_inputs(args: argparse.Namespace) -> tuple[list[dict], list[dict]]:
    sequence = args.sequence.resolve()
    results_path = (args.results or
                    sequence / 'detzero_output' / 'results.json').resolve()
    with (sequence / 'manifest.csv').open(newline='') as stream:
        manifest = list(csv.DictReader(stream))
    results = json.loads(results_path.read_text())
    if len(results) != len(manifest):
        raise ValueError(
            f'frame count mismatch: results={len(results)}, '
            f'manifest={len(manifest)}')
    for index, result in enumerate(results):
        if int(result['frame_id']) != index:
            raise ValueError(
                f'result position {index} has frame_id={result["frame_id"]}')
    return manifest, results


def valid_objects(results: list[dict], min_score: float):
    for frame in results:
        frame_index = int(frame['frame_id'])
        for obj in frame.get('objects', []):
            score = float(obj.get('score', 1.0))
            if score < min_score:
                continue
            if obj.get('track_id') is None:
                raise ValueError(
                    f'frame {frame_index}: object is missing track_id')
            source_track_id = str(obj['track_id'])
            box = np.asarray(obj['box'], dtype=float)
            if box.shape != (7,) or not np.isfinite(box).all():
                raise ValueError(
                    f'frame {frame_index}, track {obj.get("track_id")}: '
                    'box must contain seven finite values')
            if np.any(box[3:6] <= 0):
                raise ValueError(
                    f'frame {frame_index}, track {obj.get("track_id")}: '
                    'box dimensions must be positive')
            yield (frame_index, obj, normalize_class(obj['name']), box, score,
                   source_track_id)


def collect_tracks(results: list[dict], min_score: float,
                   policy: str) -> tuple[list[tuple], dict[str, dict]]:
    """Collect observations and resolve exactly one class per source track."""
    observations = list(valid_objects(results, min_score))
    grouped = {}
    for observation in observations:
        grouped.setdefault(observation[5], []).append(observation)

    tracks = {}
    for source_track_id, items in grouped.items():
        items.sort(key=lambda item: item[0])
        class_counts = Counter(item[2] for item in items)
        class_scores = Counter()
        for item in items:
            class_scores[item[2]] += item[4]
        if policy == 'strict' and len(class_counts) != 1:
            raise ValueError(
                f'track {source_track_id} has multiple classes: '
                f'{dict(class_counts)}')
        if policy == 'first':
            resolved_class = items[0][2]
        else:
            resolved_class = max(
                class_counts,
                key=lambda name: (class_counts[name], class_scores[name], name))
        tracks[source_track_id] = {
            'resolved_class': resolved_class,
            'observations': len(items),
            'first_frame': int(items[0][0]),
            'last_frame': int(items[-1][0]),
            'input_classes': dict(class_counts),
            'class_confidence_sums': {
                name: float(value) for name, value in class_scores.items()},
        }
    return observations, tracks


def cuboid(box: np.ndarray) -> dict:
    return {
        'position': {
            'x': float(box[0]), 'y': float(box[1]), 'z': float(box[2]),
        },
        'dimensions': {
            'length': float(box[3]),
            'width': float(box[4]),
            'height': float(box[5]),
        },
        'rotation': {
            'yaw': float(box[6]), 'pitch': 0.0, 'roll': 0.0,
        },
    }


def stable_track_id(scope: str, source_id: object) -> str:
    return str(uuid.uuid5(NAMESPACE, f'{scope}:track:{source_id}'))


def make_lidar_json(manifest: list[dict], results: list[dict],
                    min_score: float, class_policy: str
                    ) -> tuple[dict, Counter, dict]:
    first_ns = int(manifest[0]['timestamp_ns'])
    categories = [
        {'id': index + 1, 'name': name.lower(), 'supercategory': 'object'}
        for index, name in enumerate(CLASSES)
    ]
    category_ids = {name: index + 1 for index, name in enumerate(CLASSES)}
    frames = []
    for index, row in enumerate(manifest):
        timestamp_ns = int(row['timestamp_ns'])
        frames.append({
            'id': index,
            # CaliperGT otherwise derives the index from file_name.  Our PCD
            # names are nanosecond timestamps, not sequential frame numbers.
            'frame_index': index,
            'file_name': Path(row['lidar_pcd']).name,
            'timestamp': (timestamp_ns - first_ns) / 1e9,
            'ego_pose': None,
        })

    annotations = []
    counts = Counter()
    scope = str(first_ns)
    observations, tracks = collect_tracks(results, min_score, class_policy)
    for source_track_id, track in tracks.items():
        track['calipergt_track_id'] = stable_track_id(scope, source_track_id)
    for annotation_id, (frame_index, obj, observed_name, box, score,
                        source_track_id) in enumerate(observations, start=1):
        name = tracks[source_track_id]['resolved_class']
        counts[name] += 1
        annotations.append({
            'id': annotation_id,
            'frame_id': frame_index,
            'category_id': category_ids[name],
            'class': name.lower(),
            'cuboid_3d': cuboid(box),
            'attributes': {},
            'track_id': tracks[source_track_id]['calipergt_track_id'],
            # Every exported cuboid is a directly observed DetZero result, not
            # an interpolated pose.  CaliperGT uses this flag to initialize
            # and display the track's keyframes.
            'is_keyframe': True,
            'source': 'detzero',
            'confidence': score,
        })
    document = {
        'info': {
            'description': 'DetZero generated LiDAR 3D annotations',
            'version': '1.0',
            'year': datetime.now(timezone.utc).year,
            'contributor': 'DetZero',
            'date_created': datetime.now(timezone.utc).isoformat(),
            'coordinate_system': 'lidar',
        },
        'frames': frames,
        'annotations': annotations,
        'categories': categories,
    }
    validate_exported_tracks(document['annotations'], tracks)
    return document, counts, tracks


def select_task(metadata: dict, task_id: str | None) -> dict:
    tasks = metadata.get('tasks', [])
    if task_id:
        matches = [task for task in tasks if task.get('id') == task_id]
    else:
        matches = [task for task in tasks if task.get('taxonomy_id')]
    if len(matches) != 1:
        raise ValueError(
            f'could not select one annotated task (found {len(matches)}); '
            'pass --task-id')
    return matches[0]


def source_to_caliper_frames(metadata: dict,
                             mapping_path: Path | None) -> dict[int, dict]:
    caliper_by_index = {
        int(frame['frame_index']): frame for frame in metadata['frames']
    }
    if mapping_path is None:
        return caliper_by_index
    with mapping_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    return {
        int(row['source_frame_index']): caliper_by_index[int(row['frame_index'])]
        for row in rows
    }


def taxonomy_classes(metadata: dict) -> dict[str, str]:
    classes = metadata.get('taxonomy', {}).get('classes', [])
    result = {}
    for item in classes:
        name = str(item.get('name', item.get('id', ''))).strip().lower()
        result[name] = str(item['id'])
    return result


def make_api_payload(args: argparse.Namespace, results: list[dict]
                     ) -> tuple[dict, Counter, dict]:
    if args.metadata is None:
        raise ValueError('--metadata is required with --format api')
    metadata_path = args.metadata.resolve()
    metadata = json.loads(metadata_path.read_text())
    task = select_task(metadata, args.task_id)
    task_id = args.task_id or str(task['id'])
    taxonomy_id = args.taxonomy_id or task.get('taxonomy_id')
    if not taxonomy_id:
        raise ValueError('selected task has no taxonomy; pass --taxonomy-id')

    mapping_path = args.frame_mapping
    if mapping_path is None and (metadata_path.parent / 'frame_mapping.csv').is_file():
        mapping_path = metadata_path.parent / 'frame_mapping.csv'
    if mapping_path is not None:
        mapping_path = mapping_path.resolve()
    frames = source_to_caliper_frames(metadata, mapping_path)
    taxonomy = taxonomy_classes(metadata)
    counts = Counter()
    annotations = []
    observations, tracks = collect_tracks(
        results, args.min_score, args.track_class_policy)
    for source_track_id, track in tracks.items():
        track['calipergt_track_id'] = stable_track_id(
            task_id, source_track_id)
    for (frame_index, obj, observed_name, box, score,
         source_track_id) in observations:
        if frame_index not in frames:
            continue
        name = tracks[source_track_id]['resolved_class']
        taxonomy_name = name.lower()
        if taxonomy_name == 'van' and 'van' not in taxonomy:
            taxonomy_name = 'car'
        if taxonomy_name not in taxonomy:
            raise ValueError(
                f'CaliperGT taxonomy has no class for {name!r}')
        counts[name] += 1
        track_id = tracks[source_track_id]['calipergt_track_id']
        annotation_id = str(uuid.uuid5(
            NAMESPACE,
            f'{task_id}:annotation:{frame_index}:{source_track_id}'))
        cube = cuboid(box)
        annotations.append({
            'id': annotation_id,
            'task_id': task_id,
            'frame_id': str(frames[frame_index]['id']),
            'track_id': track_id,
            'type': 'cuboid',
            'class_id': taxonomy[taxonomy_name],
            'taxonomy_id': str(taxonomy_id),
            'data': {
                'center': cube['position'],
                'dimensions': cube['dimensions'],
                'rotation': cube['rotation'],
                'confidence': score,
            },
            'attributes': {},
            'source': 'model',
            'is_keyframe': True,
        })
    # This is the exact body accepted by POST /annotations-3d/bulk.
    validate_exported_tracks(annotations, tracks, api_format=True)
    return {'annotations': annotations}, counts, tracks


def validate_exported_tracks(annotations: list[dict], tracks: dict[str, dict],
                             api_format: bool = False) -> None:
    """Fail if a CaliperGT track ID is reused with inconsistent classes."""
    mapped_ids = [
        item['calipergt_track_id'] for item in tracks.values()
    ]
    duplicate_mappings = [
        track_id for track_id, count in Counter(mapped_ids).items()
        if count > 1
    ]
    if duplicate_mappings:
        raise ValueError(
            f'multiple source tracks map to the same CaliperGT track: '
            f'{duplicate_mappings[:10]}')
    expected = {
        item['calipergt_track_id']: item['resolved_class'].lower()
        for item in tracks.values()
    }
    observed = {}
    annotation_ids = set()
    for annotation in annotations:
        annotation_id = str(annotation['id'])
        if annotation_id in annotation_ids:
            raise ValueError(f'duplicate annotation id {annotation_id}')
        annotation_ids.add(annotation_id)
        track_id = str(annotation['track_id'])
        class_name = (str(annotation['class']).lower()
                      if not api_format else expected[track_id])
        previous = observed.setdefault(track_id, class_name)
        if previous != class_name:
            raise ValueError(
                f'CaliperGT track {track_id} contains both {previous!r} and '
                f'{class_name!r}')


def main() -> None:
    args = parse_args()
    if not math.isfinite(args.min_score) or not 0 <= args.min_score <= 1:
        raise ValueError('--min-score must be between zero and one')
    manifest, results = load_inputs(args)
    if args.format == 'api':
        document, counts, tracks = make_api_payload(args, results)
        default_name = 'calipergt_bulk_annotations.json'
    else:
        document, counts, tracks = make_lidar_json(
            manifest, results, args.min_score, args.track_class_policy)
        default_name = 'calipergt_annotations/lidar.json'
    output = (args.output or args.sequence / 'detzero_output' /
              default_name).resolve()
    if args.format == 'lidar-json' and output.name != 'lidar.json':
        raise ValueError(
            'CaliperGT only recognizes this format when the file is named '
            'exactly lidar.json')
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_suffix(output.suffix + '.tmp')
    temporary.write_text(json.dumps(document, indent=2) + '\n')
    temporary.replace(output)
    track_map = (args.track_map or
                 output.with_name('track_id_map.json')).resolve()
    track_map.parent.mkdir(parents=True, exist_ok=True)
    track_map_tmp = track_map.with_suffix(track_map.suffix + '.tmp')
    track_map_tmp.write_text(json.dumps({
        'track_class_policy': args.track_class_policy,
        'source_tracks': tracks,
    }, indent=2) + '\n')
    track_map_tmp.replace(track_map)
    archive = None
    if args.format == 'lidar-json':
        if args.make_zip:
            archive = output.parent.with_suffix('.zip')
            archive_tmp = archive.with_suffix('.zip.tmp')
            with zipfile.ZipFile(
                    archive_tmp, 'w', compression=zipfile.ZIP_DEFLATED) as bundle:
                bundle.write(output, 'lidar.json')
            archive_tmp.replace(archive)
    print(json.dumps({
        'output': str(output),
        'archive': str(archive) if archive else None,
        'track_map': str(track_map),
        'format': args.format,
        'frames': len(manifest),
        'tracks': len(tracks),
        'annotations': sum(counts.values()),
        'class_counts': dict(counts),
        'min_score': args.min_score,
        'track_class_policy': args.track_class_policy,
    }, indent=2))


if __name__ == '__main__':
    main()
