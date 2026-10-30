"""Backward-compatible temporal annotation sidecars for recorder/DetZero.

The established DetZero JSON ([{frame_id, timestamp_ns, objects}]) and the
Fisheye3DOD label_3d JSON ([{bounding_box, class, name, id}]) are unchanged.
All temporal and provenance fields live in a separate versioned sidecar.
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

CAMERA_PREFIXES = ('arc_f', 'arc_sl', 'arc_sr', 'arc_r')
SCHEMA = 'detzero_temporal_v1'
# The project's six-class Fisheye3DOD/ROSA taxonomy treats commercial vans
# labelled ConstructionVehicle by DetZero as Van.
CLASSIC_CLASS_MAP = {'ConstructionVehicle': 'Van'}


def atomic_json(path: Path, data: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + '.tmp')
    try:
        with tmp.open('w') as stream:
            json.dump(data, stream, indent=2, allow_nan=False)
            stream.write('\n')
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def _integer(row: dict, key: str, fallback=None):
    value = row.get(key)
    return fallback if value is None or value == '' else int(value)


def read_calibration(sequence: Path, calibration: Path | None = None):
    if calibration is None:
        calibration = next((p for p in (sequence / 'calibration.json',
                                        sequence.parent / 'calibration.json')
                            if p.exists()), None)
    if calibration is None:
        raise FileNotFoundError('T_BASELINK_CLIDAR calibration is required')
    doc = json.loads(calibration.read_text())
    entry = doc.get('extrinsics', {}).get('CLIDAR')
    if not entry or entry.get('from') != 'CLIDAR' or entry.get('to') != 'BASELINK':
        raise ValueError('expected extrinsics.CLIDAR: CLIDAR -> BASELINK')
    transform = np.asarray(entry['matrix'], dtype=np.float64)
    validate_pose(transform, str(calibration))
    return transform, calibration


def validate_pose(pose: np.ndarray, description: str) -> None:
    if pose.shape != (4, 4) or not np.isfinite(pose).all():
        raise ValueError('%s: expected a finite 4x4 pose' % description)
    if not np.allclose(pose[3], [0, 0, 0, 1], atol=1e-5):
        raise ValueError('%s: invalid homogeneous last row' % description)
    rotation = pose[:3, :3]
    if not np.allclose(rotation.T @ rotation, np.eye(3), atol=1e-3) or not np.isclose(
            np.linalg.det(rotation), 1, atol=1e-3):
        raise ValueError('%s: rotation must be a proper rigid transform' % description)


def load_ego_poses(sequence: Path, rows: list[dict], calibration: Path | None = None):
    """Return two pose arrays indexed by *manifest row*, not odometry row."""
    transforms = np.load(sequence / 'odometry_poses.npy')
    if transforms.ndim != 3 or transforms.shape[1:] != (4, 4):
        raise ValueError('odometry_poses.npy must contain 4x4 matrices')
    lidar_to_base, path = read_calibration(sequence, calibration)
    base = []
    for frame_id, row in enumerate(rows):
        idx = _integer(row, 'odometry_index')
        if idx is None or idx < 0 or idx >= len(transforms):
            raise ValueError('missing/invalid odometry_index at manifest row %d' % frame_id)
        pose = np.asarray(transforms[idx], dtype=np.float64)
        validate_pose(pose, 'odometry index %d' % idx)
        base.append(pose)
    base = np.stack(base)
    return base, base @ lidar_to_base, path


def track_hit_lookup(tracks: dict | None):
    """(track, frame) -> hit state, with duplicates rejected rather than guessed."""
    if tracks is None:
        return {}
    result = {}
    for raw_id, data in tracks.items():
        for idx, hit in zip(data['sample_idx'], data['hit']):
            key = int(raw_id), int(idx)
            if key in result and result[key] != bool(hit > 0):
                raise ValueError('conflicting hits for track %r' % (key,))
            result[key] = bool(hit > 0)
    return result


def provenance(obj: dict, frame_id: int, hits: dict, stage: str):
    original = obj.get('source')
    if original == 'camera_track_hold':
        source = 'camera_hold'
    elif original == 'yolo_lidar':
        source = 'lidar'
    elif original in ('lidar', 'camera_hold', 'tracker_prediction'):
        source = original
    else:
        key = int(obj['track_id']), frame_id
        if key in hits:
            source = 'lidar' if hits[key] else 'tracker_prediction'
        else:
            # Refined results with no tracker provenance are not assumed to
            # originate from an actual LiDAR detection.
            source = 'lidar' if stage == 'detect' else 'unknown'
    return {
        'source': source,
        'source_detail': str(original or ('detector' if stage == 'detect' else stage)),
        'camera': obj.get('camera'),
        'camera_confidence': obj.get('camera_confidence'),
        'prediction_source': obj.get('prediction_source'),
        'hold_age_frames': obj.get('hold_age_frames'),
        'lidar_support_points': obj.get('lidar_support'),
        'camera_confirmation': obj.get('camera_confirmation'),
        'camera_time_compensation': (obj.get('camera_time_compensation') or
                                     (obj.get('camera_confirmation') or {}).get('camera_time_compensation')),
    }


def build_temporal(rows: list[dict], results: list[dict], base: np.ndarray,
                   lidar: np.ndarray, hits: dict | None = None,
                   stage: str = 'refine', calibration_path: str = '') -> dict:
    if len(rows) != len(results) or len(base) != len(rows) or len(lidar) != len(rows):
        raise ValueError('manifest, results and pose lengths must match')
    hits = hits or {}
    frames = []
    seen_acquisitions = set()
    for index, (row, frame) in enumerate(zip(rows, results)):
        if int(frame['frame_id']) != index:
            raise ValueError('frame_id and manifest row are misaligned at %d' % index)
        recorder_ns = int(row['timestamp_ns'])
        acquisition_ns = _integer(row, 'lidar_sensor_timestamp_ns', recorder_ns)
        if acquisition_ns in seen_acquisitions:
            raise ValueError('duplicate LiDAR acquisition timestamp: %d' % acquisition_ns)
        seen_acquisitions.add(acquisition_ns)
        if 'timestamp_ns' in frame and int(frame['timestamp_ns']) != recorder_ns:
            raise ValueError('stale result timestamps at frame %d' % index)
        camera_records = {}
        for camera in CAMERA_PREFIXES:
            recorded = _integer(row, camera + '_timestamp_ns')
            clock = row.get('camera_sync_clock') or None
            sync_target = _integer(row, 'camera_sync_target_ns')
            delta = _integer(row, camera + '_delta_ns')
            method = 'manifest'
            if recorded is None and sync_target is not None and delta is not None:
                recorded = sync_target + delta
                method = 'reconstructed_from_sync_delta'
            camera_records[camera.upper()] = {
                'timestamp_ns': recorded,
                'timestamp_origin': method if recorded is not None else 'unavailable',
                'clock': 'camera_recorder_packet' if recorded is not None else None,
                'synchronization_clock': clock,
                'delta_to_sync_target_ns': delta,
                'image': row.get(camera + '_path') or None,
            }
        labels = []
        seen_tracks = set()
        for obj_index, obj in enumerate(frame['objects']):
            track_id = int(obj['track_id'])
            if track_id in seen_tracks and stage != 'detect':
                raise ValueError('duplicate track_id %d in frame %d' % (track_id, index))
            seen_tracks.add(track_id)
            box = np.asarray(obj['box'], dtype=float)
            if box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
                raise ValueError('invalid 3D box in frame %d object %d' % (index, obj_index))
            label_provenance = provenance(obj, index, hits, stage)
            camera_name = label_provenance['camera']
            camera_time = (camera_records.get(str(camera_name).upper(), {})
                           .get('timestamp_ns') if camera_name else None)
            labels.append({
                'object_index': obj_index,
                'track_id': track_id,
                'class': str(obj['name']),
                'box_frame': 'CLIDAR',
                **label_provenance,
                'camera_packet_timestamp_ns': camera_time,
                'camera_packet_minus_lidar_acquisition_ns': (
                    camera_time - acquisition_ns if camera_time is not None
                    else None),
            })
        frames.append({
            'frame_id': index,
            'acquisition_timestamp_ns': acquisition_ns,
            'recorder_timestamp_ns': recorder_ns,
            'lidar_transport_delay_ns': recorder_ns - acquisition_ns,
            'timestamp_clock': 'lidar_sensor' if row.get('lidar_sensor_timestamp_ns') else 'recorder_fallback',
            'ego_pose': {
                'odometry_index': int(row['odometry_index']),
                'reference_timestamp_ns': acquisition_ns,
                'alignment_delta_ns': _integer(row, 'odometry_delta_ns'),
                'alignment_status': row.get('odometry_status') or None,
                'T_world_BASELINK': base[index].tolist(),
                'T_world_CLIDAR': lidar[index].tolist(),
            },
            'cameras': camera_records,
            'labels': labels,
        })
    return {
        'schema': SCHEMA,
        'coordinate_convention': 'right-handed x-forward, y-left, z-up; center-origin boxes in CLIDAR',
        'calibration_path': calibration_path,
        'stage': stage,
        'frames': frames,
    }


def export_annotations(sequence: Path, rows: list[dict], results: list[dict],
                       classic_path: Path, calibration: Path | None = None,
                       tracks: dict | None = None, stage: str = 'refine',
                       write_classic_labels: bool = True) -> Path:
    """Write sidecar beside classic JSON, and optional Fisheye3DOD label_3d files.

    Classic DetZero results JSON is owned by callers and never mutated here.
    """
    base, lidar, path = load_ego_poses(sequence, rows, calibration)
    sidecar = build_temporal(rows, results, base, lidar, track_hit_lookup(tracks),
                             stage, str(path))
    sidecar_path = classic_path.with_name(classic_path.stem + '_temporal.json')
    atomic_json(sidecar_path, sidecar)
    if write_classic_labels:
        dest = classic_path.with_name(classic_path.stem + '_fisheye3dod')
        label_dir = dest / 'label_3d'
        label_dir.mkdir(parents=True, exist_ok=True)
        expected_files = {'%06d.json' % int(f['frame_id']) for f in results}
        for old_label in label_dir.glob('*.json'):
            if old_label.name not in expected_files:
                old_label.unlink()
        index_map = []
        for row, frame in zip(rows, results):
            frame_name = '%06d' % int(frame['frame_id'])
            labels = [{
                'bounding_box': [float(v) for v in obj['box']],
                'class': CLASSIC_CLASS_MAP.get(str(obj['name']), str(obj['name'])),
                'name': str(obj['name']),
                'id': int(obj['track_id']),
            } for obj in frame['objects']]
            atomic_json(label_dir / (frame_name + '.json'), labels)
            index_map.append({'frame_id': int(frame['frame_id']),
                              'legacy_frame_name': frame_name,
                              'acquisition_timestamp_ns': _integer(
                                  row, 'lidar_sensor_timestamp_ns', int(row['timestamp_ns']))})
        atomic_json(dest / 'frame_map.json', index_map)
    return sidecar_path


def main():
    """Add temporal sidecars to existing DetZero results without rerunning GPUs."""
    import argparse
    import csv
    import pickle

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', type=Path,
                        help='recorder pipeline exported sequence directory')
    parser.add_argument('--results', type=Path,
                        help='existing classic JSON (default detzero_output/results.json)')
    parser.add_argument('--tracks', type=Path,
                        help='optional tracks.pkl for reliable hit/coast provenance')
    parser.add_argument('--calibration', type=Path)
    parser.add_argument('--stage', choices=('detect', 'track', 'refine',
                                            'camera_improved'), default='refine')
    parser.add_argument('--skip-classic-labels', action='store_true')
    args = parser.parse_args()
    sequence = args.sequence.resolve()
    classic = (args.results or sequence / 'detzero_output' / 'results.json').resolve()
    with (sequence / 'manifest.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    results = json.loads(classic.read_text())
    track_path = (args.tracks or classic.parent / 'tracks.pkl').resolve()
    tracks = pickle.loads(track_path.read_bytes()) if track_path.exists() else None
    temporal_path = export_annotations(
        sequence, rows, results, classic, args.calibration, tracks,
        stage=args.stage, write_classic_labels=not args.skip_classic_labels)
    print('Temporal sidecar:', temporal_path)


if __name__ == '__main__':
    main()
