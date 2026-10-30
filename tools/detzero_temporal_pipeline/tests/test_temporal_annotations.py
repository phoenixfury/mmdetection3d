import json
from pathlib import Path

import numpy as np
import pytest

from temporal_annotations import (build_temporal, export_annotations,
                                  load_ego_poses, track_hit_lookup)


def fixture(tmp_path):
    sequence = tmp_path / 'sequence'
    sequence.mkdir()
    base = np.stack([np.eye(4), np.eye(4), np.eye(4)])
    base[1, 0, 3] = 12
    base[2, 0, 3] = 21
    np.save(sequence / 'odometry_poses.npy', base)
    transform = np.eye(4)
    transform[0, 3] = 2.5
    calibration = {'extrinsics': {'CLIDAR': {
        'from': 'CLIDAR', 'to': 'BASELINK', 'matrix': transform.tolist()}}}
    (sequence / 'calibration.json').write_text(json.dumps(calibration))
    rows = [
        {'timestamp_ns': '1800000900', 'lidar_sensor_timestamp_ns': '1800000000',
         'odometry_index': '2', 'odometry_delta_ns': '25',
         'camera_sync_clock': 'sensor', 'camera_sync_target_ns': '1800000000',
         'arc_f_timestamp_ns': '1800000005', 'arc_f_delta_ns': '5'},
        {'timestamp_ns': '2300000900', 'lidar_sensor_timestamp_ns': '2300000000',
         'odometry_index': '1', 'odometry_delta_ns': '40',
         'camera_sync_clock': 'sensor', 'camera_sync_target_ns': '2300000000',
         'arc_f_delta_ns': '-8'},
    ]
    results = [
        {'frame_id': 0, 'timestamp_ns': 1800000900, 'objects': [
            {'track_id': 7, 'name': 'ConstructionVehicle', 'score': 0.8,
             'box': [1, 2, 0, 4, 2, 2, 0]}]},
        {'frame_id': 1, 'timestamp_ns': 2300000900, 'objects': [
            {'track_id': 7, 'name': 'Car', 'source': 'camera_track_hold',
             'camera': 'ARC_F', 'camera_confidence': 0.7,
             'box': [1, 2, 0, 4, 2, 2, 0]}]},
    ]
    tracks = {'7': {'sample_idx': np.array([0, 1]),
                    'hit': np.array([1, 0])}}
    return sequence, rows, results, tracks


def test_sidecar_preserves_legacy_and_pose_index(tmp_path):
    seq, rows, results, tracks = fixture(tmp_path)
    legacy = seq / 'detzero_output/results.json'
    legacy.parent.mkdir()
    legacy.write_text(json.dumps(results))
    original = legacy.read_bytes()
    temporal_path = export_annotations(seq, rows, results, legacy, tracks=tracks)
    assert legacy.read_bytes() == original
    temporal = json.loads(temporal_path.read_text())
    assert temporal['schema'] == 'detzero_temporal_v1'
    first, second = temporal['frames']
    assert first['acquisition_timestamp_ns'] == 1800000000
    assert first['ego_pose']['T_world_BASELINK'][0][3] == 21.0  # odometry index 2
    assert first['ego_pose']['T_world_CLIDAR'][0][3] == 23.5
    assert first['labels'][0]['source'] == 'lidar'
    assert second['labels'][0]['source'] == 'camera_hold'
    assert second['cameras']['ARC_F']['timestamp_ns'] == 2299999992
    assert second['cameras']['ARC_F']['timestamp_origin'] == 'reconstructed_from_sync_delta'
    assert second['labels'][0]['camera_packet_minus_lidar_acquisition_ns'] == -8
    assert first['cameras']['ARC_F']['clock'] == 'camera_recorder_packet'
    labels = json.loads((legacy.parent / 'results_fisheye3dod' / 'label_3d'
                         / '000000.json').read_text())
    assert set(labels[0]) == {'bounding_box', 'class', 'name', 'id'}
    assert labels[0]['class'] == 'Van'  # project six-class taxonomy
    assert labels[0]['name'] == 'ConstructionVehicle'  # original retained


def test_unknown_source_not_falsely_called_lidar(tmp_path):
    seq, rows, results, _ = fixture(tmp_path)
    base, lidar, _ = load_ego_poses(seq, rows)
    temporal = build_temporal(rows, results, base, lidar, {}, stage='refine')
    assert temporal['frames'][0]['labels'][0]['source'] == 'unknown'
    assert temporal['frames'][1]['labels'][0]['source'] == 'camera_hold'


def test_tracker_coast_explicit(tmp_path):
    seq, rows, results, tracks = fixture(tmp_path)
    del results[1]['objects'][0]['source']
    base, lidar, _ = load_ego_poses(seq, rows)
    temporal = build_temporal(rows, results, base, lidar, track_hit_lookup(tracks))
    assert temporal['frames'][1]['labels'][0]['source'] == 'tracker_prediction'


def test_out_of_bounds_odometry_index_rejected(tmp_path):
    seq, rows, results, tracks = fixture(tmp_path)
    rows[1]['odometry_index'] = '9'
    with pytest.raises(ValueError, match='odometry_index'):
        load_ego_poses(seq, rows)


def test_duplicate_acquisition_and_invalid_boxes_rejected(tmp_path):
    seq, rows, results, tracks = fixture(tmp_path)
    base, lidar, _ = load_ego_poses(seq, rows)
    rows[1]['lidar_sensor_timestamp_ns'] = rows[0]['lidar_sensor_timestamp_ns']
    with pytest.raises(ValueError, match='duplicate LiDAR'):
        build_temporal(rows, results, base, lidar)
    rows[1]['lidar_sensor_timestamp_ns'] = '2300000000'
    results[1]['objects'][0]['box'][3] = float('nan')
    with pytest.raises(ValueError, match='invalid 3D box'):
        build_temporal(rows, results, base, lidar)


def test_legacy_fallback_clock_is_marked(tmp_path):
    seq, rows, results, tracks = fixture(tmp_path)
    rows[0].pop('lidar_sensor_timestamp_ns')
    base, lidar, _ = load_ego_poses(seq, rows)
    temporal = build_temporal(rows, results, base, lidar)
    assert temporal['frames'][0]['timestamp_clock'] == 'recorder_fallback'
    assert temporal['frames'][0]['acquisition_timestamp_ns'] == 1800000900


def test_multiple_frame_overwrite_cleans_previous_labels(tmp_path):
    seq, rows, results, tracks = fixture(tmp_path)
    legacy = seq / 'results.json'
    export_annotations(seq, rows, results, legacy, tracks=tracks)
    export_annotations(seq, rows[:1], results[:1], legacy, tracks=tracks)
    assert not (seq / 'results_fisheye3dod/label_3d/000001.json').exists()
