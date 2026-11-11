"""GPU-free integration smoke test for projected 3D boxes and all five videos."""
from __future__ import annotations

import csv
import json
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'integration_scripts'))
from render_improved_detzero import (main, render_bev, box_corners, kb4_project,
                                     CLASS_COLORS, safe_child)


def test_projection_geometry():
    box = np.array([10., 0., 0., 4., 2., 2., 0.])
    assert box_corners(box).shape == (8, 3)
    assert np.allclose(sorted(set(box_corners(box)[:, 0])), [8., 12.])
    model = {'K': np.array([[60., 0, 80.], [0, 60., 60.], [0, 0, 1.]]),
             'D': np.zeros(4)}
    uv, good = kb4_project(np.array([[0., 0., 8.], [1., 0., 8.]]), model)
    assert good.all() and np.allclose(uv[0], [80., 60.])
    assert uv[1, 0] > uv[0, 0]
    assert len({tuple(x) for x in CLASS_COLORS.values()}) == len(CLASS_COLORS)


def test_end_to_end_video(tmp_path):
    source = tmp_path / 'recording' / 'dataset'
    source.mkdir(parents=True)
    image = np.full((120, 160, 3), 50, dtype=np.uint8)
    # CLIDAR x forward, y left -> camera optical axis +Z.
    matrix = np.eye(4)
    matrix[:3, :3] = np.array([[0, -1, 0], [0, 0, -1], [1, 0, 0]])
    cams = ('ARC_F', 'ARC_SL', 'ARC_SR', 'ARC_R')
    calibration = {'extrinsics': {'CLIDAR': {'from': 'CLIDAR', 'to': 'BASELINK',
                                           'matrix': np.eye(4).tolist()}},
                   'intrinsics': {}, 'lidar_to_camera': {}}
    for cam in cams:
        calibration['intrinsics'][cam] = {
            'camera_matrix': [[65., 0., 80.], [0., 65., 60.], [0., 0., 1.]],
            'distortion_coefficients': [0., 0., 0., 0.], 'width': 160, 'height': 120}
        calibration['lidar_to_camera'][cam] = matrix.tolist()
    (source / 'calibration.json').write_text(json.dumps(calibration))
    times = [1700000000000000000 + i * 500000000 for i in range(3)]
    np.save(source / 'odometry_poses.npy', np.repeat(np.eye(4)[None], 3, axis=0))
    rows, frames = [], []
    for i, stamp in enumerate(times):
        img_paths = {}
        for cam in cams:
            path = f'images/{cam.lower()}_{i}.jpg'
            (source / 'images').mkdir(exist_ok=True)
            cv2.imwrite(str(source / path), image)
            img_paths[cam.lower() + '_path'] = path
        pcd_path = f'pointcloud/{i}.pcd'
        (source / 'pointcloud').mkdir(exist_ok=True)
        pts = np.array([[5., -1., 0., .5], [8., 0., 0., .9],
                        [10., 2., 0.5, .2], [12., 3., 0., .5]], dtype='<f4')
        header = ('FIELDS x y z intensity\nSIZE 4 4 4 4\nTYPE F F F F\n'
                  'COUNT 1 1 1 1\nWIDTH 4\nHEIGHT 1\nPOINTS 4\nDATA binary\n').encode()
        (source / pcd_path).write_bytes(header + pts.tobytes())
        rows.append({'timestamp_ns': stamp, 'lidar_sensor_timestamp_ns': stamp,
                     'camera_sync_clock': 'sensor', 'odometry_index': i,
                     'lidar_pcd': pcd_path, **img_paths,
                     **{cam.lower() + '_timestamp_ns': stamp for cam in cams}})
        frames.append({'frame_id': i, 'timestamp_ns': stamp,
                       'objects': [{'track_id': 10, 'name': 'Car', 'source': 'lidar',
                                        'box': [9., 0., 0., 4., 2., 2., 0.]}]})
    with (source / 'manifest.csv').open('w', newline='') as stream:
        w = csv.DictWriter(stream, fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)
    out_dir = source / 'detzero_output'
    out_dir.mkdir()
    (out_dir / 'improved_results.json').write_text(json.dumps(frames))
    main([str(source), '--camera-width', '160', '--camera-height', '120',
          '--fps', '2', '--bev-range', '20'])
    for suffix in ('all_cameras_plus_bev', 'lidar_bev', 'arc_f_boxes',
                   'arc_sl_boxes', 'arc_sr_boxes', 'arc_r_boxes'):
        path = out_dir / 'improved_videos' / (suffix + '.mp4')
        assert path.stat().st_size > 100
        reader = cv2.VideoCapture(str(path)); assert reader.isOpened()
        assert int(reader.get(cv2.CAP_PROP_FRAME_COUNT)) == 3
        assert reader.read()[0]; reader.release()
    info = json.loads((out_dir / 'improved_videos' / 'visualization_report.json').read_text())
    assert info['frames'] == 3 and info['per_class_boxes']['Car'] == 3
    assert all(info['camera_projections_drawn'].get(c, 0) == 3 for c in cams)

    diagnostic = source / 'sync_diagnostic_videos'
    main([str(source), '--camera-width', '160', '--camera-height', '120',
          '--fps', '2', '--bev-range', '20', '--no-annotations',
          '--output', str(diagnostic)])
    report = json.loads((diagnostic / 'visualization_report.json').read_text())
    assert report['frames'] == 3
    assert report['per_class_boxes'] == {}
    assert report['camera_projections_drawn'] == {}


def test_reject_unsafe_paths(tmp_path):
    import pytest
    with pytest.raises(ValueError):
        safe_child(tmp_path, '../outside.jpg')
