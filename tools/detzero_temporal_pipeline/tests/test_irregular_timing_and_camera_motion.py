"""CPU-only physical regression tests for asynchronous LiDAR/camera and DetZero."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
FILTERS = ROOT / 'DetZero/tracking/detzero_track/models/tracking_modules/kalman_filter'


def load(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def filters():
    name = 'timing_test_filters'
    pkg = types.ModuleType(name)
    pkg.__path__ = [str(FILTERS)]
    sys.modules[name] = pkg
    helper = load(f'{name}.motion_time', FILTERS / 'motion_time.py')
    kalman = load(f'{name}.kalman_filter', FILTERS / 'kalman_filter.py')
    return helper, kalman


def rotation(yaw):
    out = np.eye(4)
    out[:2, :2] = [[np.cos(yaw), -np.sin(yaw)],
                    [np.sin(yaw), np.cos(yaw)]]
    return out


def test_timestep_irregular_and_reverse(filters):
    helper, _ = filters
    assert helper.elapsed_seconds(1_150_000_000, 1_000_000_000, .5) == pytest.approx(.15)
    assert helper.elapsed_seconds(1_000_000_000, 1_150_000_000, -.5) == pytest.approx(-.15)
    assert helper.elapsed_seconds(None, None, -.5) == -.5
    with pytest.raises(ValueError, match='opposite'):
        helper.elapsed_seconds(1, 2, .5)
    with pytest.raises(ValueError, match='repeated'):
        helper.elapsed_seconds(1, 1, .5)


def test_planar_filter_propagates_actual_dt(filters):
    _, kalman = filters
    kf = kalman.KalmanFilter(
        bbox=np.array([0., 0., 0., 4., 2., 2., 0.], dtype=np.float32),
        name='Pedestrian', score=.8, frame_id=0, track_id=42,
        timestamp_ns=1_000_000_000, delta_t=.5, q=[2, 25])
    kf.x[3:5, 0] = [10., 1.]
    kf.predict(1, 1_100_000_000)
    assert kf.x[0, 0] == pytest.approx(1.)
    assert kf.x[1, 0] == pytest.approx(.1)
    assert kf.F[0, 3] == pytest.approx(.1)
    short_q = kf.Q.copy()
    kf.predict(2, 1_500_000_000)
    assert kf.x[0, 0] == pytest.approx(5.)
    assert kf.Q[0, 0] > short_q[0, 0]
    assert kf.Q[0, 3] > 0
    assert kf.info()[42]['timestamp_ns'] == 1_500_000_000


def test_reverse_custom_filter(filters):
    _, kalman = filters
    kf = kalman.KalmanFilter(
        bbox=np.array([10., 0., 0., 4., 2., 1., 0.], dtype=np.float32),
        name='Pedestrian', score=.8, frame_id=2, track_id=0,
        timestamp_ns=2_000_000_000, delta_t=-.5)
    kf.x[3, 0] = 2.
    kf.predict(1, 1_800_000_000)
    assert kf.x[0,0] == pytest.approx(9.6)
    assert np.linalg.eigvalsh(kf.Q).min() > -1e-4
    assert kf.Q[0,3] < 0  # reverse-time CV noise covariance


def test_camera_pose_and_object_velocity():
    from camera_motion import CameraMotion
    t = [1_000_000_000, 1_500_000_000]
    p0 = rotation(0)
    p1 = rotation(np.pi/2)
    p1[0, 3] = 5
    rows = [dict(lidar_sensor_timestamp_ns=t[0], camera_sync_clock='sensor',
                 arc_f_timestamp_ns=1_100_000_000),
            dict(lidar_sensor_timestamp_ns=t[1], camera_sync_clock='sensor',
                 arc_f_timestamp_ns=1_500_000_000)]
    motion = CameraMotion(rows, np.stack([p0,p1]))
    g = motion.geometry(0, 'ARC_F')
    assert g['applied']
    assert g['pose_camera'][0,3] == pytest.approx(1)
    assert np.arctan2(g['pose_camera'][1,0],g['pose_camera'][0,0]) == pytest.approx(np.pi/10)
    original = np.array([10.,0.,0.,4.,2.,2.,0.])
    static = motion.box(original, 0, 'ARC_F')
    moving = motion.box(original, 0, 'ARC_F', velocity_world=[10.,0.])
    recovered_static = (g['pose_camera'] @ np.r_[static[:3],1.])[:3]
    recovered_moving = (g['pose_camera'] @ np.r_[moving[:3],1.])[:3]
    assert recovered_static == pytest.approx([10,0,0])
    assert recovered_moving == pytest.approx([11,0,0])
    assert static[6] == pytest.approx(-np.pi/10)
    assert motion.point_transform(0, 'ARC_F', np.eye(4)) == pytest.approx(
        np.linalg.inv(g['pose_camera']) @ p0)
    details = motion.details(0, 'ARC_F', has_velocity=True)
    assert details['applied']
    assert details['object_motion_velocity_used'] is True
    assert details['camera_minus_lidar_seconds'] == pytest.approx(.1)


def test_camera_compensation_safe_fallback():
    from camera_motion import CameraMotion
    p = np.stack([np.eye(4), np.eye(4)])
    rows = [dict(lidar_sensor_timestamp_ns=1_000_000_000,
                 camera_sync_clock='read', arc_r_timestamp_ns=1_050_000_000),
            dict(lidar_sensor_timestamp_ns=1_500_000_000,
                 camera_sync_clock='sensor', arc_r_timestamp_ns=1_510_000_000)]
    motion = CameraMotion(rows, p)
    assert motion.geometry(0, 'ARC_R')['reason'] == 'camera_and_lidar_clock_not_proven_comparable'
    assert motion.geometry(1, 'ARC_R')['reason'] == 'camera_time_unbracketed_by_ego_poses'
    assert motion.box(np.array([5,0,0,2,2,2,0]), 0, 'ARC_R') == pytest.approx([5,0,0,2,2,2,0])
    blocked = CameraMotion(rows, p, enabled=False)
    assert blocked.geometry(0, 'ARC_R')['reason'] == 'disabled_by_user'


def test_camera_compensated_matching(monkeypatch):
    import importlib
    import types
    import sys
    try:
        import easydict
    except ImportError:
        stub = types.ModuleType('easydict')
        stub.EasyDict = dict
        sys.modules['easydict'] = stub
    improve = importlib.import_module('improve_detzero_annotations')
    from camera_motion import CameraMotion
    rows = [dict(lidar_sensor_timestamp_ns=1_000_000_000, camera_sync_clock='sensor',
                 arc_f_timestamp_ns=1_050_000_000),
            dict(lidar_sensor_timestamp_ns=1_500_000_000, camera_sync_clock='sensor',
                 arc_f_timestamp_ns=1_500_000_000)]
    p0, p1 = np.eye(4), np.eye(4)
    p1[0, 3] = 5
    motion = CameraMotion(rows, np.stack([p0,p1]))
    # The approximate bounding-box footprint moves by -0.5 m with ego motion.
    monkeypatch.setattr(improve, 'projected_box',
                        lambda box, transform, model: np.array(
                            [box[0], 0., box[0]+2, 100.]))
    frame = {'objects':[{'track_id':1, 'name':'Car', 'box':[5,0,0,4,2,1,0]}]}
    det = {'0:ARC_F': [{'xyxy':[4.5,0,6.5,100], 'class':'Car', 'confidence':.99}]}
    models = {key:{} for key in improve.CAMERAS}
    transforms = {key:np.eye(4) for key in improve.CAMERAS}
    evidence, _ = improve.match_frame(frame,0,det,models,transforms,60.,motion=motion)
    assert evidence[0]['camera'] == 'ARC_F'
    assert evidence[0]['overlap'] == pytest.approx(1.)


def test_heading_smoothing_uses_acquisition_elapsed_time():
    import copy
    from improve_detzero_annotations import smooth_track_headings

    frames = [
        {'objects': [{'track_id': 7, 'name': 'Car',
                      'box': [x, 0., 0., 4., 2., 1.5, np.pi]}]}
        for x in (0., 3.)
    ]
    poses = np.stack((np.eye(4), np.eye(4)))
    slow = copy.deepcopy(frames)
    smooth_track_headings(slow, poses,
                          np.array([1_000_000_000, 11_000_000_000]))
    assert abs(slow[0]['objects'][0]['box'][6]) == pytest.approx(np.pi)
    fast = copy.deepcopy(frames)
    smooth_track_headings(fast, poses,
                          np.array([1_000_000_000, 1_500_000_000]))
    assert fast[0]['objects'][0]['box'][6] == pytest.approx(0.)


def test_vehicle_consensus_removes_bus_truck_flicker():
    from improve_detzero_annotations import consensus_vehicle_name
    objects = [dict(name=name, score=.7)
               for name in ('Bus', 'Truck', 'Bus')]
    assert consensus_vehicle_name(objects, np.array([11., 2.7, 3.4])) == 'Bus'
    tie = [dict(name='Truck', score=.5), dict(name='Bus', score=.9)]
    assert consensus_vehicle_name(tie, np.array([10., 2.7, 3.4])) == 'Bus'


def test_vehicle_size_classification_uses_track_dimensions():
    from improve_detzero_annotations import classify_vehicle_by_size
    classify = classify_vehicle_by_size
    assert classify(np.array([3.7, 1.73, 1.82]), 'Car') == 'Van'
    assert classify(np.array([4.45, 1.90, 1.62]), 'Car') == 'Car'
    assert classify(np.array([5.6, 2.2, 2.2]), 'Car') == 'Truck'
    assert classify(np.array([11.8, 3.0, 3.6]), 'Car') == 'Truck'
    assert classify(np.array([11.8, 3.0, 3.6]), 'Bus') == 'Bus'
    assert classify(np.array([11.8, 3.0, 3.6]), 'Truck') == 'Truck'
    assert classify(np.array([5.6, 2.2, 2.2]), 'Bus') == 'Truck'
    assert classify(np.array([40., 8., 9.]), 'Car') == 'Car'


def test_large_car_track_is_relabelable_before_dimension_rejection():
    from improve_detzero_annotations import invalid_dimension_tracks
    frames = [{'objects': [{'track_id': 1, 'name': 'Car',
                            'box': [0., 0., 0., 5.6, 2.2, 2.2, 0.]},
                           {'track_id': 2, 'name': 'Cyclist',
                            'box': [0., 0., 0., 7., 2.2, 2.2, 0.]}]}
              for _ in range(3)]
    invalid, _ = invalid_dimension_tracks(frames)
    assert 1 not in invalid
    assert 2 in invalid


def test_size_jump_is_not_used_to_rescue_or_relabel_a_track():
    from improve_detzero_annotations import (invalid_dimension_tracks,
                                              vehicle_size_stable)
    unstable = np.array([[4.5, 1.9, 1.6], [11., 3., 3.5],
                         [4.6, 1.9, 1.6], [11., 3., 3.5]])
    assert not vehicle_size_stable(unstable)
    assert vehicle_size_stable(np.array([[5.5, 2.2, 2.2],
                                         [5.6, 2.2, 2.3],
                                         [5.7, 2.3, 2.3]]))
    frames = [{'objects': [{'track_id': 1, 'name': 'Car',
                            'box': [0., 0., 0., *dims, 0.]}]}
              for dims in unstable]
    invalid, _ = invalid_dimension_tracks(frames)
    assert 1 in invalid


def test_tiny_car_detection_cannot_confirm_phantom_bus(monkeypatch):
    import improve_detzero_annotations as postprocess
    projected = np.array([0., 0., 142., 151.])
    tiny_car = np.array([35., 110., 80., 148.])
    assert postprocess.iou(projected, tiny_car) < .12
    assert postprocess.overlap_detection(projected, tiny_car) == 1.
    assert postprocess.camera_match_score(projected, tiny_car) == 0.

    monkeypatch.setattr(postprocess, 'projected_box',
                        lambda box, transform, model: projected)
    frame = {'objects': [{'track_id': 218, 'name': 'Bus',
                          'box': [20., 0., 0., 12., 3., 4., 0.]}]}
    detections = {'0:ARC_F': [{'class': 'Car', 'confidence': .25,
                               'xyxy': tiny_car.tolist()}]}
    models = {camera: {} for camera in postprocess.CAMERAS}
    transforms = {camera: np.eye(4) for camera in postprocess.CAMERAS}
    evidence, used = postprocess.match_frame(frame, 0, detections, models,
                                             transforms, 60.)
    assert evidence[0]['match'] is None
    assert not used
    predictions = {218: (frame['objects'][0], {'velocity': None})}
    assert not postprocess.match_camera_hold_predictions(
        predictions, 0, detections, models, transforms, set(), .55)

    detections['0:ARC_F'][0]['xyxy'] = [10., 10., 130., 140.]
    evidence, used = postprocess.match_frame(frame, 0, detections, models,
                                             transforms, 60.)
    assert evidence[0]['match'] is not None
    assert used == {('ARC_F', 0)}


def test_backward_compat_reconstruct_camera_timestamps():
    from camera_motion import CameraMotion
    p = np.stack([np.eye(4), np.eye(4)])
    rows = [dict(lidar_sensor_timestamp_ns=1_000_000_000,
                 camera_sync_clock='sensor', camera_sync_target_ns=1_000_000_000,
                 arc_f_delta_ns=10_000_000),
            dict(lidar_sensor_timestamp_ns=1_500_000_000,
                 camera_sync_clock='sensor', camera_sync_target_ns=1_500_000_000,
                 arc_f_delta_ns=0)]
    motion = CameraMotion(rows,p)
    record = motion.geometry(0,'ARC_F')
    assert record['applied']
    assert record['camera_packet_timestamp_ns'] == 1_010_000_000
    assert record['camera_timestamp_origin'] == 'reconstructed_from_sync_delta'


def test_ab3dmot_filter_uses_acquisition_time(filters, monkeypatch):
    import types
    _, base = filters
    try:
        from filterpy.kalman import KalmanFilter as FilterPyKalman
    except ImportError:
        class FilterPyKalman:
            def __init__(self, dim_x, dim_z):
                self.F = np.eye(dim_x)
                self.Q = np.eye(dim_x)
                self.P = np.eye(dim_x)
                self.H = np.eye(dim_z, dim_x)
                self.x = np.zeros((dim_x,1))
            def predict(self):
                self.x = self.F @ self.x
                self.P = self.F @ self.P @ self.F.T + self.Q
            def update(self, z):
                pass
        module = types.ModuleType('filterpy')
        kalman = types.ModuleType('filterpy.kalman')
        kalman.KalmanFilter = FilterPyKalman
        module.kalman = kalman
        monkeypatch.setitem(sys.modules, 'filterpy', module)
        monkeypatch.setitem(sys.modules, 'filterpy.kalman', kalman)
    ab3dmot = load('timing_test_filters.ab3dmot', FILTERS / 'ab3dmot.py')
    box = np.array([10., 0., 3., 4., 2., 1.5, 0.], dtype=np.float32)
    tracker = ab3dmot.AB3DMOT(box,'Vehicle',.9,0,12,
                              timestamp_ns=1_000_000_000,delta_t=.5)
    tracker.kf.x[7,0] = 4.
    tracker.kf.x[9,0] = 100.  # vertical velocity must not be used
    tracker.predict(1,1_250_000_000)
    assert tracker.bbox[0] == pytest.approx(11.)
    assert tracker.bbox[2] == pytest.approx(3.)
    assert tracker.kf.F[0,7] == pytest.approx(.25)
    assert tracker.kf.F[2,9] == 0.
    assert tracker.kf.Q[0,7] > 0.
    assert tracker.info()[12]['timestamp_ns'] == 1_250_000_000
    tracker.predict(2,1_500_000_000)
    assert tracker.bbox[0] == pytest.approx(12.)
