"""CPU-only regression tests using import stubs for unavailable GPU deps."""
import importlib.util
import sys
import types
from pathlib import Path

import numpy as np
import pytest


ROOT = Path(__file__).resolve().parents[1]
SCRIPTS = ROOT / 'integration_scripts' if (ROOT / 'integration_scripts').is_dir() else ROOT


def load_module(name, relative):
    spec = importlib.util.spec_from_file_location(name, SCRIPTS / relative)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope='module')
def scripts():
    try:
        import easydict  # noqa: F401
    except ImportError:
        stub = types.ModuleType('easydict')
        stub.EasyDict = dict
        sys.modules['easydict'] = stub
    runner = load_module('run_detzero_sequence', 'run_detzero_sequence.py')
    improve = load_module('improve_detzero_annotations', 'improve_detzero_annotations.py')
    return runner, improve


def test_camera_hold_compensates_ego_height(scripts):
    _, improve = scripts
    state = {
        'track_id': 4, 'name': 'Car', 'score': .8,
        'global_center': np.array([12., 0., 3.]),
        'global_yaw': 0.0, 'velocity': np.zeros(2),
        'dimensions': np.array([4., 2., 1.5]),
        'local_z': 3.0, 'timestamp_ns': 1_000_000_000, 'hold_age': 0,
    }
    pose = np.eye(4)
    pose[2, 3] = 1.5  # vehicle climbs a slope
    obj, nxt = improve.predicted_hold_object(
        state, 1, 1_500_000_000, pose, None, None, 60., 4., 45., 'test')
    assert obj['source'] == 'camera_track_hold'
    assert obj['box'][2] == pytest.approx(1.5)
    assert nxt['global_center'][2] == pytest.approx(3.)


def test_static_camera_hold_ignores_drifting_tracker_coast(scripts):
    _, improve = scripts
    state = {
        'track_id': 4, 'name': 'Truck', 'score': .8,
        'global_center': np.array([12., 3., 1.]),
        'global_yaw': .2, 'velocity': np.zeros(2),
        'dimensions': np.array([7., 2.5, 3.]),
        'timestamp_ns': 1_000_000_000, 'last_lidar_frame': 0,
        'hold_age': 0, 'is_static': True,
    }
    drifting_coast = np.array([18., 8., 1., 7., 2.5, 3., 1.2])
    obj, nxt = improve.predicted_hold_object(
        state, 2, 2_000_000_000, np.eye(4), drifting_coast,
        np.array([6., 5.]), 60., 4., 45., 'tracker')
    assert obj['box'][:3] == pytest.approx([12., 3., 1.])
    assert obj['box'][6] == pytest.approx(.2)
    assert nxt['global_center'] == pytest.approx([12., 3., 1.])
    assert obj['hold_age_frames'] == 2


def test_camera_bus_truck_subtype_requires_track_consensus(scripts):
    from collections import Counter
    _, improve = scripts
    resolve = improve.camera_vehicle_subtype
    assert resolve(Counter(Truck=4.2, Bus=.5),
                   Counter(Truck=7, Bus=1)) == 'Truck'
    assert resolve(Counter(Bus=3.0, Truck=.4),
                   Counter(Bus=4, Truck=1)) == 'Bus'
    assert resolve(Counter(Truck=.8), Counter(Truck=1)) is None
    assert resolve(Counter(Truck=1.4, Bus=1.3),
                   Counter(Truck=3, Bus=3)) is None


def test_heavy_camera_labels_are_gated_by_lidar_dimensions(scripts):
    _, improve = scripts
    choose = improve.camera_subtype_for_dimensions
    enforce = improve.enforce_vehicle_size_class
    # A tall, non-heavy Car is a Van even when YOLO calls it Truck.
    van = np.array([4.6, 1.9, 2.0])
    assert choose('Truck', van, 'Car') == 'Van'
    assert enforce('Car', van) == 'Van'
    assert enforce('Truck', van) == 'Van'
    # A normal-height hatchback cannot become Truck or Bus.
    car = np.array([4.5, 1.9, 1.6])
    assert choose('Truck', car, 'Car') == 'Car'
    assert enforce('Truck', car) == 'Car'
    # Detector height noise on a hatchback/crossover is not enough for Van.
    tall_hatchback = np.array([4.3, 1.8, 1.78])
    assert enforce('Car', tall_hatchback) == 'Car'
    # Heavy and bus-sized tracks may use the explicit camera subtype.
    truck = np.array([6.0, 2.2, 2.4])
    bus = np.array([11.0, 2.7, 3.4])
    assert choose('Truck', truck, 'Car') == 'Truck'
    assert choose('Bus', truck, 'Car') == 'Truck'
    assert choose('Bus', bus, 'Car') == 'Bus'


def test_overlap_suppression_uses_evidence_confidence_then_age(scripts):
    _, improve = scripts

    def obj(track_id, *, source=None, score=.5, age=0, x=0.):
        value = {'track_id': track_id, 'name': 'Car', 'score': score,
                 'box': [x, 0., 0., 4.5, 1.85, 1.55, 0.]}
        if source:
            value.update(source=source, hold_age_frames=age)
        return value

    detected = obj(1, score=.2)
    predicted = obj(2, source='camera_track_hold', score=.99, age=1)
    retained, removed = improve.suppress_overlapping_boxes(
        [predicted, detected], .1)
    assert [item['track_id'] for item in retained] == [1]
    assert removed['predicted_lost_to_lidar_detection'] == 1

    newer = obj(3, source='camera_track_hold', score=.7, age=1)
    older = obj(4, source='camera_track_hold', score=.7, age=5)
    retained, removed = improve.suppress_overlapping_boxes([older, newer], .1)
    assert [item['track_id'] for item in retained] == [3]
    assert removed['older_lidar_detection'] == 1

    high = obj(5, source='camera_track_hold', score=.8, age=5)
    retained, removed = improve.suppress_overlapping_boxes([newer, high], .1)
    assert [item['track_id'] for item in retained] == [5]
    assert removed['lower_confidence'] == 1


def test_lidar_to_global_respects_rotation(scripts):
    runner, _ = scripts
    pose = np.eye(4)
    pose[:2, :2] = [[0, -1], [1, 0]]
    pose[0, 3] = 8.
    boxes = np.array([[2., 1., 3., 4., 2., 1., 0.]], dtype=float)
    world = runner.boxes_to_global(boxes, pose)
    assert world[0, :3] == pytest.approx([7., 2., 3.])
    assert world[0, 6] == pytest.approx(np.pi / 2)


def test_detector_box_fail_fast_no_silent_nan(scripts):
    runner, _ = scripts
    frame = {'frame_id': 0, 'pose': np.eye(4),
             'boxes_lidar': np.array([[np.nan, 0, 0, 4, 2, 2, 0.]]),
             'score': np.array([.8]),
             'name': np.array(['Vehicle']),
             'original_name': np.array(['Car'])}
    with pytest.raises(ValueError, match='invalid detection'):
        runner.check_frame_boxes([frame])
