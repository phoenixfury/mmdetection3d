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
