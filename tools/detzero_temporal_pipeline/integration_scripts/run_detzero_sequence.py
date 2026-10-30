#!/usr/bin/env python3
"""Run CenterPoint -> DetZero tracking/refining on an exported bag sequence."""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import json
import math
import os
import pickle
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np
import torch
from easydict import EasyDict
from tqdm import tqdm

from temporal_annotations import export_annotations, load_ego_poses


# Resolve both an integrated mmdetection3d checkout and the standalone ZIP
# layout (integration_scripts/ beside DetZero/). Environment overrides let
# users keep their CenterPoint config/weights in an existing checkout.
_SEARCH_ROOTS = (Path(__file__).resolve().parent,
                 *Path(__file__).resolve().parents)
ROOT = Path(os.environ['MMDET3D_ROOT']).expanduser().resolve() if 'MMDET3D_ROOT' in os.environ else next(
    (p for p in _SEARCH_ROOTS if (p / 'configs' / 'centerpoint').is_dir()),
    Path(__file__).resolve().parent.parent)
DETZERO = Path(os.environ['DETZERO_ROOT']).expanduser().resolve() if 'DETZERO_ROOT' in os.environ else next(
    (p / 'DetZero' for p in _SEARCH_ROOTS if (p / 'DetZero' / 'tracking').is_dir()),
    ROOT / 'DetZero')
# DetZero incorrectly uses str.strip rather than removeprefix/removesuffix when
# resolving Waymo sequence names; a numeric token survives that legacy logic.
SEQUENCE_NAME = '0000000000000000000'
CLASS_NAMES = ('Vehicle', 'Pedestrian', 'Cyclist')
VULNERABLE_ROAD_USERS = {'Pedestrian', 'Cyclist'}
REFINED_CLASSES = {'Vehicle'}
STATIC_HEADING_SPEED_MPS = 0.75
TRACKING_GAP_RESET_SECONDS = 4.0
LABEL_TO_ORIGINAL_CLASS = {
    0: 'Car',
    1: 'Truck',
    # Keep the native NuScenes fine class.  Collapsing it to Truck loses the
    # distinction used by the CaliperGT taxonomy and cannot be recovered by
    # track-level class voting later in the pipeline.
    2: 'ConstructionVehicle',
    3: 'Bus',
    4: 'Truck',  # trailer
    6: 'Cyclist',  # motorcycle
    7: 'Cyclist',  # bicycle
    8: 'Pedestrian',
}
ORIGINAL_TO_DETZERO_CLASS = {
    'Car': 'Vehicle',
    'Van': 'Vehicle',
    'Truck': 'Vehicle',
    'Bus': 'Vehicle',
    'ConstructionVehicle': 'Vehicle',
    'Pedestrian': 'Pedestrian',
    'Cyclist': 'Cyclist',
}
COLORS = {
    'Vehicle': (40, 210, 40),
    'Car': (255, 216, 0),
    'Van': (255, 121, 41),
    'Truck': (0, 145, 255),
    'Bus': (68, 23, 255),
    'ConstructionVehicle': (160, 80, 255),
    'Pedestrian': (40, 190, 240),
    'Cyclist': (230, 120, 30),
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', type=Path,
                        help='synchronized_dual_images folder')
    parser.add_argument('--output', type=Path,
                        help='default: SEQUENCE/detzero_output')
    parser.add_argument('--stage', choices=('all', 'detect', 'track', 'refine',
                                            'render'), default='all')
    parser.add_argument('--config', type=Path, default=ROOT / (
        'configs/centerpoint/centerpoint_voxel0075_second_secfpn_'
        'head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'))
    parser.add_argument('--checkpoint', type=Path, default=(
        ROOT / 'checkpoints/centerpoint_nus_voxel0075_dcn.pth'))
    parser.add_argument(
        '--calibration', type=Path,
        help='calibration.json containing T_BASELINK_CLIDAR; defaults to '
             'SEQUENCE/calibration.json or SEQUENCE/../calibration.json')
    parser.add_argument('--detzero-checkpoints', type=Path,
                        default=ROOT / 'checkpoints_det3zero')
    parser.add_argument(
        '--enable-position-refiner', action='store_true',
        help='run the Waymo-trained DetZero PRM; disabled by default because '
             'the reviewed custom-bag benchmark favors tracker positions')
    parser.add_argument('--score-threshold', type=float, default=0.5)
    parser.add_argument('--vehicle-score-threshold', type=float)
    parser.add_argument('--pedestrian-score-threshold', type=float, default=0.2)
    parser.add_argument('--cyclist-score-threshold', type=float)
    parser.add_argument(
        '--tracking-score-threshold', type=float,
        help='filter cached detections before tracking')
    parser.add_argument(
        '--fast-tracking', action='store_true',
        help='disable reverse tracking and quadratic track merging')
    parser.add_argument(
        '--reverse-max-misses', type=int,
        help='retain reverse tracking but discard a reverse track after this '
             'many unmatched frames')
    parser.add_argument(
        '--tracking-profile', choices=('default', 'camera-2hz'),
        default='camera-2hz', help='camera-2hz enables wider sparse-frame motion '
        'gates and 2 Hz Kalman/noise tuning')
    parser.add_argument(
        '--tracking-filter', choices=('ab3dmot', 'detzero-kalman'),
        default='ab3dmot',
        help='tracking state estimator (default: planar AB3DMOT)')
    parser.add_argument('--limit', type=int)
    parser.add_argument('--render-range', type=float, default=60.0)
    parser.add_argument('--render-size', type=int, default=1000)
    parser.add_argument('--fps', type=float, default=2.0)
    parser.add_argument('--skip-classic-labels', action='store_true',
                        help='write temporal sidecars but omit label_3d files')
    return parser.parse_args()


def wrap_angle(angle):
    """Wrap scalar or array angles to [-pi, pi)."""
    return (np.asarray(angle) + np.pi) % (2 * np.pi) - np.pi


def stabilize_tracks(tracks: dict, frame_times: np.ndarray | None = None,
                     detection_frames: list[dict] | None = None,
                     delta_t: float = 0.5,
                     static_speed: float = STATIC_HEADING_SPEED_MPS) -> dict:
    """Stabilize class and heading without inventing tracker geometry."""
    detections_by_frame = {}
    if detection_frames is not None:
        for frame in detection_frames:
            detections_by_frame[int(frame['frame_id'])] = (
                np.asarray(frame['name'], dtype=object),
                boxes_to_global(np.asarray(frame['boxes_lidar']), frame['pose']))

    for track in tracks.values():
        boxes = np.asarray(track['boxes_global'])
        if not len(boxes):
            continue
        # Reverse tracking can append earlier observations after later ones.
        # Every temporal operation below (class flicker, speed, yaw unwrap,
        # interpolation) requires chronological order, and np.interp in
        # particular is undefined for descending frame IDs.
        frame_ids = np.asarray(track['sample_idx'], dtype=int)
        order = np.argsort(frame_ids, kind='stable')
        if not np.array_equal(order, np.arange(len(order))):
            for key, value in list(track.items()):
                if (isinstance(value, np.ndarray) and value.ndim > 0 and
                        len(value) == len(order)):
                    track[key] = value[order]
                elif isinstance(value, list) and len(value) == len(order):
                    track[key] = [value[index] for index in order]
            boxes = np.asarray(track['boxes_global'])
            frame_ids = np.asarray(track['sample_idx'], dtype=int)
        hit = np.asarray(track['hit']) > 0
        evidence = hit if np.any(hit) else np.ones(len(boxes), dtype=bool)
        evidence_indices = np.flatnonzero(evidence)
        scores = np.asarray(track['score'], dtype=float)
        fine_names = np.asarray(
            track.get('original_name', track['name']), dtype=object)

        # Correct only a one-observation class flicker surrounded by the same
        # class. Whole-track voting performed poorly when an association spans
        # two nearby vehicle types, while this handles Bus/Truck/Bus without
        # propagating an ID-switch mistake through the complete track.
        stabilized_names = fine_names.copy()
        observed_names = fine_names[evidence_indices]
        for position in range(1, len(observed_names) - 1):
            if (observed_names[position - 1] ==
                    observed_names[position + 1] !=
                    observed_names[position]):
                stabilized_names[evidence_indices[position]] = (
                    observed_names[position - 1])
        track['original_name'] = stabilized_names

        if frame_times is None:
            times = frame_ids.astype(float) * delta_t
        else:
            times = np.asarray(frame_times, dtype=float)[frame_ids]
        centers = boxes[:, :2]
        poses = np.asarray(track['pose'])
        first, last = evidence_indices[0], evidence_indices[-1]
        elapsed = max(times[last] - times[first], delta_t)
        displacement = centers[last] - centers[first]
        # Static is a world-frame property. A vehicle can remain at nearly
        # the same relative position while both it and the ego vehicle move;
        # treating low local displacement as static freezes its LiDAR-frame
        # yaw during ego turns. Use smoothed global path speed instead of the
        # end-to-end chord, which can also be small after a curved/looping path.
        static_window = min(
            7, len(centers) if len(centers) % 2 else len(centers) - 1)
        if static_window >= 3:
            radius = static_window // 2
            kernel = np.ones(static_window, dtype=float) / static_window
            static_centers = np.column_stack([
                np.convolve(np.pad(centers[:, dim], (radius, radius),
                                   mode='edge'), kernel, mode='valid')
                for dim in range(2)
            ])
        else:
            static_centers = centers
        global_path_speed = float(np.linalg.norm(
            np.diff(static_centers[first:last + 1], axis=0), axis=1
        ).sum() / elapsed)
        nearly_static = global_path_speed < static_speed

        # Recover CenterPoint yaw on associated observations. The tracker can
        # perturb yaw even on hits, while the detector axis benchmarks better.
        detector_headings = boxes[:, 6].copy()
        coarse_name = str(track['name'][0])
        for index in evidence_indices:
            detected = detections_by_frame.get(int(frame_ids[index]))
            if detected is None:
                continue
            names, detection_boxes = detected
            candidates = np.flatnonzero(names == coarse_name)
            if not len(candidates):
                continue
            distances = np.linalg.norm(
                detection_boxes[candidates, :2] - boxes[index, :2], axis=1)
            nearest = int(candidates[np.argmin(distances)])
            if float(np.min(distances)) <= 4.0:
                detector_headings[index] = detection_boxes[nearest, 6]
        headings = detector_headings[evidence]

        if nearly_static:
            if coarse_name == 'Vehicle':
                # A world-static object's global orientation is constant.
                # Its local/LiDAR yaw must change by the inverse ego rotation
                # so that the box continues to fit the same point cloud while
                # the ego vehicle turns. Resolve occasional pi-symmetric
                # detector headings before computing a robust constant yaw.
                reference = float(headings[np.argmax(scores[evidence])])
                aligned = []
                for heading in headings:
                    alternatives = (
                        float(heading), float(wrap_angle(heading + np.pi)))
                    aligned.append(min(
                        alternatives,
                        key=lambda candidate: abs(float(wrap_angle(
                            candidate - reference)))))
                chosen_global = math.atan2(
                    float(np.mean(np.sin(aligned))),
                    float(np.mean(np.cos(aligned))))
                boxes[:, 6] = chosen_global
            else:
                # Non-vehicle static boxes keep a stable unoriented axis.
                axis = 0.5 * math.atan2(
                    float(np.mean(np.sin(2 * headings))),
                    float(np.mean(np.cos(2 * headings))))
                alternatives = (axis, float(wrap_angle(axis + np.pi)))
                chosen = min(alternatives, key=lambda candidate: float(np.sum(
                    np.abs(wrap_angle(candidate - headings)))))
                boxes[:, 6] = chosen
        else:
            # Resolve only the detector's 180-degree ambiguity; never replace
            # its measured box axis with a center-trajectory derivative.
            resolved = [float(headings[0])]
            for heading in headings[1:]:
                alternatives = (float(heading),
                                float(wrap_angle(heading + np.pi)))
                resolved.append(min(
                    alternatives,
                    key=lambda candidate: abs(float(wrap_angle(
                        candidate - resolved[-1])))))
            resolved = np.asarray(resolved)

            # Motion selects only the forward end of the measured axis.
            if coarse_name == 'Vehicle' and np.linalg.norm(displacement) >= 2.0:
                motion_yaw = math.atan2(displacement[1], displacement[0])
                mean_yaw = math.atan2(float(np.mean(np.sin(resolved))),
                                      float(np.mean(np.cos(resolved))))
                if math.cos(float(wrap_angle(mean_yaw - motion_yaw))) < 0:
                    resolved = wrap_angle(resolved + np.pi)

            boxes[:, 6] = wrap_angle(np.interp(
                times, times[evidence], np.unwrap(resolved)))
        track['boxes_global'] = boxes
    return tracks


@contextlib.contextmanager
def working_directory(path: Path):
    previous = Path.cwd()
    os.chdir(path)
    try:
        yield
    finally:
        os.chdir(previous)


def install_detzero_paths() -> None:
    for path in (DETZERO / 'utils', DETZERO / 'tracking', DETZERO / 'refining'):
        if str(path) not in sys.path:
            sys.path.insert(0, str(path))
    # DetZero predates NumPy 1.24.
    if 'bool' not in np.__dict__:
        np.bool = np.bool_
    if 'int' not in np.__dict__:
        np.int = int
    if 'float' not in np.__dict__:
        np.float = float


def world_lidar_poses(sequence: Path, calibration: Path | None) -> np.ndarray:
    """Compose exported T_world_base poses with T_base_lidar.

    Point clouds and CenterPoint boxes are expressed in CLIDAR, whereas the
    recorder exporter deliberately stores the localization pose in BASELINK.
    Treating that pose as T_world_lidar produces a lever-arm error whenever
    the ego vehicle rotates.
    """
    base_poses = np.load(sequence / 'odometry_poses.npy')
    if calibration is None:
        candidates = (sequence / 'calibration.json',
                      sequence.parent / 'calibration.json')
        calibration = next((path for path in candidates if path.exists()), None)
    if calibration is None or not calibration.exists():
        raise FileNotFoundError(
            'calibration.json is required to transform CLIDAR boxes with '
            'BASELINK odometry; pass --calibration')
    document = json.loads(calibration.read_text())
    lidar = document.get('extrinsics', {}).get('CLIDAR')
    if not lidar or lidar.get('from') != 'CLIDAR' or lidar.get('to') != 'BASELINK':
        raise ValueError(
            f'{calibration}: expected extrinsics.CLIDAR to describe '
            'T_BASELINK_CLIDAR')
    base_from_lidar = np.asarray(lidar['matrix'], dtype=np.float64)
    if base_from_lidar.shape != (4, 4):
        raise ValueError(f'{calibration}: CLIDAR matrix must be 4x4')
    return base_poses @ base_from_lidar


def read_manifest(sequence: Path, limit: int | None) -> list[dict]:
    with (sequence / 'manifest.csv').open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if limit is not None:
        rows = rows[:limit]
    if not rows:
        raise RuntimeError('manifest has no frames')
    required = {'timestamp_ns', 'lidar_pcd', 'odometry_index'}
    missing = required - rows[0].keys()
    if missing:
        raise ValueError(
            f'manifest is missing {sorted(missing)}; rerun the exporter with '
            '--odometry-dir and --proto-dir')
    return rows


def read_binary_pcd(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    marker = b'DATA binary\n'
    offset = raw.find(marker)
    if offset < 0:
        raise ValueError(f'{path}: expected binary PCD')
    points = np.frombuffer(raw, dtype=np.float32, offset=offset + len(marker))
    if points.size % 4:
        raise ValueError(f'{path}: point payload is not float32 XYZI')
    return points.reshape(-1, 4).copy()


def atomic_pickle(value, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    with temporary.open('wb') as stream:
        pickle.dump(value, stream)
    temporary.replace(path)


def detect(sequence: Path, rows: list[dict], poses: np.ndarray,
           output: Path, args: argparse.Namespace) -> list[dict]:
    # MMDetection imports an older scikit-learn release that still references
    # NumPy's removed scalar aliases.
    install_detzero_paths()
    from mmengine.config import Config
    from mmdet3d.apis import inference_detector, init_model

    result_path = output / 'detections.pkl'
    thresholds = {
        'Vehicle': (args.vehicle_score_threshold
                    if args.vehicle_score_threshold is not None
                    else args.score_threshold),
        'Pedestrian': (args.pedestrian_score_threshold
                       if args.pedestrian_score_threshold is not None
                       else args.score_threshold),
        'Cyclist': (args.cyclist_score_threshold
                    if args.cyclist_score_threshold is not None
                    else args.score_threshold),
    }
    manifest_signature = hashlib.sha256(json.dumps([
        [row.get('lidar_pcd'), row.get('timestamp_ns'),
         row.get('lidar_sensor_timestamp_ns'),
         (sequence / row['lidar_pcd']).stat().st_size,
         (sequence / row['lidar_pcd']).stat().st_mtime_ns]
        for row in rows], separators=(',', ':')).encode()).hexdigest()
    model_signature = {
        'config_path': str(args.config.resolve()),
        'config_sha256': hashlib.sha256(args.config.read_bytes()).hexdigest(),
        'checkpoint_size': args.checkpoint.stat().st_size,
        'checkpoint_mtime_ns': args.checkpoint.stat().st_mtime_ns,
    }
    completed = []
    if result_path.exists():
        cached = pickle.loads(result_path.read_bytes())
        if (cached.get('checkpoint') == str(args.checkpoint.resolve()) and
                cached.get('label_schema_version') == 4 and
                cached.get('score_thresholds') == thresholds and
                cached.get('manifest_signature') == manifest_signature and
                cached.get('model_signature') == model_signature):
            completed = cached['frames']
    start = len(completed)
    if start >= len(rows):
        return completed[:len(rows)]

    cfg = Config.fromfile(args.config)
    cfg.test_dataloader.dataset.pipeline = [
        step for step in cfg.test_dataloader.dataset.pipeline
        if step['type'] != 'LoadPointsFromMultiSweeps'
    ]
    model = init_model(cfg, str(args.checkpoint), device='cuda:0')
    for frame_index in tqdm(range(start, len(rows)), initial=start,
                            total=len(rows), desc='CenterPoint'):
        row = rows[frame_index]
        points = read_binary_pcd(sequence / row['lidar_pcd'])
        model_points = np.column_stack(
            (points, np.zeros(len(points), dtype=np.float32)))
        result, _ = inference_detector(model, model_points)
        pred = result.pred_instances_3d
        boxes = pred.bboxes_3d.tensor.detach().cpu().numpy()
        scores = pred.scores_3d.detach().cpu().numpy()
        labels = pred.labels_3d.detach().cpu().numpy()
        keep = np.array([
            int(label) in LABEL_TO_ORIGINAL_CLASS and np.isfinite(score)
            and score >= thresholds[
                ORIGINAL_TO_DETZERO_CLASS[LABEL_TO_ORIGINAL_CLASS[int(label)]]]
            for score, label in zip(scores, labels)
        ], dtype=bool)
        boxes = boxes[keep, :7].astype(np.float32)
        # An invalid detector box must not reach DetZero's cost matrices.
        valid_geometry = (np.isfinite(boxes).all(axis=1) &
                          np.all(boxes[:, 3:6] > 0, axis=1))
        if not valid_geometry.all():
            kept_indices = np.flatnonzero(keep)[valid_geometry]
            keep[:] = False
            keep[kept_indices] = True
            boxes = boxes[valid_geometry]
        # MMDetection3D LiDAR boxes use a bottom-center z origin; DetZero uses
        # geometric centers.
        if len(boxes):
            boxes[:, 2] += boxes[:, 5] / 2
        completed.append({
            'frame_id': frame_index,
            'timestamp_ns': int(row['timestamp_ns']),
            'lidar_sensor_timestamp_ns': int(
                row.get('lidar_sensor_timestamp_ns') or row['timestamp_ns']),
            'boxes_lidar': boxes,
            'score': scores[keep].astype(np.float32),
            'name': np.array([
                ORIGINAL_TO_DETZERO_CLASS[
                    LABEL_TO_ORIGINAL_CLASS[int(label)]]
                for label in labels[keep]
            ], dtype=object),
            'original_name': np.array([
                LABEL_TO_ORIGINAL_CLASS[int(label)]
                for label in labels[keep]
            ], dtype=object),
            'pose': poses[frame_index],
        })
        atomic_pickle({
            'checkpoint': str(args.checkpoint.resolve()),
            'config': str(args.config.resolve()),
            'label_schema_version': 4,
            'score_thresholds': thresholds,
            'manifest_signature': manifest_signature,
            'model_signature': model_signature,
            'frames': completed,
        }, result_path)
    del model
    torch.cuda.empty_cache()
    return completed


def boxes_to_global(boxes: np.ndarray, pose: np.ndarray) -> np.ndarray:
    result = np.zeros((len(boxes), 7), dtype=np.float32)
    if not len(boxes):
        return result
    centers = np.column_stack((boxes[:, :3], np.ones(len(boxes))))
    result[:, :3] = (centers @ pose.T)[:, :3]
    result[:, 3:7] = boxes[:, 3:7]
    result[:, 6] += math.atan2(pose[1, 0], pose[0, 0])
    return result

def check_frame_boxes(frames):
    for frame in frames:
        frame_id = int(frame['frame_id'])
        boxes = np.asarray(frame['boxes_lidar'])
        pose = np.asarray(frame['pose'])
        lengths = [len(boxes), len(frame['name']), len(frame['original_name']),
                   len(frame['score'])]
        if len(set(lengths)) != 1 or boxes.ndim != 2 or boxes.shape[1] != 7:
            raise ValueError(f'frame {frame_id}: misaligned detection arrays {lengths}')
        if pose.shape != (4, 4) or not np.isfinite(pose).all():
            raise ValueError(f'frame {frame_id}: invalid LiDAR ego pose')
        if (not np.isfinite(boxes).all() or
                np.any(boxes[:, 3:6] <= 0) or
                not np.isfinite(frame['score']).all()):
            raise ValueError(f'frame {frame_id}: invalid detection boxes/scores; '
                             'rerun detector or inspect upstream cloud')
        if not np.isfinite(boxes_to_global(boxes, pose)).all():
            raise ValueError(f'frame {frame_id}: invalid world boxes')

def track(frames: list[dict], frame_times: np.ndarray, output: Path,
          args: argparse.Namespace) -> dict:
    if frames and 'original_name' not in frames[0]:
        raise RuntimeError(
            'detections.pkl predates six-class label preservation; rerun '
            'with --stage detect before tracking')
    install_detzero_paths()
    from detzero_utils.config_utils import cfg_from_yaml_file
    from detzero_track.models import build_model
    check_frame_boxes(frames)
    config = EasyDict()
    tools = DETZERO / 'tracking' / 'tools'
    with working_directory(tools):
        cfg_from_yaml_file(
            'cfgs/tk_model_cfgs/waymo_detzero_track.yaml', config)
    config.MODEL.TRACKING.FILTER.NAME = (
        'AB3DMOT' if args.tracking_filter == 'ab3dmot' else 'KalmanFilter')
    if args.fast_tracking:
        config.MODEL.TRACKING.TRACK_MERGE.ENABLE = False
        config.MODEL.TRACKING.REVERSE_TRACKING.ENABLE = False
    if args.tracking_profile == 'camera-2hz':
        tracking = config.MODEL.TRACKING
        tracking.FILTER.DELTA_T = 0.5
        tracking.FILTER.Q = [2, 25]
        tracking.FILTER.P = [25, 400]
        tracking.FILTER.R = 0.5
        tracking.TRACK_AGE.BIRTH_AGE = 1
        # Four seconds of coasting bridges ordinary 2 Hz occlusions while
        # still bounding false continuation after an object disappears.
        tracking.TRACK_AGE.DEATH_AGE = 8
        tracking.DATA_ASSOCIATION.DISTANCE_METHOD = 'HybridBEV'
        # HybridBEV proximity scale is 8 m. These minima correspond to
        # approximately 11.1 m Vehicle, 4.8 m Pedestrian, 6.4 m Cyclist gates.
        tracking.DATA_ASSOCIATION.STAGE.FIRST_STAGE.DIST_THRESHOLD = [
            0.25, 0.55, 0.45]
        tracking.DATA_ASSOCIATION.STAGE.SECOND_STAGE.DIST_THRESHOLD = [
            0.20, 0.50, 0.40]
        tracking.REVERSE_TRACKING.MAX_MISSES = 8
        config.MODEL.POST_PROCESS.CONFIG_LIST[0].LEAST_AGE = 2
        for processor in config.MODEL.POST_PROCESS.CONFIG_LIST:
            if processor.NAME == 'velocity_optimize':
                processor.DELTA_T = 0.5  # fallback only for legacy tracks
        print('Tracking profile: camera-2hz (0.5 s, HybridBEV gates)')
    print(f'Tracking filter: {config.MODEL.TRACKING.FILTER.NAME}')
    if args.reverse_max_misses is not None:
        if args.reverse_max_misses < 1:
            raise ValueError('--reverse-max-misses must be at least 1')
        config.MODEL.TRACKING.REVERSE_TRACKING.MAX_MISSES = (
            args.reverse_max_misses)

    threshold = args.tracking_score_threshold
    if threshold is not None:
        filtered_frames = []
        for frame in frames:
            keep = np.asarray(frame['score']) >= threshold
            filtered_frames.append({
                **frame,
                'boxes_lidar': frame['boxes_lidar'][keep],
                'score': frame['score'][keep],
                'name': frame['name'][keep],
                'original_name': frame['original_name'][keep],
            })
        frames = filtered_frames
        print(f'Tracking score threshold: {threshold:g}')
    print(
        'Reverse tracking: '
        f"{config.MODEL.TRACKING.REVERSE_TRACKING.ENABLE}; "
        'track merge: '
        f"{config.MODEL.TRACKING.TRACK_MERGE.ENABLE}")
    if args.reverse_max_misses is not None:
        print(f'Reverse maximum misses: {args.reverse_max_misses}')
    counts = np.array(
    [len(frame["score"]) for frame in frames]
    )

    print(f"Total frames: {len(frames)}")
    print(f"Total detections: {counts.sum()}")
    print(f"Mean detections/frame: {counts.mean():.1f}")
    print(f"Max detections/frame: {counts.max()}")

    print("Detections by class:")
    for name in CLASS_NAMES:
        count = sum(
            np.count_nonzero(frame["name"] == name)
            for frame in frames
        )
        print(f"  {name}: {count}")
    sequence = {}
    for frame in frames:
        sequence[str(frame['frame_id'])] = {
            **frame,
            'boxes_global': boxes_to_global(
                frame['boxes_lidar'], frame['pose']),
            'lidar_sensor_timestamp_ns': int(frame['lidar_sensor_timestamp_ns']),
            'num_points': np.zeros(len(frame['score']), dtype=np.int32),
        }
    # The Kalman filter has a fixed 0.5 s step. Never associate through a
    # reversed sensor timestamp or a gap longer than the four-second coast.
    frame_deltas = np.diff(frame_times)
    split_points = list(np.flatnonzero(
        (frame_deltas <= 0) |
        (frame_deltas > TRACKING_GAP_RESET_SECONDS)) + 1)
    boundaries = [0, *split_points, len(frames)]
    tracks = {}
    next_track_id = 0
    for start, end in zip(boundaries[:-1], boundaries[1:]):
        tracker = build_model(config.MODEL)
        tracker.module_list[0].init_track_id = next_track_id
        segment = {
            str(index): sequence[str(index)]
            for index in range(start, end)
        }
        segment_tracks = tracker.forward(segment)
        tracks.update(segment_tracks)
        if segment_tracks:
            next_track_id = max(map(int, segment_tracks)) + 1
    if split_points:
        print('Tracking resets after invalid/reversed or long acquisition gaps: '
              + ', '.join(f'{index - 1}->{index}' for index in split_points))
    # Preserve DetZero's native output before the custom class/size/heading
    # stabilization.  Keeping this boundary makes tracker regressions
    # distinguishable from regressions introduced by our stabilization pass.
    atomic_pickle(tracks, output / 'tracks_native.pkl')
    stabilize_tracks(tracks, frame_times=frame_times,
                     detection_frames=frames, delta_t=0.5)
    atomic_pickle(tracks, output / 'tracks.pkl')
    (output / 'tracking_run_config.json').write_text(json.dumps({
        'tracking_filter': config.MODEL.TRACKING.FILTER.NAME,
        'tracking_profile': args.tracking_profile,
        'vertical_prediction_enabled': False,
        'height_prediction_enabled': False,
        'variable_delta_t_from_lidar_timestamps': True,
        'process_noise': 'elapsed-time-scaled CV with acceleration cross-covariance',
    }, indent=2) + '\n')
    return tracks


def points_in_box(points: np.ndarray, box: np.ndarray,
                  scale: float = 1.1) -> np.ndarray:
    relative = points[:, :3] - box[:3]
    cosine, sine = math.cos(box[6]), math.sin(box[6])
    local_x = cosine * relative[:, 0] + sine * relative[:, 1]
    local_y = -sine * relative[:, 0] + cosine * relative[:, 1]
    mask = ((np.abs(local_x) <= box[3] * scale / 2) &
            (np.abs(local_y) <= box[4] * scale / 2) &
            (np.abs(relative[:, 2]) <= box[5] * scale / 2))
    return points[mask]



def prepare_refining_data(
    sequence: Path,
    rows: list[dict],
    tracks: dict,
    output: Path,
) -> tuple[Path, dict[int, int]]:
    MAX_TRACK_LENGTH = 200

    TRACK_FIELDS = (
        'boxes_global',
        'score',
        'sample_idx',
        'hit',
        'pose',
        'matched',
        'pts',
        'gt_boxes_global',
    )

    data_root = output / 'waymo'
    image_sets = data_root / 'ImageSets'
    image_sets.mkdir(parents=True, exist_ok=True)
    (image_sets / 'test.txt').write_text(SEQUENCE_NAME + '\n')

    by_class = {name: {} for name in CLASS_NAMES}
    cloud_cache = {}
    segment_parent = {}

    next_segment_id = (
        max((int(k) for k in tracks), default=-1) + 1
    )

    def validate_track(obj: dict, track_id: int) -> None:
        lengths = {
            field: len(obj[field])
            for field in TRACK_FIELDS
        }

        if len(set(lengths.values())) != 1:
            raise ValueError(
                f"Track {track_id}: inconsistent lengths {lengths}"
            )

    def copy_segment(
        obj: dict,
        segment_id: int,
        start: int,
        end: int,
    ) -> dict:
        segment = dict(obj)
        segment['obj_id'] = segment_id

        for field in TRACK_FIELDS:
            value = obj[field][start:end]

            if isinstance(value, np.ndarray):
                segment[field] = value.copy()
            elif isinstance(value, list):
                segment[field] = [
                    item.copy() if isinstance(item, np.ndarray)
                    else item
                    for item in value
                ]
            else:
                segment[field] = value

        return segment

    original_counts = {}
    segmented_counts = {}

    for obj_id, track_info in tqdm(
        tracks.items(),
        desc='Cropping tracks',
    ):
        obj_id = int(obj_id)
        name = str(track_info['name'][0])

        if name not in by_class:
            continue
        if name not in REFINED_CLASSES:
            continue

        boxes = np.asarray(
            track_info['boxes_global'][:, :7]
        )

        obj = {
            'sequence_name': SEQUENCE_NAME,
            'obj_id': obj_id,
            'name': name,
            'boxes_global': boxes,
            'score': track_info['score'],
            'sample_idx': track_info['sample_idx'].astype(str),
            'hit': track_info['hit'],
            'pose': track_info['pose'],
            'state': track_info['state'],
            'matched': np.ones(
                len(track_info['sample_idx']),
                dtype=bool,
            ),
            'matched_tracklet': False,
            'pts': [],
            'gt_boxes_global': np.zeros_like(boxes),
        }

        total_boxes = len(boxes)

        # Validate source fields before cropping.
        source_fields = (
            'score',
            'sample_idx',
            'hit',
            'pose',
            'matched',
            'gt_boxes_global',
        )

        for field in source_fields:
            if len(obj[field]) != total_boxes:
                raise ValueError(
                    f"Track {obj_id}: {field} has "
                    f"{len(obj[field])} entries, "
                    f"expected {total_boxes}"
                )

        # -------------------------------------------------
        # 1. Crop points for ALL observations in the track.
        # -------------------------------------------------
        for frame_id, pose, box in zip(
            obj['sample_idx'],
            obj['pose'],
            obj['boxes_global'],
        ):
            frame_index = int(frame_id)

            if not 0 <= frame_index < len(rows):
                raise IndexError(
                    f"Track {obj_id}: invalid frame "
                    f"index {frame_index}"
                )

            if frame_index not in cloud_cache:
                pcd_path = (
                    sequence / rows[frame_index]['lidar_pcd']
                )

                local = read_binary_pcd(pcd_path)

                xyz1 = np.column_stack((
                    local[:, :3],
                    np.ones(len(local)),
                ))

                xyz_global = (xyz1 @ pose.T)[:, :3]

                cloud_cache[frame_index] = np.column_stack((
                    xyz_global,
                    np.tanh(local[:, 3]),
                ))

            cropped_points = points_in_box(
                cloud_cache[frame_index],
                box,
            )

            obj['pts'].append(cropped_points)

        # -------------------------------------------------
        # 2. Validate completed track before splitting.
        # -------------------------------------------------
        validate_track(obj, obj_id)

        if not np.isfinite(boxes).all():
            raise ValueError(
                f"Track {obj_id} contains invalid boxes"
            )

        original_counts[obj_id] = total_boxes
        segmented_counts[obj_id] = 0

        # -------------------------------------------------
        # 3. Split ONCE, after all points are available.
        # -------------------------------------------------
        num_segments = (
            total_boxes + MAX_TRACK_LENGTH - 1
        ) // MAX_TRACK_LENGTH

        for segment_index in range(num_segments):
            start = segment_index * MAX_TRACK_LENGTH
            end = min(
                start + MAX_TRACK_LENGTH,
                total_boxes,
            )

            if segment_index == 0:
                segment_id = obj_id
            else:
                segment_id = next_segment_id
                next_segment_id += 1

            segment = copy_segment(
                obj,
                segment_id,
                start,
                end,
            )

            validate_track(segment, segment_id)

            segment_length = len(
                segment['boxes_global']
            )

            if not 0 < segment_length <= MAX_TRACK_LENGTH:
                raise ValueError(
                    f"Segment {segment_id}: invalid length "
                    f"{segment_length}"
                )

            if segment_id in by_class[name]:
                raise ValueError(
                    f"Duplicate segment ID {segment_id}"
                )

            by_class[name][segment_id] = segment
            segment_parent[segment_id] = obj_id

            segmented_counts[obj_id] += segment_length

        if num_segments > 1:
            print(
                f"[REFINE] Track {obj_id}: "
                f"{total_boxes} boxes -> "
                f"{num_segments} segments"
            )

        # -------------------------------------------------
        # 4. Verify no observations were lost or duplicated.
        # -------------------------------------------------
        if segmented_counts[obj_id] != original_counts[obj_id]:
            raise ValueError(
                f"Track {obj_id}: "
                f"original={original_counts[obj_id]}, "
                f"segmented={segmented_counts[obj_id]}"
            )

    # -------------------------------------------------
    # 5. Final statistics and validation.
    # -------------------------------------------------
    total_original = sum(original_counts.values())
    total_segmented = sum(segmented_counts.values())

    if total_original != total_segmented:
        raise ValueError(
            f"Observation count mismatch: "
            f"original={total_original}, "
            f"segmented={total_segmented}"
        )

    print(
        f"[REFINE] Original observations: {total_original}"
    )
    print(
        f"[REFINE] Segmented observations: {total_segmented}"
    )

    for name, class_tracks in by_class.items():
        lengths = [
            len(obj['boxes_global'])
            for obj in class_tracks.values()
        ]

        if lengths:
            print(
                f"[{name}] "
                f"Segments: {len(lengths)}, "
                f"Longest: {max(lengths)}, "
                f"Mean length: {np.mean(lengths):.1f}"
            )

            if max(lengths) > MAX_TRACK_LENGTH:
                raise ValueError(
                    f"{name}: segment exceeds QUERY_NUM"
                )

        # Save only after validation succeeds.
        target = data_root / 'refining' / name
        target.mkdir(parents=True, exist_ok=True)

        atomic_pickle(
            class_tracks,
            target / f'{SEQUENCE_NAME}.pkl',
        )

    return data_root, segment_parent


def run_refiner(data_root: Path, class_name: str, kind: str,
                checkpoint: Path) -> dict:
    install_detzero_paths()
    from detzero_refine.datasets import build_dataloader
    from detzero_refine.models import build_network, load_data_to_gpu
    from detzero_utils.common_utils import create_logger
    from detzero_utils.config_utils import cfg_from_yaml_file
    from detzero_utils.model_utils import load_params_from_file

    config = EasyDict(LOCAL_RANK=0)
    config_name = f'{class_name.lower()}_{kind}_model.yaml'
    tools = DETZERO / 'refining' / 'tools'
    with working_directory(tools):
        cfg_from_yaml_file(f'cfgs/ref_model_cfgs/{config_name}', config)
    config.DATA_CONFIG.DATA_PATH = str(data_root)
    config.DATA_CONFIG.DATA_SPLIT.test = 'test'
    config.DATA_CONFIG.save_to_file = True
    logger = create_logger()
    dataset, loader, _ = build_dataloader(
        dataset_cfg=config.DATA_CONFIG, class_names=config.CLASS_NAMES,
        batch_size=1, dist=False, workers=0, logger=logger, training=False)
    print(f"[{class_name} {kind}] Validating {len(dataset)} tracks")

    bad_samples = []

    for idx, info in enumerate(dataset.track_infos):
        fields = (
            "boxes_global",
            "pts",
            "sample_idx",
            "pose",
            "score",
            "hit",
            "matched",
            "gt_boxes_global",
        )

        lengths = {
            key: len(info[key])
            for key in fields
        }

        if len(set(lengths.values())) != 1:
            bad_samples.append((idx, info["obj_id"], lengths))
            print(
                f"[BAD SAMPLE] dataset_index={idx}, "
                f"obj_id={info['obj_id']}, "
                f"lengths={lengths}"
            )

    if bad_samples:
        raise ValueError(
            f"Found {len(bad_samples)} inconsistent refinement samples"
        )

    print(f"[{class_name} {kind}] All field lengths are consistent")

    if not len(dataset):
        return {}
    model = build_network(config.MODEL, dataset).cuda().eval()
    load_params_from_file(model, filename=str(checkpoint), logger=logger,
                          to_cpu=False, fix_pretrained_weights=False)
    saved = {}
    for batch in tqdm(loader, desc=f'{class_name} {kind.upper()}'):
        load_data_to_gpu(batch)
        with torch.no_grad():
            prediction, _, _ = model(batch)
        dataset.generate_prediction_dicts(batch, prediction, saved)
    del model
    torch.cuda.empty_cache()
    return saved


def refine(sequence: Path, rows: list[dict], tracks: dict, output: Path,
           checkpoints: Path,
           enable_position_refiner: bool = False) -> list[dict]:
    if not enable_position_refiner:
        result = tracks_to_results(rows, tracks, hits_only=True)
        atomic_pickle({
            'position_refiner_enabled': False,
            'reason': ('disabled by default: reviewed custom-bag benchmark '
                       'favors stabilized tracker positions'),
        }, output / 'refiner_outputs.pkl')
        atomic_pickle(result, output / 'results.pkl')
        (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
        write_baseline_report(result, tracks, output)
        print('[Refinement] Position refiner disabled; exporting stabilized '
              'tracking results')
        return result
    data_root, segment_parent = prepare_refining_data(sequence, rows, tracks, output)
    original_names = {}
    source_geometry = {}
    for track_id, info in tracks.items():
        fine_names = info.get('original_name', info['name'])
        for index, (frame_id, fine_name) in enumerate(zip(
                info['sample_idx'], fine_names)):
            original_names[(int(track_id), int(frame_id))] = str(fine_name)
            pose = np.asarray(info['pose'][index])
            pose_yaw = math.atan2(pose[1, 0], pose[0, 0])
            global_box = np.asarray(info['boxes_global'][index])
            source_geometry[(int(track_id), int(frame_id))] = {
                'dimensions': global_box[3:6].copy(),
                'local_yaw': float(wrap_angle(global_box[6] - pose_yaw)),
            }
    combined = defaultdict(list)
    raw = {}
    for class_name in CLASS_NAMES:
        if class_name not in REFINED_CLASSES:
            raw[class_name] = {'bypassed': True}
            for track_id, info in tracks.items():
                if str(info['name'][0]) != class_name:
                    continue
                fine_names = info.get('original_name', info['name'])
                for index, frame_id in enumerate(info['sample_idx']):
                    pose = np.asarray(info['pose'][index])
                    inverse = np.linalg.inv(pose)
                    global_box = np.asarray(info['boxes_global'][index])
                    center = inverse @ np.r_[global_box[:3], 1.0]
                    local_yaw = float(wrap_angle(
                        global_box[6] - math.atan2(
                            pose[1, 0], pose[0, 0])))
                    combined[int(frame_id)].append({
                        'track_id': int(track_id),
                        'name': str(fine_names[index]),
                        'score': float(info['score'][index]),
                        'box': np.r_[center[:3], global_box[3:6],
                                     local_yaw].astype(float).tolist(),
                    })
            print(f'[{class_name}] Refiner bypassed; using stabilized '
                  'CenterPoint track geometry')
            continue
        # The GRM checkpoints are trained on Waymo and produced severe size
        # inflation on this sensor. Position refinement remains useful, but
        # geometry comes from the stabilized CenterPoint tracks.
        geometry = {}
        print(f'[{class_name}] Geometry refiner bypassed; preserving '
              'CenterPoint track dimensions')
        position = run_refiner(
            data_root, class_name, 'prm',
            checkpoints / f'{class_name.lower()}_prm_model.pth')
        raw[class_name] = {'geometry': geometry, 'position': position}
        
        geo_tracks = geometry.get(SEQUENCE_NAME, {})
        pos_tracks = position.get(SEQUENCE_NAME, {})

        # Determine which refinement segment IDs actually belong
        # to the current class.
        class_pickle = (
            data_root
            / "refining"
            / class_name
            / f"{SEQUENCE_NAME}.pkl"
        )

        with class_pickle.open("rb") as f:
            class_segments = pickle.load(f)

        allowed_ids = {str(k) for k in class_segments}

        # Normalize key types and discard predictions for other classes.
        geo_by_id = {
            str(k): v
            for k, v in geo_tracks.items()
            if str(k) in allowed_ids
        }

        pos_by_id = {
            str(k): v
            for k, v in pos_tracks.items()
            if str(k) in allowed_ids
        }

        print(
            f"[{class_name}] Expected segments: "
            f"{len(allowed_ids)}"
        )

        print(
            f"[{class_name}] GRM results: "
            f"{len(geo_by_id)}"
        )

        print(
            f"[{class_name}] PRM results: "
            f"{len(pos_by_id)}"
        )

        print(
            f"[{class_name}] Ignored other-class PRM results: "
            f"{len(pos_tracks) - len(pos_by_id)}"
        )

        missing_geo = allowed_ids - set(geo_by_id)
        missing_pos = allowed_ids - set(pos_by_id)

        if missing_pos:
            raise ValueError(
                f"{class_name}: incomplete refinement outputs. "
                f"Missing GRM IDs: {sorted(missing_geo)[:10]}, "
                f"Missing PRM IDs: {sorted(missing_pos)[:10]}"
            )

        # Use these filtered dictionaries for result combination.
        geo_tracks = geo_by_id
        pos_tracks = pos_by_id

        geo_by_id = {str(k): v for k, v in geo_tracks.items()}
        pos_by_id = {str(k): v for k, v in pos_tracks.items()}

        missing_geo = set(pos_by_id) - set(geo_by_id)
        missing_pos = set(geo_by_id) - set(pos_by_id)

        print(f"[{class_name}] GRM tracks: {len(geo_by_id)}")
        print(f"[{class_name}] PRM tracks: {len(pos_by_id)}")
        print(f"[{class_name}] Missing GRM: {len(missing_geo)}")
        print(f"[{class_name}] Missing PRM: {len(missing_pos)}")
        print(f"[{class_name}] Example missing GRM IDs: {sorted(missing_geo)[:20]}")
        missing_grm = 0
        missing_frames = 0

        for obj_id, pos in pos_tracks.items():
            geo = geo_by_id.get(str(obj_id))

            if geo is None:
                missing_grm += 1
                if geometry:
                    print(
                        f"[REFINE WARNING] Track {obj_id}: "
                        "GRM result missing, keeping PRM dimensions"
                    )

            # Associate GRM boxes with their original frame IDs.
            geo_boxes_by_frame = {}

            if geo is not None:
                for frame_id, geo_box in zip(
                    geo['frame_id'],
                    geo['boxes_lidar'],
                ):
                    frame_id = int(frame_id)

                    if frame_id in geo_boxes_by_frame:
                        raise ValueError(
                            f"Duplicate GRM frame {frame_id} "
                            f"for track {obj_id}"
                        )

                    geo_boxes_by_frame[frame_id] = np.asarray(
                        geo_box
                    ).reshape(-1, 7)[0]

            for index, frame_id in enumerate(pos['frame_id']):
                frame_id = int(frame_id)
                parent_id = segment_parent[int(obj_id)]

                box = np.asarray(
                    pos['boxes_lidar'][index]
                ).reshape(-1, 7)[0].copy()

                geo_box = geo_boxes_by_frame.get(frame_id)

                if geo_box is not None and class_name == 'Vehicle':
                    dimensions = geo_box[3:6]

                    if (
                        np.isfinite(dimensions).all()
                        and np.all(dimensions > 0)
                    ):
                        box[3:6] = dimensions
                else:
                    if geo_box is None and geometry:
                        missing_frames += 1

                source = source_geometry.get((parent_id, frame_id))
                if source is not None:
                    # Keep CenterPoint's track-median geometry. The Waymo
                    # geometry refiner is strongly out of domain here and can
                    # inflate cars/cyclists/pedestrians into 10+ metre boxes.
                    box[3:6] = source['dimensions']
                    box[6] = source['local_yaw']

                if not np.isfinite(box).all():
                    print(
                        f"[REFINE WARNING] Invalid result: "
                        f"track={obj_id}, frame={frame_id}"
                    )
                    continue

                combined[frame_id].append({
                    'track_id': parent_id,
                    'name': original_names.get(
                        (parent_id, frame_id), class_name),
                    'score': float(pos['score'][index]),
                    'box': box.astype(float).tolist(),
                })

        print(
            f"[{class_name}] Missing GRM tracks: {missing_grm}, "
            f"observations without GRM: {missing_frames}"
        )

    atomic_pickle(raw, output / 'refiner_outputs.pkl')
    result = [{
        'frame_id': index,
        'timestamp_ns': int(rows[index]['timestamp_ns']),
        'lidar_sensor_timestamp_ns': int(
            rows[index].get('lidar_sensor_timestamp_ns') or
            rows[index]['timestamp_ns']),
        'objects': combined[index],
    } for index in range(len(rows))]
    atomic_pickle(result, output / 'results.pkl')
    (output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    write_baseline_report(result, tracks, output)
    return result


def tracks_to_results(rows: list[dict], tracks: dict,
                      hits_only: bool = False,
                      max_coast_frames: int | None = None) -> list[dict]:
    combined = defaultdict(list)
    for obj_id, info in tracks.items():
        fine_names = info.get('original_name', info['name'])
        frame_ids = np.asarray(info['sample_idx'], dtype=int)
        hit_mask = np.asarray(info['hit']) > 0
        if hits_only:
            allowed = hit_mask
        elif max_coast_frames is not None and np.any(hit_mask):
            hit_frames = frame_ids[hit_mask]
            allowed = np.array([
                np.min(np.abs(hit_frames - frame_id)) <= max_coast_frames
                for frame_id in frame_ids
            ], dtype=bool)
        else:
            allowed = np.ones(len(frame_ids), dtype=bool)
        for index, frame_id in enumerate(info['sample_idx']):
            if not allowed[index]:
                continue
            pose = info['pose'][index]
            inverse = np.linalg.inv(pose)
            box_global = info['boxes_global'][index]
            center = inverse @ np.r_[box_global[:3], 1.0]
            yaw = box_global[6] + math.atan2(inverse[1, 0], inverse[0, 0])
            box = np.r_[center[:3], box_global[3:6], yaw]
            combined[int(frame_id)].append({
                'track_id': int(obj_id), 'name': str(fine_names[index]),
                'score': float(info['score'][index]),
                'box': box.astype(float).tolist(),
            })
    return [{
        'frame_id': i,
        'timestamp_ns': int(row['timestamp_ns']),
        'lidar_sensor_timestamp_ns': int(
            row.get('lidar_sensor_timestamp_ns') or row['timestamp_ns']),
        'objects': combined[i],
    } for i, row in enumerate(rows)]


def detections_to_results(rows: list[dict], frames: list[dict]) -> list[dict]:
    """Export thresholded CenterPoint boxes before temporal tracking."""
    result = []
    next_detection_id = 0
    for row, frame in zip(rows, frames):
        fine_names = frame.get('original_name', frame['name'])
        objects = []
        for box, name, score in zip(
                frame['boxes_lidar'], fine_names, frame['score']):
            objects.append({
                'track_id': next_detection_id,
                'name': str(name),
                'score': float(score),
                'box': np.asarray(box, dtype=float).tolist(),
            })
            next_detection_id += 1
        result.append({
            'frame_id': int(frame['frame_id']),
            'timestamp_ns': int(row['timestamp_ns']),
            'lidar_sensor_timestamp_ns': int(
                row.get('lidar_sensor_timestamp_ns') or row['timestamp_ns']),
            'objects': objects,
        })
    return result


def write_baseline_report(results: list[dict], tracks: dict,
                          output: Path) -> dict:
    objects = [obj for frame in results for obj in frame['objects']]
    by_track = defaultdict(list)
    for frame in results:
        for obj in frame['objects']:
            by_track[int(obj['track_id'])].append(
                (int(frame['frame_id']), obj))
    distances = [float(np.linalg.norm(np.asarray(obj['box'][:2], float)))
                 for obj in objects]
    heading_flips = sum(int(np.sum(np.abs(wrap_angle(np.diff(
        np.asarray(track['boxes_global'])[:, 6]))) > np.pi / 2))
        for track in tracks.values())
    static_vehicle_tracks = 0
    static_heading_updates = 0
    moving_heading_flips = 0
    frame_times = np.array([
        frame.get('lidar_sensor_timestamp_ns', frame['timestamp_ns'])
        for frame in results
    ], dtype=np.float64)
    frame_times = (frame_times - frame_times[0]) * 1e-9
    for track in tracks.values():
        if str(track['name'][0]) != 'Vehicle':
            continue
        boxes = np.asarray(track['boxes_global'])
        hit_indices = np.flatnonzero(np.asarray(track['hit']) > 0)
        first, last = hit_indices[0], hit_indices[-1]
        frame_ids = np.asarray(track['sample_idx'], dtype=int)
        times = frame_times[frame_ids]
        elapsed = max(times[last] - times[first], 0.5)
        window = min(
            7, len(boxes) if len(boxes) % 2 else len(boxes) - 1)
        if window >= 3:
            radius = window // 2
            kernel = np.ones(window, dtype=float) / window
            smooth_centers = np.column_stack([
                np.convolve(np.pad(boxes[:, dim], (radius, radius),
                                   mode='edge'), kernel, mode='valid')
                for dim in range(2)
            ])
        else:
            smooth_centers = boxes[:, :2]
        global_path_speed = float(np.linalg.norm(
            np.diff(smooth_centers[first:last + 1], axis=0), axis=1
        ).sum() / elapsed)
        if global_path_speed < STATIC_HEADING_SPEED_MPS:
            static_vehicle_tracks += 1
            static_heading_updates += int(np.count_nonzero(
                np.abs(wrap_angle(np.diff(boxes[:, 6]))) > 1e-6))
        else:
            pose_yaws = np.array([
                math.atan2(pose[1, 0], pose[0, 0])
                for pose in np.asarray(track['pose'])
            ])
            local_yaws = wrap_angle(boxes[:, 6] - pose_yaws)
            moving_heading_flips += int(np.count_nonzero(
                np.abs(wrap_angle(np.diff(local_yaws))) > np.pi / 2))
    report = {
        'frames': len(results),
        'annotations': len(objects),
        'tracks': len(by_track),
        'class_annotations': dict(Counter(obj['name'] for obj in objects)),
        'class_tracks': dict(Counter(
            items[0][1]['name'] for items in by_track.values())),
        'mixed_class_tracks': sum(len({obj['name'] for _, obj in items}) > 1
                                  for items in by_track.values()),
        'duplicate_track_frames': sum(
            len(items) != len({frame_id for frame_id, _ in items})
            for items in by_track.values()),
        'global_heading_jumps_over_90_deg': heading_flips,
        'static_or_nearly_static_vehicle_tracks': static_vehicle_tracks,
        'static_vehicle_heading_updates': static_heading_updates,
        'static_heading_frame': 'global',
        'static_heading_speed_threshold_mps': STATIC_HEADING_SPEED_MPS,
        'moving_vehicle_local_heading_jumps_over_90_deg': (
            moving_heading_flips),
        'tracking_gap_reset_seconds': TRACKING_GAP_RESET_SECONDS,
        'maximum_distance_m': max(distances, default=0.0),
        'annotations_beyond_60m': sum(value > 60 for value in distances),
        'camera_fusion_applied': False,
        'distance_filter_applied': False,
    }
    (output / 'baseline_report.json').write_text(
        json.dumps(report, indent=2) + '\n')
    return report


def box_corners(box: np.ndarray) -> np.ndarray:
    dx, dy = box[3] / 2, box[4] / 2
    corners = np.array([[dx, dy], [dx, -dy], [-dx, -dy], [-dx, dy]])
    cosine, sine = math.cos(box[6]), math.sin(box[6])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return corners @ rotation.T + box[:2]


def render(sequence: Path, rows: list[dict], results: list[dict],
           output: Path, args: argparse.Namespace) -> None:
    image_dir = output / 'visualizations'
    image_dir.mkdir(parents=True, exist_ok=True)
    size, radius = args.render_size, args.render_range
    scale = size / (2 * radius)

    def pixel(xy):
        return np.column_stack((size / 2 - xy[:, 1] * scale,
                                size / 2 - xy[:, 0] * scale)).astype(np.int32)

    video_path = output / 'detzero_results.mp4'
    writer = cv2.VideoWriter(str(video_path),
                             cv2.VideoWriter_fourcc(*'mp4v'), args.fps,
                             (size, size))
    for row, frame in tqdm(zip(rows, results), total=len(rows), desc='Rendering'):
        points = read_binary_pcd(sequence / row['lidar_pcd'])
        canvas = np.full((size, size, 3), 18, dtype=np.uint8)
        mask = ((np.abs(points[:, 0]) < radius) &
                (np.abs(points[:, 1]) < radius))
        pixels = pixel(points[mask, :2])
        canvas[pixels[:, 1], pixels[:, 0]] = (100, 100, 100)
        for obj in frame['objects']:
            box = np.asarray(obj['box'])
            color = COLORS[obj['name']]
            corners = pixel(box_corners(box))
            cv2.polylines(canvas, [corners], True, color, 2, cv2.LINE_AA)
            center = pixel(box[None, :2])[0]
            tip_xy = box[:2] + max(2.0, box[3] * .7) * np.array(
                [math.cos(box[6]), math.sin(box[6])])
            tip = pixel(tip_xy[None])[0]
            cv2.arrowedLine(canvas, center, tip, color, 3, cv2.LINE_AA,
                            tipLength=.25)
            cv2.putText(canvas, f"{obj['name']} {obj['track_id']}",
                        tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, .45,
                        color, 1, cv2.LINE_AA)
        cv2.putText(canvas, f"frame {frame['frame_id']:04d}", (15, 28),
                    cv2.FONT_HERSHEY_SIMPLEX, .7, (240, 240, 240), 2)
        image_path = image_dir / f"{row['timestamp_ns']}.jpg"
        cv2.imwrite(str(image_path), canvas, [cv2.IMWRITE_JPEG_QUALITY, 92])
        writer.write(canvas)
    writer.release()


def main() -> None:
    args = parse_args()
    sequence = args.sequence.resolve()
    output = (args.output or sequence / 'detzero_output').resolve()
    output.mkdir(parents=True, exist_ok=True)
    rows = read_manifest(sequence, args.limit)
    acquisition_ns = np.array([
        int(row.get('lidar_sensor_timestamp_ns') or row['timestamp_ns'])
        for row in rows
    ], dtype=np.int64)
    frame_times = (acquisition_ns - acquisition_ns[0]).astype(float) * 1e-9
    _, poses, _ = load_ego_poses(sequence, rows, args.calibration)

    detections_path, tracks_path = output / 'detections.pkl', output / 'tracks.pkl'
    if args.stage in ('all', 'detect'):
        frames = detect(sequence, rows, poses, output, args)
    else:
        frames = pickle.loads(detections_path.read_bytes())['frames'][:len(rows)]
    # CenterPoint boxes are in the LiDAR frame and remain valid when odometry
    # synchronization is repaired. Always refresh cached frame poses, including
    # the cache-return path inside ``--stage all``; otherwise tracking silently
    # uses poses embedded before the synchronization fix.
    if len(frames) != len(rows):
        raise ValueError('cached detections and manifest have different frame counts')
    for frame_index, (frame, row) in enumerate(zip(frames, rows)):
        if int(frame['frame_id']) != frame_index:
            raise ValueError('cached detections misaligned at frame %d' % frame_index)
        if int(frame.get('timestamp_ns', -1)) != int(row['timestamp_ns']):
            raise ValueError(
                'cached detections belong to a different LiDAR frame at %d' %
                frame_index)
        frame['pose'] = poses[frame_index]
        frame['timestamp_ns'] = int(row['timestamp_ns'])
        frame['lidar_sensor_timestamp_ns'] = int(
            row.get('lidar_sensor_timestamp_ns') or row['timestamp_ns'])
    if args.stage == 'detect':
        detection_results = detections_to_results(rows, frames)
        atomic_pickle(detection_results, output / 'detection_results.pkl')
        (output / 'detection_results.json').write_text(
            json.dumps(detection_results, indent=2) + '\n')
        print(f'Detections: {detections_path}')
        export_annotations(sequence, rows, detection_results,
                           output / 'detection_results.json', args.calibration,
                           stage='detect',
                           write_classic_labels=not args.skip_classic_labels)
        print(f'Detection results: {output / "detection_results.json"}')
        return
    if args.stage in ('all', 'track'):
        tracks = track(frames, frame_times, output, args)
        # Preserve the direct tracker output before PRM refinement overwrites
        # results.json.  This also makes stage-to-stage evaluation reproducible
        # without rerunning the tracker or reverse-engineering tracks.pkl.
        tracking_results = tracks_to_results(rows, tracks, hits_only=True)
        atomic_pickle(tracking_results, output / 'tracking_results.pkl')
        (output / 'tracking_results.json').write_text(
            json.dumps(tracking_results, indent=2) + '\n')
        native_tracks = pickle.loads((output / 'tracks_native.pkl').read_bytes())
        for filename, value in (
                ('tracking_native_results.json',
                 tracks_to_results(rows, native_tracks)),
                ('tracking_native_hits_results.json',
                 tracks_to_results(rows, native_tracks, hits_only=True)),
                ('tracking_hits_results.json',
                 tracks_to_results(rows, tracks, hits_only=True)),
                *[(f'tracking_native_coast_{coast}_results.json',
                   tracks_to_results(
                       rows, native_tracks, max_coast_frames=coast))
                  for coast in (1, 2, 4)],
                *[(f'tracking_coast_{coast}_results.json',
                   tracks_to_results(rows, tracks, max_coast_frames=coast))
                  for coast in (1, 2, 4)]):
            (output / filename).write_text(json.dumps(value, indent=2) + '\n')
    else:
        tracks = pickle.loads(tracks_path.read_bytes())
        stabilize_tracks(tracks, frame_times=frame_times,
                         detection_frames=frames, delta_t=0.5)
        atomic_pickle(tracks, tracks_path)
    if args.stage in ('all', 'track'):
        export_annotations(sequence, rows, tracking_results,
                           output / 'tracking_results.json', args.calibration,
                           tracks=tracks, stage='track',
                           write_classic_labels=(args.stage == 'track' and
                                                 not args.skip_classic_labels))
    if args.stage == 'track':
        print(f'Tracks: {tracks_path}')
        print(f'Tracking results: {output / "tracking_results.json"}')
        return
    if args.stage in ('all', 'refine'):
        results = refine(sequence, rows, tracks, output,
                         args.detzero_checkpoints.resolve(),
                         enable_position_refiner=args.enable_position_refiner)
    elif (output / 'results.pkl').exists():
        results = pickle.loads((output / 'results.pkl').read_bytes())
    else:
        results = tracks_to_results(rows, tracks, hits_only=True)
    if args.stage in ('all', 'refine'):
        export_annotations(sequence, rows, results, output / 'results.json',
                           args.calibration, tracks=tracks, stage='refine',
                           write_classic_labels=not args.skip_classic_labels)
    if args.stage == 'refine':
        print(f'Results: {output / "results.json"}')
        return
    if args.stage in ('all', 'render'):
        render(sequence, rows, results, output, args)
    print(f'DetZero output: {output}')


if __name__ == '__main__':
    main()
