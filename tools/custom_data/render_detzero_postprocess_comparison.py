#!/usr/bin/env python3
"""Render DetZero/postprocess comparison and four-camera projection videos.

Outputs:
  1. detzero_vs_postprocess.mp4 -- side-by-side LiDAR bird's-eye views.
  2. detzero_postprocess_cameras.mp4 -- both BEVs plus four native fisheye
     images with only the final postprocessed cuboids projected into them.

Camera projection deliberately uses the native recorder images: calibration
intrinsics are defined in their native pixel coordinates.  The separate
800x800 Fisheye3DOD images use a different canonical projection model.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import sys
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

TEMPORAL_SCRIPTS = (Path(__file__).resolve().parents[1] /
                    'detzero_temporal_pipeline/integration_scripts')
sys.path.insert(0, str(TEMPORAL_SCRIPTS))
from camera_motion import CameraMotion
from temporal_annotations import load_ego_poses
from improve_detzero_annotations import tracker_prediction_support


CAMERAS = (
    ('ARC_F', 'arc_f_path', 'FRONT'),
    ('ARC_R', 'arc_r_path', 'REAR'),
    ('ARC_SL', 'arc_sl_path', 'LEFT'),
    ('ARC_SR', 'arc_sr_path', 'RIGHT'),
)
CLASS_COLORS = {
    'Car': (0, 220, 255),
    'Van': (255, 150, 40),
    'Truck': (0, 130, 255),
    'Bus': (80, 40, 255),
    'ConstructionVehicle': (210, 80, 255),
    'Pedestrian': (60, 220, 255),
    'Cyclist': (255, 170, 30),
}
BASELINE_COLOR = (0, 190, 255)       # amber, BGR
POSTPROCESS_COLOR = (40, 230, 40)    # green
REMOVED_COLOR = (40, 40, 240)        # red
ADDED_COLOR = (255, 220, 40)         # cyan
BOX_EDGES = (
    (0, 1), (1, 3), (3, 2), (2, 0),
    (4, 5), (5, 7), (7, 6), (6, 4),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', type=Path,
                        help='recorder dataset containing manifest.csv')
    parser.add_argument('--baseline', type=Path,
                        help='default: SEQUENCE/detzero_output/results.json')
    parser.add_argument('--postprocessed', type=Path,
                        help='default: detzero_output/improved_results.json')
    parser.add_argument('--calibration', type=Path,
                        help='default: SEQUENCE/../calibration.json')
    parser.add_argument('--output-dir', type=Path,
                        help='default: detzero_output/comparison_videos')
    parser.add_argument('--fps', type=float, default=2.)
    parser.add_argument('--range', type=float, default=60., dest='bev_range')
    parser.add_argument('--start', type=int, default=0)
    parser.add_argument('--end', type=int,
                        help='exclusive frame index; default: end of sequence')
    parser.add_argument('--codec', default='mp4v',
                        help='four-character OpenCV codec (default: mp4v)')
    return parser.parse_args()


def read_rows(path: Path) -> list[dict]:
    with path.open(newline='') as stream:
        return list(csv.DictReader(stream))


def read_cloud(path: Path) -> np.ndarray:
    raw = path.read_bytes()
    marker = b'DATA binary\n'
    offset = raw.find(marker)
    if offset < 0:
        raise ValueError(f'{path}: expected binary PCD')
    points = np.frombuffer(raw, dtype=np.float32, offset=offset + len(marker))
    if points.size % 4:
        raise ValueError(f'{path}: point payload is not XYZI float32')
    return points.reshape(-1, 4)


def object_map(frame: dict) -> dict[str, dict]:
    result = {}
    for item in frame.get('objects', []):
        track_id = str(item['track_id'])
        if track_id in result:
            raise ValueError(
                f'frame {frame.get("frame_id")}: duplicate track {track_id}')
        box = np.asarray(item['box'], dtype=float)
        if box.shape != (7,) or not np.isfinite(box).all():
            raise ValueError(
                f'frame {frame.get("frame_id")}, track {track_id}: invalid box')
        result[track_id] = item
    return result


def corners_3d(box: np.ndarray) -> np.ndarray:
    half = box[3:6] / 2
    local = np.asarray([
        [sx * half[0], sy * half[1], sz * half[2]]
        for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)
    ])
    cosine, sine = math.cos(box[6]), math.sin(box[6])
    rotation = np.asarray([
        [cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]
    ])
    return local @ rotation.T + box[:3]


def corners_bev(box: np.ndarray) -> np.ndarray:
    half_l, half_w = box[3] / 2, box[4] / 2
    local = np.asarray([
        [half_l, half_w], [half_l, -half_w],
        [-half_l, -half_w], [-half_l, half_w],
    ])
    cosine, sine = math.cos(box[6]), math.sin(box[6])
    rotation = np.asarray([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + box[:2]


def calibration(path: Path) -> tuple[dict, dict]:
    document = json.loads(path.read_text())
    models, transforms = {}, {}
    for name, _, _ in CAMERAS:
        intrinsic = document['intrinsics'][name]
        models[name] = {
            'K': np.asarray(intrinsic['camera_matrix'], dtype=float),
            'D': np.asarray(intrinsic['distortion_coefficients'], dtype=float),
            'width': int(intrinsic['width']),
            'height': int(intrinsic['height']),
        }
        transforms[name] = np.asarray(
            document['lidar_to_camera'][name], dtype=float)
        if transforms[name].shape != (4, 4):
            raise ValueError(f'{path}: {name} lidar_to_camera must be 4x4')
    return models, transforms


def project(points_lidar: np.ndarray, transform: np.ndarray,
            model: dict) -> tuple[np.ndarray, np.ndarray]:
    camera = (np.column_stack((points_lidar, np.ones(len(points_lidar)))) @
              transform.T)[:, :3]
    x, y, z = camera.T
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)
    k1, k2, k3, k4 = model['D']
    theta2 = theta * theta
    radius = theta * (1 + theta2 * (
        k1 + theta2 * (k2 + theta2 * (k3 + theta2 * k4))))
    scale = np.divide(radius, rho, out=np.ones_like(radius), where=rho > 1e-12)
    matrix = model['K']
    pixels = np.column_stack((
        matrix[0, 0] * x * scale + matrix[0, 2],
        matrix[1, 1] * y * scale + matrix[1, 2],
    ))
    valid = (np.isfinite(camera).all(axis=1) &
             (np.linalg.norm(camera, axis=1) > 1e-6) &
             (theta <= np.deg2rad(110)))
    return pixels, valid


def fit_image(image: np.ndarray, width: int, height: int
              ) -> tuple[np.ndarray, float, int, int]:
    scale = min(width / image.shape[1], height / image.shape[0])
    resized_width = max(1, round(image.shape[1] * scale))
    resized_height = max(1, round(image.shape[0] * scale))
    resized = cv2.resize(image, (resized_width, resized_height),
                         interpolation=cv2.INTER_AREA)
    canvas = np.full((height, width, 3), 12, dtype=np.uint8)
    offset_x = (width - resized_width) // 2
    offset_y = (height - resized_height) // 2
    canvas[offset_y:offset_y + resized_height,
           offset_x:offset_x + resized_width] = resized
    return canvas, scale, offset_x, offset_y


def draw_camera_box(panel: np.ndarray, item: dict, transform: np.ndarray,
                    model: dict, fit: tuple[float, int, int], color: tuple,
                    thickness: int, label: bool = False) -> None:
    box = np.asarray(item['box'], dtype=float)
    corners = corners_3d(box)
    # Include center and the front-heading point for an orientation arrow.
    front = box[:3].copy()
    front[:2] += box[3] * .6 * np.asarray(
        [math.cos(box[6]), math.sin(box[6])])
    pixels, valid = project(np.vstack((corners, box[:3], front)),
                            transform, model)
    scale, offset_x, offset_y = fit
    pixels[:, 0] = pixels[:, 0] * scale + offset_x
    pixels[:, 1] = pixels[:, 1] * scale + offset_y
    pixels = np.rint(pixels).astype(np.int32)
    height, width = panel.shape[:2]
    margin = max(width, height) * .35
    inside = ((pixels[:, 0] >= -margin) & (pixels[:, 0] < width + margin) &
              (pixels[:, 1] >= -margin) & (pixels[:, 1] < height + margin))
    for first, second in BOX_EDGES:
        if valid[first] and valid[second] and inside[first] and inside[second]:
            cv2.line(panel, tuple(pixels[first]), tuple(pixels[second]),
                     color, thickness, cv2.LINE_AA)
    if valid[8] and valid[9] and inside[8] and inside[9]:
        cv2.arrowedLine(panel, tuple(pixels[8]), tuple(pixels[9]), color,
                        thickness, cv2.LINE_AA, tipLength=.25)
    if label and valid[8] and inside[8]:
        hold = ' HOLD' if item.get('source') == 'camera_track_hold' else ''
        text = f"{item['name']} T:{item['track_id']}{hold}"
        cv2.putText(panel, text, tuple(pixels[8] + np.array([4, -4])),
                    cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1, cv2.LINE_AA)


def put_label(image: np.ndarray, text: str, origin: tuple[int, int],
              color=(245, 245, 245), scale=.62, thickness=2) -> None:
    x, y = origin
    (width, height), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, scale, thickness)
    cv2.rectangle(image, (x - 5, y - height - 7),
                  (x + width + 5, y + baseline + 5), (10, 10, 10), -1)
    cv2.putText(image, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX,
                scale, color, thickness, cv2.LINE_AA)


def render_bev(points: np.ndarray, objects: dict[str, dict],
               comparison_ids: set[str], mode: str, size: tuple[int, int],
               bev_range: float, title: str, frame_id: int) -> np.ndarray:
    width, height = size
    image = np.full((height, width, 3), 17, dtype=np.uint8)
    center = np.asarray([width / 2, height / 2])
    scale = .92 * min(width, height) / (2 * bev_range)

    def pixel(xy: np.ndarray) -> np.ndarray:
        return np.column_stack((center[0] - xy[:, 1] * scale,
                                center[1] - xy[:, 0] * scale)).astype(np.int32)

    for radius in (10, 20, 40, 60):
        if radius <= bev_range:
            cv2.circle(image, tuple(center.astype(int)), round(radius * scale),
                       (45, 45, 45), 1, cv2.LINE_AA)
    cv2.line(image, (int(center[0]), 0), (int(center[0]), height),
             (38, 38, 38), 1)
    cv2.line(image, (0, int(center[1])), (width, int(center[1])),
             (38, 38, 38), 1)
    keep = ((np.abs(points[:, 0]) <= bev_range) &
            (np.abs(points[:, 1]) <= bev_range))
    cloud_pixels = pixel(points[keep, :2])
    inside = ((cloud_pixels[:, 0] >= 0) & (cloud_pixels[:, 0] < width) &
              (cloud_pixels[:, 1] >= 0) & (cloud_pixels[:, 1] < height))
    cloud_pixels = cloud_pixels[inside]
    image[cloud_pixels[:, 1], cloud_pixels[:, 0]] = (105, 105, 105)
    for track_id, item in objects.items():
        box = np.asarray(item['box'], dtype=float)
        if np.linalg.norm(box[:2]) > bev_range * 1.25:
            continue
        if mode == 'baseline':
            color = (BASELINE_COLOR if track_id in comparison_ids
                     else REMOVED_COLOR)
        else:
            color = (POSTPROCESS_COLOR if track_id in comparison_ids
                     else ADDED_COLOR)
        corners = pixel(corners_bev(box))
        cv2.polylines(image, [corners], True, color, 2, cv2.LINE_AA)
        center_pixel = pixel(box[None, :2])[0]
        tip_xy = box[:2] + max(1.5, box[3] * .6) * np.asarray(
            [math.cos(box[6]), math.sin(box[6])])
        tip = pixel(tip_xy[None])[0]
        cv2.arrowedLine(image, tuple(center_pixel), tuple(tip), color, 2,
                        cv2.LINE_AA, tipLength=.25)
        cv2.putText(image, f"{item['name']} T:{track_id}",
                    tuple(corners[0]), cv2.FONT_HERSHEY_SIMPLEX, .34,
                    color, 1, cv2.LINE_AA)
    put_label(image, f'{title} | frame {frame_id} | boxes {len(objects)}',
              (12, 27))
    return image


def camera_panel(sequence: Path, row: dict, camera: tuple,
                 post: dict[str, dict], models: dict, transforms: dict,
                 frame_index: int, motion: CameraMotion,
                 velocities: dict,
                 size=(640, 540)) -> np.ndarray:
    name, column, title = camera
    image_path = sequence / row[column]
    source = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if source is None:
        raise FileNotFoundError(f'cannot read {image_path}')
    panel, scale, offset_x, offset_y = fit_image(source, *size)
    for item in post.values():
        color = (ADDED_COLOR if item.get('source') == 'camera_track_hold'
                 else POSTPROCESS_COLOR)
        velocity = velocities.get((int(item['track_id']), frame_index))
        projected = dict(item)
        projected['box'] = motion.box(
            np.asarray(item['box'], dtype=float), frame_index, name,
            velocity).tolist()
        draw_camera_box(panel, projected, transforms[name], models[name],
                        (scale, offset_x, offset_y), color, 2,
                        label=True)
    delta_ms = float(row.get(column.replace('_path', '_delta_ns'), 0)) / 1e6
    put_label(panel, f'{title}  camera delta {delta_ms:+.1f} ms', (12, 27))
    return panel


def writer(path: Path, fps: float, size: tuple[int, int], codec: str):
    if len(codec) != 4:
        raise ValueError('--codec must contain exactly four characters')
    value = cv2.VideoWriter(
        str(path), cv2.VideoWriter_fourcc(*codec), fps, size)
    if not value.isOpened():
        raise RuntimeError(f'cannot open video writer for {path}')
    return value


def validate(rows: list[dict], baseline: list[dict], post: list[dict]) -> None:
    if not (len(rows) == len(baseline) == len(post)):
        raise ValueError(
            f'frame count mismatch: manifest={len(rows)}, '
            f'baseline={len(baseline)}, postprocessed={len(post)}')
    for index, (before, after) in enumerate(zip(baseline, post)):
        if int(before['frame_id']) != index or int(after['frame_id']) != index:
            raise ValueError(f'results are not position-aligned at frame {index}')


def main() -> None:
    args = arguments()
    if args.fps <= 0 or args.bev_range <= 0:
        raise ValueError('--fps and --range must be positive')
    sequence = args.sequence.resolve()
    output = (args.output_dir or sequence / 'detzero_output' /
              'comparison_videos').resolve()
    baseline_path = (args.baseline or sequence / 'detzero_output' /
                     'results.json').resolve()
    post_path = (args.postprocessed or sequence / 'detzero_output' /
                 'improved_results.json').resolve()
    calibration_path = (args.calibration or sequence.parent /
                        'calibration.json').resolve()
    rows = read_rows(sequence / 'manifest.csv')
    baseline = json.loads(baseline_path.read_text())
    post = json.loads(post_path.read_text())
    validate(rows, baseline, post)
    models, transforms = calibration(calibration_path)
    _, poses, _ = load_ego_poses(sequence, rows, calibration_path)
    motion = CameraMotion(rows, poses)
    _, velocities = tracker_prediction_support(
        sequence / 'detzero_output/tracks.pkl')
    end = len(rows) if args.end is None else args.end
    if not 0 <= args.start < end <= len(rows):
        raise ValueError(
            f'invalid frame interval [{args.start}, {end}) for {len(rows)} frames')

    output.mkdir(parents=True, exist_ok=True)
    bev_path = output / 'detzero_vs_postprocess.mp4'
    dashboard_path = output / 'detzero_postprocess_cameras.mp4'
    bev_video = writer(bev_path, args.fps, (1600, 800), args.codec)
    dashboard_video = writer(
        dashboard_path, args.fps, (1920, 1080), args.codec)
    totals = Counter()
    try:
        for index in tqdm(range(args.start, end), desc='Rendering videos'):
            row = rows[index]
            points = read_cloud(sequence / row['lidar_pcd'])
            before = object_map(baseline[index])
            after = object_map(post[index])
            common = set(before).intersection(after)
            totals['baseline'] += len(before)
            totals['postprocessed'] += len(after)
            totals['retained'] += len(common)
            totals['removed'] += len(set(before) - set(after))
            totals['added'] += len(set(after) - set(before))

            before_large = render_bev(
                points, before, common, 'baseline', (800, 800),
                args.bev_range, 'DETZERO', index)
            after_large = render_bev(
                points, after, common, 'post', (800, 800),
                args.bev_range, 'POSTPROCESSED', index)
            comparison = np.hstack((before_large, after_large))
            put_label(comparison,
                      'amber=retained baseline  red=removed  green=post  cyan=added',
                      (490, 785), scale=.55)
            bev_video.write(comparison)

            dashboard = np.full((1080, 1920, 3), 12, dtype=np.uint8)
            dashboard[0:540, 0:640] = render_bev(
                points, before, common, 'baseline', (640, 540),
                args.bev_range, 'DETZERO', index)
            dashboard[540:1080, 0:640] = render_bev(
                points, after, common, 'post', (640, 540),
                args.bev_range, 'POSTPROCESSED', index)
            positions = ((640, 0), (1280, 0), (640, 540), (1280, 540))
            for camera, (x, y) in zip(CAMERAS, positions):
                dashboard[y:y + 540, x:x + 640] = camera_panel(
                    sequence, row, camera, after, models, transforms,
                    index, motion, velocities)
            put_label(dashboard,
                      'postprocessed projections: green=LiDAR  cyan=camera-held',
                      (665, 1062), scale=.52)
            dashboard_video.write(dashboard)
    finally:
        bev_video.release()
        dashboard_video.release()

    report = {
        'sequence': str(sequence), 'baseline': str(baseline_path),
        'postprocessed': str(post_path), 'calibration': str(calibration_path),
        'frame_interval': [args.start, end], 'frames': end - args.start,
        'fps': args.fps, 'bev_range_m': args.bev_range,
        'camera_projection_time': 'camera packet timestamp: ego SLERP and tracker velocity',
        'counts': dict(totals),
        'videos': {'bev_comparison': str(bev_path),
                   'camera_dashboard': str(dashboard_path)},
    }
    report_path = output / 'render_report.json'
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
