#!/usr/bin/env python3
"""Visualize improved DetZero 3D labels in four native fisheye views + LiDAR BEV.

Expected box: [x,y,z,length,width,height,yaw] in CLIDAR at LiDAR acquisition.
Images remain native (KB4 projection); the OCam training export is independent.
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from collections import Counter
from pathlib import Path

import cv2
import numpy as np

from camera_motion import CameraMotion
from temporal_annotations import load_ego_poses

CAMERAS = {
    'ARC_F': 'arc_f_path', 'ARC_SL': 'arc_sl_path',
    'ARC_SR': 'arc_sr_path', 'ARC_R': 'arc_r_path',
}
# BGR; shared by every video and the legend. ConstructionVehicle is mapped
# to Van by the training dataset, but kept distinct in raw visualizations.
CLASS_COLORS = {
    'Car': (70, 220, 70),
    'Van': (0, 165, 255),
    'Truck': (70, 70, 245),
    'Bus': (190, 75, 195),
    'Pedestrian': (0, 235, 235),
    'Cyclist': (240, 225, 20),
    'ConstructionVehicle': (225, 110, 125),
}
BOX_EDGES = (
    (0, 1), (2, 3), (4, 5), (6, 7),
    (0, 2), (1, 3), (4, 6), (5, 7),
    (0, 4), (1, 5), (2, 6), (3, 7),
)


def safe_child(root: Path, relative: str) -> Path:
    path = Path(relative)
    if not relative or path.is_absolute() or '..' in path.parts:
        raise ValueError(f'unsafe dataset path: {relative!r}')
    output = (root / path).resolve()
    try:
        output.relative_to(root.resolve())
    except ValueError as error:
        raise ValueError(f'path escapes dataset: {relative!r}') from error
    if not output.is_file():
        raise FileNotFoundError(output)
    return output


def read_xyzi_pcd(path: Path) -> np.ndarray:
    """Read exact binary float32 XYZI PCD format emitted by recorder_pipeline."""
    with path.open('rb') as stream:
        header = []
        for _ in range(128):
            line = stream.readline()
            if not line:
                raise ValueError(f'{path}: truncated PCD header')
            header.append(line.decode('ascii', 'replace').strip())
            if line.strip().lower() == b'data binary':
                break
        else:
            raise ValueError(f'{path}: DATA binary not found')
        fields = next((v.split()[1:] for v in header if v.upper().startswith('FIELDS ')), None)
        sizes = next((v.split()[1:] for v in header if v.upper().startswith('SIZE ')), None)
        types = next((v.split()[1:] for v in header if v.upper().startswith('TYPE ')), None)
        counts = next((v.split()[1:] for v in header if v.upper().startswith('COUNT ')), None)
        if fields != ['x', 'y', 'z', 'intensity'] or sizes != ['4'] * 4 or types != ['F'] * 4 or (counts is not None and counts != ['1'] * 4):
            raise ValueError(f'{path}: requires float32 x y z intensity binary PCD')
        raw = stream.read()
    if len(raw) % 16:
        raise ValueError(f'{path}: invalid XYZI byte count')
    declared = next((v.split()[1] for v in header if v.upper().startswith('POINTS ')), None)
    if declared is not None and int(declared) != len(raw) // 16:
        raise ValueError(f'{path}: point count disagrees with PCD header')
    points = np.frombuffer(raw, dtype='<f4').reshape(-1, 4)
    return points[np.isfinite(points[:, :3]).all(axis=1)]


def box_corners(box: np.ndarray) -> np.ndarray:
    half = box[3:6] / 2.0
    local = np.array([[sx * half[0], sy * half[1], sz * half[2]]
                      for sz in (-1, 1) for sy in (-1, 1)
                      for sx in (-1, 1)], dtype=np.float64)
    yaw = float(box[6])
    c, s = math.cos(yaw), math.sin(yaw)
    R = np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]])
    return local @ R.T + box[:3]


def kb4_project(xyz: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    """Same KB4 projection and 110-degree FOV limit as improve annotations."""
    x, y, z = xyz.T
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)
    k1, k2, k3, k4 = model['D']
    t2 = theta * theta
    radius = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    factor = np.divide(radius, rho, out=np.ones_like(radius), where=rho > 1e-12)
    K = model['K']
    uv = np.column_stack([K[0, 0] * x * factor + K[0, 2],
                          K[1, 1] * y * factor + K[1, 2]])
    good = (np.isfinite(xyz).all(axis=1) & (np.linalg.norm(xyz, axis=1) > 1e-6)
            & (theta <= math.radians(110)) & np.isfinite(uv).all(axis=1))
    return uv, good


def get_calibration(path: Path) -> tuple[dict, dict]:
    doc = json.loads(path.read_text())
    models, transforms = {}, {}
    for name in CAMERAS:
        intr = doc['intrinsics'][name]
        K = np.asarray(intr['camera_matrix'], dtype=np.float64)
        D = np.asarray(intr['distortion_coefficients'], dtype=np.float64).reshape(-1)
        T = np.asarray(doc['lidar_to_camera'][name], dtype=np.float64)
        if K.shape != (3, 3) or D.shape != (4,) or T.shape != (4, 4) or not all(np.isfinite(a).all() for a in (K, D, T)):
            raise ValueError(f'{path}: invalid projection calibration for {name}')
        models[name] = {'K': K, 'D': D, 'width': int(intr['width']), 'height': int(intr['height'])}
        transforms[name] = T
    return models, transforms


def draw_box_camera(image: np.ndarray, box: np.ndarray, T: np.ndarray,
                    model: dict, color: tuple[int, int, int], label: str) -> bool:
    xyz = box_corners(box)
    projected_xyz = (np.column_stack((xyz, np.ones(8))) @ T.T)[:, :3]
    uv, valid = kb4_project(projected_xyz, model)
    # Two visible corners can belong to a box that is almost entirely behind
    # the camera/FOV boundary. Drawing those corners creates misleading,
    # empty-looking line fragments in fisheye images. Keep the same minimum
    # visibility rule used by annotation matching.
    if sum(valid) < 4:
        return False
    h, w = image.shape[:2]
    drawn = False
    for i, j in BOX_EDGES:
        if not (valid[i] and valid[j]):
            continue
        a, b = tuple(np.round(uv[i]).astype(int)), tuple(np.round(uv[j]).astype(int))
        # Guard against antipodal fisheye seams drawing across the image.
        if np.linalg.norm(uv[i] - uv[j]) > max(w, h) * 1.4:
            continue
        ok, c, d = cv2.clipLine((0, 0, w, h), a, b)
        if ok:
            cv2.line(image, c, d, color, 2, cv2.LINE_AA)
            drawn = True
    if drawn:
        good = uv[valid]
        x = int(np.clip(np.min(good[:, 0]), 3, w - 180)) if w > 185 else 3
        y = int(np.clip(np.min(good[:, 1]) - 8, 16, h - 6))
        cv2.putText(image, label, (x, y), cv2.FONT_HERSHEY_SIMPLEX, .55, color, 2, cv2.LINE_AA)
    return drawn


def coords(points: np.ndarray, radius: float, size: int) -> np.ndarray:
    return np.round(np.column_stack([
        size * .5 - points[:, 1] * size / (2 * radius),
        size * .5 - points[:, 0] * size / (2 * radius),
    ])).astype(np.int32)


def render_bev(points: np.ndarray, objects: list[dict], size: int, radius: float,
               frame_index: int) -> np.ndarray:
    img = np.full((size, size, 3), 23, dtype=np.uint8)
    step = 10
    for distance in range(-int(radius) + step, int(radius), step):
        pixel = int(round(size * .5 - distance * size / (2 * radius)))
        if 0 <= pixel < size:
            cv2.line(img, (pixel, 0), (pixel, size - 1), (40, 40, 40), 1)
            cv2.line(img, (0, pixel), (size - 1, pixel), (40, 40, 40), 1)
    cloud = points[:, :2]
    keep = np.isfinite(cloud).all(axis=1) & (np.abs(cloud) <= radius).all(axis=1)
    pix = coords(cloud[keep], radius, size)
    good = (pix[:, 0] >= 0) & (pix[:, 0] < size) & (pix[:, 1] >= 0) & (pix[:, 1] < size)
    pix = pix[good]
    img[pix[:, 1], pix[:, 0]] = (118, 118, 118)
    ego = tuple(coords(np.zeros((1, 2)), radius, size)[0])
    cv2.circle(img, ego, 5, (255, 255, 255), -1)
    cv2.arrowedLine(img, ego, (ego[0], max(0, ego[1] - int(size * 6 / (2 * radius)))),
                    (255, 255, 255), 2, tipLength=.35)
    for obj in objects:
        box = np.asarray(obj['box'], dtype=float)
        color = CLASS_COLORS.get(obj['name'], (230, 230, 230))
        quad = box_corners(box)[[0, 1, 3, 2], :2]
        vertices = coords(quad, radius, size)
        cv2.polylines(img, [vertices], True, color, 2, cv2.LINE_AA)
        center = tuple(coords(box[None, :2], radius, size)[0])
        tip = np.array([box[0] + max(1.0, box[3] * .6) * math.cos(box[6]),
                        box[1] + max(1.0, box[3] * .6) * math.sin(box[6])])
        arrow_tip = tuple(coords(tip[None, :], radius, size)[0])
        cv2.arrowedLine(img, center, arrow_tip, color, 2, cv2.LINE_AA, tipLength=.25)
        cv2.putText(img, f"{obj['name']} #{obj['track_id']}",
                    (int(vertices[0, 0]), int(vertices[0, 1]) - 5),
                    cv2.FONT_HERSHEY_SIMPLEX, .45, color, 1, cv2.LINE_AA)
    cv2.putText(img, f'LiDAR BEV | frame {frame_index:05d} | +/-{radius:g}m',
                (12, 28), cv2.FONT_HERSHEY_SIMPLEX, .62, (255, 255, 255), 2, cv2.LINE_AA)
    y = 54
    for cls, color in CLASS_COLORS.items():
        cv2.putText(img, cls, (12, y), cv2.FONT_HERSHEY_SIMPLEX, .48, color, 1, cv2.LINE_AA)
        y += 21
    return img


def letterbox(img: np.ndarray, width: int, height: int, title: str) -> np.ndarray:
    h, w = img.shape[:2]
    scale = min(width / w, height / h)
    resized = cv2.resize(img, (max(1, round(w * scale)), max(1, round(h * scale))), interpolation=cv2.INTER_AREA)
    out = np.full((height, width, 3), 14, dtype=np.uint8)
    x, y = (width - resized.shape[1]) // 2, (height - resized.shape[0]) // 2
    out[y:y + resized.shape[0], x:x + resized.shape[1]] = resized
    cv2.putText(out, title, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, .7, (255, 255, 255), 2, cv2.LINE_AA)
    return out


def make_writer(path: Path, fps: float, size: tuple[int, int]) -> cv2.VideoWriter:
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*'mp4v'), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f'cannot open MP4 writer: {path} (check OpenCV FFmpeg codecs)')
    return writer


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('sequence', type=Path, help='converted/dataset directory')
    parser.add_argument('--calibration', type=Path, help='recorder calibration.json')
    parser.add_argument('--annotations', type=Path, help='default: detzero_output/improved_results.json')
    parser.add_argument('--no-annotations', action='store_true',
                        help='render synchronized cameras and LiDAR BEV without boxes')
    parser.add_argument('--output', type=Path, help='default: detzero_output/improved_videos')
    parser.add_argument('--fps', type=float, default=2.)
    parser.add_argument('--stride', type=int, default=1)
    parser.add_argument('--limit', type=int)
    parser.add_argument('--camera-width', type=int, default=640)
    parser.add_argument('--camera-height', type=int, default=480)
    parser.add_argument('--bev-range', type=float, default=60.)
    parser.add_argument('--max-bev-points', type=int, default=160000)
    parser.add_argument('--camera-time-offset-ms', type=float, default=0.)
    parser.add_argument('--no-camera-time-compensation', action='store_true')
    args = parser.parse_args(argv)
    if args.fps <= 0 or args.stride < 1 or args.bev_range <= 0 or args.camera_width < 100 or args.camera_height < 100 or args.max_bev_points < 1:
        parser.error('invalid FPS / dimensions / stride / BEV settings')
    sequence = args.sequence.expanduser().resolve()
    calibration = (args.calibration or next((p for p in (sequence / 'calibration.json', sequence.parent / 'calibration.json') if p.is_file()), sequence / 'calibration.json')).resolve()
    if args.no_annotations and args.annotations is not None:
        parser.error('--no-annotations cannot be combined with --annotations')
    source = (args.annotations or sequence / 'detzero_output/improved_results.json').resolve()
    output = (args.output or sequence / 'detzero_output/improved_videos').resolve()
    with (sequence / 'manifest.csv').open(newline='') as handle:
        rows = list(csv.DictReader(handle))
    frames = ([{'frame_id': i, 'timestamp_ns': int(row['timestamp_ns']),
                'objects': []} for i, row in enumerate(rows)]
              if args.no_annotations else json.loads(source.read_text()))
    if not isinstance(frames, list) or len(frames) != len(rows):
        raise ValueError('improved annotation count must exactly match manifest')
    for i, (row, frame) in enumerate(zip(rows, frames)):
        if int(frame['frame_id']) != i or int(frame['timestamp_ns']) != int(row['timestamp_ns']):
            raise ValueError(f'frame {i}: annotations not synchronized with manifest')
    models, transforms = get_calibration(calibration)
    _, poses, _ = load_ego_poses(sequence, rows, calibration)
    motion = CameraMotion(rows, poses, enabled=not args.no_camera_time_compensation,
                          offset_ms=args.camera_time_offset_ms)
    output.mkdir(parents=True, exist_ok=True)
    cw, ch = args.camera_width, args.camera_height
    bev_size = ch * 2
    combined_size = (cw * 2 + bev_size, ch * 2)
    files = {'combined': output / 'all_cameras_plus_bev.mp4',
             'bev': output / 'lidar_bev.mp4',
             **{camera: output / f'{camera.lower()}_boxes.mp4' for camera in CAMERAS}}
    writers = {}
    seen = Counter()
    drawn = Counter()
    processed = 0
    try:
        writers['combined'] = make_writer(files['combined'], args.fps, combined_size)
        writers['bev'] = make_writer(files['bev'], args.fps, (bev_size, bev_size))
        for cam in CAMERAS:
            writers[cam] = make_writer(files[cam], args.fps, (cw, ch))
        for idx in range(0, len(rows), args.stride):
            if args.limit is not None and processed >= args.limit:
                break
            row, frame = rows[idx], frames[idx]
            points = read_xyzi_pcd(safe_child(sequence, row['lidar_pcd']))
            if len(points) > args.max_bev_points:
                points = points[np.linspace(0, len(points) - 1, args.max_bev_points, dtype=int)]
            valid_objs = []
            for obj in frame['objects']:
                box = np.asarray(obj['box'], dtype=float)
                if box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
                    raise ValueError(f'frame {idx}: invalid box for track {obj.get("track_id")}')
                seen[str(obj['name'])] += 1
                valid_objs.append(obj)
            bev = render_bev(points, valid_objs, bev_size, args.bev_range, idx)
            writers['bev'].write(bev)
            camera_tiles = {}
            for camera, column in CAMERAS.items():
                path = safe_child(sequence, row[column])
                image = cv2.imread(str(path), cv2.IMREAD_COLOR)
                if image is None:
                    raise ValueError(f'cannot decode camera frame {path}')
                expected = models[camera]
                if image.shape[1] != expected['width'] or image.shape[0] != expected['height']:
                    raise ValueError(f'{path}: image dimensions {image.shape[1]}x{image.shape[0]} differ from KB4 calibration {expected["width"]}x{expected["height"]}')
                for obj in valid_objs:
                    box = np.asarray(obj['box'], dtype=float)
                    # Improved-result boxes are at LiDAR time. Reproject using
                    # the same ego-time compensation as the annotation matcher.
                    velocity_world = obj.get('velocity_world')
                    projected = motion.box(box, idx, camera, velocity_world)
                    label = f"{obj['name']} #{obj['track_id']}"
                    if obj.get('source') == 'camera_track_hold':
                        label += ' [hold]'
                    if draw_box_camera(image, projected, transforms[camera],
                                       expected, CLASS_COLORS.get(obj['name'], (240, 240, 240)), label):
                        drawn[camera] += 1
                tile = letterbox(image, cw, ch, camera)
                camera_tiles[camera] = tile
                writers[camera].write(tile)
            left_top = np.hstack((camera_tiles['ARC_F'], camera_tiles['ARC_SL']))
            left_bottom = np.hstack((camera_tiles['ARC_R'], camera_tiles['ARC_SR']))
            combined = np.hstack((np.vstack((left_top, left_bottom)), bev))
            writers['combined'].write(combined)
            processed += 1
            if processed % 50 == 0:
                print(f'visualization frames: {processed}', flush=True)
    finally:
        for writer in writers.values():
            writer.release()
    if processed == 0:
        raise ValueError('nothing rendered')
    report = {'frames': processed, 'source': str(source), 'calibration': str(calibration),
              'camera_time_compensation': not args.no_camera_time_compensation,
              'caution': 'camera packet timestamps approximate exposure; moving LiDAR points are not deskewed',
              'per_class_boxes': dict(seen), 'camera_projections_drawn': dict(drawn),
              'camera_motion_status': motion.status_counts,
              'class_colors_bgr': {key: list(value) for key, value in CLASS_COLORS.items()},
              'videos': {key: str(path) for key, path in files.items()}}
    (output / 'visualization_report.json').write_text(json.dumps(report, indent=2) + '\n')
    print(json.dumps({'frames': processed, 'videos': report['videos']}, indent=2))


if __name__ == '__main__':
    main()
