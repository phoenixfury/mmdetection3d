#!/usr/bin/env python3
"""Post-process DetZero tracks with fisheye YOLO and LiDAR evidence."""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from run_detzero_sequence import read_binary_pcd
from temporal_annotations import export_annotations, load_ego_poses
from camera_motion import CameraMotion


CAMERAS = {
    "ARC_F": ("arc_f_path", "arc_f_native"),
    "ARC_SL": ("arc_sl_path", "arc_sl_native"),
    "ARC_SR": ("arc_sr_path", "arc_sr_native"),
    "ARC_R": ("arc_r_path", "arc_r_native"),
}
YOLO_CLASSES = {0: "Pedestrian", 1: "Cyclist", 2: "Car",
                3: "Cyclist", 5: "Bus", 7: "Truck"}
VEHICLES = {"Car", "Van", "Truck", "Bus", "ConstructionVehicle"}
ALLOWED_RELABELS = {
    "Car": VEHICLES, "Van": VEHICLES, "Truck": VEHICLES, "Bus": VEHICLES,
    "ConstructionVehicle": VEHICLES,
    "Pedestrian": {"Pedestrian", "Cyclist"},
    "Cyclist": {"Pedestrian", "Cyclist"},
}
MAX_DIMS = {
    "Car": np.array([6.2, 2.7, 2.6]),
    "Van": np.array([7.5, 2.8, 3.4]),
    "Truck": np.array([15.0, 3.5, 4.5]),
    "Bus": np.array([15.0, 3.5, 4.5]),
    "ConstructionVehicle": np.array([15.0, 3.5, 4.5]),
    "Pedestrian": np.array([1.2, 1.2, 2.5]),
    "Cyclist": np.array([3.0, 1.5, 2.5]),
}
CANONICAL_DIMS = {
    "Car": np.array([4.5, 1.85, 1.55]),
    "Van": np.array([5.5, 2.1, 2.4]),
    "Truck": np.array([7.0, 2.5, 3.0]),
    "Bus": np.array([11.0, 2.7, 3.4]),
    "ConstructionVehicle": np.array([7.0, 2.5, 3.0]),
    "Pedestrian": np.array([0.6, 0.6, 1.75]),
    "Cyclist": np.array([2.0, 0.8, 1.7]),
}
# User-specified road-vehicle dimensions (length, width, height).  The
# tolerances account for CenterPoint's systematically inflated height/width
# estimates on these bags; the nominal intervals are kept separate so an
# especially clear Van can still be distinguished from a small Car.
VEHICLE_SIZE_PROFILES = {
    "Van": (([3.4, 1.7, 1.7], [4.0, 1.75, 1.9], [.25, .15, .15]),),
    "Car": (([3.9, 1.7, 1.4], [4.7, 2.2, 1.5], [.25, .15, .25]),),
    "Truck": (
        ([5.3, 1.85, 1.8], [5.8, 1.9, 2.0], [.3, .55, .65]),
        ([5.3, 1.85, 1.8], [12.0, 2.55, 4.0], [.3, .65, .45]),
    ),
    "Bus": (([10.0, 2.55, 3.81], [14.0, 2.55, 3.81],
             [1.5, .75, 1.1]),),
}
MIN_VEHICLE_SIZE_LIDAR_FRAMES = 3
# Reject a complete track only for a clearly impossible size.  Exact class
# envelopes remain in use for dimension repair/reclassification, while this
# tolerance prevents centimetre-scale detector noise from deleting a bus or
# lorry for its entire lifetime.
TRACK_REJECTION_DIMENSION_FACTOR = 1.15
STATIC_GLOBAL_SPEED_MPS = 0.75
DEFAULT_CAMERA_HOLD_MIN_OVERLAP = 0.55
DEFAULT_CAMERA_HOLD_NO_LIDAR_MIN_OVERLAP = 0.95
DEFAULT_CAMERA_HOLD_MAX_LIDAR_RESIDUAL_M = 2.0
DEFAULT_CAMERA_HOLD_MAX_DT_SECONDS = 4.0
DEFAULT_CAMERA_HOLD_MAX_SPEED_MPS = 45.0
# A camera-confirmed coast must never occupy the ego vehicle itself.  Keep
# this deliberately smaller than the complete body envelope so nearby cars
# passing alongside the platform remain valid.
DEFAULT_EGO_PROTECTED_FRONT_M = 2.5
DEFAULT_EGO_PROTECTED_REAR_M = 1.5
DEFAULT_EGO_PROTECTED_HALF_WIDTH_M = 0.9
DEFAULT_MAX_EGO_FOOTPRINT_OVERLAP_M2 = 0.25
# Detection coverage alone is not evidence for a huge projected 3D box: a
# small car can sit entirely inside a phantom bus's 2D rectangle.  Preserve
# moderate partial visibility while rejecting extreme scale disagreement.
MAX_CAMERA_PROJECTION_TO_DETECTION_AREA_RATIO = 5.0
MIN_IOU_FOR_LARGE_CAMERA_AREA_MISMATCH = 0.12


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("sequence", type=Path)
    parser.add_argument("--results", type=Path)
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--calibration", type=Path)
    parser.add_argument("--weights", type=Path, required=True)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--cache", type=Path)
    parser.add_argument("--max-distance", type=float, default=60.0)
    parser.add_argument("--confirm-distance", type=float, default=60.0)
    parser.add_argument("--yolo-confidence", type=float, default=.12)
    parser.add_argument("--add-confidence", type=float, default=.35)
    parser.add_argument("--image-size", type=int, default=768)
    parser.add_argument("--batch-size", type=int, default=4)
    parser.add_argument("--device", default="auto")
    parser.add_argument("--limit", type=int)
    parser.add_argument('--skip-classic-labels', action='store_true',
                        help='write temporal sidecar without Fisheye3DOD label files')
    parser.add_argument("--yolo-only", action="store_true")
    parser.add_argument("--max-yolo-jobs", type=int,
                        help="process at most this many uncached images")
    parser.add_argument("--keep-unconfirmed", action="store_true")
    parser.add_argument(
        "--camera-confirm-classes", nargs="+",
        choices=sorted(MAX_DIMS), default=sorted(MAX_DIMS),
        help="classes that require a compatible camera match; default: all "
             "vehicle, pedestrian, and cyclist classes")
    parser.add_argument(
        "--vehicle-track-confirm-min-lidar-frames", type=int, default=3,
        help="minimum LiDAR detections for camera confirmation to validate a "
             "complete vehicle track")
    parser.add_argument(
        "--vehicle-track-confirm-min-score", type=float, default=.5,
        help="minimum peak LiDAR score for complete vehicle-track validation")
    parser.add_argument(
        "--vehicle-track-confirm-min-camera-confidence", type=float,
        default=.2, help="minimum camera confidence for vehicle-track validation")
    parser.add_argument(
        "--vehicle-track-confirm-min-overlap", type=float, default=.5,
        help="minimum projection overlap for vehicle-track validation")
    additions = parser.add_mutually_exclusive_group()
    additions.add_argument(
        "--add-camera-objects", action="store_true",
        help="add unmatched YOLO objects using LiDAR clustering (off by default)")
    additions.add_argument(
        "--no-additions", dest="add_camera_objects", action="store_false",
        help=argparse.SUPPRESS)
    parser.set_defaults(add_camera_objects=False)
    camera_hold = parser.add_mutually_exclusive_group()
    camera_hold.add_argument(
        "--camera-track-hold", dest="camera_track_hold", action="store_true",
        help="keep a LiDAR-originated track through missed LiDAR frames while "
             "its predicted projection is confirmed by a camera (default)")
    camera_hold.add_argument(
        "--no-camera-track-hold", dest="camera_track_hold",
        action="store_false",
        help="disable camera-confirmed track prediction")
    parser.set_defaults(camera_track_hold=True)
    parser.add_argument(
        "--camera-hold-min-overlap", type=float,
        default=DEFAULT_CAMERA_HOLD_MIN_OVERLAP,
        help="minimum projected-box IoU/detection coverage for a held track")
    parser.add_argument(
        "--camera-hold-no-lidar-min-overlap", type=float,
        default=DEFAULT_CAMERA_HOLD_NO_LIDAR_MIN_OVERLAP,
        help="stricter projection overlap when the camera frustum contains "
             "too few compatible LiDAR points")
    parser.add_argument(
        "--camera-hold-min-lidar-points", type=int, default=3,
        help="minimum point cluster supporting a camera-held box")
    parser.add_argument(
        "--camera-hold-max-lidar-residual", type=float,
        default=DEFAULT_CAMERA_HOLD_MAX_LIDAR_RESIDUAL_M,
        help="maximum XY distance in metres from prediction to frustum cluster")
    parser.add_argument(
        "--camera-hold-max-position-correction", type=float, default=2.0,
        help="maximum LiDAR frustum correction per frame in metres")
    parser.add_argument(
        "--camera-hold-max-dt", type=float,
        default=DEFAULT_CAMERA_HOLD_MAX_DT_SECONDS,
        help="maximum time gap in seconds for one prediction step")
    parser.add_argument(
        "--camera-hold-max-speed", type=float,
        default=DEFAULT_CAMERA_HOLD_MAX_SPEED_MPS,
        help="maximum global prediction speed in m/s")
    parser.add_argument(
        "--ego-protected-front", type=float,
        default=DEFAULT_EGO_PROTECTED_FRONT_M,
        help="protected ego footprint distance forward of the LiDAR in metres")
    parser.add_argument(
        "--ego-protected-rear", type=float,
        default=DEFAULT_EGO_PROTECTED_REAR_M,
        help="protected ego footprint distance behind the LiDAR in metres")
    parser.add_argument(
        "--ego-protected-half-width", type=float,
        default=DEFAULT_EGO_PROTECTED_HALF_WIDTH_M,
        help="protected ego footprint half-width in metres")
    parser.add_argument(
        "--camera-hold-max-ego-overlap", type=float,
        default=DEFAULT_MAX_EGO_FOOTPRINT_OVERLAP_M2,
        help="maximum allowed intersection area between a held box and the "
             "protected ego footprint in square metres")
    parser.add_argument(
        '--no-camera-time-compensation', action='store_true',
        help='disable LiDAR-to-camera-time ego and constant-velocity compensation')
    parser.add_argument(
        '--camera-time-offset-ms', type=float, default=0.,
        help='camera clock/exposure correction in ms; requires external calibration')
    parser.add_argument(
        '--camera-time-max-offset-ms', type=float, default=200.,
        help='maximum |camera packet time - LiDAR acquisition| for compensation')
    parser.add_argument(
        '--camera-time-max-pose-gap-s', type=float, default=4.,
        help='maximum bracket span permitted for interpolating ego poses')
    parser.add_argument(
        "--smooth-headings", action="store_true",
        help="apply the legacy camera-pass heading smoother (off by default)")
    return parser.parse_args()


def atomic_json(value: object, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(value, indent=2) + "\n")
    temporary.replace(path)


def wrap(angle: float) -> float:
    return (angle + math.pi) % (2 * math.pi) - math.pi


def footprint_corners(box: np.ndarray) -> np.ndarray:
    """Return the four local-frame ground-plane corners of a 3D box."""
    x, y, _, length, width, _, yaw = np.asarray(box, dtype=float)[:7]
    base = np.array([
        [length / 2, width / 2], [length / 2, -width / 2],
        [-length / 2, -width / 2], [-length / 2, width / 2],
    ])
    cosine, sine = math.cos(float(yaw)), math.sin(float(yaw))
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return (base @ rotation.T + np.array([x, y])).astype(np.float32)


def ego_footprint_overlap(box: np.ndarray, front: float, rear: float,
                          half_width: float) -> float:
    """Intersection area with a conservative protected ego body core."""
    ego = np.array([
        [-rear, -half_width], [front, -half_width],
        [front, half_width], [-rear, half_width],
    ], dtype=np.float32)
    area, _ = cv2.intersectConvexConvex(footprint_corners(box), ego)
    return float(max(area, 0.0))


def load_rows(sequence: Path, limit: int | None) -> list[dict]:
    with (sequence / "manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    return rows if limit is None else rows[:limit]


def calibration_models(path: Path) -> tuple[dict, dict]:
    raw = json.loads(path.read_text())
    models = {}
    transforms = {}
    for name in CAMERAS:
        intrinsic = raw["intrinsics"][name]
        models[name] = {
            "K": np.asarray(intrinsic["camera_matrix"], dtype=float),
            "D": np.asarray(intrinsic["distortion_coefficients"], dtype=float),
            "width": int(intrinsic["width"]),
            "height": int(intrinsic["height"]),
        }
        transforms[name] = np.asarray(raw["lidar_to_camera"][name], dtype=float)
    return models, transforms


def kb4_project(camera_xyz: np.ndarray, model: dict) -> tuple[np.ndarray, np.ndarray]:
    x, y, z = camera_xyz.T
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)
    k1, k2, k3, k4 = model["D"]
    t2 = theta * theta
    radius = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    scale = np.divide(radius, rho, out=np.ones_like(radius), where=rho > 1e-12)
    matrix = model["K"]
    pixels = np.column_stack((matrix[0, 0] * x * scale + matrix[0, 2],
                              matrix[1, 1] * y * scale + matrix[1, 2]))
    valid = (np.isfinite(camera_xyz).all(axis=1) &
             (np.linalg.norm(camera_xyz, axis=1) > 1e-6) &
             (theta <= np.deg2rad(110)))
    return pixels, valid


def corners_3d(box: np.ndarray) -> np.ndarray:
    half = box[3:6] / 2
    local = np.asarray([[sx * half[0], sy * half[1], sz * half[2]]
                        for sz in (-1, 1) for sy in (-1, 1)
                        for sx in (-1, 1)])
    cosine, sine = math.cos(box[6]), math.sin(box[6])
    rotation = np.asarray([[cosine, -sine, 0], [sine, cosine, 0], [0, 0, 1]])
    return local @ rotation.T + box[:3]


def projected_box(box: np.ndarray, transform: np.ndarray,
                  model: dict) -> np.ndarray | None:
    corners = corners_3d(box)
    camera = (np.column_stack((corners, np.ones(8))) @ transform.T)[:, :3]
    pixels, valid = kb4_project(camera, model)
    if np.count_nonzero(valid) < 4:
        return None
    pixels = pixels[valid]
    width, height = model["width"], model["height"]
    result = np.array([pixels[:, 0].min(), pixels[:, 1].min(),
                       pixels[:, 0].max(), pixels[:, 1].max()])
    result[[0, 2]] = np.clip(result[[0, 2]], 0, width - 1)
    result[[1, 3]] = np.clip(result[[1, 3]], 0, height - 1)
    if (result[2] - result[0]) * (result[3] - result[1]) < 1:
        return None
    return result


def iou(first: np.ndarray, second: np.ndarray) -> float:
    x1, y1 = np.maximum(first[:2], second[:2])
    x2, y2 = np.minimum(first[2:], second[2:])
    intersection = max(0., x2 - x1) * max(0., y2 - y1)
    union = ((first[2] - first[0]) * (first[3] - first[1]) +
             (second[2] - second[0]) * (second[3] - second[1]) - intersection)
    return intersection / max(union, 1.)


def overlap_detection(projection: np.ndarray, detection: np.ndarray) -> float:
    x1, y1 = np.maximum(projection[:2], detection[:2])
    x2, y2 = np.minimum(projection[2:], detection[2:])
    intersection = max(0., x2 - x1) * max(0., y2 - y1)
    detection_area = max(1., (detection[2] - detection[0]) *
                         (detection[3] - detection[1]))
    return intersection / detection_area


def camera_match_score(projection: np.ndarray, detection: np.ndarray) -> float:
    """2D overlap, guarded against a tiny detection validating a huge box."""
    projection_area = max(1., (projection[2] - projection[0]) *
                          (projection[3] - projection[1]))
    detection_area = max(1., (detection[2] - detection[0]) *
                         (detection[3] - detection[1]))
    box_iou = iou(projection, detection)
    if (projection_area / detection_area >
            MAX_CAMERA_PROJECTION_TO_DETECTION_AREA_RATIO and
            box_iou < MIN_IOU_FOR_LARGE_CAMERA_AREA_MISMATCH):
        return 0.0
    return max(box_iou, overlap_detection(projection, detection))


def run_yolo(rows: list[dict], sequence: Path, args: argparse.Namespace,
             cache_path: Path) -> dict:
    # YOLO predictions depend on the model, preprocessing, input images, and
    # thresholds. A cache keyed only by "frame:camera" silently reuses stale
    # detections when any of these change (including replacing a JPG in place).
    cache_meta_path = cache_path.with_name(cache_path.name + '.meta.json')
    fingerprint = hashlib.sha256()
    fingerprint.update(json.dumps({
        'schema': 2,
        'dataset': str(sequence.resolve()),
        'weights': str(args.weights.resolve()),
        'weights_size': args.weights.stat().st_size,
        'weights_mtime_ns': args.weights.stat().st_mtime_ns,
        'image_size': args.image_size,
        'yolo_confidence': args.yolo_confidence,
        'class_mapping': YOLO_CLASSES,
        'iou': .65,
    }, sort_keys=True).encode())
    for row in rows:
        for camera, columns in CAMERAS.items():
            column = next((name for name in columns if row.get(name)), None)
            if column is None:
                raise KeyError('manifest row has no image path for ' + camera)
            image_path = (sequence / row[column]).resolve()
            try:
                image_path.relative_to(sequence.resolve())
            except ValueError as error:
                raise ValueError('unsafe camera image path: %s' % row[column]) from error
            stat = image_path.stat()
            fingerprint.update(json.dumps((row[column], stat.st_size,
                                           stat.st_mtime_ns)).encode())
    cache_fingerprint = fingerprint.hexdigest()
    stored = (json.loads(cache_meta_path.read_text())
              if cache_meta_path.exists() else {})
    cache = (json.loads(cache_path.read_text())
             if cache_path.exists() and stored.get('fingerprint') == cache_fingerprint
             else {})
    if stored.get('fingerprint') != cache_fingerprint:
        atomic_json({'fingerprint': cache_fingerprint,
                     'schema': 'yolo_frame_cache_v2'}, cache_meta_path)
        atomic_json(cache, cache_path)
    jobs = []
    for frame_index, row in enumerate(rows):
        for camera, columns in CAMERAS.items():
            key = f"{frame_index}:{camera}"
            if key not in cache:
                column = next((name for name in columns if row.get(name)), None)
                if column is None:
                    raise KeyError(
                        f"manifest row has no image path for {camera}: "
                        f"expected one of {columns}")
                jobs.append((key, sequence / row[column]))
    if not jobs:
        return cache
    if args.max_yolo_jobs is not None:
        jobs = jobs[:args.max_yolo_jobs]

    import torch
    from ultralytics import YOLO
    device = args.device
    if device == "auto":
        device = 0 if torch.cuda.is_available() else "cpu"
    model = YOLO(str(args.weights.resolve()))
    for start in range(0, len(jobs), args.batch_size):
        batch = jobs[start:start + args.batch_size]
        predict_options = dict(
            imgsz=args.image_size, conf=args.yolo_confidence, iou=.65,
            classes=sorted(YOLO_CLASSES), device=device,
            batch=len(batch), verbose=False)
        if device != "cpu":
            predict_options["half"] = True
        predictions = model.predict(
            [str(path) for _, path in batch], **predict_options)
        for (key, _), prediction in zip(batch, predictions):
            records = []
            if prediction.boxes is not None:
                for xyxy, confidence, label in zip(
                        prediction.boxes.xyxy.cpu().numpy(),
                        prediction.boxes.conf.cpu().numpy(),
                        prediction.boxes.cls.cpu().numpy()):
                    records.append({"xyxy": [float(value) for value in xyxy],
                                    "confidence": float(confidence),
                                    "class": YOLO_CLASSES[int(label)]})
            cache[key] = records
        if start % 80 == 0 or start + len(batch) == len(jobs):
            atomic_json(cache, cache_path)
            print(f"YOLO {min(start + len(batch), len(jobs))}/{len(jobs)}",
                  flush=True)
    atomic_json(cache, cache_path)
    return cache


def invalid_dimension_tracks(results: list[dict]) -> tuple[set[int], dict]:
    dimensions = defaultdict(list)
    names = defaultdict(list)
    for frame in results:
        for obj in frame["objects"]:
            track_id = int(obj["track_id"])
            dimensions[track_id].append(np.asarray(obj["box"], float)[3:6])
            names[track_id].append(str(obj["name"]))
    invalid, details = set(), {}
    for track_id, values in dimensions.items():
        name = Counter(names[track_id]).most_common(1)[0][0]
        values_array = np.asarray(values)
        median = np.median(values_array, axis=0)
        observed_maximum = np.max(values_array, axis=0)
        limit = MAX_DIMS.get(name)
        # Reject a complete trajectory only when its representative size is
        # impossible.  A single oversized observation is commonly an
        # association/merge glitch and must not erase an otherwise valid
        # long-lived track; it is repaired later from the track median.
        # An overlarge Car can be a correctly sized Truck/Bus with the wrong
        # class.  Give the track-level size classifier a chance to relabel it;
        # genuinely impossible dimensions are still rejected here.
        plausible_vehicle_size = (
            name in VEHICLES and len(values) >= MIN_VEHICLE_SIZE_LIDAR_FRAMES
            and vehicle_size_stable(values_array)
            and bool(vehicle_size_candidates(median)))
        violates_limit = (
            limit is not None and np.any(
                median > limit * TRACK_REJECTION_DIMENSION_FACTOR)
            and not plausible_vehicle_size)
        if violates_limit:
            invalid.add(track_id)
            details[str(track_id)] = {
                "class": name,
                "median_dimensions": median.tolist(),
                "observed_maximum_dimensions": observed_maximum.tolist(),
                "allowed_maximum": limit.tolist(),
                "track_rejection_maximum": (
                    limit * TRACK_REJECTION_DIMENSION_FACTOR).tolist(),
            }
    return invalid, details


def repair_refiner_dimensions(results: list[dict], tracks_path: Path) -> int:
    """Restore source dimensions when a class refiner leaves its envelope."""
    tracks = pickle.loads(tracks_path.read_bytes())
    source = {}
    for track_id, track in tracks.items():
        for frame_id, box in zip(track["sample_idx"], track["boxes_global"]):
            source[(int(track_id), int(frame_id))] = np.asarray(box, float)[3:6]
    repaired = 0
    for frame in results:
        frame_id = int(frame["frame_id"])
        for obj in frame["objects"]:
            limit = MAX_DIMS.get(obj["name"])
            if limit is None:
                continue
            refined = np.asarray(obj["box"], float)[3:6]
            original = source.get((int(obj["track_id"]), frame_id))
            if (np.any(refined > limit) and original is not None and
                    np.all(original > 0) and np.all(original <= limit)):
                obj["box"][3:6] = original.tolist()
                repaired += 1
    return repaired


def match_frame(frame: dict, frame_index: int, detections: dict,
                models: dict, transforms: dict, max_distance: float,
                motion: CameraMotion | None = None,
                velocities: dict | None = None) -> tuple[dict, set]:
    evidence = {}
    candidates = []
    for object_index, obj in enumerate(frame["objects"]):
        box = np.asarray(obj["box"], dtype=float)
        if np.linalg.norm(box[:2]) > max_distance:
            continue
        visible = []
        for camera in CAMERAS:
            velocity = ((velocities or {}).get((int(obj['track_id']), frame_index)))
            projected = motion.box(box, frame_index, camera, velocity) if motion else box
            projection = projected_box(projected, transforms[camera], models[camera])
            if projection is None:
                continue
            area = (projection[2] - projection[0]) * (projection[3] - projection[1])
            if area < 100:
                continue
            visible.append({"camera": camera, "area": float(area)})
            for detection_index, detection in enumerate(
                    detections.get(f"{frame_index}:{camera}", [])):
                # Camera evidence may correct a class within a compatible
                # family, but a car must not validate a spurious cyclist (or
                # vice versa).  Leave incompatible detections unused so the
                # LiDAR-backed addition pass can recover the camera object.
                allowed = ALLOWED_RELABELS.get(obj["name"], {obj["name"]})
                if detection["class"] not in allowed:
                    continue
                target = np.asarray(detection["xyxy"], dtype=float)
                score = camera_match_score(projection, target)
                if score >= .25:
                    candidates.append((score * detection["confidence"],
                                       object_index, camera, detection_index,
                                       detection, score))
        evidence[object_index] = {"visible": visible, "match": None}
    # Enforce one-to-one camera evidence. Without this, several overlapping
    # 3D hypotheses can all be confirmed by the same YOLO box.
    used = set()
    assigned_objects = set()
    for _, object_index, camera, detection_index, detection, score in sorted(
            candidates, key=lambda item: item[0], reverse=True):
        detection_key = (camera, detection_index)
        if object_index in assigned_objects or detection_key in used:
            continue
        used.add(detection_key)
        assigned_objects.add(object_index)
        evidence[object_index].update(
            {"match": detection, "camera": camera, "overlap": float(score),
             "detection_key": detection_key})
    return evidence, used


def tracker_prediction_support(
        tracks_path: Path) -> tuple[dict[tuple[int, int], np.ndarray],
                                    dict[tuple[int, int], np.ndarray]]:
    """Load DetZero global coast boxes and per-frame global velocities."""
    tracks = pickle.loads(tracks_path.read_bytes())
    coast_boxes = {}
    velocities = {}
    for raw_track_id, track in tracks.items():
        track_id = int(raw_track_id)
        boxes = np.asarray(track["boxes_global"], dtype=float)
        frame_ids = np.asarray(track["sample_idx"], dtype=int)
        hits = np.asarray(track["hit"], dtype=float) > 0
        if not (len(boxes) == len(frame_ids) == len(hits)):
            raise ValueError(f"track {track_id}: inconsistent tracker arrays")
        for index, frame_id in enumerate(frame_ids):
            key = (track_id, int(frame_id))
            if boxes.shape[1] >= 9:
                velocity = boxes[index, 7:9].copy()
                if np.isfinite(velocity).all():
                    velocities[key] = velocity
            if not hits[index]:
                coast_boxes[key] = boxes[index, :7].copy()
    return coast_boxes, velocities


def global_track_state(obj: dict, frame_index: int, timestamp_ns: int,
                       pose: np.ndarray, velocity: np.ndarray | None,
                       previous: dict | None = None) -> dict:
    """Create a global-frame motion state from a camera-confirmed LiDAR hit."""
    box = np.asarray(obj["box"], dtype=float)
    center = (pose @ np.r_[box[:3], 1.0])[:3]
    pose_yaw = math.atan2(pose[1, 0], pose[0, 0])
    global_yaw = wrap(float(box[6]) + pose_yaw)
    if velocity is None or not np.isfinite(velocity).all():
        velocity = np.zeros(2, dtype=float)
        if previous is not None:
            elapsed = (timestamp_ns - previous["last_lidar_timestamp_ns"]) * 1e-9
            if elapsed > 1e-3:
                velocity = ((center[:2] - previous["last_lidar_center"][:2]) /
                            elapsed)
    else:
        velocity = np.asarray(velocity, dtype=float).copy()
    if np.linalg.norm(velocity) < STATIC_GLOBAL_SPEED_MPS:
        velocity[:] = 0.0
    return {
        "track_id": int(obj["track_id"]), "name": str(obj["name"]),
        "score": float(obj.get("score", 0.0)),
        "dimensions": box[3:6].copy(), "global_center": center,
        "global_yaw": global_yaw, "velocity": velocity,
        # Vertical state is not observable robustly from the 2D camera hold.
        # Keep the exact local z from the most recent real LiDAR observation.
        "local_z": float(box[2]),
        "last_frame": frame_index, "timestamp_ns": int(timestamp_ns),
        "last_lidar_center": center.copy(),
        "last_lidar_timestamp_ns": int(timestamp_ns), "hold_age": 0,
    }


def predicted_hold_object(state: dict, frame_index: int, timestamp_ns: int,
                          pose: np.ndarray, tracker_box: np.ndarray | None,
                          tracker_velocity: np.ndarray | None,
                          max_distance: float, max_dt: float,
                          max_speed: float,
                          tracker_source: str) -> tuple[dict, dict] | None:
    """Predict one local-frame box while keeping motion in the global frame."""
    elapsed = (timestamp_ns - state["timestamp_ns"]) * 1e-9
    if elapsed <= 0 or elapsed > max_dt:
        return None
    velocity = np.asarray(
        tracker_velocity if tracker_velocity is not None else state["velocity"],
        dtype=float).copy()
    speed = float(np.linalg.norm(velocity))
    if speed < STATIC_GLOBAL_SPEED_MPS:
        velocity[:] = 0.0
    elif speed > max_speed:
        velocity *= max_speed / speed
    if tracker_box is not None:
        global_center = np.asarray(tracker_box[:3], dtype=float).copy()
        global_yaw = wrap(float(tracker_box[6]))
        source = tracker_source
    else:
        global_center = np.asarray(state["global_center"], dtype=float).copy()
        global_center[:2] += velocity * elapsed
        global_yaw = float(state["global_yaw"])
        source = "global_constant_velocity"
    inverse_pose = np.linalg.inv(pose)
    local_center = (inverse_pose @ np.r_[global_center, 1.0])[:3]
    # The global Z remains fixed for planar motion. Unlike preserving local Z,
    # this correctly handles ego elevation/pitch changes between acquisitions.
    if np.linalg.norm(local_center[:2]) > max_distance:
        return None
    pose_yaw = math.atan2(pose[1, 0], pose[0, 0])
    local_yaw = wrap(global_yaw - pose_yaw)
    hold_age = int(state["hold_age"]) + 1
    obj = {
        "track_id": int(state["track_id"]), "name": str(state["name"]),
        "score": float(state["score"]),
        "box": np.r_[local_center, state["dimensions"], local_yaw].tolist(),
        "source": "camera_track_hold", "prediction_source": source,
        "hold_age_frames": hold_age,
    }
    next_state = {
        **state, "global_center": global_center, "global_yaw": global_yaw,
        "velocity": velocity, "last_frame": frame_index,
        "timestamp_ns": int(timestamp_ns), "hold_age": hold_age,
    }
    return obj, next_state


def match_camera_hold_predictions(
        predictions: dict[int, tuple[dict, dict]], frame_index: int,
        detections: dict, models: dict, transforms: dict, used: set,
        min_overlap: float, motion: CameraMotion | None = None) -> dict[int, dict]:
    """Assign unused compatible camera detections to predicted track boxes."""
    candidates = []
    for track_id, (obj, _) in predictions.items():
        box = np.asarray(obj["box"], dtype=float)
        allowed = ALLOWED_RELABELS.get(obj["name"], {obj["name"]})
        for camera in CAMERAS:
            velocity = predictions[track_id][1].get('velocity')
            projected = motion.box(box, frame_index, camera, velocity) if motion else box
            projection = projected_box(projected, transforms[camera], models[camera])
            if projection is None:
                continue
            area = ((projection[2] - projection[0]) *
                    (projection[3] - projection[1]))
            if area < 100:
                continue
            for detection_index, detection in enumerate(
                    detections.get(f"{frame_index}:{camera}", [])):
                detection_key = (camera, detection_index)
                if detection_key in used or detection["class"] not in allowed:
                    continue
                target = np.asarray(detection["xyxy"], dtype=float)
                overlap = camera_match_score(projection, target)
                if overlap >= min_overlap:
                    candidates.append((overlap * detection["confidence"],
                                       track_id, detection_key, camera,
                                       detection, overlap))
    assignments = {}
    assigned_tracks = set()
    for _, track_id, detection_key, camera, detection, overlap in sorted(
            candidates, key=lambda item: item[0], reverse=True):
        if track_id in assigned_tracks or detection_key in used:
            continue
        used.add(detection_key)
        assigned_tracks.add(track_id)
        assignments[track_id] = {
            "camera": camera, "detection": detection,
            "detection_key": detection_key, "overlap": float(overlap),
        }
    return assignments


def lidar_hold_support(
        sequence: Path, row: dict, camera: str, detection: dict,
        predicted_box: np.ndarray, model: dict, transform: np.ndarray,
        max_distance: float, projection_cache: dict, min_points: int,
        max_residual: float, max_correction: float,
        projection_transform: np.ndarray | None = None
        ) -> tuple[np.ndarray, int, float, float] | None:
    """Validate/correct a camera hold with sparse points in its 2D frustum."""
    if camera not in projection_cache:
        points = read_binary_pcd(sequence / row["lidar_pcd"])
        xyz = points[:, :3].astype(float)
        ranges = np.linalg.norm(xyz[:, :2], axis=1)
        keep = ranges <= max_distance
        xyz, ranges = xyz[keep], ranges[keep]
        point_transform = (projection_transform if projection_transform is not None
                           else transform)
        camera_xyz = (
            np.column_stack((xyz, np.ones(len(xyz)))) @ point_transform.T)[:, :3]
        pixels, valid = kb4_project(camera_xyz, model)
        projection_cache[camera] = (xyz, ranges, pixels, valid)
    xyz, ranges, pixels, valid = projection_cache[camera]
    target = np.asarray(detection["xyxy"], dtype=float)
    margin_x = .03 * (target[2] - target[0])
    margin_y = .03 * (target[3] - target[1])
    half_height = predicted_box[5] / 2
    spatial_residual = np.linalg.norm(
        xyz[:, :2] - predicted_box[None, :2], axis=1)
    inside = (
        valid & (pixels[:, 0] >= target[0] + margin_x) &
        (pixels[:, 0] <= target[2] - margin_x) &
        (pixels[:, 1] >= target[1] + margin_y) &
        (pixels[:, 1] <= target[3] - margin_y) &
        (spatial_residual <= max_residual) &
        # Reject road returns at the lower cuboid face while retaining sparse
        # returns slightly outside the detector's estimated vertical extent.
        (xyz[:, 2] >= predicted_box[2] - half_height + .12) &
        (xyz[:, 2] <= predicted_box[2] + half_height + .4))
    candidates = xyz[inside]
    candidate_ranges = ranges[inside]
    if len(candidates) < min_points:
        return None
    order = np.argsort(candidate_ranges)
    splits = np.flatnonzero(np.diff(candidate_ranges[order]) > 1.25) + 1
    groups = [group for group in np.split(order, splits)
              if len(group) >= min_points]
    if not groups:
        return None
    centers = [np.median(candidates[group], axis=0) for group in groups]
    distances = [np.linalg.norm(center[:2] - predicted_box[:2])
                 for center in centers]
    best_index = int(np.argmin(distances))
    residual = float(distances[best_index])
    if residual > max_residual:
        return None
    cluster = candidates[groups[best_index]]
    cluster_center = centers[best_index]

    # Sparse returns commonly lie on the visible surface, not at the cuboid
    # center. Apply only the smallest shift needed to bring their median well
    # inside the predicted footprint; never replace depth with the raw median.
    delta = cluster_center[:2] - predicted_box[:2]
    cosine, sine = math.cos(predicted_box[6]), math.sin(predicted_box[6])
    rotation = np.asarray([[cosine, sine], [-sine, cosine]])
    local_delta = rotation @ delta
    interior = .75 * predicted_box[3:5] / 2
    local_shift = np.sign(local_delta) * np.maximum(
        np.abs(local_delta) - interior, 0.0)
    shift = rotation.T @ local_shift
    correction = float(np.linalg.norm(shift))
    if correction > max_correction:
        shift *= max_correction / correction
        correction = max_correction
    corrected = predicted_box.copy()
    corrected[:2] += shift
    return corrected, len(cluster), residual, correction


def lidar_candidate(sequence: Path, row: dict, camera: str, detection: dict,
                    model: dict, transform: np.ndarray,
                    max_distance: float, projection_cache: dict
                    ) -> tuple[np.ndarray, int] | None:
    if camera not in projection_cache:
        points = read_binary_pcd(sequence / row["lidar_pcd"])
        xyz = points[:, :3].astype(float)
        ranges = np.linalg.norm(xyz[:, :2], axis=1)
        keep = ranges <= max_distance
        xyz, ranges = xyz[keep], ranges[keep]
        point_transform = (projection_transform if projection_transform is not None
                           else transform)
        camera_xyz = (
            np.column_stack((xyz, np.ones(len(xyz)))) @ point_transform.T)[:, :3]
        pixels, valid = kb4_project(camera_xyz, model)
        projection_cache[camera] = (xyz, ranges, pixels, valid)
    xyz, ranges, pixels, valid = projection_cache[camera]
    target = np.asarray(detection["xyxy"], float)
    margin_x = .08 * (target[2] - target[0])
    margin_y = .08 * (target[3] - target[1])
    inside = (valid & (pixels[:, 0] >= target[0] + margin_x) &
              (pixels[:, 0] <= target[2] - margin_x) &
              (pixels[:, 1] >= target[1] + margin_y) &
              (pixels[:, 1] <= target[3] - margin_y))
    candidates, candidate_ranges = xyz[inside], ranges[inside]
    if len(candidates) < 6:
        return None
    order = np.argsort(candidate_ranges)
    sorted_ranges = candidate_ranges[order]
    splits = np.flatnonzero(np.diff(sorted_ranges) > 1.5) + 1
    groups = np.split(order, splits)
    groups = [group for group in groups if len(group) >= 6]
    if not groups:
        return None
    group = min(groups, key=lambda indices: np.median(candidate_ranges[indices]))
    cluster = candidates[group]
    center = np.median(cluster, axis=0)
    return center, len(cluster)


def smooth_track_headings(results: list[dict], poses: np.ndarray,
                          frame_times_ns: np.ndarray) -> int:
    if len(frame_times_ns) != len(results):
        raise ValueError('heading timestamps and results differ in length')
    observations = defaultdict(list)
    for frame_index, frame in enumerate(results):
        for obj in frame["objects"]:
            if obj["name"] in VEHICLES:
                observations[int(obj["track_id"])].append((frame_index, obj))
    changed = 0
    for items in observations.values():
        if len(items) < 2:
            continue
        global_centers, global_yaws = [], []
        for frame_index, obj in items:
            box = np.asarray(obj["box"], float)
            pose = poses[frame_index]
            global_centers.append((pose @ np.r_[box[:3], 1])[:3])
            global_yaws.append(wrap(box[6] + math.atan2(pose[1, 0], pose[0, 0])))
        centers = np.asarray(global_centers)
        elapsed = (int(frame_times_ns[items[-1][0]]) -
                   int(frame_times_ns[items[0][0]])) / 1e9
        if elapsed <= 0:
            raise ValueError('heading timestamps must increase within a track')
        speed = np.linalg.norm(centers[-1, :2] - centers[0, :2]) / elapsed
        if speed > .8 and np.linalg.norm(centers[-1, :2] - centers[0, :2]) > 2:
            delta = centers[-1, :2] - centers[0, :2]
            chosen = math.atan2(delta[1], delta[0])
        else:
            # Double-angle averaging finds the stable axis, then the majority
            # direction resolves the 180-degree ambiguity.
            axis = .5 * math.atan2(np.mean(np.sin(2 * np.asarray(global_yaws))),
                                   np.mean(np.cos(2 * np.asarray(global_yaws))))
            alternatives = (axis, wrap(axis + math.pi))
            chosen = min(alternatives, key=lambda candidate: sum(
                abs(wrap(candidate - yaw)) for yaw in global_yaws))
        for (frame_index, obj), old_global in zip(items, global_yaws):
            pose_yaw = math.atan2(poses[frame_index][1, 0], poses[frame_index][0, 0])
            new_yaw = wrap(chosen - pose_yaw)
            if abs(wrap(new_yaw - float(obj["box"][6]))) > math.radians(90):
                changed += 1
            obj["box"][6] = new_yaw
    return changed


def consensus_vehicle_name(objects: list[dict], dimensions: np.ndarray) -> str:
    """Choose one LiDAR-supported subtype for every frame of a vehicle ID."""
    counts = Counter(obj['name'] for obj in objects)
    scores = defaultdict(float)
    for obj in objects:
        scores[obj['name']] += float(obj.get('score', 0.))
    candidates = [name for name in counts if name in VEHICLES]
    if not candidates:
        raise ValueError('vehicle track has no vehicle-class observations')
    plausible = [name for name in candidates
                 if np.all(dimensions <= MAX_DIMS[name])]
    return max(plausible or candidates,
               key=lambda name: (counts[name], scores[name], name))


def vehicle_size_candidates(dimensions: np.ndarray) -> dict[str, float]:
    """Return dimensionally plausible classes and nominal-range penalties."""
    dims = np.asarray(dimensions, dtype=float)
    if dims.shape != (3,) or not np.all(np.isfinite(dims)) or np.any(dims <= 0):
        return {}
    candidates = {}
    for name, profiles in VEHICLE_SIZE_PROFILES.items():
        penalties = []
        for lower, upper, tolerance in profiles:
            lower = np.asarray(lower)
            upper = np.asarray(upper)
            tolerance = np.asarray(tolerance)
            if np.all(dims >= lower - tolerance) and np.all(dims <= upper + tolerance):
                outside = np.maximum(lower - dims, 0) + np.maximum(dims - upper, 0)
                penalties.append(float(np.linalg.norm(outside / tolerance)))
        if penalties:
            candidates[name] = min(penalties)
    return candidates


def vehicle_size_stable(values: np.ndarray) -> bool:
    """Reject size evidence from tracks that likely merged different objects."""
    values = np.asarray(values, dtype=float)
    if (values.ndim != 2 or values.shape[1] != 3 or
            len(values) < MIN_VEHICLE_SIZE_LIDAR_FRAMES or
            not np.all(np.isfinite(values)) or np.any(values <= 0)):
        return False
    median = np.median(values, axis=0)
    spread = (np.percentile(values, 90, axis=0) -
              np.percentile(values, 10, axis=0)) / median
    return bool(np.all(spread <= np.array([.35, .35, .4])))


def classify_vehicle_by_size(dimensions: np.ndarray, lidar_name: str) -> str:
    """Relabel a whole track only when its median size is informative.

    Rigid trucks and buses overlap in all three dimensions.  Their LiDAR
    subtype wins that tie; no size-only rule can reliably separate them.
    """
    candidates = vehicle_size_candidates(dimensions)
    if not candidates:
        return lidar_name
    if len(candidates) == 1:
        return next(iter(candidates))
    if lidar_name in ("Bus", "Truck") and lidar_name in candidates:
        return lidar_name
    if "Car" in candidates and "Van" in candidates:
        if (candidates["Van"] + .75 < candidates["Car"] and
                lidar_name in ("Car", "Van", "ConstructionVehicle")):
            return "Van"
        if (candidates["Car"] + .75 < candidates["Van"] and
                lidar_name in ("Car", "Van", "ConstructionVehicle")):
            return "Car"
        return lidar_name
    if "Bus" in candidates and "Truck" in candidates:
        # With no LiDAR Bus/Truck vote, the broader Truck class is safer than
        # pretending dimensions can distinguish a box truck from a bus.
        return "Truck"
    return lidar_name if lidar_name in candidates else min(candidates,
                                                             key=candidates.get)


def main() -> None:
    args = parse_args()
    if not (0 < args.camera_hold_min_overlap <= 1 and
            0 < args.camera_hold_no_lidar_min_overlap <= 1):
        raise ValueError("camera hold overlap thresholds must be in (0, 1]")
    if args.camera_hold_max_dt <= 0 or args.camera_hold_max_speed <= 0:
        raise ValueError("camera hold maximum dt and speed must be positive")
    if (args.camera_hold_min_lidar_points < 1 or
            args.camera_hold_max_lidar_residual <= 0 or
            args.camera_hold_max_position_correction < 0):
        raise ValueError("invalid camera hold LiDAR-support parameters")
    if (args.ego_protected_front <= 0 or args.ego_protected_rear <= 0 or
            args.ego_protected_half_width <= 0 or
            args.camera_hold_max_ego_overlap < 0):
        raise ValueError("invalid protected ego-footprint parameters")
    if (args.vehicle_track_confirm_min_lidar_frames < 1 or
            not 0 <= args.vehicle_track_confirm_min_score <= 1 or
            not 0 <= args.vehicle_track_confirm_min_camera_confidence <= 1 or
            not 0 <= args.vehicle_track_confirm_min_overlap <= 1):
        raise ValueError("invalid vehicle-track confirmation parameters")
    sequence = args.sequence.resolve()
    output = (args.output or sequence / "detzero_output" /
              "improved_results.json").resolve()
    results_path = (args.results or sequence / "detzero_output" /
                    "results.json").resolve()
    calibration_path = (args.calibration or next(
        (p for p in (sequence / 'calibration.json',
                     sequence.parent / 'calibration.json') if p.exists()),
        sequence / 'calibration.json')).resolve()
    cache_path = (args.cache or output.with_name("yolo_detections.json")).resolve()
    rows = load_rows(sequence, args.limit)
    results = json.loads(results_path.read_text())[:len(rows)]
    _, poses, _ = load_ego_poses(sequence, rows, calibration_path)
    if len(results) != len(rows):
        raise ValueError('results count differs from manifest')
    for frame_index, (frame, row) in enumerate(zip(results, rows)):
        if int(frame['frame_id']) != frame_index:
            raise ValueError('results and manifest frame indices are misaligned')
        if int(frame['timestamp_ns']) != int(row['timestamp_ns']):
            raise ValueError('results and manifest timestamps differ at frame %d' % frame_index)
    models, transforms = calibration_models(calibration_path)
    if (not np.isfinite(args.camera_time_offset_ms) or
            not np.isfinite(args.camera_time_max_offset_ms) or
            args.camera_time_max_offset_ms <= 0 or
            not np.isfinite(args.camera_time_max_pose_gap_s) or
            args.camera_time_max_pose_gap_s <= 0):
        raise ValueError('invalid camera temporal compensation arguments')
    motion = CameraMotion(
        rows, poses, enabled=not args.no_camera_time_compensation,
        offset_ms=args.camera_time_offset_ms,
        max_offset_ms=args.camera_time_max_offset_ms,
        max_pose_gap_s=args.camera_time_max_pose_gap_s)
    cache_path.parent.mkdir(parents=True, exist_ok=True)
    detections = run_yolo(rows, sequence, args, cache_path)
    expected_keys = {f"{frame_index}:{camera}"
                     for frame_index in range(len(rows)) for camera in CAMERAS}
    cached_keys = expected_keys.intersection(detections)
    if args.yolo_only:
        print(f"YOLO cache: {len(cached_keys)}/{len(expected_keys)}", flush=True)
        raise SystemExit(0 if len(cached_keys) == len(expected_keys) else 3)
    if len(cached_keys) != len(expected_keys):
        raise RuntimeError(
            f"YOLO cache incomplete: {len(cached_keys)}/{len(expected_keys)}")

    tracks_path = (args.tracks or results_path.parent / "tracks.pkl").resolve()
    tracking_config_path = results_path.parent / "tracking_run_config.json"
    tracking_config = (json.loads(tracking_config_path.read_text())
                       if tracking_config_path.exists() else {})
    tracker_filter = tracking_config.get("tracking_filter", "unknown")
    tracker_prediction_source = (
        "ab3dmot_planar" if tracker_filter == "AB3DMOT" else "tracker_coast")
    dimension_fallbacks = repair_refiner_dimensions(results, tracks_path)
    tracker_tracks = pickle.loads(tracks_path.read_bytes())
    tracker_coasts, tracker_velocities = tracker_prediction_support(tracks_path)
    lidar_track_ids = {
        int(obj["track_id"]) for frame in results for obj in frame["objects"]
    }
    invalid_tracks, invalid_details = invalid_dimension_tracks(results)
    # Camera validation applies to the physical LiDAR track, not independently
    # to every frame. A bus at a fisheye seam can be missed for several frames
    # even though CenterPoint follows it cleanly. Require a multi-frame,
    # sufficiently confident LiDAR track plus at least one strong one-to-one
    # camera match before retaining its unmatched LiDAR observations.
    frame_evidence = []
    vehicle_track_stats = defaultdict(
        lambda: {"lidar_frames": 0, "max_score": 0.0, "camera_matches": []})
    for frame_index, frame in enumerate(results):
        evidence, _ = match_frame(
            frame, frame_index, detections, models, transforms,
            args.max_distance, motion=motion, velocities=tracker_velocities)
        frame_evidence.append(evidence)
        for object_index, obj in enumerate(frame["objects"]):
            if obj["name"] not in VEHICLES:
                continue
            track_id = int(obj["track_id"])
            stats = vehicle_track_stats[track_id]
            stats["lidar_frames"] += 1
            stats["max_score"] = max(
                stats["max_score"], float(obj.get("score", 0.0)))
            record = evidence.get(object_index, {})
            match = record.get("match")
            if match is not None:
                stats["camera_matches"].append({
                    "frame": frame_index,
                    "confidence": float(match["confidence"]),
                    "overlap": float(record["overlap"]),
                })
    camera_validated_vehicle_tracks = {
        track_id for track_id, stats in vehicle_track_stats.items()
        if (track_id not in invalid_tracks and
            stats["lidar_frames"] >=
                args.vehicle_track_confirm_min_lidar_frames and
            stats["max_score"] >= args.vehicle_track_confirm_min_score and
            any(match["confidence"] >=
                    args.vehicle_track_confirm_min_camera_confidence and
                match["overlap"] >= args.vehicle_track_confirm_min_overlap
                for match in stats["camera_matches"]))
    }
    class_votes = defaultdict(Counter)
    generic_car_subtype_validations = 0
    confirmation_classes = set(args.camera_confirm_classes)
    removed_range = removed_unconfirmed = removed_dimensions = 0
    removed_unconfirmed_by_class = Counter()
    retained_by_vehicle_track_confirmation = 0
    camera_held_by_class = Counter()
    camera_hold_prediction_sources = Counter()
    camera_hold_overlaps = []
    camera_hold_lidar_points = []
    camera_hold_lidar_residuals = []
    camera_hold_position_corrections = []
    camera_held_with_lidar = camera_held_without_lidar = 0
    camera_holds_rejected_ego_overlap = 0
    maximum_camera_hold_age = 0
    active_states = {}
    additions = []
    next_track_id = 1 + max((int(obj["track_id"]) for frame in results
                             for obj in frame["objects"]), default=0)
    previous_additions = []
    for frame_index, (row, frame) in enumerate(zip(rows, results)):
        evidence = frame_evidence[frame_index]
        original_track_ids = {
            int(obj["track_id"]) for obj in frame["objects"]
        }
        timestamp_ns = int(frame.get(
            "lidar_sensor_timestamp_ns", frame.get("timestamp_ns",
                                                    row["timestamp_ns"])))
        kept = []
        used = set()
        next_active_states = {}
        for object_index, obj in enumerate(frame["objects"]):
            track_id = int(obj["track_id"])
            distance = float(np.linalg.norm(np.asarray(obj["box"][:2], float)))
            if distance > args.max_distance:
                removed_range += 1
                continue
            if track_id in invalid_tracks:
                removed_dimensions += 1
                continue
            record = evidence.get(object_index, {"visible": [], "match": None})
            match = record["match"]
            if match is not None:
                matched_name = match["class"]
                # The reviewed bag shows LiDAR vehicle subtypes are much more
                # accurate than fisheye YOLO votes. Camera road-vehicle labels
                # validate existence only; they never rewrite Car/Truck/Bus.
                # In particular, generic YOLO Car validates Truck/Bus without
                # collapsing the LiDAR-supported subtype.
                if obj["name"] in {"Pedestrian", "Cyclist"}:
                    class_votes[track_id][matched_name] += (
                        match["confidence"] * max(record["overlap"], .25))
                elif matched_name == "Car":
                    generic_car_subtype_validations += 1
            elif (not args.keep_unconfirmed and
                  obj["name"] in confirmation_classes and
                  distance <= args.confirm_distance and
                  not (obj["name"] in VEHICLES and
                       track_id in camera_validated_vehicle_tracks)):
                removed_unconfirmed += 1
                removed_unconfirmed_by_class[obj["name"]] += 1
                continue
            elif (match is None and obj["name"] in VEHICLES and
                  track_id in camera_validated_vehicle_tracks):
                retained_by_vehicle_track_confirmation += 1
            kept.append(obj)
            if match is not None:
                # Do not confuse camera confirmation of an actual LiDAR box
                # with camera-only track propagation. Preserve both signals.
                obj['camera_confirmation'] = {
                    'camera': record['camera'],
                    'camera_time_compensation': motion.details(
                        frame_index, record['camera'],
                        (track_id, frame_index) in tracker_velocities),
                    'confidence': float(match['confidence']),
                    'projection_overlap': float(record['overlap']),
                }
                used.add(record["detection_key"])
                if (args.camera_track_hold and
                        track_id in lidar_track_ids):
                    next_active_states[track_id] = global_track_state(
                        obj, frame_index, timestamp_ns, poses[frame_index],
                        tracker_velocities.get((track_id, frame_index)),
                        active_states.get(track_id))
        frame["objects"] = kept
        projection_cache = {}

        if args.camera_track_hold:
            predictions = {}
            for track_id, state in active_states.items():
                if (state["last_frame"] != frame_index - 1 or
                        track_id in original_track_ids):
                    continue
                prediction = predicted_hold_object(
                    state, frame_index, timestamp_ns, poses[frame_index],
                    tracker_coasts.get((track_id, frame_index)),
                    tracker_velocities.get((track_id, frame_index)),
                    args.max_distance, args.camera_hold_max_dt,
                    args.camera_hold_max_speed, tracker_prediction_source)
                if prediction is not None:
                    overlap = ego_footprint_overlap(
                        np.asarray(prediction[0]["box"], float),
                        args.ego_protected_front, args.ego_protected_rear,
                        args.ego_protected_half_width)
                    if overlap > args.camera_hold_max_ego_overlap:
                        camera_holds_rejected_ego_overlap += 1
                        continue
                    predictions[track_id] = prediction
            hold_matches = match_camera_hold_predictions(
                predictions, frame_index, detections, models, transforms,
                used, args.camera_hold_min_overlap, motion=motion)
            for track_id, match_record in hold_matches.items():
                obj, state = predictions[track_id]
                support = lidar_hold_support(
                    sequence, row, match_record["camera"],
                    match_record["detection"], np.asarray(obj["box"], float),
                    models[match_record["camera"]],
                    transforms[match_record["camera"]], args.max_distance,
                    projection_cache, args.camera_hold_min_lidar_points,
                    args.camera_hold_max_lidar_residual,
                    args.camera_hold_max_position_correction,
                    projection_transform=motion.point_transform(
                        frame_index, match_record['camera'],
                        transforms[match_record['camera']]))
                if support is None:
                    if (match_record["overlap"] <
                            args.camera_hold_no_lidar_min_overlap):
                        used.discard(match_record["detection_key"])
                        continue
                    obj["lidar_support"] = 0
                    camera_held_without_lidar += 1
                else:
                    corrected, point_count, residual, correction = support
                    corrected_at_camera = motion.box(
                        corrected, frame_index, match_record['camera'],
                        state.get('velocity'))
                    corrected_projection = projected_box(
                        corrected_at_camera, transforms[match_record["camera"]],
                        models[match_record["camera"]])
                    target = np.asarray(
                        match_record["detection"]["xyxy"], dtype=float)
                    corrected_overlap = (0.0 if corrected_projection is None
                        else camera_match_score(corrected_projection, target))
                    if corrected_overlap < args.camera_hold_min_overlap:
                        used.discard(match_record["detection_key"])
                        continue
                    ego_overlap = ego_footprint_overlap(
                        corrected, args.ego_protected_front,
                        args.ego_protected_rear,
                        args.ego_protected_half_width)
                    if ego_overlap > args.camera_hold_max_ego_overlap:
                        camera_holds_rejected_ego_overlap += 1
                        used.discard(match_record["detection_key"])
                        continue
                    match_record["overlap"] = float(corrected_overlap)
                    obj["box"] = corrected.tolist()
                    obj["lidar_support"] = int(point_count)
                    obj["lidar_cluster_residual_m"] = residual
                    obj["lidar_position_correction_m"] = correction
                    global_center = (
                        poses[frame_index] @ np.r_[corrected[:3], 1.0])[:3]
                    state["global_center"] = global_center
                    camera_held_with_lidar += 1
                    camera_hold_lidar_points.append(point_count)
                    camera_hold_lidar_residuals.append(residual)
                    camera_hold_position_corrections.append(correction)
                obj['camera_time_compensation'] = motion.details(
                    frame_index, match_record['camera'],
                    state.get('velocity') is not None)
                obj["camera"] = match_record["camera"]
                obj["camera_class"] = match_record["detection"]["class"]
                obj["camera_confidence"] = float(
                    match_record["detection"]["confidence"])
                obj["camera_projection_overlap"] = match_record["overlap"]
                frame["objects"].append(obj)
                next_active_states[track_id] = state
                camera_held_by_class[obj["name"]] += 1
                camera_hold_prediction_sources[obj["prediction_source"]] += 1
                camera_hold_overlaps.append(match_record["overlap"])
                maximum_camera_hold_age = max(
                    maximum_camera_hold_age, int(obj["hold_age_frames"]))
        active_states = next_active_states

        if args.add_camera_objects:
            frame_additions = []
            used_addition_track_ids = set()
            for camera in CAMERAS:
                for detection_index, detection in enumerate(
                        detections.get(f"{frame_index}:{camera}", [])):
                    if ((camera, detection_index) in used or
                            detection["confidence"] < args.add_confidence):
                        continue
                    candidate = lidar_candidate(
                        sequence, row, camera, detection, models[camera],
                        transforms[camera], args.max_distance,
                        projection_cache,
                        projection_transform=motion.point_transform(
                            frame_index, camera, transforms[camera]))
                    if candidate is None:
                        continue
                    center, support = candidate
                    if np.linalg.norm(center[:2]) > args.max_distance:
                        continue
                    name = detection["class"]
                    global_center = (poses[frame_index] @ np.r_[center, 1])[:3]
                    if any(np.linalg.norm(global_center[:2] - old["global"][:2]) < 2.0
                           for old in frame_additions):
                        continue
                    compatible = [(np.linalg.norm(global_center[:2] - old["global"][:2]), old)
                                  for old in previous_additions
                                  if (old["class"] == name and
                                      old["track_id"] not in used_addition_track_ids)]
                    compatible = [item for item in compatible if item[0] <= 10.]
                    if compatible:
                        track_id = min(compatible, key=lambda item: item[0])[1]["track_id"]
                    else:
                        track_id, next_track_id = next_track_id, next_track_id + 1
                    used_addition_track_ids.add(track_id)
                    dims = CANONICAL_DIMS[name]
                    obj = {"track_id": track_id, "name": name,
                           "score": float(detection["confidence"]),
                           "box": [*map(float, center), *map(float, dims), 0.0],
                           "source": "yolo_lidar", "lidar_support": support,
                           "camera": camera,
                           "camera_time_compensation": motion.details(
                               frame_index, camera)}
                    frame["objects"].append(obj)
                    frame_additions.append({"global": global_center, "class": name,
                                            "track_id": track_id})
                    additions.append({"frame_id": frame_index, "camera": camera,
                                      "track_id": track_id, "class": name,
                                      "confidence": detection["confidence"],
                                      "lidar_support": support})
            previous_additions = frame_additions

    track_dimensions = defaultdict(list)
    lidar_track_dimensions = defaultdict(list)
    track_objects = defaultdict(list)
    for frame in results:
        for obj in frame["objects"]:
            track_id = int(obj["track_id"])
            track_dimensions[track_id].append(
                np.asarray(obj["box"], float)[3:6])
            if obj.get("source") != "camera_track_hold":
                lidar_track_dimensions[track_id].append(
                    np.asarray(obj["box"], float)[3:6])
            track_objects[track_id].append(obj)
    track_median_dimensions = {
        track_id: np.median(values, axis=0)
        for track_id, values in track_dimensions.items()
    }

    relabeled = Counter()
    size_relabels = Counter()
    size_ambiguous = 0
    size_unstable = 0
    for track_id, objects in track_objects.items():
        current_name = Counter(obj["name"] for obj in objects).most_common(1)[0][0]
        dimensions = track_median_dimensions[track_id]
        if current_name in VEHICLES:
            # A physical ID has one class for its whole life. Previous code
            # required 70% agreement and left ambiguous Bus/Truck/Car tracks
            # flickering. Rank native LiDAR votes by frame count then score;
            # use dimensions only to break ties among plausible classes.
            # Generic YOLO Car is existence evidence, not a subtype vote.
            resolved_name = consensus_vehicle_name(objects, dimensions)
            lidar_sizes = lidar_track_dimensions[track_id]
            if vehicle_size_stable(lidar_sizes):
                lidar_median = np.median(lidar_sizes, axis=0)
                candidates = vehicle_size_candidates(lidar_median)
                if len(candidates) > 1:
                    size_ambiguous += 1
                size_name = classify_vehicle_by_size(lidar_median,
                                                     resolved_name)
                if size_name != resolved_name:
                    size_relabels[f"{resolved_name}->{size_name}"] += 1
                    resolved_name = size_name
            elif len(lidar_sizes) >= MIN_VEHICLE_SIZE_LIDAR_FRAMES:
                size_unstable += 1
            for obj in objects:
                if obj["name"] != resolved_name:
                    relabeled[f"{obj['name']}->{resolved_name}"] += 1
                    obj["name"] = resolved_name
                if np.any(np.asarray(obj["box"], float)[3:6] >
                          MAX_DIMS[resolved_name]):
                    replacement = (dimensions if np.all(
                        dimensions <= MAX_DIMS[resolved_name]) else
                        CANONICAL_DIMS[resolved_name])
                    obj["box"][3:6] = replacement.tolist()
            continue
        resolved_name = current_name
        votes = class_votes.get(track_id)
        if votes:
            voted_name, vote = votes.most_common(1)[0]
            total = sum(votes.values())
            allowed = ALLOWED_RELABELS.get(current_name, {current_name})
            if voted_name in allowed and vote >= .25 and vote / total >= .55:
                resolved_name = voted_name
        replacement_dimensions = (dimensions if
            np.all(dimensions <= MAX_DIMS[resolved_name]) else
            CANONICAL_DIMS[resolved_name])
        for obj in objects:
            old_name = obj["name"]
            if old_name != resolved_name:
                relabeled[f"{old_name}->{resolved_name}"] += 1
                obj["name"] = resolved_name
            if np.any(np.asarray(obj["box"], float)[3:6] >
                      MAX_DIMS[resolved_name]):
                obj["box"][3:6] = replacement_dimensions.tolist()
    frame_times_ns = np.asarray([
        int(row.get('lidar_sensor_timestamp_ns') or row['timestamp_ns'])
        for row in rows], dtype=np.int64)
    heading_flips = (smooth_track_headings(results, poses, frame_times_ns)
                     if args.smooth_headings else 0)
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(results, output)
    export_annotations(sequence, rows, results, output, calibration_path,
                       tracks=tracker_tracks,
                       stage='camera_improved',
                       write_classic_labels=not args.skip_classic_labels)
    report = {
        "input": str(results_path), "output": str(output), "frames": len(results),
        "max_distance_m": args.max_distance,
        "camera_confirmation_required_classes": sorted(confirmation_classes),
        "vehicle_track_confirmation": {
            "minimum_lidar_frames": (
                args.vehicle_track_confirm_min_lidar_frames),
            "minimum_peak_lidar_score": args.vehicle_track_confirm_min_score,
            "minimum_camera_confidence": (
                args.vehicle_track_confirm_min_camera_confidence),
            "minimum_projection_overlap": (
                args.vehicle_track_confirm_min_overlap),
            "validated_tracks": len(camera_validated_vehicle_tracks),
            "retained_unmatched_lidar_annotations": (
                retained_by_vehicle_track_confirmation),
        },
        "removed": {"over_distance": removed_range,
                    "invalid_track_dimensions": removed_dimensions,
                    "camera_unconfirmed": removed_unconfirmed,
                    "camera_unconfirmed_by_class": dict(
                        removed_unconfirmed_by_class)},
        "invalid_dimension_tracks": invalid_details,
        "refiner_dimension_fallbacks": dimension_fallbacks,
        "generic_car_vehicle_subtype_validations": (
            generic_car_subtype_validations),
        "camera_match_geometry": {
            "maximum_projection_to_detection_area_ratio_without_iou":
                MAX_CAMERA_PROJECTION_TO_DETECTION_AREA_RATIO,
            "minimum_iou_for_larger_area_ratio":
                MIN_IOU_FOR_LARGE_CAMERA_AREA_MISMATCH,
        },
        "vehicle_size_classification": {
            "minimum_lidar_frames": MIN_VEHICLE_SIZE_LIDAR_FRAMES,
            "dimension_order": "length_width_height_m",
            "relabelled_tracks": dict(size_relabels),
            "tracks_with_overlapping_size_profiles": size_ambiguous,
            "tracks_with_unstable_dimensions_skipped": size_unstable,
        },
        "relabeled": dict(relabeled), "heading_flips_corrected": heading_flips,
        "heading_smoothing_applied": args.smooth_headings,
        "camera_time_compensation": {
            "enabled": not args.no_camera_time_compensation,
            "camera_time_offset_ms": args.camera_time_offset_ms,
            "maximum_camera_lidar_delta_ms": args.camera_time_max_offset_ms,
            "maximum_pose_bracket_seconds": args.camera_time_max_pose_gap_s,
            "statuses_by_frame_camera": motion.status_counts,
            "camera_timestamp_kind": "recorder_packet_not_verified_exposure",
            "object_motion_model": "constant_world_xy_velocity",
            "untracked_object_motion": "assumed_static",
            "point_cloud_motion_model": "stationary_scene_ego_warp_only",
        },
        "camera_track_hold": {
            "enabled": args.camera_track_hold,
            "tracker_filter": tracker_filter,
            "minimum_projection_overlap": args.camera_hold_min_overlap,
            "minimum_projection_overlap_without_lidar": (
                args.camera_hold_no_lidar_min_overlap),
            "minimum_lidar_support_points": args.camera_hold_min_lidar_points,
            "maximum_lidar_cluster_residual_m": (
                args.camera_hold_max_lidar_residual),
            "maximum_position_correction_m": (
                args.camera_hold_max_position_correction),
            "maximum_prediction_step_seconds": args.camera_hold_max_dt,
            "maximum_prediction_speed_mps": args.camera_hold_max_speed,
            "protected_ego_footprint_m": {
                "front": args.ego_protected_front,
                "rear": args.ego_protected_rear,
                "half_width": args.ego_protected_half_width,
                "maximum_allowed_overlap_m2": (
                    args.camera_hold_max_ego_overlap),
            },
            "rejected_ego_footprint_overlap": (
                camera_holds_rejected_ego_overlap),
            "held_annotations": sum(camera_held_by_class.values()),
            "held_by_class": dict(camera_held_by_class),
            "prediction_sources": dict(camera_hold_prediction_sources),
            "maximum_continuous_hold_frames": maximum_camera_hold_age,
            "held_with_lidar_support": camera_held_with_lidar,
            "held_without_lidar_support": camera_held_without_lidar,
            "lidar_support_points": ({
                "minimum": int(np.min(camera_hold_lidar_points)),
                "mean": float(np.mean(camera_hold_lidar_points)),
                "median": float(np.median(camera_hold_lidar_points)),
            } if camera_hold_lidar_points else {}),
            "lidar_cluster_residual_m": ({
                "mean": float(np.mean(camera_hold_lidar_residuals)),
                "median": float(np.median(camera_hold_lidar_residuals)),
                "maximum": float(np.max(camera_hold_lidar_residuals)),
            } if camera_hold_lidar_residuals else {}),
            "position_correction_m": ({
                "mean": float(np.mean(camera_hold_position_corrections)),
                "median": float(np.median(camera_hold_position_corrections)),
                "maximum": float(np.max(camera_hold_position_corrections)),
            } if camera_hold_position_corrections else {}),
            "projection_overlap": ({
                "minimum": float(np.min(camera_hold_overlaps)),
                "mean": float(np.mean(camera_hold_overlaps)),
                "median": float(np.median(camera_hold_overlaps)),
            } if camera_hold_overlaps else {}),
        },
        "camera_additions_enabled": args.add_camera_objects,
        "added_from_yolo_lidar": len(additions),
        "added_by_class": dict(Counter(item["class"] for item in additions)),
        "final_annotations": sum(len(frame["objects"]) for frame in results),
        "final_classes": dict(Counter(obj["name"] for frame in results
                                      for obj in frame["objects"])),
    }
    atomic_json({"summary": report, "additions": additions},
                output.with_name(f"{output.stem}_report.json"))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
