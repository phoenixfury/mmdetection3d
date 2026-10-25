#!/usr/bin/env python3
"""Post-process DetZero tracks with fisheye YOLO and LiDAR evidence."""

from __future__ import annotations

import argparse
import csv
import json
import math
import pickle
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

from run_detzero_sequence import read_binary_pcd


CAMERAS = {
    "ARC_F": "arc_f_native", "ARC_SL": "arc_sl_native",
    "ARC_SR": "arc_sr_native", "ARC_R": "arc_r_native",
}
YOLO_CLASSES = {0: "Pedestrian", 1: "Cyclist", 2: "Car",
                3: "Cyclist", 5: "Bus", 7: "Truck"}
VEHICLES = {"Car", "Van", "Truck", "Bus"}
ALLOWED_RELABELS = {
    "Car": VEHICLES, "Van": VEHICLES, "Truck": VEHICLES, "Bus": VEHICLES,
    "Pedestrian": {"Pedestrian", "Cyclist"},
    "Cyclist": {"Pedestrian", "Cyclist"},
}
MAX_DIMS = {
    "Car": np.array([6.2, 2.7, 2.6]),
    "Van": np.array([7.5, 2.8, 3.4]),
    "Truck": np.array([15.0, 3.5, 4.5]),
    "Bus": np.array([15.0, 3.5, 4.5]),
    "Pedestrian": np.array([1.2, 1.2, 2.5]),
    "Cyclist": np.array([3.0, 1.5, 2.5]),
}
CANONICAL_DIMS = {
    "Car": np.array([4.5, 1.85, 1.55]),
    "Van": np.array([5.5, 2.1, 2.4]),
    "Truck": np.array([7.0, 2.5, 3.0]),
    "Bus": np.array([11.0, 2.7, 3.4]),
    "Pedestrian": np.array([0.6, 0.6, 1.75]),
    "Cyclist": np.array([2.0, 0.8, 1.7]),
}
# Reject a complete track only for a clearly impossible size.  Exact class
# envelopes remain in use for dimension repair/reclassification, while this
# tolerance prevents centimetre-scale detector noise from deleting a bus or
# lorry for its entire lifetime.
TRACK_REJECTION_DIMENSION_FACTOR = 1.15


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
    parser.add_argument("--yolo-only", action="store_true")
    parser.add_argument("--max-yolo-jobs", type=int,
                        help="process at most this many uncached images")
    parser.add_argument("--keep-unconfirmed", action="store_true")
    parser.add_argument(
        "--camera-confirm-classes", nargs="+",
        choices=sorted(MAX_DIMS), default=["Pedestrian", "Cyclist"],
        help="classes that require a compatible camera match; default: "
             "Pedestrian Cyclist")
    additions = parser.add_mutually_exclusive_group()
    additions.add_argument(
        "--add-camera-objects", action="store_true",
        help="add unmatched YOLO objects using LiDAR clustering (off by default)")
    additions.add_argument(
        "--no-additions", dest="add_camera_objects", action="store_false",
        help=argparse.SUPPRESS)
    parser.set_defaults(add_camera_objects=False)
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


def run_yolo(rows: list[dict], sequence: Path, args: argparse.Namespace,
             cache_path: Path) -> dict:
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    jobs = []
    for frame_index, row in enumerate(rows):
        for camera, column in CAMERAS.items():
            key = f"{frame_index}:{camera}"
            if key not in cache:
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
        violates_limit = any(
            item_name in MAX_DIMS and np.any(
                value > MAX_DIMS[item_name] * TRACK_REJECTION_DIMENSION_FACTOR)
            for value, item_name in zip(values_array, names[track_id])
        )
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
                models: dict, transforms: dict, max_distance: float) -> tuple[dict, set]:
    evidence = {}
    candidates = []
    for object_index, obj in enumerate(frame["objects"]):
        box = np.asarray(obj["box"], dtype=float)
        if np.linalg.norm(box[:2]) > max_distance:
            continue
        visible = []
        for camera in CAMERAS:
            projection = projected_box(box, transforms[camera], models[camera])
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
                score = max(iou(projection, target),
                            overlap_detection(projection, target))
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
            {"match": detection, "camera": camera, "overlap": float(score)})
    return evidence, used


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
        camera_xyz = (
            np.column_stack((xyz, np.ones(len(xyz)))) @ transform.T)[:, :3]
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


def smooth_track_headings(results: list[dict], poses: np.ndarray) -> int:
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
        elapsed = max(1., (items[-1][0] - items[0][0]) / 2.)
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


def main() -> None:
    args = parse_args()
    sequence = args.sequence.resolve()
    output = (args.output or sequence / "detzero_output" /
              "improved_results.json").resolve()
    results_path = (args.results or sequence / "detzero_output" /
                    "results.json").resolve()
    calibration_path = (args.calibration or sequence.parent /
                        "calibration.json").resolve()
    cache_path = (args.cache or output.with_name("yolo_detections.json")).resolve()
    rows = load_rows(sequence, args.limit)
    results = json.loads(results_path.read_text())[:len(rows)]
    poses = np.load(sequence / "odometry_poses.npy")[:len(rows)]
    models, transforms = calibration_models(calibration_path)
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
    dimension_fallbacks = repair_refiner_dimensions(results, tracks_path)
    invalid_tracks, invalid_details = invalid_dimension_tracks(results)
    class_votes = defaultdict(Counter)
    generic_car_subtype_validations = 0
    confirmation_classes = set(args.camera_confirm_classes)
    removed_range = removed_unconfirmed = removed_dimensions = 0
    additions = []
    next_track_id = 1 + max((int(obj["track_id"]) for frame in results
                             for obj in frame["objects"]), default=0)
    previous_additions = []
    for frame_index, (row, frame) in enumerate(zip(rows, results)):
        evidence, used = match_frame(frame, frame_index, detections, models,
                                     transforms, args.max_distance)
        kept = []
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
                  distance <= args.confirm_distance):
                removed_unconfirmed += 1
                continue
            kept.append(obj)
        frame["objects"] = kept

        if args.add_camera_objects:
            frame_additions = []
            used_addition_track_ids = set()
            projection_cache = {}
            for camera in CAMERAS:
                for detection_index, detection in enumerate(
                        detections.get(f"{frame_index}:{camera}", [])):
                    if ((camera, detection_index) in used or
                            detection["confidence"] < args.add_confidence):
                        continue
                    candidate = lidar_candidate(
                        sequence, row, camera, detection, models[camera],
                        transforms[camera], args.max_distance,
                        projection_cache)
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
                           "source": "yolo_lidar", "lidar_support": support}
                    frame["objects"].append(obj)
                    frame_additions.append({"global": global_center, "class": name,
                                            "track_id": track_id})
                    additions.append({"frame_id": frame_index, "camera": camera,
                                      "track_id": track_id, "class": name,
                                      "confidence": detection["confidence"],
                                      "lidar_support": support})
            previous_additions = frame_additions

    track_dimensions = defaultdict(list)
    track_objects = defaultdict(list)
    for frame in results:
        for obj in frame["objects"]:
            track_id = int(obj["track_id"])
            track_dimensions[track_id].append(
                np.asarray(obj["box"], float)[3:6])
            track_objects[track_id].append(obj)
    track_median_dimensions = {
        track_id: np.median(values, axis=0)
        for track_id, values in track_dimensions.items()
    }

    relabeled = Counter()
    for track_id, objects in track_objects.items():
        current_name = Counter(obj["name"] for obj in objects).most_common(1)[0][0]
        dimensions = track_median_dimensions[track_id]
        if current_name in VEHICLES:
            # Preserve the tracker's per-observation LiDAR subtype. Applying
            # another whole-track majority here recreates the same ID-switch
            # class regression already avoided in stabilize_tracks().
            for obj in objects:
                object_name = obj["name"]
                if np.any(np.asarray(obj["box"], float)[3:6] >
                          MAX_DIMS[object_name]):
                    replacement = (dimensions if np.all(
                        dimensions <= MAX_DIMS[object_name]) else
                        CANONICAL_DIMS[object_name])
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
        # Do not invent Van/Truck labels from dimensions. The reviewed data
        # has no Van taxonomy (many apparent vans are ConstructionVehicle),
        # and CenterPoint's track-level Car/Truck/Bus vote is more accurate.
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
    heading_flips = smooth_track_headings(results, poses) if args.smooth_headings else 0
    output.parent.mkdir(parents=True, exist_ok=True)
    atomic_json(results, output)
    report = {
        "input": str(results_path), "output": str(output), "frames": len(results),
        "max_distance_m": args.max_distance,
        "camera_confirmation_required_classes": sorted(confirmation_classes),
        "removed": {"over_distance": removed_range,
                    "invalid_track_dimensions": removed_dimensions,
                    "camera_unconfirmed": removed_unconfirmed},
        "invalid_dimension_tracks": invalid_details,
        "refiner_dimension_fallbacks": dimension_fallbacks,
        "generic_car_vehicle_subtype_validations": (
            generic_car_subtype_validations),
        "relabeled": dict(relabeled), "heading_flips_corrected": heading_flips,
        "heading_smoothing_applied": args.smooth_headings,
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
