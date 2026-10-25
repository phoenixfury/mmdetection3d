#!/usr/bin/env python3
"""Evaluate DetZero frame results against reviewed CaliperGT cuboids."""

from __future__ import annotations

import argparse
import json
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment


CLASS_MAP = {
    "car": "Car", "van": "Van", "truck": "Truck", "bus": "Bus",
    "construction_vehicle": "ConstructionVehicle",
    "construction vehicle": "ConstructionVehicle",
    "constructionvehicle": "ConstructionVehicle",
    "pedestrian": "Pedestrian", "cyclist": "Cyclist",
}
COMPATIBLE_CLASS_PAIRS = {("Van", "ConstructionVehicle")}


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("reference", type=Path)
    parser.add_argument("results", type=Path, nargs="+")
    parser.add_argument("--names", nargs="+")
    parser.add_argument("--max-center-distance", type=float, default=3.0)
    parser.add_argument("--max-range", type=float, default=60.0)
    parser.add_argument("--exclude-classes", nargs="+", default=[])
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def wrap(value: float) -> float:
    return (value + math.pi) % (2 * math.pi) - math.pi


def quantiles(values: list[float]) -> dict:
    if not values:
        return {}
    array = np.asarray(values, dtype=float)
    return {
        "mean": float(array.mean()), "p50": float(np.percentile(array, 50)),
        "p90": float(np.percentile(array, 90)),
        "p95": float(np.percentile(array, 95)),
    }


def reference_frames(path: Path, max_range: float,
                     excluded: set[str]) -> tuple[dict, Counter]:
    document = json.loads(path.read_text())
    frames = defaultdict(list)
    classes = Counter()
    for item in document["annotations"]:
        cube = item["cuboid_3d"]
        box = np.asarray([
            cube["position"]["x"], cube["position"]["y"],
            cube["position"]["z"], cube["dimensions"]["length"],
            cube["dimensions"]["width"], cube["dimensions"]["height"],
            cube["rotation"]["yaw"],
        ], dtype=float)
        if np.linalg.norm(box[:2]) > max_range:
            continue
        name = CLASS_MAP.get(str(item["class"]).lower(), str(item["class"]))
        if name in excluded:
            continue
        frames[int(item["frame_id"])].append((box, name))
        classes[name] += 1
    return frames, classes


def evaluate(path: Path, reference: dict, reference_classes: Counter,
             max_distance: float, max_range: float,
             excluded: set[str]) -> dict:
    predictions = json.loads(path.read_text())
    prediction_count = matched_count = class_correct = compatible_correct = 0
    prediction_classes = Counter()
    matched_prediction_classes = Counter()
    matched_reference_classes = Counter()
    matched_classes = Counter()
    center_errors, dimension_errors = [], []
    yaw_errors, axis_yaw_errors = [], []
    errors_by_reference_class = defaultdict(
        lambda: {"center": [], "dimension": [], "axis_yaw": []})
    thresholds = (0.5, 1.0, 2.0, 3.0)
    matches_at_threshold = Counter()

    for frame in predictions:
        frame_id = int(frame["frame_id"])
        predicted = []
        for item in frame["objects"]:
            box = np.asarray(item["box"], dtype=float)
            if np.linalg.norm(box[:2]) > max_range:
                continue
            name = CLASS_MAP.get(str(item["name"]).lower(), str(item["name"]))
            if name in excluded:
                continue
            predicted.append((box, name))
            prediction_classes[name] += 1
        prediction_count += len(predicted)
        target = reference.get(frame_id, [])
        if not predicted or not target:
            continue
        pred_boxes = np.stack([item[0] for item in predicted])
        target_boxes = np.stack([item[0] for item in target])
        distances = np.linalg.norm(
            pred_boxes[:, None, :3] - target_boxes[None, :, :3], axis=2)
        costs = distances.copy()
        costs[costs > max_distance] = 1e6
        rows, columns = linear_sum_assignment(costs)
        for pred_index, target_index in zip(rows, columns):
            distance = float(distances[pred_index, target_index])
            for threshold in thresholds:
                if distance <= threshold:
                    matches_at_threshold[threshold] += 1
            if distance > max_distance:
                continue
            predicted_box, predicted_name = predicted[pred_index]
            target_box, target_name = target[target_index]
            matched_count += 1
            center_errors.append(distance)
            dimension_errors.append(float(np.mean(np.abs(
                predicted_box[3:6] - target_box[3:6]))))
            yaw_error = abs(wrap(float(predicted_box[6] - target_box[6])))
            yaw_errors.append(math.degrees(yaw_error))
            axis_yaw_errors.append(math.degrees(min(
                yaw_error, abs(math.pi - yaw_error))))
            matched_classes[f"{predicted_name}->{target_name}"] += 1
            matched_prediction_classes[predicted_name] += 1
            matched_reference_classes[target_name] += 1
            errors_by_reference_class[target_name]["center"].append(distance)
            errors_by_reference_class[target_name]["dimension"].append(
                dimension_errors[-1])
            errors_by_reference_class[target_name]["axis_yaw"].append(
                axis_yaw_errors[-1])
            class_correct += predicted_name == target_name
            compatible_correct += (
                predicted_name == target_name or
                (predicted_name, target_name) in COMPATIBLE_CLASS_PAIRS)

    reference_count = sum(reference_classes.values())
    precision = matched_count / prediction_count if prediction_count else 0.0
    recall = matched_count / reference_count if reference_count else 0.0
    result = {
        "path": str(path), "max_range_m": max_range,
        "max_center_distance_m": max_distance,
        "predictions": prediction_count, "references": reference_count,
        "matched": matched_count, "precision": precision, "recall": recall,
        "f1": (2 * precision * recall / (precision + recall)
               if precision + recall else 0.0),
        "matches_by_center_threshold_m": {
            str(key): matches_at_threshold[key] for key in thresholds},
        "center_error_m": quantiles(center_errors),
        "dimension_mae_m": quantiles(dimension_errors),
        "directed_yaw_error_deg": quantiles(yaw_errors),
        "axis_yaw_error_deg": quantiles(axis_yaw_errors),
        "class_accuracy_on_matches": (
            class_correct / matched_count if matched_count else 0.0),
        "class_accuracy_with_van_construction_equivalence": (
            compatible_correct / matched_count if matched_count else 0.0),
        "prediction_classes": dict(prediction_classes),
        "unmatched_predictions_by_class": {
            name: count - matched_prediction_classes[name]
            for name, count in prediction_classes.items()},
        "unmatched_references_by_class": {
            name: count - matched_reference_classes[name]
            for name, count in reference_classes.items()},
        "errors_by_reference_class": {
            name: {
                "matched": len(values["center"]),
                "center_error_m": quantiles(values["center"]),
                "dimension_mae_m": quantiles(values["dimension"]),
                "axis_yaw_error_deg": quantiles(values["axis_yaw"]),
            }
            for name, values in errors_by_reference_class.items()
        },
        "matched_class_pairs": dict(matched_classes),
    }
    return result


def main() -> None:
    args = arguments()
    if args.names and len(args.names) != len(args.results):
        raise ValueError("--names must have one entry per results file")
    excluded = {
        CLASS_MAP.get(name.lower(), name) for name in args.exclude_classes
    }
    reference, reference_classes = reference_frames(
        args.reference.resolve(), args.max_range, excluded)
    names = args.names or [path.stem for path in args.results]
    evaluations = {
        name: evaluate(path.resolve(), reference, reference_classes,
                       args.max_center_distance, args.max_range, excluded)
        for name, path in zip(names, args.results)
    }
    report = {
        "reference": str(args.reference.resolve()),
        "reference_classes": dict(reference_classes),
        "excluded_classes": sorted(excluded),
        "evaluations": evaluations,
    }
    rendered = json.dumps(report, indent=2) + "\n"
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(rendered)
    print(rendered, end="")


if __name__ == "__main__":
    main()
