#!/usr/bin/env python3
"""Append/replace synchronized bags without rebuilding existing image assets.

The canonical, ROSA and Fisheye3DOD PKLs are republished together. Samples
are ordered by LiDAR *acquisition* time, with a chronological tail validation
split and a purge gap. Old sequences retain their recorded synchronization
clock, including valid read-time/recorder-clock exports.
"""

from __future__ import annotations

import argparse
from bisect import bisect_left
import copy
import csv
import json
import os
import re
import shlex
import sys
from collections import Counter, defaultdict
from pathlib import Path

import mmengine
import numpy as np
from ocamcamera import OcamCamera

from build_improved_detzero_rosa_dataset import (
    CAMERAS, CATEGORIES, CLASSES, CLASS_MAP, DEFAULT_OCAM_CALIBRATION,
    DEFAULT_OUTPUT, build_rosa_anchor, materialize, sha256,
)

# The scripts live together inside integration_scripts/ in both standalone and
# mmdetection3d-installed layouts. The repository root is configurable.
ROOT = Path(os.environ["MMDET3D_ROOT"]).expanduser().resolve() if os.environ.get("MMDET3D_ROOT") else next(
    (parent for parent in Path(__file__).resolve().parents
     if (parent / "tools" / "custom_data").is_dir()),
    Path(__file__).resolve().parent.parent)
sys.path.insert(0, str(Path(__file__).resolve().parent))
from temporal_annotations import load_ego_poses, validate_pose  # noqa: E402

RECORDER_CAMERAS = ("ARC_F", "ARC_R", "ARC_SL", "ARC_SR")


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dataset", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--add-sequence", action="append", nargs=2,
                        metavar=("NAME", "SYNCHRONIZED_DATASET"), default=[])
    parser.add_argument("--replace-sequence", action="append", nargs=2,
                        metavar=("NAME", "SYNCHRONIZED_DATASET"), default=[])
    parser.add_argument("--remove-sequence", action="append", default=[],
                        metavar="NAME",
                        help="remove a sequence from annotations and recompute splits")
    parser.add_argument("--ocam-calibration", type=Path,
                        default=DEFAULT_OCAM_CALIBRATION)
    parser.add_argument("--source-calibration", type=Path,
                        help="sensor calibration JSON for incoming recordings")
    parser.add_argument("--storage", choices=("hardlink", "copy", "symlink"),
                        default="hardlink")
    parser.add_argument("--val-fraction", type=float, default=.1)
    parser.add_argument("--purge-seconds", type=float, default=5.)
    parser.add_argument("--max-scene-gap-seconds", type=float, default=4.)
    parser.add_argument("--refit-anchors", action="store_true",
                        help="refit ROSA anchors from training boxes; automatic on split migration/replacement")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def source_metadata(sequence: Path, calibration: Path | None = None) -> tuple[dict[str, tuple[int, np.ndarray]], int]:
    with (sequence / "manifest.csv").open(newline="") as stream:
        rows = list(csv.DictReader(stream))
    if not rows or any(manifest_clock(row) not in ("sensor", "read") for row in rows):
        raise ValueError(f"{sequence}: unsupported camera synchronization clock")
    _, lidar_poses, _ = load_ego_poses(sequence, rows, calibration)
    metadata = {}
    for row, pose in zip(rows, lidar_poses):
        token = str(row["timestamp_ns"])
        clock = manifest_clock(row)
        order_clock = row.get("lidar_order_clock") or clock
        acquisition = int(row["lidar_sensor_timestamp_ns"] if order_clock == "sensor"
                          else row.get("camera_sync_target_ns") or row["timestamp_ns"])
        target = int(row.get("camera_sync_target_ns") or
                     (row["lidar_sensor_timestamp_ns"] if clock == "sensor"
                      else row["timestamp_ns"]))
        if target <= 0:
            raise ValueError(f"{sequence}: invalid camera synchronization target")
        if token in metadata or acquisition <= 0:
            raise ValueError(f"{sequence}: duplicate token or invalid acquisition time")
        metadata[token] = (acquisition, pose)
    if len({value[0] for value in metadata.values()}) != len(metadata):
        raise ValueError(f"{sequence}: duplicate LiDAR acquisition timestamp")
    return metadata, len(rows)


def manifest_clock(row: dict[str, str]) -> str | None:
    """Recover the proven sensor clock used by early recorder manifests."""
    clock = row.get("camera_sync_clock")
    if clock in ("sensor", "read"):
        return clock
    if (row.get("lidar_sensor_timestamp_ns") not in (None, "") and
            all(row.get(camera.lower() + "_delta_ns") not in (None, "")
                for camera in RECORDER_CAMERAS)):
        return "sensor"
    return None


def attach_temporal(sample: dict, metadata: dict) -> None:
    token = str(sample["token"])
    if token not in metadata:
        raise ValueError(f"missing manifest token {token}")
    acquisition, pose = metadata[token]
    sample.setdefault("metainfo", {}).update({
        "acquisition_timestamp_ns": acquisition,
        "record_timestamp_ns": int(token),
        "T_world_CLIDAR": pose.astype(np.float64),
    })


def load_preparation(name: str, sequence: Path) -> dict:
    """Report the exact preparation command instead of a missing-file traceback."""
    template_root = sequence / "fisheye3dod_inference_2hz"
    required = (template_root / "preparation.json",
                template_root / "fisheye3dod_infos_test.pkl")
    missing = [path for path in required if not path.is_file()]
    if missing:
        calibration = next((path for path in (sequence / "calibration.json",
                            sequence.parent / "calibration.json") if path.is_file()),
                           sequence.parent / "calibration.json")
        command = shlex.join([
            "conda", "run", "--no-capture-output", "-n", "openmmlab", "python",
            str(ROOT / "tools/custom_data/prepare_fisheye3dod_inference.py"),
            str(sequence), str(calibration), "--sync", str(sequence / "manifest.csv"),
            "--output", str(template_root), "--sample-rate", "0",
        ])
        raise FileNotFoundError(
            f"{name}: missing prepared native-camera export: "
            f"{', '.join(str(path) for path in missing)}\n"
            f"Generate it from this sensor-synchronized dataset before importing:\n{command}\n"
            "Do not copy the old read-time-synchronized export into this directory.")
    preparation = json.loads(required[0].read_text())
    if preparation.get("image_mode") not in ("native", "canonical-ocam"):
        raise ValueError(f"{name}: unsupported prepared image mode")
    return preparation


def load_sequence(name: str, sequence: Path, output: Path, ocam: OcamCamera,
                  storage: str, dry_run: bool, next_id: int,
                  replacing: bool, source_calibration: Path | None = None) -> tuple[list[dict], dict, int]:
    annotation = sequence / "detzero_output/improved_results.json"
    template_root = sequence / "fisheye3dod_inference_2hz"
    template_path = template_root / "fisheye3dod_infos_test.pkl"
    preparation = load_preparation(name, sequence)
    frames = json.loads(annotation.read_text())
    templates = mmengine.load(template_path)["data_list"]
    by_token = {str(item["token"]): item for item in templates}
    if len(by_token) != len(templates):
        raise ValueError(f"{name}: duplicate template token")
    metadata, manifest_count = source_metadata(sequence, source_calibration)
    with (sequence / "manifest.csv").open(newline="") as stream:
        manifest_rows = list(csv.DictReader(stream))
    order_clock_by_token = {str(row["timestamp_ns"]):
                            row.get("lidar_order_clock") or manifest_clock(row)
                            for row in manifest_rows}
    if len(frames) != manifest_count or len(frames) != len(templates):
        raise ValueError(f"{name}: annotation, manifest and OCam frame counts differ")
    tokens = [str(frame["timestamp_ns"]) for frame in frames]
    if set(tokens) != set(by_token) or set(tokens) != set(metadata):
        raise ValueError(f"{name}: annotation, manifest and OCam tokens differ")
    if len(set(tokens)) != len(tokens):
        raise ValueError(f"{name}: duplicate annotation token")

    votes = defaultdict(Counter)
    scores = defaultdict(Counter)
    for frame in frames:
        for obj in frame["objects"]:
            cls = CLASS_MAP.get(str(obj["name"]))
            if cls is None:
                raise ValueError(f"{name}: unsupported class {obj['name']}")
            track = int(obj["track_id"])
            votes[track][cls] += 1
            scores[track][cls] += float(obj.get("score", 0.))
    classes = {track: max(CLASSES, key=lambda cls: (
        counts[cls], scores[track][cls], -CATEGORIES[cls]))
        for track, counts in votes.items()}
    ids = {track: next_id + idx for idx, track in enumerate(sorted(votes))}
    samples = []
    counts = Counter()
    sources = Counter()
    asset_prefix = f"{name}__sensor" if replacing else name
    for index, frame in enumerate(frames):
        if int(frame["frame_id"]) != index:
            raise ValueError(f"{name}: unexpected frame_id at index {index}")
        token = str(frame["timestamp_ns"])
        frame_time = (int(frame.get("lidar_sensor_timestamp_ns", -1))
                      if order_clock_by_token.get(token) == "sensor"
                      else int(frame.get("timestamp_ns", -1)))
        if frame_time != metadata[token][0]:
            raise ValueError(f"{name} frame {index}: stale annotation acquisition timestamp")
        sample = copy.deepcopy(by_token[token])
        sample["token"] = token
        sample.setdefault("metainfo", {})["sequence_id"] = name
        sample["metainfo"]["source_frame_id"] = index
        attach_temporal(sample, metadata)
        for camera in CAMERAS:
            info = sample["cam_info"]["cam_fisheye"][camera]
            model = info.get("cam2img")
            image_size = tuple(info.get("image_size", ()))
            if preparation["image_mode"] == "native":
                matrix = np.asarray(model)
                if (len(image_size) != 2 or min(image_size) <= 0 or
                        matrix.shape != (4, 4) or not np.isfinite(matrix).all()):
                    raise ValueError(f"{name}: {camera} has invalid native calibration")
            elif (ocam is None or image_size != (800, 800) or
                  not isinstance(model, OcamCamera) or any(
                      not np.allclose(value, vars(model).get(key, np.nan))
                      for key, value in vars(ocam).items())):
                raise ValueError(f"{name}: {camera} does not match the combined OCam model")
            image = template_root / str(info["cam_path"])
            if not image.is_file():
                raise FileNotFoundError(image)
            relative = Path("images") / camera / f"{asset_prefix}_{token}.jpg"
            info["cam_path"] = relative.as_posix()
            if preparation["image_mode"] == "canonical-ocam":
                info["image_size"] = (800, 800)
                info["cam2img"] = copy.deepcopy(ocam)
            if not dry_run:
                materialize(image, output / relative, storage, overwrite=False)
        boxes, names, instance_ids = [], [], []
        frame_tracks = set()
        for obj in frame["objects"]:
            track = int(obj["track_id"])
            if track in frame_tracks:
                raise ValueError(f"{name} frame {index}: duplicate track {track}")
            frame_tracks.add(track)
            box = np.asarray(obj["box"], dtype=np.float32)
            if box.shape != (7,) or not np.isfinite(box).all() or np.any(box[3:6] <= 0):
                raise ValueError(f"{name} frame {index}: invalid 3D box")
            cls = classes[track]
            boxes.append(box)
            names.append(cls)
            instance_ids.append(ids[track])
            counts[cls] += 1
            sources[str(obj.get("source", "lidar"))] += 1
        sample["annos"] = {
            "gt_boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
            "gt_names": np.asarray(names),
            "gt_mesh": np.asarray([f"detzero.{cls.lower()}" for cls in names]),
            "gt_ids": np.asarray(instance_ids, dtype=np.int64),
        }
        samples.append(sample)
    with (sequence / "manifest.csv").open(newline="") as stream:
        sync_rows = list(csv.DictReader(stream))
    uses_read_clock = any(manifest_clock(row) == "read" for row in sync_rows)
    uses_sensor_order = all((row.get("lidar_order_clock") or
                             manifest_clock(row)) == "sensor"
                            for row in sync_rows)
    summary = {
        "name": name, "dataset": str(sequence),
        "annotation_file": str(annotation), "annotation_sha256": sha256(annotation),
        "template_info": str(template_path), "frames": len(samples),
        "boxes": sum(counts.values()), "class_counts": dict(counts),
        "source_counts": dict(sources),
        "image_mode": preparation["image_mode"],
        "camera_sync_clock": "read" if uses_read_clock else "sensor",
        "temporal_clock": ("lidar_sensor_timestamp_ns" if uses_sensor_order
                           else "camera_sync_target_ns"),
    }
    return samples, summary, next_id + len(ids)


def assign_chronological_split(samples: list[dict], fraction: float,
                               purge_seconds: float,
                               max_gap_seconds: float = 4.,
                               membership: dict[str, str] | None = None) -> tuple[list[dict], list[dict], int]:
    if membership is not None:
        if set(membership) != {str(sample['token']) for sample in samples}:
            raise ValueError('Explicit split membership must cover exactly all sample tokens')
        if set(membership.values()) - {'train','gap','val'}:
            raise ValueError('Unknown explicit split partition')
    grouped = defaultdict(list)
    for sample in samples:
        grouped[str(sample["metainfo"]["sequence_id"])].append(sample)
    train, val = [], []
    purged = 0
    for name, group in grouped.items():
        group.sort(key=lambda item: int(item["metainfo"]["acquisition_timestamp_ns"]))
        if len(group) < 3:
            raise ValueError(f"{name}: at least three samples needed for temporal split")
        times = [int(item["metainfo"]["acquisition_timestamp_ns"]) for item in group]
        if len(set(times)) != len(times) or times[0] <= 0:
            raise ValueError(f"{name}: duplicate/invalid acquisition timestamps")
        cutoff = times[0] + round((times[-1] - times[0]) * (1 - fraction))
        val_start = min(len(group) - 1, max(1, bisect_left(times, cutoff)))
        boundary = times[val_start]
        train_end = boundary - round(purge_seconds * 1e9)
        partitions = {"train": [], "gap": [], "val": []}
        for idx, sample in enumerate(group):
            part = (membership[str(sample['token'])] if membership is not None else
                    "val" if idx >= val_start else ("train" if times[idx] < train_end else "gap"))
            partitions[part].append(sample)
        if membership is not None:
            rank={'train':0,'gap':1,'val':2}
            order=[rank[membership[str(sample['token'])]] for sample in group]
            if order != sorted(order):
                raise ValueError(f'{name}: explicit split is not a chronological tail holdout')
        if not partitions["train"] or not partitions["val"]:
            raise ValueError(f"{name}: empty train/val after purge gap")
        for part, members in partitions.items():
            chunks = []
            for sample in members:
                if not chunks or (
                        sample["metainfo"]["acquisition_timestamp_ns"] -
                        chunks[-1][-1]["metainfo"]["acquisition_timestamp_ns"]
                        > round(max_gap_seconds * 1e9)):
                    chunks.append([])
                chunks[-1].append(sample)
            for segment, chunk in enumerate(chunks):
                for idx, sample in enumerate(chunk):
                    sample["metainfo"].update({
                        "scene_name": f"{name}_{part}_{segment:03d}",
                        "temporal_split": part, "frame_id": str(idx),
                        "prev_id": str(chunk[idx-1]["token"]) if idx else "",
                        "next_id": str(chunk[idx+1]["token"]) if idx+1 < len(chunk) else "",
                    })
        train.extend(partitions["train"])
        val.extend(partitions["val"])
        purged += len(partitions["gap"])
    key = lambda item: int(item["metainfo"]["acquisition_timestamp_ns"])
    samples.sort(key=key)
    train.sort(key=key)
    val.sort(key=key)
    for idx, sample in enumerate(samples):
        sample["sample_idx"] = idx
    return train, val, purged


def publish_pickle(path: Path, data: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    try:
        mmengine.dump(data, str(tmp), file_format="pkl")
        os.replace(tmp, path)
    finally:
        tmp.unlink(missing_ok=True)


def publish_view(source: Path, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_name(destination.name + ".tmp")
    try:
        tmp.unlink(missing_ok=True)
        os.link(source, tmp)
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)


def main() -> None:
    args = arguments()
    if not args.add_sequence and not args.replace_sequence and not args.remove_sequence:
        raise ValueError("provide --add-sequence, --replace-sequence, or --remove-sequence")
    if (not 0 < args.val_fraction < 1 or args.purge_seconds < 0 or
            args.max_scene_gap_seconds <= 0):
        raise ValueError("invalid validation fraction or purge seconds")
    output = args.dataset.expanduser().resolve()
    summary_path = output / "summary.json"
    summary = json.loads(summary_path.read_text())
    previous_split = summary.get("split", {})
    all_path = output / "annotations/fisheye3dod_infos_all.pkl"
    recorded_hash = summary.get("annotation_sha256", {}).get("all")
    if recorded_hash and sha256(all_path) != recorded_hash:
        raise ValueError("canonical all PKL differs from summary; restore a consistent dataset before updating")
    info = mmengine.load(all_path)
    samples = info["data_list"]
    old_sequences = {entry["name"]: entry for entry in summary["sequences"]}
    sample_sequences = {str(item["metainfo"]["sequence_id"]) for item in samples}
    if (len(old_sequences) != len(summary["sequences"]) or
            sample_sequences != set(old_sequences)):
        raise ValueError("summary sequence names do not match the canonical all PKL")
    additions = [(name, Path(path).expanduser().resolve(), False)
                 for name, path in args.add_sequence]
    replacements = [(name, Path(path).expanduser().resolve(), True)
                    for name, path in args.replace_sequence]
    operations = additions + replacements
    names = [name for name, _, _ in operations] + list(args.remove_sequence)
    if any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]*", name) for name in names):
        raise ValueError("sequence names may contain only letters, numbers, '_' and '-'")
    if len(names) != len(set(names)):
        raise ValueError("duplicate sequence name in request")
    for name, _, replacing in operations:
        if (name in old_sequences) != replacing:
            raise ValueError(f"{name}: use {'--replace-sequence' if name in old_sequences else '--add-sequence'}")
    missing_removals = set(args.remove_sequence) - set(old_sequences)
    if missing_removals:
        raise ValueError("unknown sequence(s) to remove: " + ", ".join(sorted(missing_removals)))
    # Check every requested source before materializing any new images.
    image_mode = summary.get("image_mode", "canonical-ocam")
    for name, source, _ in operations:
        incoming_mode = load_preparation(name, source)["image_mode"]
        if incoming_mode != image_mode:
            raise ValueError(
                f"{name}: image mode {incoming_mode} differs from dataset mode {image_mode}")
    retained_names = (set(old_sequences) -
                      {name for name, _, replacing in replacements} -
                      set(args.remove_sequence))
    samples = [sample for sample in samples
               if str(sample["metainfo"]["sequence_id"]) in retained_names]
    for name in retained_names:
        members = [sample for sample in samples
                   if sample["metainfo"]["sequence_id"] == name]
        if (old_sequences[name].get("camera_sync_clock") == "sensor" and
                all("acquisition_timestamp_ns" in item["metainfo"] and
                    "T_world_CLIDAR" in item["metainfo"] for item in members)):
            for item in members:
                pose = np.asarray(item["metainfo"]["T_world_CLIDAR"])
                validate_pose(pose, f"{name}: stored LiDAR pose")
            continue
        source = Path(old_sequences[name]["dataset"])
        metadata, _ = source_metadata(source)
        for item in members:
            attach_temporal(item, metadata)
        # Preserve the clock recorded at import time. Older combined datasets
        # do not carry per-sample clock metadata, so inferring it from members
        # can silently convert a valid read-clock sequence back to sensor time.
        clock = old_sequences[name].get("camera_sync_clock", "sensor")
        old_sequences[name].update(
            camera_sync_clock=clock,
            temporal_clock=("camera_sync_target_ns" if clock == "read"
                            else "lidar_sensor_timestamp_ns"))

    calibration = args.ocam_calibration.expanduser().resolve()
    ocam = None
    if image_mode == "canonical-ocam":
        expected = summary.get("ocam_calibration_sha256")
        if expected and sha256(calibration) != expected:
            raise ValueError("OCam calibration differs from existing combined dataset")
        ocam = OcamCamera(filename=str(calibration), fov=220.)
        if (ocam.width, ocam.height) != (800, 800):
            raise ValueError("expected 800x800 OCam calibration")
    next_id = max((int(item) for sample in info["data_list"]
                   for item in sample["annos"]["gt_ids"]), default=-1) + 1
    new_summaries = []
    for name, source, replacing in operations:
        new, entry, next_id = load_sequence(
            name, source, output, ocam, args.storage, args.dry_run,
            next_id, replacing, args.source_calibration)
        samples.extend(new)
        new_summaries.append(entry)
    if len({str(sample["token"]) for sample in samples}) != len(samples):
        raise ValueError("duplicate recorder tokens across sequences")
    train, val, purged = assign_chronological_split(
        samples, args.val_fraction, args.purge_seconds, args.max_scene_gap_seconds)
    refit_anchors = (args.refit_anchors or bool(replacements) or
                     bool(args.remove_sequence) or
                     previous_split.get("method") != "per-sequence chronological tail holdout" or
                     previous_split.get("fraction_basis") != "elapsed acquisition time within each BAG" or
                     previous_split.get("val_fraction") != args.val_fraction or
                     previous_split.get("purge_seconds") != args.purge_seconds)
    summary["rosa_anchor"]["update"] = "refit from training split" if refit_anchors else "preserved"
    counts = Counter(str(cls) for sample in samples for cls in sample["annos"]["gt_names"])
    sources = Counter()
    entries = [old_sequences[name] for name in old_sequences if name in retained_names] + new_summaries
    for entry in entries:
        sources.update(entry.get("source_counts", {}))
        members = [item for item in samples
                   if item["metainfo"]["sequence_id"] == entry["name"]]
        by_split = defaultdict(list)
        for item in members:
            by_split[item["metainfo"]["temporal_split"]].append(
                int(item["metainfo"]["acquisition_timestamp_ns"]))
        entry["temporal_split"] = {
            split: {"frames": len(times), "first_acquisition_ns": min(times),
                    "last_acquisition_ns": max(times)}
            for split, times in by_split.items()
        }
    summary.update({
        "output": str(output), "dry_run": args.dry_run, "storage": args.storage,
        "image_mode": image_mode,
        "ocam_calibration": (str(calibration) if image_mode == "canonical-ocam" else None),
        "frames": len(samples), "train_frames": len(train),
        "val_frames": len(val), "purged_frames": purged,
        "empty_frames": sum(not len(item["annos"]["gt_names"]) for item in samples),
        "boxes": sum(counts.values()), "class_counts": dict(counts),
        "source_counts": dict(sources),
        "instances": len({int(item) for sample in samples
                          for item in sample["annos"]["gt_ids"]}),
        "split": {"method": "per-sequence chronological tail holdout",
                  "clock": "lidar_sensor_timestamp_ns",
                  "val_fraction": args.val_fraction,
                  "purge_seconds": args.purge_seconds,
                  "fraction_basis": "elapsed acquisition time within each BAG",
                  "max_scene_gap_seconds": args.max_scene_gap_seconds,
                  "scene_links_cross_split": False},
        "sequences": entries,
        "temporal_metadata": "acquisition_timestamp_ns and T_world_CLIDAR",
    })
    if args.dry_run:
        print(json.dumps(summary, indent=2))
        return
    if refit_anchors:
        anchor_info = summary["rosa_anchor"]
        anchor, training_boxes = build_rosa_anchor(
            train, int(anchor_info["shape"][0]), int(anchor_info["seed"]))
        anchor_path = output / anchor_info["path"]
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        tmp_anchor = anchor_path.with_name(anchor_path.stem + ".tmp.npy")
        np.save(tmp_anchor, anchor)
        os.replace(tmp_anchor, anchor_path)
        anchor_info.update(training_boxes=training_boxes, sha256=sha256(anchor_path))
    metainfo = copy.deepcopy(info["metainfo"])
    metainfo["version"] = "improved-detzero-acquisition-temporal-v2"
    paths = {}
    for split, data in (("all", samples), ("train", train), ("val", val)):
        path = output / "annotations" / f"fisheye3dod_infos_{split}.pkl"
        publish_pickle(path, {"metainfo": metainfo, "data_list": data})
        paths[split] = path
        for directory in ("fisheye3dod", "rosa", "ImageSets-improved-detzero"):
            publish_view(path, output / directory / path.name)
    summary["annotation_sha256"] = {name: sha256(path) for name, path in paths.items()}
    summary["format_views_byte_identical"] = True
    summary["rosa_anchor"]["split_note"] = (
        "Fitted only to this chronological training split" if refit_anchors else
        "Existing training-only anchors preserved; use --refit-anchors to include new bags")
    tmp = summary_path.with_name("summary.json.tmp")
    tmp.write_text(json.dumps(summary, indent=2) + "\n")
    os.replace(tmp, summary_path)
    print(json.dumps({"frames": len(samples), "train": len(train),
                      "val": len(val), "purged": purged,
                      "sequences": [entry["name"] for entry in summary["sequences"]]}, indent=2))


if __name__ == "__main__":
    try:
        main()
    except (FileNotFoundError, ValueError) as error:
        raise SystemExit(str(error))
