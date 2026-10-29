#!/usr/bin/env python3
"""Build one shared Fisheye3DOD/ROSA dataset from improved DetZero labels.

The image assets and samples are generated once.  The Fisheye3DOD and ROSA
annotation files are byte-identical views of the same samples and therefore
cannot silently acquire different train/validation splits.
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import os
import shutil
from collections import Counter, defaultdict
from pathlib import Path

import mmengine
import numpy as np
from ocamcamera import OcamCamera


ROOT = Path(__file__).resolve().parents[2]
CUSTOM = ROOT / "data" / "Custom"
CAMERAS = (
    "fisheye_camera_front",
    "fisheye_camera_left",
    "fisheye_camera_right",
    "fisheye_camera_rear",
)
CLASSES = ("Car", "Van", "Truck", "Bus", "Pedestrian", "Cyclist")
CATEGORIES = {name: index for index, name in enumerate(CLASSES)}
CLASS_MAP = {
    "Car": "Car",
    "Van": "Van",
    "Truck": "Truck",
    "Bus": "Bus",
    "Pedestrian": "Pedestrian",
    "Cyclist": "Cyclist",
    # The reviewed custom taxonomy used ConstructionVehicle for van-like
    # commercial vehicles; ROSA/Fisheye3DOD exposes Van instead.
    "ConstructionVehicle": "Van",
}
DEFAULT_SEQUENCES = (
    ("16_09", CUSTOM / "BAG_2026_09_16_09_58_56_converted" / "dataset"),
    ("30_09", CUSTOM / "BAG_2026_09_30_10_10_54_converted" / "dataset"),
)
DEFAULT_OUTPUT = CUSTOM / "combined_improved_detzero_rosa"
DEFAULT_OCAM_CALIBRATION = ROOT / "data" / "Fisheye3DODdataset" / "calib_results.txt"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--sequence", action="append", nargs=2, metavar=("NAME", "DATASET"),
        help="repeatable sequence name and synchronized dataset directory")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument(
        "--ocam-calibration", type=Path, default=DEFAULT_OCAM_CALIBRATION,
        help="released 800x800 OCam calibration used to remap the images")
    parser.add_argument(
        "--storage", choices=("hardlink", "copy", "symlink"),
        default="hardlink")
    parser.add_argument("--val-fraction", type=float, default=.2)
    parser.add_argument("--split-block-seconds", type=float, default=40.0)
    parser.add_argument("--num-anchors", type=int, default=300)
    parser.add_argument("--anchor-seed", type=int, default=3407)
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def materialize(source: Path, destination: Path, mode: str,
                overwrite: bool) -> None:
    if destination.exists() or destination.is_symlink():
        try:
            if source.samefile(destination):
                return
        except OSError:
            pass
        if not overwrite:
            if (not destination.is_symlink() and destination.is_file() and
                    destination.stat().st_size == source.stat().st_size and
                    sha256(destination) == sha256(source)):
                return
            raise FileExistsError(f"conflicting output file: {destination}")
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    if mode == "copy":
        shutil.copy2(source, destination)
    elif mode == "symlink":
        destination.symlink_to(os.path.relpath(source, destination.parent))
    else:
        try:
            os.link(source, destination)
        except OSError:
            shutil.copy2(source, destination)


def publish_identical(source: Path, destination: Path,
                      overwrite: bool) -> None:
    """Publish an exact hardlink/copy without the asset size shortcut."""
    if destination.exists() or destination.is_symlink():
        try:
            if source.samefile(destination):
                return
        except OSError:
            pass
        if not overwrite:
            raise FileExistsError(f"conflicting output file: {destination}")
        destination.unlink()
    destination.parent.mkdir(parents=True, exist_ok=True)
    try:
        os.link(source, destination)
    except OSError:
        shutil.copy2(source, destination)


def temporal_split(samples: list[dict], val_fraction: float,
                   block_seconds: float) -> tuple[list[dict], list[dict]]:
    cycle = max(2, round(1.0 / val_fraction))
    block_ns = max(1, round(block_seconds * 1e9))
    grouped = defaultdict(list)
    for sample in samples:
        grouped[sample["metainfo"]["sequence_id"]].append(sample)
    train, val = [], []
    for sequence_samples in grouped.values():
        sequence_samples.sort(key=lambda item: int(item["token"]))
        start = int(sequence_samples[0]["token"])
        for sample in sequence_samples:
            block = (int(sample["token"]) - start) // block_ns
            (val if block % cycle == cycle - 1 else train).append(sample)
    if not train or not val:
        raise RuntimeError("temporal split produced an empty partition")
    return train, val


def validate_source(sequence_name: str, frames: list[dict],
                    templates: list[dict]) -> None:
    if len(frames) != len(templates):
        raise ValueError(
            f"{sequence_name}: annotations={len(frames)} != "
            f"templates={len(templates)}")
    for index, (frame, template) in enumerate(zip(frames, templates)):
        if int(frame["frame_id"]) != index:
            raise ValueError(
                f"{sequence_name}: frame {index} has id {frame['frame_id']}")
        if int(frame["timestamp_ns"]) != int(template["token"]):
            raise ValueError(
                f"{sequence_name}: token mismatch at frame {index}: "
                f"{frame['timestamp_ns']} != {template['token']}")


def build_rosa_anchor(samples: list[dict], num_anchors: int,
                      seed: int) -> tuple[np.ndarray, int]:
    """Fit ROSA Sparse4D anchors from this exact training split."""
    if num_anchors <= 0:
        raise ValueError("--num-anchors must be positive")
    centers = []
    for sample in samples:
        boxes = np.asarray(sample["annos"]["gt_boxes"], dtype=np.float32)
        if len(boxes):
            centers.append(boxes[:, :3])
    if not centers:
        raise RuntimeError("cannot generate ROSA anchors from an empty GT set")
    centers_array = np.concatenate(centers, axis=0)
    if len(centers_array) < num_anchors:
        raise ValueError(
            f"requested {num_anchors} anchors from only "
            f"{len(centers_array)} training boxes")

    # The openmmlab environment can contain an old sklearn with deprecated
    # NumPy aliases.  Define only the missing aliases before importing it.
    if "bool" not in np.__dict__:
        np.bool = np.bool_  # type: ignore[attr-defined]
    if "int" not in np.__dict__:
        np.int = int  # type: ignore[attr-defined]
    if "float" not in np.__dict__:
        np.float = float  # type: ignore[attr-defined]
    from sklearn.cluster import KMeans

    kmeans = KMeans(
        n_clusters=num_anchors,
        random_state=seed,
        n_init=10,
    ).fit(centers_array)
    anchor = np.zeros((num_anchors, 11), dtype=np.float32)
    anchor[:, :3] = kmeans.cluster_centers_.astype(np.float32)
    return anchor, len(centers_array)


def write_benchmark_readme(output: Path, num_anchors: int,
                           anchor_seed: int) -> None:
    anchor_name = f"kmeans{num_anchors}_seed{anchor_seed}.npy"
    text = f"""# Shared Fisheye3DOD + ROSA dataset

Both format views contain the same samples, images, boxes, IDs, and temporal
train/validation split.  Run these commands from the mmdetection3d root.

## Fisheye3DOD, two GPUs

```bash
DUAL_FISHEYE_ROOT={output} CUDA_VISIBLE_DEVICES=0,1 \\
torchrun --nproc_per_node=2 tools/train.py \\
projects/Fisheye3DOD/configs/custom_fisheye_bevdet_train_dual_export.py \\
--launcher pytorch
```

## ROSA, two GPUs

```bash
ROSA_DATA_ROOT={output} CUDA_VISIBLE_DEVICES=0,1 \\
PYTHONPATH=$PWD:$PWD/projects/ROSA:$PWD/projects/Fisheye3DOD \\
FISHEYE3DOD_OFFICIAL_ROOT=$PWD/projects/Fisheye3DOD \\
torchrun --nproc_per_node=2 projects/ROSA/tools/run.py train \\
projects/ROSA/configs/fisheye3dod/rosa_vov99_dual_export_24e.py \\
--launcher pytorch --amp
```

ROSA anchor: `rosa/anchors/{anchor_name}`
"""
    (output / "BENCHMARK.md").write_text(text)


def main() -> None:
    args = parse_args()
    if not 0 < args.val_fraction < 1:
        raise ValueError("--val-fraction must be between zero and one")
    sequences = DEFAULT_SEQUENCES if args.sequence is None else tuple(
        (name, Path(path)) for name, path in args.sequence)
    output = args.output.expanduser().resolve()
    ocam_calibration = args.ocam_calibration.expanduser().resolve()
    if not ocam_calibration.is_file():
        raise FileNotFoundError(ocam_calibration)
    ocam = OcamCamera(filename=str(ocam_calibration), fov=220.0)
    if (ocam.width, ocam.height) != (800, 800):
        raise ValueError(
            f"expected an 800x800 OCam model, got {ocam.width}x{ocam.height}")

    all_samples = []
    all_tokens = set()
    instance_ids: dict[tuple[str, int], int] = {}
    class_counts = Counter()
    source_counts = Counter()
    summaries = []
    global_index = 0

    for sequence_name, raw_root in sequences:
        sequence = raw_root.expanduser().resolve()
        annotations_path = (
            sequence / "detzero_output" / "camera_validated_results.json")
        report_path = annotations_path.with_name(
            "camera_validated_results_report.json")
        template_root = sequence / "fisheye3dod_inference_2hz"
        template_path = template_root / "fisheye3dod_infos_test.pkl"
        frames = json.loads(annotations_path.read_text())
        template_info = mmengine.load(template_path)
        raw_templates = template_info["data_list"]
        templates_by_token = {
            int(template["token"]): template for template in raw_templates
        }
        if len(templates_by_token) != len(raw_templates):
            raise ValueError(f"{sequence_name}: duplicate template tokens")
        frame_tokens = {int(frame["timestamp_ns"]) for frame in frames}
        if frame_tokens != set(templates_by_token):
            missing = sorted(frame_tokens - set(templates_by_token))[:5]
            extra = sorted(set(templates_by_token) - frame_tokens)[:5]
            raise ValueError(
                f"{sequence_name}: annotation/template token sets differ: "
                f"missing={missing}, extra={extra}")
        # Template exporters may order a few late frames by transport/read
        # order. Join by the acquisition timestamp used by the synchronized
        # annotation instead of trusting list position.
        templates = [templates_by_token[int(frame["timestamp_ns"])]
                     for frame in frames]
        validate_source(sequence_name, frames, templates)

        # A track is one physical object, so it must have one class across its
        # lifetime. Frame count is the primary vote and summed detector score
        # breaks ties. This removes isolated Car/Van/Truck/Bus class flips.
        track_votes: dict[int, Counter] = defaultdict(Counter)
        track_scores: dict[int, Counter] = defaultdict(Counter)
        for frame in frames:
            for obj in frame["objects"]:
                raw_name = str(obj["name"])
                name = CLASS_MAP.get(raw_name)
                if name is None:
                    raise ValueError(
                        f"{sequence_name}: unsupported class {raw_name!r}")
                track_id = int(obj["track_id"])
                track_votes[track_id][name] += 1
                track_scores[track_id][name] += float(obj.get("score", 0.0))
        track_classes = {
            track_id: max(
                CLASSES,
                key=lambda name: (
                    votes[name],
                    track_scores[track_id][name],
                    -CATEGORIES[name],
                ),
            )
            for track_id, votes in track_votes.items()
        }
        conflicted_tracks = sum(len(votes) > 1 for votes in track_votes.values())
        relabeled_boxes = 0

        sequence_counts = Counter()
        sequence_sources = Counter()
        for frame_index, (frame, template) in enumerate(zip(frames, templates)):
            token = str(template["token"])
            if token in all_tokens:
                raise ValueError(f"duplicate token across sequences: {token}")
            all_tokens.add(token)
            sample = copy.deepcopy(template)
            sample["sample_idx"] = global_index
            sample["token"] = token
            sample.setdefault("metainfo", {})
            sample["metainfo"].update({
                "sequence_id": sequence_name,
                "scene_name": sequence_name,
                "frame_id": str(frame_index),
                "prev_id": (str(templates[frame_index - 1]["token"])
                            if frame_index else ""),
                "next_id": (str(templates[frame_index + 1]["token"])
                            if frame_index + 1 < len(templates) else ""),
            })

            for camera in CAMERAS:
                camera_info = sample["cam_info"]["cam_fisheye"][camera]
                source = template_root / str(camera_info["cam_path"])
                if not source.is_file():
                    raise FileNotFoundError(source)
                relative = Path("images") / camera / (
                    f"{sequence_name}_{token}.jpg")
                camera_info["cam_path"] = relative.as_posix()
                camera_info["image_size"] = (800, 800)
                camera_info["cam2img"] = copy.deepcopy(ocam)
                if not args.dry_run:
                    materialize(source, output / relative, args.storage,
                                args.overwrite)

            boxes, names, ids = [], [], []
            for obj in frame["objects"]:
                raw_name = str(obj["name"])
                name = CLASS_MAP.get(raw_name)
                if name is None:
                    raise ValueError(
                        f"{sequence_name} frame {frame_index}: unsupported "
                        f"class {raw_name!r}")
                consensus_name = track_classes[int(obj["track_id"])]
                if name != consensus_name:
                    relabeled_boxes += 1
                name = consensus_name
                box = np.asarray(obj["box"], dtype=np.float32)
                if (box.shape != (7,) or not np.isfinite(box).all() or
                        np.any(box[3:6] <= 0)):
                    raise ValueError(
                        f"{sequence_name} frame {frame_index}: invalid box")
                key = (sequence_name, int(obj["track_id"]))
                if key not in instance_ids:
                    instance_ids[key] = len(instance_ids)
                boxes.append(box)
                names.append(name)
                ids.append(instance_ids[key])
                sequence_counts[name] += 1
                sequence_sources[str(obj.get("source", "lidar"))] += 1

            sample["annos"] = {
                "gt_boxes": np.asarray(boxes, dtype=np.float32).reshape(-1, 7),
                "gt_names": np.asarray(names),
                "gt_mesh": np.asarray(
                    [f"detzero.{name.lower()}" for name in names]),
                "gt_ids": np.asarray(ids, dtype=np.int64),
            }
            all_samples.append(sample)
            global_index += 1

        class_counts.update(sequence_counts)
        source_counts.update(sequence_sources)
        summaries.append({
            "name": sequence_name,
            "dataset": str(sequence),
            "annotation_file": str(annotations_path),
            "annotation_sha256": sha256(annotations_path),
            "postprocess_report": str(report_path),
            "template_info": str(template_path),
            "frames": len(frames),
            "empty_frames": sum(not frame["objects"] for frame in frames),
            "boxes": sum(sequence_counts.values()),
            "class_counts": dict(sequence_counts),
            "source_counts": dict(sequence_sources),
            "track_class_conflicts_resolved": conflicted_tracks,
            "boxes_relabelled_to_track_majority": relabeled_boxes,
        })

    train, val = temporal_split(
        all_samples, args.val_fraction, args.split_block_seconds)
    rosa_anchor, anchor_training_boxes = build_rosa_anchor(
        train, args.num_anchors, args.anchor_seed)
    metainfo = {
        "categories": CATEGORIES,
        "classes": CLASSES,
        "dataset": "combined_improved_detzero_rosa",
        "version": "improved-detzero-track-validated-v1",
    }
    summary = {
        "output": str(output),
        "dry_run": args.dry_run,
        "storage": args.storage,
        "frames": len(all_samples),
        "train_frames": len(train),
        "val_frames": len(val),
        "empty_frames": sum(not len(item["annos"]["gt_names"])
                            for item in all_samples),
        "boxes": sum(class_counts.values()),
        "instances": len(instance_ids),
        "class_counts": dict(class_counts),
        "source_counts": dict(source_counts),
        "ocam_calibration": str(ocam_calibration),
        "ocam_calibration_sha256": sha256(ocam_calibration),
        "construction_vehicle_mapping": "Van",
        "format_views": {
            "fisheye3dod": "fisheye3dod/fisheye3dod_infos_{split}.pkl",
            "rosa": "rosa/fisheye3dod_infos_{split}.pkl",
            "canonical": "annotations/fisheye3dod_infos_{split}.pkl",
        },
        "rosa_anchor": {
            "path": (
                f"rosa/anchors/kmeans{args.num_anchors}_"
                f"seed{args.anchor_seed}.npy"),
            "shape": list(rosa_anchor.shape),
            "training_boxes": anchor_training_boxes,
            "seed": args.anchor_seed,
        },
        "split": {
            "method": "per-sequence temporal blocks",
            "val_fraction": args.val_fraction,
            "block_seconds": args.split_block_seconds,
        },
        "sequences": summaries,
    }

    if not args.dry_run:
        canonical_dir = output / "annotations"
        canonical_dir.mkdir(parents=True, exist_ok=True)
        canonical_files = {}
        for name, data in (("all", all_samples), ("train", train),
                           ("val", val)):
            canonical = canonical_dir / f"fisheye3dod_infos_{name}.pkl"
            mmengine.dump({"metainfo": metainfo, "data_list": data}, canonical)
            canonical_files[name] = canonical

        # Keep explicit format directories for server jobs, plus the legacy
        # directory consumed by the already-tested ROSA configuration.
        for directory in (
                output / "fisheye3dod",
                output / "rosa",
                output / "ImageSets-improved-detzero"):
            for name, canonical in canonical_files.items():
                publish_identical(
                    canonical,
                    directory / f"fisheye3dod_infos_{name}.pkl",
                    args.overwrite)

        anchor_dir = output / "rosa" / "anchors"
        anchor_dir.mkdir(parents=True, exist_ok=True)
        anchor_path = anchor_dir / (
            f"kmeans{args.num_anchors}_seed{args.anchor_seed}.npy")
        np.save(anchor_path, rosa_anchor)
        shutil.copy2(ocam_calibration, output / "calib_results.txt")

        summary["annotation_sha256"] = {
            name: sha256(path) for name, path in canonical_files.items()
        }
        summary["rosa_anchor"]["sha256"] = sha256(anchor_path)
        summary["format_views_byte_identical"] = True
        (output / "summary.json").write_text(
            json.dumps(summary, indent=2) + "\n")
        write_benchmark_readme(
            output, args.num_anchors, args.anchor_seed)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
