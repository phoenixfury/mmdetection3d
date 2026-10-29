#!/usr/bin/env python3
"""Render three detectors on four fisheye cameras and the LiDAR BEV."""

from __future__ import annotations

import argparse
from collections import Counter
import json
import os
from pathlib import Path
import shutil
import subprocess

os.environ.setdefault("MPLCONFIGDIR", "/tmp/three-model-benchmark-mpl")
import cv2
import mmengine
import numpy as np
from ocamcamera import OcamCamera


CAMERAS = (
    ("fisheye_camera_front", "FRONT"),
    ("fisheye_camera_left", "LEFT"),
    ("fisheye_camera_right", "RIGHT"),
    ("fisheye_camera_rear", "REAR"),
)
COLORS = {
    "Car": (255, 220, 0),
    "Van": (255, 90, 40),
    "Truck": (0, 165, 255),
    "Bus": (40, 40, 255),
    "Pedestrian": (0, 255, 120),
    "Cyclist": (255, 0, 210),
}
BOX_EDGES = ((0, 1), (0, 2), (1, 3), (2, 3),
             (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-root", type=Path, required=True)
    parser.add_argument("--info", type=Path)
    parser.add_argument("--manual", type=Path, required=True)
    parser.add_argument("--automatic", type=Path, required=True)
    parser.add_argument("--rosa", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--score-threshold", type=float, default=0.10)
    parser.add_argument("--bev-range", type=float, default=50.0)
    parser.add_argument("--point-stride", type=int, default=2)
    parser.add_argument("--fps", type=float, default=2.0)
    parser.add_argument("--start-frame", type=int, default=0)
    parser.add_argument("--end-frame", type=int)
    parser.add_argument("--frame-step", type=int, default=1)
    parser.add_argument("--no-ground-truth", action="store_true")
    parser.add_argument("--no-video", action="store_true")
    parser.add_argument("--source-dataset", action="append", default=[],
                        metavar="NAME=PATH")
    return parser.parse_args()


def parse_sources(data_root: Path, values: list[str]) -> dict[str, Path]:
    sources = {}
    summary_path = data_root / "summary.json"
    if summary_path.is_file():
        for item in json.loads(summary_path.read_text()).get("sequences", []):
            sources[str(item["name"])] = Path(item["dataset"]).expanduser()
    for value in values:
        if "=" not in value:
            raise ValueError(f"expected NAME=PATH, got {value!r}")
        name, path = value.split("=", 1)
        sources[name] = Path(path).expanduser().resolve()
    return sources


def resolve_lidar(data_root: Path, info_path: Path, sample: dict,
                  sources: dict[str, Path]) -> Path:
    relative = Path(str(sample["pc_info"]["pts_path"]))
    sequence = str(sample.get("metainfo", {}).get("sequence_id", ""))
    candidates = [data_root / relative, info_path.parent / relative,
                  data_root / "lidar_pcd" / relative.name]
    if sequence in sources:
        source = sources[sequence]
        candidates.extend((source / "lidar_pcd" / relative.name,
                           source / "fisheye3dod_inference_2hz" / relative,
                           source / relative))
    for candidate in candidates:
        if candidate.resolve().is_file():
            return candidate.resolve()
    raise FileNotFoundError(
        f"LiDAR cloud not found for {sequence} token {sample['token']}")


def load_points(path: Path) -> np.ndarray:
    payload = path.read_bytes()
    marker = b"DATA binary\n"
    offset = payload.find(marker)
    if offset >= 0:
        payload = payload[offset + len(marker):]
    values = np.frombuffer(payload, dtype="<f4")
    if values.size % 4:
        raise ValueError(f"{path}: expected float32 XYZI")
    points = values.reshape(-1, 4)
    return points[np.isfinite(points).all(axis=1)]


def detections_from_fisheye(path: Path, samples: list[dict]) -> list[list]:
    payload = mmengine.load(path)
    rows = payload["data_list"] if isinstance(payload, dict) else payload
    by_token = {str(row["token"]): row for row in rows}
    result = []
    for sample in samples:
        row = by_token[str(sample["token"])]
        annos = row["annos"]
        result.append([
            (str(name), np.asarray(box, dtype=float)[:7], float(score))
            for name, box, score in zip(
                annos["pred_names"], annos["pred_boxes"],
                annos["pred_scores"])
        ])
    return result


def detections_from_rosa(path: Path, sample_count: int) -> list[list]:
    payload = json.loads(path.read_text())
    frames = payload["frames"]
    if len(frames) != sample_count:
        raise RuntimeError(f"ROSA frame count {len(frames)} != {sample_count}")
    if payload.get('box_origin', 'bottom_center') == 'bottom_center':
        for frame in frames:
            for box in frame['boxes_3d']:
                box[2] += box[5] / 2
    elif payload['box_origin'] != 'gravity_center':
        raise ValueError('Unsupported ROSA box_origin')
    return [[
        (str(name), np.asarray(box, dtype=float)[:7], float(score))
        for name, box, score in zip(
            frame["names"], frame["boxes_3d"], frame["scores"])
    ] for frame in frames]


def corners_3d(box: np.ndarray) -> np.ndarray:
    half_l, half_w, half_h = box[3:6] / 2
    local = np.asarray([
        [sx * half_l, sy * half_w, sz * half_h]
        for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)
    ])
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.asarray([[cosine, -sine, 0],
                           [sine, cosine, 0], [0, 0, 1]])
    return local @ rotation.T + box[:3]


def corners_2d(box: np.ndarray) -> np.ndarray:
    half_l, half_w = box[3] / 2, box[4] / 2
    local = np.asarray([[half_l, half_w], [half_l, -half_w],
                        [-half_l, -half_w], [-half_l, half_w]])
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    return local @ np.asarray([[cosine, -sine],
                               [sine, cosine]]).T + box[:2]


def dashed_line(image: np.ndarray, first: tuple[int, int],
                second: tuple[int, int], color: tuple[int, int, int],
                thickness: int = 1, pieces: int = 8) -> None:
    first_array = np.asarray(first, dtype=float)
    delta = np.asarray(second, dtype=float) - first_array
    for index in range(0, pieces, 2):
        start = first_array + delta * (index / pieces)
        end = first_array + delta * ((index + 1) / pieces)
        cv2.line(image, tuple(start.astype(int)), tuple(end.astype(int)),
                 color, thickness, cv2.LINE_AA)


def project_box(box: np.ndarray, camera: dict,
                ocam: OcamCamera) -> np.ndarray:
    corners = corners_3d(box)
    homogeneous = np.column_stack((corners, np.ones(8)))
    camera_xyz = (np.asarray(camera["lidar2cam"], dtype=float) @
                  homogeneous.T)[:3]
    return np.asarray(ocam.world2cam(camera_xyz), dtype=float).T


def draw_camera(image: np.ndarray, camera: dict, detections: list,
                ground_truth: list, ocam: OcamCamera,
                threshold: float, show_gt: bool) -> np.ndarray:
    height, width = image.shape[:2]
    rows = []
    if show_gt:
        rows.extend(("GT", name, box, 1.0) for name, box in ground_truth)
    rows.extend(("PRED", name, box, score)
                for name, box, score in detections if score >= threshold)
    for source, name, box, score in rows:
        pixels = project_box(box, camera, ocam)
        color = (245, 245, 245) if source == "GT" else COLORS.get(
            name, (0, 255, 255))
        valid_corners = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
                         (pixels[:, 1] >= 0) & (pixels[:, 1] < height) &
                         np.isfinite(pixels).all(axis=1))
        for first, second in BOX_EDGES:
            if not (valid_corners[first] and valid_corners[second]):
                continue
            p1 = tuple(np.rint(pixels[first]).astype(int))
            p2 = tuple(np.rint(pixels[second]).astype(int))
            if np.linalg.norm(pixels[first] - pixels[second]) > width * .65:
                continue
            if source == "GT":
                dashed_line(image, p1, p2, color, 1)
            else:
                cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
        if source == "PRED" and np.any(valid_corners):
            anchor = tuple(np.rint(pixels[np.flatnonzero(valid_corners)[0]])
                           .astype(int))
            cv2.putText(image, f"{name} {score:.2f}", anchor,
                        cv2.FONT_HERSHEY_SIMPLEX, .38, color, 1, cv2.LINE_AA)
    return image


def bev_pixel(xy: np.ndarray, size: int, limit: float) -> np.ndarray:
    margin = 24
    scale = (size - 2 * margin) / (2 * limit)
    result = np.empty_like(xy, dtype=float)
    result[..., 0] = size / 2 + xy[..., 0] * scale
    result[..., 1] = size / 2 - xy[..., 1] * scale
    return result


def draw_bev(points: np.ndarray, detections: list, ground_truth: list,
             args: argparse.Namespace, size: int = 700) -> np.ndarray:
    image = np.full((size, size, 3), 7, dtype=np.uint8)
    center = size // 2
    for radius_m in range(10, int(args.bev_range) + 1, 10):
        radius = int(radius_m / args.bev_range * (size - 48) / 2)
        cv2.circle(image, (center, center), radius, (38, 38, 38), 1,
                   cv2.LINE_AA)
    cv2.line(image, (center, 20), (center, size - 20), (30, 30, 30), 1)
    cv2.line(image, (20, center), (size - 20, center), (30, 30, 30), 1)

    visible = ((np.abs(points[:, 0]) <= args.bev_range) &
               (np.abs(points[:, 1]) <= args.bev_range) &
               (points[:, 2] >= -3.0) & (points[:, 2] <= 3.0))
    draw = points[visible][::max(1, args.point_stride)]
    pixels = np.rint(bev_pixel(draw[:, :2], size, args.bev_range)).astype(int)
    gray = np.clip((draw[:, 2] + 2.5) / 4 * 210 + 35, 35, 245).astype(np.uint8)
    inside = ((pixels[:, 0] >= 0) & (pixels[:, 0] < size) &
              (pixels[:, 1] >= 0) & (pixels[:, 1] < size))
    pixels, gray = pixels[inside], gray[inside]
    image[pixels[:, 1], pixels[:, 0]] = np.column_stack((gray, gray, gray))

    rows = []
    if not args.no_ground_truth:
        rows.extend(("GT", name, box, 1.0) for name, box in ground_truth)
    rows.extend(("PRED", name, box, score)
                for name, box, score in detections
                if score >= args.score_threshold)
    for source, name, box, score in rows:
        polygon = np.vstack((corners_2d(box), corners_2d(box)[0]))
        polygon_pixels = np.rint(bev_pixel(
            polygon, size, args.bev_range)).astype(int)
        color = (245, 245, 245) if source == "GT" else COLORS.get(
            name, (0, 255, 255))
        for first in range(4):
            p1, p2 = tuple(polygon_pixels[first]), tuple(polygon_pixels[first + 1])
            if source == "GT":
                dashed_line(image, p1, p2, color, 1)
            else:
                cv2.line(image, p1, p2, color, 2, cv2.LINE_AA)
        center_pixel = tuple(np.rint(bev_pixel(
            box[None, :2], size, args.bev_range)[0]).astype(int))
        heading_xy = box[:2] + np.asarray(
            [np.cos(box[6]), np.sin(box[6])]) * box[3] / 2
        heading_pixel = tuple(np.rint(bev_pixel(
            heading_xy[None], size, args.bev_range)[0]).astype(int))
        cv2.line(image, center_pixel, heading_pixel, color,
                 1 if source == "GT" else 2, cv2.LINE_AA)
        if source == "PRED":
            cv2.putText(image, f"{name} {score:.2f}", center_pixel,
                        cv2.FONT_HERSHEY_SIMPLEX, .34, color, 1, cv2.LINE_AA)
    cv2.drawMarker(image, (center, center), (0, 255, 140),
                   cv2.MARKER_TRIANGLE_UP, 15, 2, cv2.LINE_AA)
    cv2.putText(image, f"LiDAR BEV +/-{args.bev_range:.0f} m",
                (18, 22), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (230, 230, 230), 1, cv2.LINE_AA)
    return image


def ground_truth(sample: dict) -> list:
    annos = sample.get("annos", {})
    return [(str(name), np.asarray(box, dtype=float)[:7])
            for name, box in zip(annos.get("gt_names", []),
                                 annos.get("gt_boxes", []))]


def render_model(sample: dict, points: np.ndarray, detections: list,
                 ocam: OcamCamera, data_root: Path,
                 args: argparse.Namespace, title: str,
                 metrics: dict[str, float]) -> np.ndarray:
    canvas = np.full((900, 1600, 3), 12, dtype=np.uint8)
    gt = ground_truth(sample)
    bev = draw_bev(points, detections, gt, args)
    canvas[130:830, 30:730] = bev
    camera_info = sample["cam_info"]["cam_fisheye"]
    for position, (camera_name, label) in enumerate(CAMERAS):
        camera = camera_info[camera_name]
        image_path = data_root / str(camera["cam_path"])
        image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
        if image is None:
            raise FileNotFoundError(image_path)
        image = draw_camera(image, camera, detections, gt, ocam,
                            args.score_threshold, not args.no_ground_truth)
        image = cv2.resize(image, (410, 410), interpolation=cv2.INTER_AREA)
        x = 760 + (position % 2) * 420
        y = 55 + (position // 2) * 420
        canvas[y:y + 410, x:x + 410] = image
        cv2.rectangle(canvas, (x, y), (x + 409, y + 409),
                      (90, 90, 90), 1)
        cv2.putText(canvas, label, (x + 8, y + 22),
                    cv2.FONT_HERSHEY_SIMPLEX, .65, (255, 255, 255), 2,
                    cv2.LINE_AA)
    sequence = sample.get("metainfo", {}).get("sequence_id", "")
    frame_id = sample.get("metainfo", {}).get("frame_id", "")
    cv2.putText(canvas, title, (30, 40), cv2.FONT_HERSHEY_SIMPLEX,
                .9, (255, 255, 255), 2, cv2.LINE_AA)
    cv2.putText(canvas,
                f"mAP {metrics['mAP']:.4f}  NDS {metrics['NDS']:.4f}  "
                f"score >= {args.score_threshold:.2f}",
                (30, 76), cv2.FONT_HERSHEY_SIMPLEX, .58,
                (200, 230, 255), 1, cv2.LINE_AA)
    cv2.putText(canvas,
                f"{sequence} frame {frame_id} | token {sample['token']} | "
                f"pred {sum(score >= args.score_threshold for _, _, score in detections)}",
                (30, 105), cv2.FONT_HERSHEY_SIMPLEX, .47,
                (190, 190, 190), 1, cv2.LINE_AA)
    cv2.putText(canvas, "solid/color: prediction   dashed/white: validation GT",
                (760, 890), cv2.FONT_HERSHEY_SIMPLEX, .42,
                (190, 190, 190), 1, cv2.LINE_AA)
    return canvas


def metrics_panel(metrics: dict[str, dict[str, float]]) -> np.ndarray:
    panel = np.full((540, 960, 3), 15, dtype=np.uint8)
    cv2.putText(panel, "COMMON 376-FRAME VALIDATION SPLIT",
                (45, 70), cv2.FONT_HERSHEY_SIMPLEX, .9,
                (255, 255, 255), 2, cv2.LINE_AA)
    y = 145
    for name, values in metrics.items():
        cv2.putText(panel, name, (55, y), cv2.FONT_HERSHEY_SIMPLEX,
                    .72, (225, 225, 225), 2, cv2.LINE_AA)
        cv2.putText(panel,
                    f"mAP {values['mAP']:.4f}   NDS {values['NDS']:.4f}",
                    (430, y), cv2.FONT_HERSHEY_SIMPLEX, .72,
                    (0, 220, 255), 2, cv2.LINE_AA)
        y += 95
    cv2.putText(panel, "All boxes use the same LiDAR frame and OCam calibration.",
                (55, 465), cv2.FONT_HERSHEY_SIMPLEX, .55,
                (180, 180, 180), 1, cv2.LINE_AA)
    return panel


def encode_video(frames: Path, destination: Path, fps: float) -> None:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        raise RuntimeError("ffmpeg was not found")
    subprocess.run([
        ffmpeg, "-y", "-loglevel", "warning", "-framerate", str(fps),
        "-i", str(frames / "frame_%06d.jpg"), "-c:v", "libx264",
        "-pix_fmt", "yuv420p", "-crf", "20", destination.as_posix(),
    ], check=True)


def main() -> None:
    args = parse_args()
    data_root = args.data_root.expanduser().resolve()
    info_path = (args.info or data_root / "rosa" /
                 "fisheye3dod_infos_val.pkl").resolve()
    samples = mmengine.load(info_path)["data_list"]
    end = len(samples) - 1 if args.end_frame is None else args.end_frame
    if (args.start_frame < 0 or end < args.start_frame or
            end >= len(samples) or args.frame_step < 1):
        raise ValueError("invalid frame range")
    indices = list(range(args.start_frame, end + 1, args.frame_step))
    predictions = {
        "Fisheye3DOD manual": detections_from_fisheye(
            args.manual.resolve(), samples),
        "Fisheye3DOD automatic": detections_from_fisheye(
            args.automatic.resolve(), samples),
        "ROSA automatic": detections_from_rosa(
            args.rosa.resolve(), len(samples)),
    }
    metrics = {
        "Fisheye3DOD manual": {"mAP": .0120, "NDS": .0379},
        "Fisheye3DOD automatic": {"mAP": .0281, "NDS": .0882},
        "ROSA automatic": {"mAP": .0566, "NDS": .1467},
    }
    output = args.output_dir.expanduser().resolve()
    model_directories = {
        name: output / name.lower().replace(" ", "_") / "frames"
        for name in predictions
    }
    combined_frames = output / "comparison" / "frames"
    for directory in (*model_directories.values(), combined_frames):
        directory.mkdir(parents=True, exist_ok=True)
    sources = parse_sources(data_root, args.source_dataset)
    ocam = OcamCamera(filename=str(data_root / "calib_results.txt"), fov=220.0)
    counts = {name: Counter() for name in predictions}
    panel = metrics_panel(metrics)
    for output_index, sample_index in enumerate(
            mmengine.track_iter_progress(indices)):
        sample = samples[sample_index]
        points = load_points(resolve_lidar(
            data_root, info_path, sample, sources))
        rendered = {}
        for name, model_predictions in predictions.items():
            detections = model_predictions[sample_index]
            counts[name].update(
                cls for cls, _, score in detections
                if score >= args.score_threshold)
            image = render_model(sample, points, detections, ocam,
                                 data_root, args, name, metrics[name])
            rendered[name] = image
            cv2.imwrite(str(model_directories[name] /
                            f"frame_{output_index:06d}.jpg"), image,
                        [cv2.IMWRITE_JPEG_QUALITY, 92])
        mosaic = np.full((1080, 1920, 3), 10, dtype=np.uint8)
        for position, name in enumerate(predictions):
            small = cv2.resize(rendered[name], (960, 540),
                               interpolation=cv2.INTER_AREA)
            x, y = (position % 2) * 960, (position // 2) * 540
            mosaic[y:y + 540, x:x + 960] = small
        mosaic[540:1080, 960:1920] = panel
        cv2.imwrite(str(combined_frames / f"frame_{output_index:06d}.jpg"),
                    mosaic, [cv2.IMWRITE_JPEG_QUALITY, 92])
    videos = {}
    if not args.no_video:
        for name, frames in model_directories.items():
            destination = frames.parent / "predictions_4cam_lidar.mp4"
            encode_video(frames, destination, args.fps)
            videos[name] = str(destination)
        comparison_video = output / "comparison" / "three_models.mp4"
        encode_video(combined_frames, comparison_video, args.fps)
        videos["comparison"] = str(comparison_video)
    summary = {
        "validation_samples": len(samples),
        "rendered_frames": len(indices),
        "score_threshold": args.score_threshold,
        "metrics": metrics,
        "displayed_class_counts": {name: dict(value)
                                   for name, value in counts.items()},
        "videos": videos,
        "prediction_files": {
            "Fisheye3DOD manual": str(args.manual.resolve()),
            "Fisheye3DOD automatic": str(args.automatic.resolve()),
            "ROSA automatic": str(args.rosa.resolve()),
        },
    }
    (output / "summary.json").write_text(
        json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
