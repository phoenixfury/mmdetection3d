#!/usr/bin/env python3
"""Run Fisheye3DOD inference for an inclusive frame range and visualize it."""

from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import mmengine
import numpy as np
from ocamcamera import OcamCamera
from PIL import Image


CAMERAS = (
    ('fisheye_camera_front', 'FRONT'),
    ('fisheye_camera_left', 'LEFT'),
    ('fisheye_camera_right', 'RIGHT'),
    ('fisheye_camera_rear', 'REAR'),
)
COLORS = {
    'Car': '#00d8ff', 'Van': '#2979ff', 'Truck': '#ff9100',
    'Bus': '#ff1744', 'Pedestrian': '#76ff03', 'Cyclist': '#d500f9',
}
BOX_EDGES = ((0, 1), (0, 2), (1, 3), (2, 3),
             (4, 5), (4, 6), (5, 7), (6, 7),
             (0, 4), (1, 5), (2, 6), (3, 7))


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path,
                        help='prepared custom Fisheye3DOD dataset')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--start-frame', type=int, required=True,
                        help='zero-based first frame index (inclusive)')
    parser.add_argument('--end-frame', type=int, required=True,
                        help='zero-based last frame index (inclusive)')
    parser.add_argument('--config', type=Path, default=Path(
        'projects/Fisheye3DOD/configs/custom_fisheye_bevdet.py'))
    parser.add_argument('--info', type=Path,
                        help='defaults to DATASET/fisheye3dod_infos_test.pkl')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--score-threshold', type=float, default=0.10)
    parser.add_argument('--bev-range', type=float, default=48.0)
    parser.add_argument('--point-stride', type=int, default=3)
    parser.add_argument('--target-ocam', type=Path, default=Path(
        'data/Fisheye3DODdataset/calib_results.txt'))
    parser.add_argument('--video-fps', type=float, default=5.0)
    parser.add_argument('--skip-inference', action='store_true',
                        help='reuse OUTPUT/predictions.pkl')
    parser.add_argument('--no-video', action='store_true')
    return parser.parse_args()


def corners_2d(box: np.ndarray) -> np.ndarray:
    half_l, half_w = box[3] / 2, box[4] / 2
    local = np.array([[half_l, half_w], [half_l, -half_w],
                      [-half_l, -half_w], [-half_l, half_w],
                      [half_l, half_w]])
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.array([[cosine, -sine], [sine, cosine]])
    return local @ rotation.T + box[:2]


def corners_3d(box: np.ndarray) -> np.ndarray:
    half_l, half_w, half_h = box[3:6] / 2
    local = np.array([
        [sx * half_l, sy * half_w, sz * half_h]
        for sz in (-1, 1) for sy in (-1, 1) for sx in (-1, 1)])
    cosine, sine = np.cos(box[6]), np.sin(box[6])
    rotation = np.array([[cosine, -sine, 0],
                         [sine, cosine, 0], [0, 0, 1]])
    return local @ rotation.T + box[:3]


def load_points(path: Path) -> np.ndarray:
    if path.suffix.lower() == '.pcd':
        payload = path.read_bytes()
        marker = b'DATA binary\n'
        offset = payload.find(marker)
        if offset < 0:
            raise ValueError(f'{path}: expected binary PCD')
        raw = np.frombuffer(payload, dtype=np.float32,
                            offset=offset + len(marker))
    else:
        raw = np.fromfile(path, dtype=np.float32)
    if len(raw) % 4:
        raise ValueError(f'{path}: expected float32 XYZI')
    points = raw.reshape(-1, 4)
    valid = np.isfinite(points).all(axis=1)
    valid &= np.linalg.norm(points[:, :3], axis=1) > 0.1
    return points[valid]


def run_inference(args: argparse.Namespace, subset_path: Path,
                  output: Path) -> Path:
    raw_results = output / 'raw_results.pkl'
    run_dir = output / 'run'
    command = [
        sys.executable, 'tools/test.py', str(args.config.resolve()),
        str(args.checkpoint.resolve()), '--work-dir', str(run_dir.resolve()),
        '--cfg-options',
        f'test_dataloader.dataset.ann_file={subset_path.resolve()}',
        f'test_dataloader.dataset.data_root={args.dataset.resolve()}',
        'test_dataloader.dataset.load_eval_anns=False',
        'test_evaluator._delete_=True',
        'test_evaluator.type=DumpResults',
        f'test_evaluator.out_file_path={raw_results.resolve()}',
        'test_dataloader.num_workers=2',
        'test_dataloader.persistent_workers=False',
    ]
    environment = os.environ.copy()
    environment.setdefault('MPLCONFIGDIR', '/tmp/mplconfig')
    print('Running:', ' '.join(command), flush=True)
    subprocess.run(command, check=True, env=environment)
    candidates = sorted(run_dir.glob('*/save_detection/epoch_0.pkl'),
                        key=lambda path: path.stat().st_mtime)
    if not candidates:
        raise FileNotFoundError(f'no SaveDetectionHook output below {run_dir}')
    predictions = output / 'predictions.pkl'
    shutil.copy2(candidates[-1], predictions)
    return predictions


def draw_camera(axis: object, image: np.ndarray, camera: dict,
                detections: list[tuple[str, np.ndarray, float]],
                ocam: OcamCamera, score_threshold: float, title: str) -> None:
    axis.imshow(image)
    axis.set_title(title, fontsize=11, weight='bold')
    axis.axis('off')
    height, width = image.shape[:2]
    lidar_to_camera = np.asarray(camera['lidar2cam'], dtype=float)
    for name, box, score in detections:
        if score < score_threshold:
            continue
        corners = corners_3d(box)
        homogeneous = np.column_stack((corners, np.ones(8)))
        camera_xyz = (lidar_to_camera @ homogeneous.T)[:3]
        pixels = ocam.world2cam(camera_xyz).T
        color = COLORS.get(name, '#ffffff')
        for first, second in BOX_EDGES:
            segment = pixels[[first, second]]
            valid = np.isfinite(segment).all() and (segment >= 0).all()
            valid &= (segment[:, 0] < width).all()
            valid &= (segment[:, 1] < height).all()
            valid &= np.linalg.norm(segment[0] - segment[1]) < width * 0.6
            if valid:
                axis.plot(segment[:, 0], segment[:, 1], color=color,
                          linewidth=1.7, alpha=0.9)
        visible = ((pixels[:, 0] >= 0) & (pixels[:, 0] < width) &
                   (pixels[:, 1] >= 0) & (pixels[:, 1] < height))
        if np.any(visible):
            anchor = pixels[np.flatnonzero(visible)[0]]
            axis.text(anchor[0], anchor[1], f'{name} {score:.2f}',
                      color=color, fontsize=7,
                      bbox=dict(facecolor='black', alpha=0.5, pad=1,
                                edgecolor='none'))


def render_frame(dataset: Path, sample: dict, prediction: dict,
                 output_path: Path, ocam: OcamCamera,
                 args: argparse.Namespace, original_index: int) -> int:
    annos = prediction['annos']
    detections = [
        (str(name), np.asarray(box, dtype=float), float(score))
        for name, box, score in zip(
            annos['pred_names'], annos['pred_boxes'], annos['pred_scores'])
        if float(score) >= args.score_threshold
    ]
    figure = plt.figure(figsize=(18, 10), constrained_layout=True)
    grid = figure.add_gridspec(2, 3, width_ratios=(1, 1, 1.35))
    camera_info = sample['cam_info']['cam_fisheye']
    for position, (camera_name, title) in enumerate(CAMERAS):
        axis = figure.add_subplot(grid[position // 2, position % 2])
        camera = camera_info[camera_name]
        image = np.asarray(Image.open(dataset / camera['cam_path']).convert('RGB'))
        draw_camera(axis, image, camera, detections, ocam,
                    args.score_threshold, title)

    axis = figure.add_subplot(grid[:, 2])
    points = load_points(dataset / sample['pc_info']['pts_path'])
    limit = args.bev_range
    visible = ((np.abs(points[:, 0]) <= limit) &
               (np.abs(points[:, 1]) <= limit))
    draw = points[visible][::max(1, args.point_stride)]
    axis.scatter(draw[:, 0], draw[:, 1], s=0.25, c=draw[:, 2],
                 cmap='gray', vmin=-3, vmax=2, alpha=0.65)
    for name, box, score in detections:
        polygon = corners_2d(box)
        color = COLORS.get(name, '#ffffff')
        axis.plot(polygon[:, 0], polygon[:, 1], color=color, linewidth=2)
        axis.fill(polygon[:, 0], polygon[:, 1], color=color, alpha=0.08)
        axis.text(box[0], box[1], f'{name} {score:.2f}', color=color,
                  fontsize=8, weight='bold')
    axis.plot(0, 0, marker='^', color='red', markersize=9)
    axis.set(xlim=(-limit, limit), ylim=(-limit, limit), aspect='equal',
             xlabel='LiDAR X [m]', ylabel='LiDAR Y [m]', title='LiDAR BEV')
    axis.grid(alpha=0.15)
    figure.suptitle(
        f'Frame {original_index} · token {sample["token"]} · '
        f'{len(detections)} detections ≥ {args.score_threshold:.2f}',
        fontsize=14, weight='bold')
    figure.savefig(output_path, dpi=110)
    plt.close(figure)
    return len(detections)


def make_video(frames_dir: Path, output: Path, start: int,
               fps: float) -> Path | None:
    ffmpeg = shutil.which('ffmpeg')
    if ffmpeg is None:
        print('ffmpeg is unavailable; skipping video')
        return None
    video = output / 'predictions.mp4'
    command = [
        ffmpeg, '-y', '-framerate', str(fps), '-start_number', str(start),
        '-i', str(frames_dir / 'frame_%06d.png'), '-c:v', 'libx264',
        '-pix_fmt', 'yuv420p', '-crf', '20', str(video),
    ]
    subprocess.run(command, check=True)
    return video


def main() -> None:
    args = parse_args()
    dataset = args.dataset.resolve()
    info_path = (args.info or dataset / 'fisheye3dod_infos_test.pkl').resolve()
    source = mmengine.load(info_path)
    samples = source['data_list']
    if args.start_frame < 0 or args.end_frame < args.start_frame:
        raise ValueError('require 0 <= start-frame <= end-frame')
    if args.end_frame >= len(samples):
        raise IndexError(
            f'end-frame {args.end_frame} exceeds final index {len(samples) - 1}')

    output = (args.output or Path(
        f'work_dirs/custom_fisheye_bevdet_frames_'
        f'{args.start_frame:04d}_{args.end_frame:04d}')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    selected = samples[args.start_frame:args.end_frame + 1]
    subset_path = output / 'subset_info.pkl'
    mmengine.dump({'metainfo': source.get('metainfo', {}),
                   'data_list': selected}, subset_path)

    predictions_path = output / 'predictions.pkl'
    if not args.skip_inference:
        predictions_path = run_inference(args, subset_path, output)
    if not predictions_path.exists():
        raise FileNotFoundError(predictions_path)
    predictions = mmengine.load(predictions_path)['data_list']
    by_token = {str(item['token']): item for item in predictions}
    if len(by_token) != len(selected):
        raise RuntimeError(
            f'expected {len(selected)} predictions, found {len(by_token)}')

    frames_dir = output / 'visualizations'
    frames_dir.mkdir(parents=True, exist_ok=True)
    ocam = OcamCamera(filename=str(args.target_ocam.resolve()), fov=220.0)
    counts = Counter()
    detection_total = 0
    for offset, sample in enumerate(mmengine.track_iter_progress(selected)):
        original_index = args.start_frame + offset
        prediction = by_token[str(sample['token'])]
        annos = prediction['annos']
        for name, score in zip(annos['pred_names'], annos['pred_scores']):
            if float(score) >= args.score_threshold:
                counts[str(name)] += 1
        detection_total += render_frame(
            dataset, sample, prediction,
            frames_dir / f'frame_{original_index:06d}.png', ocam, args,
            original_index)

    video = None if args.no_video else make_video(
        frames_dir, output, args.start_frame, args.video_fps)
    summary = {
        'dataset': str(dataset), 'checkpoint': str(args.checkpoint.resolve()),
        'source_info': str(info_path), 'start_frame': args.start_frame,
        'end_frame': args.end_frame, 'frames': len(selected),
        'score_threshold': args.score_threshold,
        'detections': detection_total, 'classes': dict(counts),
        'predictions': str(predictions_path),
        'visualizations': str(frames_dir),
        'video': None if video is None else str(video),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
