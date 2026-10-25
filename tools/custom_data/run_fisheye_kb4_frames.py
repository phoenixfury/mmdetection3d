#!/usr/bin/env python3
"""Run and visualize native-KB4 or canonical-OCam Fisheye3DOD inference."""

from __future__ import annotations

import argparse
import csv
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

DEFAULTS = {
    'native': {
        'checkpoint': Path('projects/Fisheye3DOD/weights/native_epoch_20.pth'),
        'config': Path(
            'projects/Fisheye3DOD/configs/custom_fisheye_bevdet_infer_native.py'),
    },
    'ocam': {
        'checkpoint': Path('projects/Fisheye3DOD/weights/ocam_epoch_20.pth'),
        'config': Path(
            'projects/Fisheye3DOD/configs/custom_fisheye_bevdet_infer_ocam.py'),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('dataset', type=Path,
                        help='prepared custom Fisheye3DOD dataset')
    parser.add_argument('checkpoint', type=Path, nargs='?',
                        help='defaults to the epoch-20 checkpoint for --mode')
    parser.add_argument('--mode', choices=('native', 'ocam'), default='native',
                        help='input image/calibration geometry (default: native)')
    parser.add_argument('--start-frame', type=int, required=True,
                        help='zero-based first frame index (inclusive)')
    parser.add_argument('--end-frame', type=int, required=True,
                        help='zero-based last frame index (inclusive)')
    parser.add_argument('--config', type=Path,
                        help='defaults to the inference config for --mode')
    parser.add_argument('--info', type=Path,
                        help='defaults to DATASET/fisheye3dod_infos_test.pkl')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--score-threshold', type=float, default=0.10)
    parser.add_argument('--bev-range', type=float, default=48.0)
    parser.add_argument('--point-stride', type=int, default=3)
    parser.add_argument('--calibration', type=Path,
        help='native per-camera calibration JSON; inferred from metadata.json')
    parser.add_argument('--ocam-calibration', type=Path,
        help='OCamCalib file; inferred from metadata.json')
    parser.add_argument('--video-fps', type=float, default=5.0)
    parser.add_argument('--skip-inference', action='store_true',
                        help='reuse OUTPUT/predictions.pkl')
    parser.add_argument('--no-video', action='store_true')
    return parser.parse_args()


CAMERA_FRAMES = {
    'fisheye_camera_front': 'ARC_F',
    'fisheye_camera_left': 'ARC_SL',
    'fisheye_camera_right': 'ARC_SR',
    'fisheye_camera_rear': 'ARC_R',
}

CLASSES = ('Car', 'Van', 'Truck', 'Bus', 'Pedestrian', 'Cyclist')


def first_existing(candidates: list[Path], description: str) -> Path:
    for candidate in candidates:
        resolved = candidate.expanduser().resolve()
        if resolved.is_file():
            return resolved
    checked = '\n  '.join(str(path) for path in candidates)
    raise FileNotFoundError(f'{description} not found; checked:\n  {checked}')


def resolve_mode_inputs(args: argparse.Namespace, dataset: Path) -> dict:
    metadata_path = dataset / 'metadata.json'
    metadata = (json.loads(metadata_path.read_text())
                if metadata_path.is_file() else {})

    calibration_candidates = []
    if args.calibration is not None:
        calibration_candidates.append(args.calibration)
    calibration_candidates.append(dataset.parent / 'calibration.json')
    if metadata.get('source_calibration'):
        calibration_candidates.append(Path(metadata['source_calibration']))
    calibration = first_existing(
        calibration_candidates, 'native camera calibration')

    ocam_candidates = []
    if args.ocam_calibration is not None:
        ocam_candidates.append(args.ocam_calibration)
    if metadata.get('target_ocam'):
        ocam_candidates.append(Path(metadata['target_ocam']))
    ocam_candidates.extend([
        Path('data/Fisheye3DODdataset/calib_results.txt'),
        Path('data/Custom/combined_caliper_fisheye3dod/calib_results.txt'),
    ])
    ocam_calibration = first_existing(
        ocam_candidates, 'canonical OCam calibration')

    default = DEFAULTS[args.mode]
    args.checkpoint = (args.checkpoint or default['checkpoint']).resolve()
    args.config = (args.config or default['config']).resolve()
    args.calibration = calibration
    args.ocam_calibration = ocam_calibration
    for path, label in ((args.checkpoint, 'checkpoint'),
                        (args.config, 'config')):
        if not path.is_file():
            raise FileNotFoundError(f'{label} not found: {path}')
    return metadata


def build_inference_info(dataset: Path, mode: str, calibration_path: Path,
                         destination: Path) -> dict:
    manifest_path = dataset / 'manifest.csv'
    if not manifest_path.is_file():
        raise FileNotFoundError(
            f'no --info was supplied and manifest is missing: {manifest_path}')
    with manifest_path.open(newline='') as stream:
        rows = list(csv.DictReader(stream))
    if not rows:
        raise RuntimeError(f'no frames in {manifest_path}')

    calibration = json.loads(calibration_path.read_text())
    image_column = 'native' if mode == 'native' else 'ocam_800x800'
    data_list = []
    for sample_index, row in enumerate(rows):
        token = str(row['timestamp_ns'])
        cameras = {}
        for camera_name, frame in CAMERA_FRAMES.items():
            relative = Path(row[f'{frame.lower()}_{image_column}'])
            if not (dataset / relative).is_file():
                raise FileNotFoundError(dataset / relative)
            intrinsic = calibration['intrinsics'][frame]
            lidar2cam = np.asarray(
                calibration['lidar_to_camera'][frame], dtype=np.float32)
            cam2img = np.eye(4, dtype=np.float32)
            if mode == 'native':
                cam2img[:3, :3] = np.asarray(
                    intrinsic['camera_matrix'], dtype=np.float32)
                image_size = (int(intrinsic['width']),
                              int(intrinsic['height']))
            else:
                image_size = (800, 800)
            cameras[camera_name] = {
                'cam_path': relative.as_posix(),
                'image_size': image_size,
                'cam2img': cam2img,
                'lidar2img': np.eye(4, dtype=np.float32),
                'lidar2cam': lidar2cam,
                'cam2lidar': np.linalg.inv(lidar2cam).astype(np.float32),
            }

        lidar_relative = Path(row['lidar_pcd'])
        if not (dataset / lidar_relative).is_file():
            raise FileNotFoundError(dataset / lidar_relative)
        data_list.append({
            'sample_idx': sample_index,
            'token': token,
            'metainfo': {
                'scene_name': dataset.parent.name,
                'vehicle_name': 'ego',
                'weather': 'unknown',
                'frame_id': token,
                'prev_id': (str(rows[sample_index - 1]['timestamp_ns'])
                            if sample_index else ''),
                'next_id': (str(rows[sample_index + 1]['timestamp_ns'])
                            if sample_index + 1 < len(rows) else ''),
            },
            'pc_info': {
                'num_features': 4,
                'pts_path': lidar_relative.as_posix(),
                'semantic_pts_path': lidar_relative.as_posix(),
            },
            'cam_info': {'cam_fisheye': cameras},
        })

    info = {
        'metainfo': {
            'categories': {name: index for index, name in enumerate(CLASSES)},
            'classes': CLASSES,
            'dataset': 'custom_fisheye3dod_inference',
            'version': f'synchronized-dual-images-{mode}',
        },
        'data_list': data_list,
    }
    mmengine.dump(info, destination)
    print(f'Generated unlabeled {mode} info: {destination} '
          f'({len(data_list)} frames)', flush=True)
    return info


def load_calibration(path: Path) -> dict:
    raw = json.loads(path.read_text())
    models = {}
    for frame in CAMERA_FRAMES.values():
        data = raw['intrinsics'][frame]
        K = np.asarray(data['camera_matrix'], dtype=np.float64)
        D = np.asarray(data['distortion_coefficients'], dtype=np.float64)
        if K.shape != (3, 3) or D.shape != (4,):
            raise ValueError(f'{frame}: expected 3x3 K and four KB4 distortion coefficients')
        models[frame] = {'K': K, 'D': D, 'width': int(data['width']),
                         'height': int(data['height'])}
    return models


def validate_training_config(args: argparse.Namespace) -> None:
    from mmengine.config import Config
    cfg = Config.fromfile(str(args.config.resolve()))
    transform = cfg.model.view_transform
    calibration_path = transform.get('kb4_calibration_path', None)
    frames = tuple(transform.get('kb4_camera_frames', ()))
    expected_frames = tuple(CAMERA_FRAMES.values())
    image_size = list(transform.get('image_size', ()))
    if args.mode == 'native':
        if not calibration_path:
            raise RuntimeError(
                'Native mode requires model.view_transform.'
                'kb4_calibration_path')
        if frames != expected_frames:
            raise RuntimeError(
                f'Config KB4 camera order is {frames}; expected '
                f'{expected_frames}')
        if image_size != [1536, 1920]:
            raise RuntimeError(
                f'native config image_size is {image_size}, expected '
                '[1536, 1920]')
        loaded_path = Path(calibration_path).resolve()
        if not loaded_path.is_file():
            raise FileNotFoundError(f'Configured KB4 calibration missing: {loaded_path}')
        if loaded_path.read_bytes() != args.calibration.resolve().read_bytes():
            raise RuntimeError('Visualization --calibration differs from model KB4 calibration. '
                               'Pass the same JSON file to both for valid overlays.')
        model_calibration = loaded_path
    else:
        if calibration_path:
            raise RuntimeError(
                'OCam mode config still enables KB4 calibration; use the '
                'OCam inference config')
        if image_size != [800, 800]:
            raise RuntimeError(
                f'OCam config image_size is {image_size}, expected [800, 800]')
        model_calibration = Path(transform['ocam_path']).resolve()
        if not model_calibration.is_file():
            raise FileNotFoundError(
                f'Configured OCam calibration missing: {model_calibration}')
        if (model_calibration.read_bytes() !=
                args.ocam_calibration.resolve().read_bytes()):
            raise RuntimeError(
                'Visualization --ocam-calibration differs from the model '
                'OCam calibration')
    print(f'Mode: {args.mode}; model calibration: {model_calibration}',
          flush=True)


def kb4_project(camera_xyz: np.ndarray, model: dict,
                image_width: int, image_height: int,
                max_angle_degrees: float = 110.0) -> np.ndarray:
    """Project Nx3 optical-frame points via OpenCV-equivalent KB4 convention.

    x right, y down, z forward. Reject rays farther than half the nominal
    220-degree fisheye FoV. Scaling assumes a direct full-image resize.
    """
    xyz = np.asarray(camera_xyz, dtype=np.float64)
    if xyz.ndim != 2 or xyz.shape[1] != 3:
        raise ValueError(f'camera_xyz must be N x 3; got {xyz.shape}')
    x, y, z = xyz.T
    rho = np.hypot(x, y)
    theta = np.arctan2(rho, z)
    k1, k2, k3, k4 = model['D']
    t2 = theta * theta
    radius = theta * (1 + t2 * (k1 + t2 * (k2 + t2 * (k3 + t2 * k4))))
    scale = np.divide(radius, rho, out=np.ones_like(radius), where=rho > 1e-12)
    K = model['K']
    u = (K[0, 0] * x * scale + K[0, 2]) * (image_width / model['width'])
    v = (K[1, 1] * y * scale + K[1, 2]) * (image_height / model['height'])
    uv = np.stack((u, v), axis=-1)
    valid = (np.isfinite(xyz).all(axis=1) &
             (np.linalg.norm(xyz, axis=1) > 1e-8) &
             (theta <= np.deg2rad(max_angle_degrees)))
    uv[~valid] = np.nan
    return uv


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
            raise ValueError(f'{path}: expected binary PCD data')
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
        f'test_dataloader.dataset.data_root={args.dataset.resolve()}',
        f'test_dataloader.dataset.ann_file={subset_path.resolve()}',
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
                projection: object, mode: str, score_threshold: float,
                title: str) -> None:
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
        if mode == 'native':
            pixels = kb4_project(
                camera_xyz.T, projection, width, height)
        else:
            pixels = projection.world2cam(camera_xyz).T
            ray_angles = np.arctan2(
                np.linalg.norm(camera_xyz[:2], axis=0), camera_xyz[2])
            pixels[ray_angles > np.deg2rad(110.0)] = np.nan
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
                 output_path: Path, projections: dict,
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
        frame = CAMERA_FRAMES[camera_name]
        draw_camera(axis, image, camera, detections, projections[frame],
                    args.mode, args.score_threshold, title)

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
    args.dataset = dataset
    metadata = resolve_mode_inputs(args, dataset)
    os.environ['FISHEYE_INFERENCE_DATA_ROOT'] = str(dataset)
    os.environ['FISHEYE_NATIVE_CALIBRATION'] = str(args.calibration)
    os.environ['FISHEYE_OCAM_CALIBRATION'] = str(args.ocam_calibration)
    validate_training_config(args)
    if args.start_frame < 0 or args.end_frame < args.start_frame:
        raise ValueError('require 0 <= start-frame <= end-frame')

    output = (args.output or Path(
        f'work_dirs/custom_fisheye_bevdet_{args.mode}_frames_'
        f'{args.start_frame:04d}_{args.end_frame:04d}')).resolve()
    output.mkdir(parents=True, exist_ok=True)
    if args.info is not None:
        info_path = args.info.resolve()
        source = mmengine.load(info_path)
    else:
        existing_info = dataset / 'fisheye3dod_infos_test.pkl'
        info_path = (output / f'fisheye3dod_infos_{args.mode}.pkl'
                     if (dataset / 'manifest.csv').is_file()
                     else existing_info)
        source = (mmengine.load(info_path) if info_path.is_file() else
                  build_inference_info(
                      dataset, args.mode, args.calibration, info_path))
    samples = source['data_list']
    if args.end_frame >= len(samples):
        raise IndexError(
            f'end-frame {args.end_frame} exceeds final index {len(samples) - 1}')

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
    if args.mode == 'native':
        projections = load_calibration(args.calibration)
    else:
        ocam = OcamCamera(
            filename=str(args.ocam_calibration),
            fov=float(metadata.get('target_ocam_fov_degrees', 220.0)))
        projections = {frame: ocam for frame in CAMERA_FRAMES.values()}
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
            frames_dir / f'frame_{original_index:06d}.png', projections, args,
            original_index)

    video = None if args.no_video else make_video(
        frames_dir, output, args.start_frame, args.video_fps)
    summary = {
        'mode': args.mode, 'dataset': str(dataset),
        'config': str(args.config), 'checkpoint': str(args.checkpoint),
        'source_info': str(info_path), 'start_frame': args.start_frame,
        'end_frame': args.end_frame, 'frames': len(selected),
        'score_threshold': args.score_threshold,
        'native_calibration': str(args.calibration),
        'ocam_calibration': (str(args.ocam_calibration)
                             if args.mode == 'ocam' else None),
        'detections': detection_total, 'classes': dict(counts),
        'predictions': str(predictions_path),
        'visualizations': str(frames_dir),
        'video': None if video is None else str(video),
    }
    (output / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
