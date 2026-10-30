#!/usr/bin/env python3
"""Run a ROSA checkpoint on an unlabeled synchronized bag and render its video.

Input is the 800x800 canonical-OCam export produced by
``prepare_fisheye3dod_inference.py``. Predictions and the video use the exact
same ordered samples and LiDAR clouds.
"""
from __future__ import annotations

import argparse
import gc
import json
import os
from pathlib import Path
import subprocess

os.environ.setdefault('CUBLAS_WORKSPACE_CONFIG', ':4096:8')

import cv2
import mmengine
import numpy as np

from render_three_model_fisheye_lidar_comparison import (
    detections_from_rosa, load_points, render_model, resolve_lidar,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = (ROOT / 'projects/ROSA/configs/fisheye3dod/'
                  'rosa_vov99_baseline_adapt_8e.py')


def arguments() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('prepared', type=Path,
                        help='canonical 800x800 synchronized camera dataset')
    parser.add_argument('checkpoint', type=Path)
    parser.add_argument('--config', type=Path, default=DEFAULT_CONFIG)
    parser.add_argument('--output', type=Path)
    parser.add_argument('--ocam-calibration', type=Path,
                        default=ROOT / 'data/Fisheye3DODdataset/calib_results.txt')
    parser.add_argument('--score-threshold', type=float, default=.1)
    parser.add_argument('--bev-range', type=float, default=60.)
    parser.add_argument('--point-stride', type=int, default=4)
    parser.add_argument('--frame-step', type=int, default=1)
    parser.add_argument('--fps', type=float, default=2.)
    parser.add_argument('--workers', type=int, default=2)
    parser.add_argument('--render-only', action='store_true')
    return parser.parse_args()


def infer(args: argparse.Namespace, info_path: Path, raw_path: Path) -> None:
    # This environment still carries packages that refer to removed NumPy
    # aliases while importing the Fisheye3DOD adapter.
    for name, value in (('float', float), ('int', int), ('bool', bool)):
        if name not in np.__dict__:
            setattr(np, name, value)
    import torch
    from mmengine.config import Config
    from mmengine.runner import Runner

    os.environ.setdefault('TORCH_FORCE_NO_WEIGHTS_ONLY_LOAD', '1')
    config = Config.fromfile(str(args.config))
    config.test_dataloader.batch_size = 1
    config.test_dataloader.num_workers = args.workers
    config.test_dataloader.dataset.ann_file = str(info_path)
    config.test_dataloader.dataset.data_root = str(args.prepared)
    config.test_dataloader.dataset.test_mode = True
    config.test_dataloader.dataset.load_eval_anns = False
    config.test_evaluator = dict(type='DumpResults', out_file_path=str(raw_path))
    config.load_from = str(args.checkpoint)
    config.resume = False
    config.launcher = 'none'
    config.work_dir = str(args.output / 'work')
    runner = Runner.from_cfg(config)
    runner.load_or_resume()
    runner.test()
    del runner
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def render(args: argparse.Namespace, info_path: Path,
           portable_path: Path) -> tuple[int, Path]:
    from ocamcamera import OcamCamera

    info = mmengine.load(info_path)
    samples = info['data_list']
    payload = json.loads(portable_path.read_text())
    predicted_tokens = [str(frame['token']) for frame in payload['frames']]
    expected_tokens = [str(sample['token']) for sample in samples]
    if predicted_tokens != expected_tokens:
        raise ValueError('prediction tokens and synchronized sample order differ')
    predictions = detections_from_rosa(portable_path, len(samples))
    ocam = OcamCamera(filename=str(args.ocam_calibration), fov=220.)
    destination = args.output / 'rosa_synchronized_four_cameras_lidar.mp4'
    command = [
        'ffmpeg', '-y', '-loglevel', 'warning', '-f', 'rawvideo',
        '-pixel_format', 'bgr24', '-video_size', '1600x900',
        '-framerate', str(args.fps), '-i', 'pipe:0', '-an', '-c:v', 'libx264',
        '-preset', 'veryfast', '-crf', '22', '-pix_fmt', 'yuv420p',
        str(destination),
    ]
    process = subprocess.Popen(command, stdin=subprocess.PIPE)
    drawn = 0
    try:
        assert process.stdin is not None
        for index in range(0, len(samples), args.frame_step):
            sample = samples[index]
            points = load_points(resolve_lidar(
                args.prepared, info_path, sample, {}))
            image = render_model(
                sample, points, predictions[index], ocam, args.prepared,
                args, f'ROSA fine-tuned | frame {index}', None)
            process.stdin.write(image.tobytes())
            drawn += 1
            if drawn % 100 == 0:
                print(f'rendered {drawn}/{len(range(0, len(samples), args.frame_step))}',
                      flush=True)
        process.stdin.close()
        if process.wait() != 0:
            raise RuntimeError('ffmpeg failed while encoding the video')
    except Exception:
        process.kill()
        process.wait()
        raise
    return drawn, destination


def main() -> None:
    args = arguments()
    args.prepared = args.prepared.resolve()
    args.checkpoint = args.checkpoint.resolve()
    args.config = args.config.resolve()
    args.output = (args.output or args.prepared / 'rosa_inference').resolve()
    args.ocam_calibration = args.ocam_calibration.resolve()
    args.no_ground_truth = True
    if args.frame_step < 1 or args.point_stride < 1 or args.fps <= 0:
        raise ValueError('frame-step, point-stride and fps must be positive')
    if not args.checkpoint.is_file() or not args.ocam_calibration.is_file():
        raise FileNotFoundError('checkpoint or OCam calibration is missing')
    info_path = args.prepared / 'fisheye3dod_infos_test.pkl'
    if not info_path.is_file():
        raise FileNotFoundError(info_path)
    args.output.mkdir(parents=True, exist_ok=True)
    raw_path = args.output / 'predictions.pkl'
    portable_path = args.output / 'predictions.json'
    if not args.render_only:
        infer(args, info_path, raw_path)
        from projects.ROSA.tools.infer_fisheye_checkpoints import export_portable_json
        frames, boxes, classes = export_portable_json(
            raw_path, portable_path, args.checkpoint, .01)
        (args.output / 'prediction_summary.json').write_text(json.dumps({
            'checkpoint': str(args.checkpoint), 'frames': frames,
            'boxes_above_0.01': boxes, 'classes': classes,
        }, indent=2) + '\n')
    count, video = render(args, info_path, portable_path)
    print(f'Video: {video} ({count} frames)', flush=True)


if __name__ == '__main__':
    main()
