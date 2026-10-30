#!/usr/bin/env python3
"""Render any ROSA model pair on the same four cameras and LiDAR cloud.

Uses compare_retrained_fisheye.py's comparison.json. Model directories must
contain gravity-center predictions.json (or predictions_gravity_center.json).
Generates one video per model plus a side-by-side comparison for every model
against --baseline. No NMS or hidden score calibration is applied.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

from render_three_model_fisheye_lidar_comparison import (
    OcamCamera, cv2, load_points, mmengine, np, parse_sources,
    render_model, resolve_lidar,
)


def align_predictions(directory, samples):
    paths = [directory / 'predictions_gravity_center.json',
             directory / 'predictions.json']
    path = next((p for p in paths if p.is_file()), None)
    if path is None:
        raise FileNotFoundError(f'No portable predictions in {directory}')
    payload = json.loads(path.read_text())
    if payload.get('box_origin') != 'gravity_center':
        raise ValueError(f'{path}: export explicit gravity-center boxes first')
    rows = payload['frames']
    by_token = {str(row['token']): row for row in rows}
    tokens = [str(s['token']) for s in samples]
    if len(rows) != len(by_token) or set(by_token) != set(tokens):
        raise ValueError(f'{path}: prediction/validation tokens differ')
    result = []
    for token in tokens:
        row = by_token[token]
        result.append([(str(name), np.asarray(box, dtype=float)[:7], float(score))
                       for name, box, score in zip(row['names'], row['boxes_3d'], row['scores'])])
    return result, payload['score_threshold']


def writer(destination, width, height, fps):
    return subprocess.Popen([
        'ffmpeg', '-y', '-loglevel', 'error', '-threads', '2',
        '-f', 'rawvideo', '-pixel_format', 'bgr24', '-video_size', f'{width}x{height}',
        '-framerate', str(fps), '-i', '-', '-an', '-c:v', 'libx264',
        '-threads', '2', '-preset', 'fast', '-crf', '20', '-pix_fmt', 'yuv420p',
        str(destination)], stdin=subprocess.PIPE)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--data-root', type=Path, required=True)
    parser.add_argument('--comparison', type=Path, required=True)
    parser.add_argument('--model', action='append', required=True)
    parser.add_argument('--baseline', default='baseline')
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--score-threshold', type=float, default=.1)
    parser.add_argument('--bev-range', type=float, default=50.)
    parser.add_argument('--point-stride', type=int, default=2)
    parser.add_argument('--fps', type=float, default=2.)
    parser.add_argument('--start-frame', type=int, default=0)
    parser.add_argument('--end-frame', type=int)
    parser.add_argument('--frame-step', type=int, default=1)
    parser.add_argument('--no-ground-truth', action='store_true')
    parser.add_argument('--source-dataset', action='append', default=[])
    args = parser.parse_args()
    if args.baseline not in args.model:
        parser.error('--baseline must be included in --model')
    if args.frame_step < 1 or args.fps <= 0:
        parser.error('frame-step and fps must be positive')
    args.output_dir.mkdir(parents=True, exist_ok=True)
    info_path = args.data_root / 'rosa/fisheye3dod_infos_val.pkl'
    samples = mmengine.load(info_path)['data_list']
    comparison = json.loads(args.comparison.read_text())
    if comparison.get('validation_sha256') != hashlib.sha256(info_path.read_bytes()).hexdigest():
        raise ValueError('Comparison metrics belong to different validation annotations')
    models = {name: comparison['models'][name] for name in args.model}
    predictions = {}
    for name, record in models.items():
        predictions[name], minimum = align_predictions(Path(record['directory']), samples)
        if args.score_threshold < minimum:
            raise ValueError(f'{name}: requested threshold below portable export floor {minimum}')
    ocam = OcamCamera(filename=str(args.data_root / 'calib_results.txt'), fov=220.)
    sources = parse_sources(args.data_root, args.source_dataset)
    end = len(samples) if args.end_frame is None else args.end_frame
    if args.start_frame < 0 or end > len(samples) or end <= args.start_frame:
        raise ValueError('Invalid frame interval (end-frame is exclusive)')
    indices = list(range(args.start_frame, end, args.frame_step))
    if not indices:
        raise ValueError('No selected frames')
    pairs = [name for name in models if name != args.baseline]
    writers = {name: writer(args.output_dir / f'{name}.mp4', 1600, 900, args.fps)
               for name in models}
    for name in pairs:
        writers[f'{args.baseline}_vs_{name}'] = writer(
            args.output_dir / f'{args.baseline}_vs_{name}.mp4', 3200, 900, args.fps)
    manifest = dict(validation_samples=len(samples), rendered_frames=len(indices),
                    score_threshold=args.score_threshold, nms=False, fps=args.fps,
                    box_origin='gravity_center', gt='white dashed pseudo-labels',
                    timeline='validation order; discontinuous temporal blocks and two bags',
                    models=models, frame_tokens=[str(samples[i]['token']) for i in indices])
    failed = False
    try:
        for count, index in enumerate(indices):
            sample = samples[index]
            points = load_points(resolve_lidar(args.data_root, info_path, sample, sources))
            panels = {}
            for name, record in models.items():
                panels[name] = render_model(sample, points, predictions[name][index],
                    ocam, args.data_root, args, name.replace('_', ' '), record['metrics'])
                writers[name].stdin.write(panels[name].tobytes())
                if count == 0:
                    cv2.imwrite(str(args.output_dir / f'{name}_preview.jpg'), panels[name])
            for name in pairs:
                combined = np.concatenate([panels[args.baseline], panels[name]], axis=1)
                writers[f'{args.baseline}_vs_{name}'].stdin.write(combined.tobytes())
                if count == 0:
                    cv2.imwrite(str(args.output_dir / f'{args.baseline}_vs_{name}_preview.jpg'), combined)
            if count % 25 == 0 or count + 1 == len(indices):
                print(f'Rendered {count + 1}/{len(indices)}', flush=True)
    finally:
        for process in writers.values():
            process.stdin.close()
        for process in writers.values():
            failed |= process.wait() != 0
    if failed:
        raise RuntimeError('Video encoding failed')
    (args.output_dir / 'manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')


if __name__ == '__main__':
    main()
