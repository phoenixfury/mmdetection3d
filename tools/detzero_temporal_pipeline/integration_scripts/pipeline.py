#!/usr/bin/env python3
"""One command: raw BAG -> synchronized dataset -> DetZero -> improve -> videos -> combined PKLs.

Run this script from any working directory. All internal code is colocated in
this package. User-supplied BAGs, calibrations, checkpoints and the external
OCam preparation implementation are deliberately NOT fabricated.
"""
from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

PACKAGE = Path(__file__).resolve().parent.parent
SCRIPTS = Path(__file__).resolve().parent
STAGES = ('extract', 'detzero', 'improve', 'visualize', 'prepare', 'dataset', 'audit')


def arguments(argv: list[str] | None = None) -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    p.add_argument('--bag', type=Path, help='raw BAG directory; required only when extracting')
    p.add_argument('--sequence-name', help='unique source name in combined dataset')
    p.add_argument('--converted', type=Path, help='converted BAG directory; default: PACKAGE/runs/BAG_NAME/converted')
    p.add_argument('--dataset', type=Path, help='existing exported synchronized dataset; default: CONVERTED/dataset')
    p.add_argument('--mmdet3d-root', type=Path, default=Path(os.environ.get('MMDET3D_ROOT', '.')),
                   help='mmdetection3d checkout for CenterPoint configs and OCam adapter')
    p.add_argument('--centerpoint-config', type=Path,
                   help='CenterPoint MMDetection3D config .py; default: repo configs/centerpoint/...')
    p.add_argument('--centerpoint-checkpoint', type=Path,
                   help='default: MMDET3D_ROOT/checkpoints/centerpoint_nus_voxel0075_dcn.pth, then package assets')
    p.add_argument('--detzero-checkpoints', type=Path,
                   help='default: MMDET3D_ROOT/checkpoints_det3zero, then package assets')
    p.add_argument('--enable-position-refiner', action='store_true')
    p.add_argument('--yolo-weights', type=Path,
                   help='YOLO .pt file or directory containing one .pt file; default: MMDET3D_ROOT/checkpoints/yolo')
    p.add_argument('--calibration', type=Path,
                   help='source calibration override; default: PACKAGE/assets/calibration/calibration.json')
    p.add_argument('--ocam-calibration', type=Path,
                   help='default: package asset, then MMDET3D_ROOT/data/Fisheye3DODdataset/calib_results.txt')
    p.add_argument('--prepare-script', type=Path,
                   help='OCam preparer override; default PACKAGE/external or MMDET_ROOT/tools/custom_data')
    p.add_argument('--lidar-decoder', type=Path, default=PACKAGE / 'recorder_pipeline/build/decode_lidar')
    p.add_argument('--lidar-to-base', nargs=16, type=float, help='measured CLIDAR -> BASELINK matrix')
    p.add_argument('--combined-output', type=Path,
                   help='combined dataset root; omitted means no OCam preparation or dataset building')
    p.add_argument('--dataset-mode', choices=('create', 'append', 'replace', 'extend'), default='create',
                   help='replace atomically refreshes an existing named sequence; extend creates a new output preserving a base split')
    p.add_argument('--base-dataset', type=Path, help='existing combined dataset root for --dataset-mode extend')
    p.add_argument('--sample-rate', type=float, default=2.)
    p.add_argument('--tolerance-ms', type=float, default=20.)
    p.add_argument('--camera-sync-clock', choices=('auto', 'sensor', 'read'), default='auto',
                   help='timestamp domain used to pair LiDAR with camera packets')
    p.add_argument('--fps', type=float, default=2.)
    p.add_argument('--video-stride', type=int, default=1)
    p.add_argument('--video-limit', type=int)
    p.add_argument('--lidar-workers', type=int, default=4,
                   help='parallel PCL LiDAR chunk decoders')
    p.add_argument('--start-at', choices=STAGES, default='extract')
    p.add_argument('--stop-after', choices=STAGES,
                   help='default dataset if --combined-output, otherwise visualize')
    p.add_argument('--overwrite-extraction', action='store_true')
    p.add_argument('--extract-args', default='', help='extra recorder arguments as shell-quoted string')
    p.add_argument('--detzero-args', default='', help='extra DetZero args')
    p.add_argument('--improve-args', default='', help='extra annotation improvement args')
    p.add_argument('--video-args', default='', help='extra video args')
    p.add_argument('--prepare-args', default='', help='extra OCam preparation args')
    p.add_argument('--dataset-args', default='', help='extra combined dataset builder args')
    p.add_argument('--audit-fraction', type=float, default=.1,
                   help='random fraction from every combined BAG in the single audit video')
    p.add_argument('--audit-seed', type=int, default=3407)
    p.add_argument('--dry-run', action='store_true', help='show commands and missing assets, do not execute')
    p.add_argument('--check', action='store_true', help='preflight required files and report missing inputs')
    return p.parse_args(argv)


def choose_preparer(args: argparse.Namespace, repo: Path) -> Path:
    if args.prepare_script:
        return args.prepare_script.expanduser().resolve()
    local = PACKAGE / 'external/prepare_fisheye3dod_inference.py'
    if local.is_file():
        return local.resolve()
    return (repo / 'tools/custom_data/prepare_fisheye3dod_inference.py').resolve()


def first_existing(*paths: Path) -> Path:
    """Return the first existing candidate, retaining the first for diagnostics."""
    return next((path for path in paths if path.exists()), paths[0])


def resolve_yolo(value: Path | None, repo: Path) -> Path:
    candidate = (value.expanduser().resolve() if value else
                 first_existing(repo / 'checkpoints/yolo', PACKAGE / 'assets/yolo11m-seg.pt'))
    if candidate.is_dir():
        weights = sorted(candidate.glob('*.pt'))
        if len(weights) != 1:
            raise ValueError(
                f'{candidate}: expected exactly one YOLO .pt file, found {len(weights)}; '
                'pass --yolo-weights FILE')
        return weights[0].resolve()
    return candidate.resolve()


def cuda_available() -> bool:
    try:
        import torch
        return bool(torch.cuda.is_available() and torch.cuda.device_count())
    except (ImportError, RuntimeError):
        return False


def commands(args: argparse.Namespace) -> tuple[list[tuple[str, list[str], Path]], dict]:
    repo = args.mmdet3d_root.expanduser().resolve()
    bag = args.bag.expanduser().resolve() if args.bag else None
    sequence_name = args.sequence_name or (bag.name if bag else None)
    if not sequence_name:
        sequence_name = 'sequence'
    converted = (args.converted or PACKAGE / 'runs' / sequence_name / 'converted').expanduser().resolve()
    dataset = (args.dataset or converted / 'dataset').expanduser().resolve()
    calibration = (args.calibration.expanduser().resolve() if args.calibration else
                   first_existing(PACKAGE / 'assets/calibration/calibration.json',
                                  converted / 'calibration.json').resolve())
    ocam = (args.ocam_calibration.expanduser().resolve() if args.ocam_calibration else
            first_existing(PACKAGE / 'assets/calibration/calib_results.txt',
                           repo / 'data/Fisheye3DODdataset/calib_results.txt').resolve())
    preparer = choose_preparer(args, repo)
    combined = args.combined_output.expanduser().resolve() if args.combined_output else None
    checkpoint = (args.centerpoint_checkpoint.expanduser().resolve()
                  if args.centerpoint_checkpoint else
                  first_existing(repo / 'checkpoints/centerpoint_nus_voxel0075_dcn.pth',
                                 PACKAGE / 'assets/centerpoint_nus_voxel0075_dcn.pth').resolve())
    yolo = resolve_yolo(args.yolo_weights, repo)
    detzero_checkpoints = (args.detzero_checkpoints.expanduser().resolve()
                           if args.detzero_checkpoints else
                           first_existing(repo / 'checkpoints_det3zero',
                                          PACKAGE / 'assets/checkpoints_det3zero').resolve())
    decoder = args.lidar_decoder.expanduser().resolve()
    config = ((args.centerpoint_config.expanduser().resolve() if args.centerpoint_config else
               repo / 'configs/centerpoint/centerpoint_voxel0075_second_secfpn_'
                      'head-dcn-circlenms_8xb4-cyclic-20e_nus-3d.py'))
    if args.dataset_mode in ('create', 'extend') and combined is not None and combined.exists():
        raise FileExistsError(f'{combined}: already exists; choose --dataset-mode append or new output')
    if args.dataset_mode in ('append', 'replace') and combined is not None and not combined.is_dir():
        raise FileNotFoundError(
            f'{combined}: {args.dataset_mode} requires an existing combined dataset')
    if args.dataset_mode == 'extend':
        if args.base_dataset is None:
            raise ValueError('--dataset-mode extend requires --base-dataset')
        if not args.base_dataset.expanduser().resolve().is_dir():
            raise FileNotFoundError(f'base dataset missing: {args.base_dataset}')
        if combined is not None and combined.exists():
            raise FileExistsError(f'extend output already exists: {combined}')
    if args.video_limit is not None and args.video_limit < 1:
        raise ValueError('--video-limit must be positive')
    if args.sample_rate <= 0 or args.tolerance_ms < 0 or args.fps <= 0 or args.video_stride < 1:
        raise ValueError('sample rate, fps, stride must be positive; tolerance cannot be negative')
    # More transparent than indexing tricks: stages in a closed interval.
    if not 0 < args.audit_fraction <= 1:
        raise ValueError('--audit-fraction must be in (0, 1]')
    start, end = STAGES.index(args.start_at), STAGES.index(args.stop_after or ('audit' if combined else 'visualize'))
    if start > end:
        raise ValueError('--start-at is after --stop-after')
    selected = STAGES[start:end + 1]
    if 'dataset' in selected and combined is None:
        raise ValueError('dataset stage requires --combined-output')
    if 'audit' in selected and combined is None:
        raise ValueError('audit stage requires --combined-output')
    if 'prepare' in selected and combined is None:
        raise ValueError('prepare stage requires --combined-output')
    if 'extract' in selected and bag is None:
        raise ValueError('extract stage requires --bag')
    cmd = []
    if 'extract' in selected:
        line = [sys.executable, '-m', 'recorder_pipeline.cli', 'all', str(bag),
                '--output', str(converted), '--lidar-decoder', str(decoder),
                '--calibration-fallback', str(calibration),
                '--lidar-workers', str(args.lidar_workers),
                '--camera-sync-clock', args.camera_sync_clock, '--sample-rate', str(args.sample_rate),
                '--tolerance-ms', str(args.tolerance_ms)]
        if args.lidar_to_base:
            line += ['--lidar-to-base', *map(str, args.lidar_to_base)]
        if args.overwrite_extraction:
            line += ['--overwrite']
        line += shlex.split(args.extract_args)
        cmd.append(('extract', line, repo))
    if 'detzero' in selected:
        line = [sys.executable, str(SCRIPTS / 'run_detzero_sequence.py'), str(dataset),
                '--stage', 'all', '--calibration', str(calibration),
                '--checkpoint', str(checkpoint), '--config', str(config),
                '--detzero-checkpoints', str(detzero_checkpoints),
                '--tracking-profile', 'camera-2hz', '--tracking-filter', 'ab3dmot']
        if args.enable_position_refiner:
            line += ['--enable-position-refiner']
        line += shlex.split(args.detzero_args)
        cmd.append(('detzero', line, repo))
    if 'improve' in selected:
        cmd.append(('improve', [sys.executable, str(SCRIPTS / 'improve_detzero_annotations.py'),
                               str(dataset), '--weights', str(yolo), '--calibration', str(calibration),
                               *shlex.split(args.improve_args)], repo))
    if 'visualize' in selected:
        line = [sys.executable, str(SCRIPTS / 'render_improved_detzero.py'), str(dataset),
                '--calibration', str(calibration), '--fps', str(args.fps),
                '--stride', str(args.video_stride)]
        if args.video_limit is not None:
            line += ['--limit', str(args.video_limit)]
        line += shlex.split(args.video_args)
        cmd.append(('visualize', line, repo))
    if 'prepare' in selected:
        cmd.append(('prepare', [sys.executable, str(preparer), str(dataset), str(calibration),
                                '--sync', str(dataset / 'manifest.csv'),
                                '--output', str(dataset / 'fisheye3dod_inference_2hz'),
                                '--sample-rate', '0', *shlex.split(args.prepare_args)], repo))
    if 'dataset' in selected:
        if args.dataset_mode == 'create':
            line = [sys.executable, str(SCRIPTS / 'build_combined_dataset.py'),
                    '--output', str(combined), '--ocam-calibration', str(ocam),
                    '--sequence', sequence_name, str(dataset),
                    '--source-calibration', str(calibration)]
        elif args.dataset_mode == 'append':
            line = [sys.executable, str(SCRIPTS / 'append_improved_detzero_rosa_dataset.py'),
                    '--dataset', str(combined), '--ocam-calibration', str(ocam),
                    '--add-sequence', sequence_name, str(dataset), '--refit-anchors',
                    '--source-calibration', str(calibration)]
        elif args.dataset_mode == 'replace':
            line = [sys.executable, str(SCRIPTS / 'append_improved_detzero_rosa_dataset.py'),
                    '--dataset', str(combined), '--ocam-calibration', str(ocam),
                    '--replace-sequence', sequence_name, str(dataset), '--refit-anchors',
                    '--source-calibration', str(calibration)]
        else:
            line = [sys.executable, str(SCRIPTS / 'build_chronological_combined_dataset.py'),
                    '--base', str(args.base_dataset.expanduser().resolve()),
                    '--output', str(combined), '--add-sequence', sequence_name, str(dataset),
                    '--source-calibration', str(calibration)]
        line += shlex.split(args.dataset_args)
        cmd.append(('dataset', line, repo))
    if 'audit' in selected:
        cmd.append(('audit', [sys.executable,
                              str(SCRIPTS / 'render_combined_projection_audit.py'),
                              str(combined), '--calibration', str(calibration),
                              '--fraction', str(args.audit_fraction),
                              '--seed', str(args.audit_seed), '--fps', str(args.fps)], repo))
    assets = {'raw_bag': bag, 'existing_dataset': dataset,
              'calibration': calibration, 'centerpoint_config': config,
              'centerpoint_checkpoint': checkpoint, 'yolo_weights': yolo,
              'detzero_checkpoints': detzero_checkpoints,
              'lidar_decoder': decoder, 'ocam_preparer': preparer,
              'ocam_calibration': ocam, 'combined_output': combined}
    needed = {}
    for key, path, stage in (
        ('raw_bag', bag, 'extract'), ('lidar_decoder', decoder, 'extract'),
        ('existing_dataset', dataset / 'manifest.csv', 'detzero'),
        ('calibration', calibration, 'detzero'),
        ('centerpoint_config', config, 'detzero'),
        ('centerpoint_checkpoint', checkpoint, 'detzero'),
        ('detzero_checkpoints', detzero_checkpoints, 'never'),
        ('yolo_weights', yolo, 'improve'),
        ('ocam_preparer', preparer, 'prepare'),
        ('ocam_calibration', ocam, 'dataset'),
    ):
        # Outputs of previous stages are not required to exist before running.
        requires = (stage in selected or
                    (key == 'existing_dataset' and any(step in selected for step in STAGES[1:])) or
                    (key == 'calibration' and any(step in selected for step in ('improve', 'visualize', 'prepare', 'dataset'))))
        if 'extract' in selected and key in ('existing_dataset', 'calibration'):
            requires = False
        if key == 'detzero_checkpoints':
            requires = args.enable_position_refiner and 'detzero' in selected
        if key == 'lidar_decoder' and '--lidar-encoding raw' in args.extract_args:
            requires = False
        needed[key] = {'path': str(path) if path else None,
                       'required_now': requires,
                       'available': bool(path and path.exists())}
    cuda_required = 'detzero' in selected
    needed['cuda'] = {
        'path': 'PyTorch CUDA device',
        'required_now': cuda_required,
        'available': cuda_available() if cuda_required else True,
    }
    if 'detzero' not in selected and 'improve' in selected:
        needed['detzero_results'] = {
            'path': str(dataset / 'detzero_output/results.json'), 'required_now': True,
            'available': (dataset / 'detzero_output/results.json').is_file()}
    if 'improve' not in selected and any(step in selected for step in ('visualize', 'dataset')):
        needed['improved_results'] = {
            'path': str(dataset / 'detzero_output/improved_results.json'), 'required_now': True,
            'available': (dataset / 'detzero_output/improved_results.json').is_file()}
    if 'prepare' not in selected and 'dataset' in selected:
        needed['ocam_template'] = {
            'path': str(dataset / 'fisheye3dod_inference_2hz/fisheye3dod_infos_test.pkl'),
            'required_now': True,
            'available': (dataset / 'fisheye3dod_inference_2hz/fisheye3dod_infos_test.pkl').is_file()}
    return cmd, {'package': str(PACKAGE), 'repo': str(repo), 'dataset': str(dataset),
                 'converted': str(converted), 'sequence_name': sequence_name,
                 'selected_stages': list(selected), 'assets': assets, 'preflight': needed}


def main(argv: list[str] | None = None) -> None:
    a = arguments(argv)
    cmd, details = commands(a)
    missing = [name for name, state in details['preflight'].items()
               if state['required_now'] and not state['available']]
    if a.check or a.dry_run:
        # Keep JSON serializable; paths in assets are advisory only.
        info = {key: val for key, val in details.items() if key != 'assets'}
        info['missing'] = missing
        info['commands'] = [{'stage': name, 'command': shlex.join(line), 'cwd': str(cwd)}
                            for name, line, cwd in cmd]
        print(json.dumps(info, indent=2))
        if a.check:
            raise SystemExit(1 if missing else 0)
        return
    if missing:
        raise SystemExit('Missing required inputs: ' + ', '.join(
            f'{key}={details["preflight"][key]["path"]}' for key in missing))
    env = os.environ.copy()
    env['MMDET3D_ROOT'] = str(Path(details['repo']).resolve())
    env['DETZERO_ROOT'] = str(PACKAGE / 'DetZero')
    repository = Path(details['repo']).resolve()
    env['PYTHONPATH'] = os.pathsep.join(filter(None, [
        str(SCRIPTS), str(PACKAGE / 'external'),
        str(PACKAGE / 'recorder_pipeline/src'),
        str(repository), str(repository / 'tools/custom_data'),
        str(repository / 'projects/Fisheye3DOD'),
        str(repository / 'projects/ROSA'), env.get('PYTHONPATH')]))
    for name, line, cwd in cmd:
        print(f'\n===== {name.upper()} =====\n{shlex.join(line)}', flush=True)
        subprocess.run(line, cwd=cwd if cwd.is_dir() else PACKAGE, env=env, check=True)
    print('\nComplete. Video output: ' + str(Path(details['dataset']) / 'detzero_output/improved_videos'), flush=True)
    if a.combined_output is not None:
        print('Combined dataset: ' + str(a.combined_output.expanduser().resolve()), flush=True)


if __name__ == '__main__':
    main()
