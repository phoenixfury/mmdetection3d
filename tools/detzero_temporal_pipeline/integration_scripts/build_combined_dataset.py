#!/usr/bin/env python3
"""Create a new ROSA/Fisheye3DOD combined dataset from improved DetZero BAGs.

Unlike the historical initial builder, this uses the same sensor-clock importer
and chronological tail holdout as incremental appending. Never overwrites an
existing dataset; publish via a staging directory after a successful build.
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import re
import shutil
import tempfile
from collections import Counter, defaultdict
from pathlib import Path

import mmengine
import numpy as np
from ocamcamera import OcamCamera

from append_improved_detzero_rosa_dataset import (
    load_sequence, assign_chronological_split, publish_pickle, publish_view,
)
from build_improved_detzero_rosa_dataset import (
    CLASSES, CATEGORIES, DEFAULT_OCAM_CALIBRATION, sha256,
    build_rosa_anchor, write_benchmark_readme,
)


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--sequence', nargs=2, metavar=('NAME', 'SYNCHRONIZED_DATASET'),
                   action='append', required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--ocam-calibration', type=Path, default=DEFAULT_OCAM_CALIBRATION)
    p.add_argument('--source-calibration', type=Path, help='calibration JSON for this incoming BAG')
    p.add_argument('--val-fraction', type=float, default=.1)
    p.add_argument('--purge-seconds', type=float, default=5.)
    p.add_argument('--max-scene-gap-seconds', type=float, default=4.)
    p.add_argument('--storage', choices=('hardlink', 'copy', 'symlink'), default='hardlink')
    p.add_argument('--num-anchors', type=int, default=300)
    p.add_argument('--anchor-seed', type=int, default=3407)
    p.add_argument('--dry-run', action='store_true')
    a = p.parse_args(argv)
    if not 0 < a.val_fraction < 1 or a.purge_seconds < 0 or a.max_scene_gap_seconds <= 0:
        p.error('invalid chronological split parameters')
    if a.num_anchors < 1:
        p.error('--num-anchors must be positive')
    sequences = [(name, Path(path).expanduser().resolve()) for name, path in a.sequence]
    if len({name for name, _ in sequences}) != len(sequences) or any(
        not re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_-]*', name) for name, _ in sequences
    ):
        p.error('sequence names must be unique and consist of letters, digits, - or _')
    output = a.output.expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'dataset exists; use append or choose a new output: {output}')
    modes = {json.loads((sequence / 'fisheye3dod_inference_2hz' /
                         'preparation.json').read_text()).get('image_mode')
             for _, sequence in sequences}
    if len(modes) != 1 or modes - {'native', 'canonical-ocam'}:
        raise ValueError(f'sequences must use one supported image mode, got {modes}')
    image_mode = next(iter(modes))
    calibration = a.ocam_calibration.expanduser().resolve()
    ocam = None
    if image_mode == 'canonical-ocam':
        if not calibration.is_file():
            raise FileNotFoundError(f'800x800 OCam calibration missing: {calibration}')
        ocam = OcamCamera(filename=str(calibration), fov=220.)
        if (ocam.width, ocam.height) != (800, 800):
            raise ValueError('shared training image calibration must be 800x800')
    output.parent.mkdir(parents=True, exist_ok=True)
    staging = (Path(tempfile.mkdtemp(prefix=f'.{output.name}.building-', dir=output.parent))
               if not a.dry_run else output)
    try:
        samples = []
        entries = []
        next_id = 0
        for name, sequence in sequences:
            new_samples, entry, next_id = load_sequence(name, sequence, staging, ocam,
                                                         a.storage, a.dry_run, next_id, False, a.source_calibration)
            samples.extend(new_samples)
            entries.append(entry)
        if len({str(s['token']) for s in samples}) != len(samples):
            raise ValueError('timestamp token collision across recordings')
        train, val, purged = assign_chronological_split(samples, a.val_fraction,
                                                        a.purge_seconds, a.max_scene_gap_seconds)
        if not train or not val:
            raise ValueError('empty chronological train or validation set')
        total_counts = Counter(str(n) for s in samples for n in s['annos']['gt_names'])
        source_counts = Counter()
        for entry in entries:
            source_counts.update(entry.get('source_counts', {}))
            by_split = defaultdict(list)
            for sample in samples:
                if sample['metainfo']['sequence_id'] == entry['name']:
                    by_split[sample['metainfo']['temporal_split']].append(
                        int(sample['metainfo']['acquisition_timestamp_ns']))
            entry['temporal_split'] = {
                key: dict(frames=len(ts), first_acquisition_ns=min(ts),
                          last_acquisition_ns=max(ts)) for key, ts in by_split.items()
            }
        anchor_name = f'kmeans{a.num_anchors}_seed{a.anchor_seed}.npy'
        summary = {
            'output': str(output), 'storage': a.storage, 'dry_run': a.dry_run,
            'frames': len(samples), 'train_frames': len(train), 'val_frames': len(val),
            'purged_frames': purged,
            'empty_frames': sum(len(s['annos']['gt_names']) == 0 for s in samples),
            'boxes': sum(total_counts.values()),
            'class_counts': dict(total_counts), 'source_counts': dict(source_counts),
            'instances': len({int(i) for sample in samples for i in sample['annos']['gt_ids']}),
            'image_mode': image_mode,
            'ocam_calibration': (str(calibration) if image_mode == 'canonical-ocam' else None),
            'ocam_calibration_sha256': (sha256(calibration) if image_mode == 'canonical-ocam' else None),
            'construction_vehicle_mapping': 'Van',
            'format_views': {
                'fisheye3dod': 'fisheye3dod/fisheye3dod_infos_{split}.pkl',
                'rosa': 'rosa/fisheye3dod_infos_{split}.pkl',
                'canonical': 'annotations/fisheye3dod_infos_{split}.pkl'},
            'rosa_anchor': {
                'path': 'rosa/anchors/' + anchor_name,
                'shape': [a.num_anchors, 11], 'seed': a.anchor_seed,
                'update': 'training split only'},
            'split': {'method': 'per-sequence chronological tail holdout',
                      'clock': 'lidar_sensor_timestamp_ns',
                      'val_fraction': a.val_fraction, 'purge_seconds': a.purge_seconds,
                      'fraction_basis': 'elapsed acquisition time within each BAG',
                      'max_scene_gap_seconds': a.max_scene_gap_seconds,
                      'scene_links_cross_split': False},
            'sequences': entries,
            'temporal_metadata': 'acquisition_timestamp_ns and T_world_CLIDAR',
            'format_views_byte_identical': True,
        }
        if a.dry_run:
            print(json.dumps(summary, indent=2))
            return
        anchor, count = build_rosa_anchor(train, a.num_anchors, a.anchor_seed)
        anchor_path = staging / summary['rosa_anchor']['path']
        anchor_path.parent.mkdir(parents=True, exist_ok=True)
        np.save(anchor_path, anchor)
        summary['rosa_anchor'].update(training_boxes=count, sha256=sha256(anchor_path))
        metadata = {'categories': CATEGORIES, 'classes': CLASSES,
                    'dataset': 'combined_improved_detzero_rosa',
                    'version': 'improved-detzero-acquisition-temporal-v2'}
        annotation_files = {}
        for split, rows in (('all', samples), ('train', train), ('val', val)):
            path = staging / 'annotations' / f'fisheye3dod_infos_{split}.pkl'
            publish_pickle(path, {'metainfo': copy.deepcopy(metadata), 'data_list': rows})
            annotation_files[split] = path
            for view in ('fisheye3dod', 'rosa', 'ImageSets-improved-detzero'):
                publish_view(path, staging / view / path.name)
        summary['annotation_sha256'] = {split: sha256(path) for split, path in annotation_files.items()}
        shutil.copy2(calibration, staging / 'calib_results.txt')
        write_benchmark_readme(staging, a.num_anchors, a.anchor_seed)
        benchmark = staging / 'BENCHMARK.md'
        benchmark.write_text(benchmark.read_text().replace(str(staging), str(output)))
        (staging / 'summary.json').write_text(json.dumps(summary, indent=2) + '\n')
        if output.exists():
            raise FileExistsError(f'dataset destination created during build: {output}')
        os.rename(staging, output)
        print(json.dumps({'output': str(output), 'frames': len(samples),
                          'train': len(train), 'val': len(val), 'purged': purged}, indent=2))
    except Exception:
        if not a.dry_run:
            print(f'Incomplete dataset kept for inspection: {staging}', flush=True)
        raise


if __name__ == '__main__':
    main()
