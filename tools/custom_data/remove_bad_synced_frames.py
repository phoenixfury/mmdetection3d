#!/usr/bin/env python3
"""Remove complete synchronized samples by original dataset index."""

from __future__ import annotations

import argparse
import copy
import csv
import json
import shutil
from collections import Counter
from pathlib import Path

import mmengine


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset', type=Path)
    parser.add_argument('--ranges', required=True,
                        help='inclusive original indices, e.g. 107-120,162-164')
    parser.add_argument('--quarantine', type=Path)
    parser.add_argument('--apply', action='store_true')
    return parser.parse_args()


def parse_ranges(value: str) -> set[int]:
    result = set()
    for part in value.split(','):
        bounds = [int(item) for item in part.strip().split('-', 1)]
        lo, hi = (bounds[0], bounds[0]) if len(bounds) == 1 else bounds
        if lo < 0 or hi < lo:
            raise ValueError(f'invalid range: {part}')
        result.update(range(lo, hi + 1))
    return result


def dump_pickle_atomic(document: dict, path: Path) -> None:
    temporary = path.with_suffix(path.suffix + '.tmp')
    mmengine.dump(document, temporary, file_format='pkl')
    temporary.replace(path)


def main() -> None:
    args = parse_args()
    root = args.dataset.resolve()
    quarantine = (args.quarantine or root / '_removed_bad_frames').resolve()
    all_path = root / 'groundtruth/fisheye3dod_infos_all.pkl'
    authoritative = mmengine.load(all_path)
    samples = authoritative['data_list']
    bad_indices = parse_ranges(args.ranges)
    if not bad_indices or max(bad_indices) >= len(samples):
        raise IndexError(f'indices must be within 0..{len(samples) - 1}')

    bad = [(index, samples[index]) for index in sorted(bad_indices)]
    bad_tokens = {str(sample['token']) for _, sample in bad}
    image_paths = []
    for _, sample in bad:
        for camera in sample['cam_info']['cam_fisheye'].values():
            image_paths.append(root / camera['cam_path'])
    annotation_paths = [root / 'manual_annotations' / 'frames' / f'{token}.json'
                        for token in sorted(bad_tokens)]

    print(f'samples to remove: {len(bad)}')
    print(f'camera images to remove: {sum(path.exists() for path in image_paths)}')
    print(f'manual frame JSONs to remove: '
          f'{sum(path.exists() for path in annotation_paths)}')
    if not args.apply:
        print('dry run; pass --apply to modify the dataset')
        return
    if quarantine.exists():
        raise FileExistsError(f'quarantine already exists: {quarantine}')

    moved = []
    for path in image_paths + annotation_paths:
        if not path.exists():
            continue
        relative = path.relative_to(root)
        destination = quarantine / relative
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(path), str(destination))
        moved.append(relative.as_posix())

    kept_all = [sample for sample in samples
                if str(sample['token']) not in bad_tokens]
    ordered_tokens = [str(sample['token']) for sample in kept_all]
    token_to_index = {token: index for index, token in enumerate(ordered_tokens)}

    pickle_paths = [
        root / 'fisheye3dod_infos_test.pkl',
        root / 'groundtruth/fisheye3dod_infos_all.pkl',
        root / 'groundtruth/fisheye3dod_infos_train.pkl',
        root / 'groundtruth/fisheye3dod_infos_val.pkl',
        root / 'manual_annotations/fisheye3dod_infos_manual_all.pkl',
        root / 'manual_annotations/fisheye3dod_infos_manual_train.pkl',
        root / 'manual_annotations/fisheye3dod_infos_manual_val.pkl',
    ]
    metadata_paths = pickle_paths + [
        root / 'groundtruth/annotation_audit.csv',
        root / 'groundtruth/summary.json',
        root / 'preparation.json',
    ]
    for path in metadata_paths:
        destination = quarantine / 'metadata_original' / path.relative_to(root)
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    pickle_counts = {}
    filtered_documents = {}
    for path in pickle_paths:
        document = mmengine.load(path)
        filtered = []
        for original in document['data_list']:
            token = str(original['token'])
            if token in bad_tokens:
                continue
            sample = copy.deepcopy(original)
            index = token_to_index[token]
            sample['sample_idx'] = index
            sample['metainfo']['prev_id'] = ordered_tokens[index - 1] if index else ''
            sample['metainfo']['next_id'] = (
                ordered_tokens[index + 1] if index + 1 < len(ordered_tokens) else '')
            filtered.append(sample)
        document['data_list'] = filtered
        filtered_documents[path] = document
        pickle_counts[str(path.relative_to(root))] = len(filtered)

    audit_path = root / 'groundtruth/annotation_audit.csv'
    with audit_path.open(newline='') as stream:
        reader = csv.DictReader(stream)
        fieldnames = reader.fieldnames
        audit_rows = [row for row in reader if row['token'] not in bad_tokens]

    for path, document in filtered_documents.items():
        dump_pickle_atomic(document, path)
    audit_tmp = audit_path.with_suffix('.csv.tmp')
    with audit_tmp.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fieldnames)
        writer.writeheader(); writer.writerows(audit_rows)
    audit_tmp.replace(audit_path)

    preparation_path = root / 'preparation.json'
    preparation = json.loads(preparation_path.read_text())
    preparation['samples'] = len(kept_all)
    preparation_path.write_text(json.dumps(preparation, indent=2) + '\n')

    summary_path = root / 'groundtruth/summary.json'
    summary = json.loads(summary_path.read_text())
    all_gt = filtered_documents[all_path]['data_list']
    names = [str(name) for sample in all_gt
             for name in sample.get('annos', {}).get('gt_names', [])]
    summary['samples'] = len(all_gt)
    summary['labeled_samples'] = sum(
        len(sample.get('annos', {}).get('gt_names', [])) > 0 for sample in all_gt)
    summary['excluded_empty_samples'] = summary['samples'] - summary['labeled_samples']
    summary['train_samples'] = pickle_counts['groundtruth/fisheye3dod_infos_train.pkl']
    summary['val_samples'] = pickle_counts['groundtruth/fisheye3dod_infos_val.pkl']
    summary['accepted_boxes'] = len(names)
    summary['class_counts'] = dict(Counter(names))
    summary['filter_counts'] = dict(Counter(row['reason'] for row in audit_rows))
    summary['accepted_class_sources'] = dict(Counter(
        row['class_source'] for row in audit_rows if row['reason'] == 'accepted'))
    summary_path.write_text(json.dumps(summary, indent=2) + '\n')

    manifest = {
        'original_samples': len(samples), 'remaining_samples': len(kept_all),
        'removed_samples': len(bad), 'ranges': args.ranges,
        'removed': [{'original_index': index, 'token': str(sample['token'])}
                    for index, sample in bad],
        'moved_files': moved, 'pickle_counts': pickle_counts,
    }
    quarantine.mkdir(parents=True, exist_ok=True)
    (quarantine / 'removal_manifest.json').write_text(
        json.dumps(manifest, indent=2) + '\n')
    print(f'removed {len(bad)} synchronized samples; {len(kept_all)} remain')
    print(f'quarantined {len(moved)} files under {quarantine}')


if __name__ == '__main__':
    main()
