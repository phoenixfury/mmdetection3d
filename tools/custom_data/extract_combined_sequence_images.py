#!/usr/bin/env python3
"""Extract one sequence's camera images from a combined FishEye dataset."""

from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


CAMERAS = (
    'fisheye_camera_front',
    'fisheye_camera_left',
    'fisheye_camera_right',
    'fisheye_camera_rear',
)
VARIANT_DIRS = {
    'native': 'images_native',
    'ocam': 'images_ocam_800x800',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        'combined', type=Path,
        help='combined_caliper_fisheye3dod dataset root')
    parser.add_argument(
        'output', type=Path,
        help='standalone output root')
    parser.add_argument('--sequence', default='16_09')
    parser.add_argument(
        '--variant', choices=('native', 'ocam', 'both'), default='both')
    parser.add_argument(
        '--storage', choices=('copy', 'hardlink', 'symlink'), default='copy',
        help='how to materialize images (default: copy)')
    parser.add_argument(
        '--keep-prefix', action='store_true',
        help='keep SEQUENCE_ at the start of output filenames')
    parser.add_argument(
        '--include-calibration', action='store_true',
        help='also copy calibration.json and calib_results.txt when present')
    parser.add_argument('--overwrite', action='store_true')
    parser.add_argument('--dry-run', action='store_true')
    return parser.parse_args()


def selected_variants(value: str) -> tuple[str, ...]:
    return ('native', 'ocam') if value == 'both' else (value,)


def sequence_files(directory: Path, prefix: str) -> dict[str, Path]:
    files = {}
    for source in sorted(directory.glob(f'{prefix}*.jpg')):
        token = source.stem[len(prefix):]
        if not token or not token.isdigit():
            raise ValueError(f'unexpected sequence filename: {source}')
        if token in files:
            raise ValueError(f'duplicate token {token} in {directory}')
        files[token] = source
    if not files:
        raise FileNotFoundError(
            f'no images matching {prefix}*.jpg in {directory}')
    return files


def materialize(source: Path, destination: Path, storage: str,
                overwrite: bool) -> str:
    if destination.exists() or destination.is_symlink():
        if not overwrite:
            if (destination.is_file() and not destination.is_symlink()
                    and destination.stat().st_size == source.stat().st_size):
                return 'skipped'
            raise FileExistsError(
                f'{destination} already exists; pass --overwrite to replace it')
        destination.unlink()

    destination.parent.mkdir(parents=True, exist_ok=True)
    if storage == 'copy':
        shutil.copy2(source, destination)
    elif storage == 'hardlink':
        os.link(source, destination)
    else:
        destination.symlink_to(
            os.path.relpath(source, start=destination.parent))
    return 'written'


def main() -> None:
    args = parse_args()
    combined = args.combined.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if combined == output:
        raise ValueError('output must differ from the combined dataset root')
    if not combined.is_dir():
        raise FileNotFoundError(combined)

    prefix = f'{args.sequence}_'
    variants = selected_variants(args.variant)
    inventory: dict[str, dict[str, dict[str, Path]]] = {}
    expected_tokens: set[str] | None = None

    for variant in variants:
        directory_name = VARIANT_DIRS[variant]
        by_camera = {}
        for camera in CAMERAS:
            files = sequence_files(combined / directory_name / camera, prefix)
            tokens = set(files)
            if expected_tokens is None:
                expected_tokens = tokens
            elif tokens != expected_tokens:
                missing = sorted(expected_tokens - tokens)[:5]
                extra = sorted(tokens - expected_tokens)[:5]
                raise RuntimeError(
                    f'token mismatch for {variant}/{camera}: '
                    f'missing={missing}, extra={extra}')
            by_camera[camera] = files
        inventory[variant] = by_camera

    assert expected_tokens is not None
    counts = {'written': 0, 'skipped': 0}
    for variant in variants:
        directory_name = VARIANT_DIRS[variant]
        for camera in CAMERAS:
            for token, source in inventory[variant][camera].items():
                name = source.name if args.keep_prefix else f'{token}.jpg'
                destination = output / directory_name / camera / name
                if args.dry_run:
                    counts['written'] += 1
                else:
                    status = materialize(
                        source, destination, args.storage, args.overwrite)
                    counts[status] += 1

    calibration_files = []
    if args.include_calibration:
        for name in ('calibration.json', 'calib_results.txt'):
            source = combined / name
            if source.is_file():
                calibration_files.append(name)
                if not args.dry_run:
                    status = materialize(
                        source, output / name, 'copy', args.overwrite)
                    counts[status] += 1

    summary = {
        'source': str(combined),
        'output': str(output),
        'sequence': args.sequence,
        'variants': list(variants),
        'storage': args.storage,
        'frames': len(expected_tokens),
        'cameras': list(CAMERAS),
        'images': len(expected_tokens) * len(CAMERAS) * len(variants),
        'filename_prefix_preserved': args.keep_prefix,
        'calibration_files': calibration_files,
        'written': counts['written'],
        'skipped': counts['skipped'],
        'dry_run': args.dry_run,
    }
    if not args.dry_run:
        output.mkdir(parents=True, exist_ok=True)
        (output / 'extraction_summary.json').write_text(
            json.dumps(summary, indent=2) + '\n')
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
