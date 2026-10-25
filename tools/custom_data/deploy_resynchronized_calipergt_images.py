#!/usr/bin/env python3
"""Replace a CaliperGT scene's camera files with synchronized images."""

from __future__ import annotations

import argparse
import json
import os
import re
import shutil
from pathlib import Path


CAMERAS = {
    'front': 'fisheye_camera_front',
    'left': 'fisheye_camera_left',
    'right': 'fisheye_camera_right',
    'rear': 'fisheye_camera_rear',
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('source', type=Path,
                        help='corrected images_native directory')
    parser.add_argument('scene', type=Path,
                        help='CaliperGT uploaded scene directory')
    parser.add_argument('--apply', action='store_true')
    return parser.parse_args()


def indexed_targets(directory: Path) -> dict[int, Path]:
    result = {}
    for path in directory.glob('*.jpg'):
        match = re.match(r'(\d+)_', path.name)
        if match:
            index = int(match.group(1))
            if index in result:
                raise ValueError(f'duplicate frame {index} in {directory}')
            result[index] = path
    return result


def main() -> None:
    args = parse_args()
    source = args.source.resolve()
    scene = args.scene.resolve()
    camera_root = scene / 'cameras'
    plans = []
    for camera, model_name in CAMERAS.items():
        sources = sorted((source / model_name).glob('*.jpg'),
                         key=lambda path: int(path.stem))
        targets = indexed_targets(camera_root / camera)
        if len(sources) != len(targets):
            raise ValueError(
                f'{camera}: {len(sources)} sources != {len(targets)} targets')
        expected = set(range(len(sources)))
        if set(targets) != expected:
            raise ValueError(f'{camera}: target frame indices are not contiguous')
        plans.extend((sources[index], targets[index])
                     for index in range(len(sources)))

    report = {
        'source': str(source), 'scene': str(scene),
        'frames': len(plans) // len(CAMERAS),
        'images': len(plans), 'applied': bool(args.apply),
    }
    print(json.dumps(report, indent=2))
    if not args.apply:
        return

    backup = scene / 'cameras_read_sync_backup'
    if backup.exists():
        raise FileExistsError(f'backup already exists: {backup}')
    shutil.copytree(camera_root, backup, copy_function=os.link)
    for source_path, target_path in plans:
        temporary = target_path.with_suffix('.sync.tmp.jpg')
        shutil.copy2(source_path, temporary)
        temporary.replace(target_path)


if __name__ == '__main__':
    main()
