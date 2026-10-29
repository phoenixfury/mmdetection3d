"""Atomic binary XYZI to PCD v0.7 conversion (no coordinate transform)."""
from __future__ import annotations

import os
import shutil
from pathlib import Path


def pcd_header(points: int) -> bytes:
    return (
        '# .PCD v0.7 - Point Cloud Data file format\n'
        'VERSION 0.7\n'
        'FIELDS x y z intensity\n'
        'SIZE 4 4 4 4\n'
        'TYPE F F F F\n'
        'COUNT 1 1 1 1\n'
        f'WIDTH {points}\n'
        'HEIGHT 1\n'
        'VIEWPOINT 0 0 0 1 0 0 0\n'
        f'POINTS {points}\n'
        'DATA binary\n'
    ).encode('ascii')


def convert_bin_to_pcd(source: Path, destination: Path, overwrite: bool) -> int:
    size = source.stat().st_size
    if size == 0 or size % 16:
        raise ValueError(f'{source}: expected nonempty float32 XYZI, got {size} bytes')
    points = size // 16
    if destination.exists() and not overwrite:
        raise FileExistsError(f'{destination} already exists; use --overwrite')
    destination.parent.mkdir(parents=True, exist_ok=True)
    tmp = destination.with_suffix('.tmp.pcd')
    try:
        with tmp.open('wb') as out, source.open('rb') as inp:
            out.write(pcd_header(points))
            shutil.copyfileobj(inp, out, length=8*1024*1024)
        os.replace(tmp, destination)
    finally:
        tmp.unlink(missing_ok=True)
    return points
