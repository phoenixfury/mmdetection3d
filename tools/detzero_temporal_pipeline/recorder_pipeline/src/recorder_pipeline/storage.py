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


def reuse_binary_pcd(source: Path, destination: Path, overwrite: bool) -> int:
    """Reuse a validated legacy XYZI cloud when its source BIN was pruned."""
    with source.open('rb') as stream:
        header = {}
        for _ in range(32):
            line = stream.readline(256)
            if not line or len(line) >= 256:
                raise ValueError(f'{source}: invalid PCD header')
            fields = line.decode('ascii').strip().split()
            if fields and not fields[0].startswith('#'):
                header[fields[0]] = fields[1:]
            if fields[:2] == ['DATA', 'binary']:
                break
        else:
            raise ValueError(f'{source}: missing binary PCD payload')
        payload_offset = stream.tell()
    if (header.get('FIELDS') != ['x', 'y', 'z', 'intensity'] or
            header.get('SIZE') != ['4', '4', '4', '4'] or
            header.get('TYPE') != ['F', 'F', 'F', 'F'] or
            header.get('COUNT') != ['1', '1', '1', '1'] or
            header.get('DATA') != ['binary']):
        raise ValueError(f'{source}: expected binary float32 XYZI PCD')
    points = int(header['POINTS'][0])
    if points <= 0 or source.stat().st_size - payload_offset != points * 16:
        raise ValueError(f'{source}: invalid XYZI point count or payload size')
    if destination.exists() and not overwrite:
        raise FileExistsError(f'{destination} already exists; use --overwrite')
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_suffix('.tmp.pcd')
    try:
        try:
            os.link(source, temporary)
        except OSError:
            shutil.copy2(source, temporary)
        os.replace(temporary, destination)
    finally:
        temporary.unlink(missing_ok=True)
    return points
