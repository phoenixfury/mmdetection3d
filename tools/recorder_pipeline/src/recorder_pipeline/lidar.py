"""Raw PointCloud decoder. PCL octree decoding is a separate C++ backend.

Raw decoding MUST be explicitly requested: protobuf PointCloud.data carries no
encoding discriminator. Bytes divisible by 16 do not prove that data is raw.
"""
from __future__ import annotations

import csv
import os
from pathlib import Path
import numpy as np

from .bag_io import BagFormatError, iter_bag_topic_samples
from .protos import lidar_pb2


FIELDNAMES = ('timestamp_ns','sensor_timestamp_ns','path','chunk','scan_in_chunk','points','encoding','intensity_source')


def decode_raw(bag: Path, output: Path, limit: int = 0,
               overwrite: bool = False, components: int = 4) -> int:
    if components not in (3, 4):
        raise ValueError('raw point layout supports XYZ(3) or XYZI(4) components')
    if limit < 0:
        raise ValueError('limit must be >= 0')
    source = bag/'ouster__points'
    chunks = sorted(source.glob('*.bin'))
    if not chunks:
        raise FileNotFoundError(f'no LiDAR binary chunks in {source}')
    manifest = output/'lidar_manifest.csv'
    if manifest.exists() and not overwrite:
        raise FileExistsError(f'{manifest} exists; use --overwrite')
    cloud_dir = output/'lidar'
    cloud_dir.mkdir(parents=True, exist_ok=True)
    seen = set()
    total = 0
    temporary_manifest = output/'lidar_manifest.tmp.csv'
    try:
        with temporary_manifest.open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=FIELDNAMES)
            writer.writeheader()
            for wall_ns, sample, chunk, n in iter_bag_topic_samples(source):
                msg = lidar_pb2.PointCloud()
                sample_data = sample.data
                if not sample_data:
                    raise BagFormatError(f'{chunk} sample {n}: empty bag.Sample.data')
                try:
                    msg.ParseFromString(sample_data)
                except Exception as error:
                    raise BagFormatError(f'{chunk} sample {n}: invalid PointCloud: {error}') from error
                data = msg.data
                if msg.rows <= 0 or not data:
                    raise BagFormatError(f'{chunk} sample {n}: empty points or invalid rows')
                expected = msg.rows * components * 4
                if len(data) != expected:
                    raise BagFormatError(
                        f'{chunk} sample {n}: {len(data)} raw bytes, expected {expected} '
                        f'for {msg.rows} points * {components} floats; check --lidar-encoding pcl '
                        f'or --raw-components')
                if wall_ns in seen:
                    raise BagFormatError(f'duplicate LiDAR index timestamp {wall_ns}')
                seen.add(wall_ns)
                cloud = np.frombuffer(data, dtype='<f4').reshape(-1, components)
                if not np.isfinite(cloud[:, :3]).all():
                    raise BagFormatError(f'{chunk} sample {n}: nonfinite XYZ values')
                if components == 3:
                    xyzi = np.zeros((msg.rows, 4), dtype='<f4')
                    xyzi[:, :3] = cloud
                    raw_bytes = memoryview(xyzi).cast('B')
                else:
                    # msg.data is already validated packed little-endian float32
                    # XYZI. Avoid building a duplicate array and calling tobytes().
                    raw_bytes = data
                relative = Path('lidar')/f'{wall_ns}.bin'
                path = output/relative
                if path.exists() and not overwrite:
                    raise FileExistsError(f'{path} already exists; use --overwrite')
                tmp = path.with_suffix('.tmp.bin')
                try:
                    with tmp.open('wb') as destination:
                        destination.write(raw_bytes)
                    os.replace(tmp, path)
                finally:
                    tmp.unlink(missing_ok=True)
                writer.writerow(dict(timestamp_ns=wall_ns, sensor_timestamp_ns=msg.timestamp,
                                     path=relative.as_posix(), chunk=chunk.name,
                                     scan_in_chunk=n, points=msg.rows, encoding='raw',
                                     intensity_source='source' if components == 4 else 'zeros'))
                total += 1
                if limit and total >= limit:
                    break
        if not total:
            raise BagFormatError('no raw LiDAR samples were decoded')
        os.replace(temporary_manifest, manifest)
    finally:
        temporary_manifest.unlink(missing_ok=True)
    return total
