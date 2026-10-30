#!/usr/bin/env python3
"""Decode the four surround-camera MP4/H.264 chunks to timestamped images."""

from __future__ import annotations

import argparse
import csv
import json
import re
import shutil
import subprocess
import sys
import tempfile
from collections import deque
from pathlib import Path

from PIL import Image
from tqdm import tqdm

from .bag_io import read_index


from .config import CAMERA_DIRS


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument('--cameras', nargs='+', choices=CAMERA_DIRS,
                        default=list(CAMERA_DIRS))
    parser.add_argument('--format', choices=('jpg', 'png'), default='jpg')
    parser.add_argument('--jpeg-quality', type=int, default=95)
    parser.add_argument('--limit', type=int, help='maximum frames per camera')
    parser.add_argument(
        '--corrupt-policy', choices=('skip', 'error'), default='skip',
        help='how to handle packets that FFmpeg cannot decode (default: skip)')
    parser.add_argument('--overwrite', action='store_true')
    return parser.parse_args()


def packet_timing(chunk: Path) -> tuple[list[int], str]:
    """Return packet PTS values and stream time base for loss-safe decoding."""
    command = [
        'ffprobe', '-v', 'error', '-select_streams', 'v:0', '-show_packets',
        '-show_entries', 'packet=pts:stream=time_base', '-of', 'json',
        str(chunk),
    ]
    result = subprocess.run(command, check=True, capture_output=True, text=True)
    document = json.loads(result.stdout)
    packets = document.get('packets', [])
    try:
        pts_values = [int(packet['pts']) for packet in packets]
    except (KeyError, TypeError, ValueError) as error:
        raise RuntimeError(f'{chunk}: a video packet has no usable PTS') from error
    streams = document.get('streams', [])
    if len(streams) != 1:
        raise RuntimeError(f'{chunk}: expected one video stream, found {len(streams)}')
    time_base = str(streams[0].get('time_base', ''))
    match = re.fullmatch(r'(\d+)/(\d+)', time_base)
    if not match or int(match.group(1)) <= 0 or int(match.group(2)) <= 0:
        raise RuntimeError(f'{chunk}: invalid video stream time base {time_base!r}')
    # FFmpeg 8 removed the special ``demux`` value for -enc_time_base.
    # An explicit rational works on both old and new FFmpeg releases.
    return pts_values, time_base.replace('/', ':')


def decode_camera(args: argparse.Namespace, sensor: str) -> list[dict[str, object]]:
    source = args.bag / CAMERA_DIRS[sensor]
    chunks = sorted(source.glob('*.h264'))
    if not chunks:
        raise FileNotFoundError(f'no .h264 chunks found in {source}')
    image_dir = args.output / 'images' / sensor.lower()
    image_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, object]] = []
    indexed_total = sum(
        len(read_index(chunk.with_suffix('.idx'))) for chunk in chunks)
    progress_total = (indexed_total if args.limit is None else
                      min(indexed_total, args.limit))
    progress = tqdm(total=progress_total, desc=sensor, unit='frame',
                    dynamic_ncols=True, disable=getattr(args, 'quiet_progress', False))

    try:
        for chunk in chunks:
            timestamps = read_index(chunk.with_suffix('.idx'))
            pts_values, time_base = packet_timing(chunk)
            if len(pts_values) != len(timestamps):
                raise RuntimeError(
                    f'{chunk}: found {len(pts_values)} video packets but index has '
                    f'{len(timestamps)} timestamps; cannot map frames safely')
            if len(set(pts_values)) != len(pts_values):
                raise RuntimeError(f'{chunk}: duplicate packet PTS values')
            pts_to_index = {pts: index for index, pts in enumerate(pts_values)}
            remaining = None if args.limit is None else args.limit - len(rows)
            expected = len(timestamps) if remaining is None else min(
                len(timestamps), remaining)
            # Stage on the destination filesystem: rename is then O(1), not a
            # potentially expensive cross-filesystem copy from /tmp.
            with tempfile.TemporaryDirectory(prefix='.camera-decode-', dir=image_dir) as temporary:
                # Output names are deliberately sequential. Original PTS values
                # are captured by showinfo before setpts replaces a corrupt or
                # non-monotonic timeline with encoder-safe sequential values.
                pattern = str(Path(temporary) / f'%08d.{args.format}')
                # -vsync 0 is supported by Ubuntu FFmpeg 4.4 and later.
                # It avoids output-frame duplication and preserves PTS mapping.
                command = [
                    'ffmpeg', '-v', 'info', '-copyts', '-threads',
                    str(getattr(args, 'ffmpeg_threads', 1)), '-i', str(chunk),
                    '-vf', 'showinfo,setpts=N', '-vsync', '0',
                    '-enc_time_base', time_base, '-progress', 'pipe:2',
                    '-nostats',
                ]
                if remaining is not None:
                    command.extend(['-frames:v', str(remaining)])
                if args.format == 'jpg':
                    # FFmpeg's qscale runs from 2 (best practical quality) to 31.
                    qscale = max(
                        2, min(31, round((100 - args.jpeg_quality) * 0.29 + 2)))
                    command.extend(['-q:v', str(qscale)])
                command.append(pattern)

                process = subprocess.Popen(
                    command, stdout=subprocess.DEVNULL, stderr=subprocess.PIPE,
                    text=True, bufsize=1)
                decoded_pts: list[int] = []
                stderr_tail: deque[str] = deque(maxlen=20)
                reported = 0
                assert process.stderr is not None
                for line in process.stderr:
                    line = line.rstrip()
                    stderr_tail.append(line)
                    timing_match = re.search(
                        r'Parsed_showinfo_.+? n:\s*(\d+) pts:\s*(-?\d+)', line)
                    if timing_match:
                        frame_number = int(timing_match.group(1))
                        if frame_number == len(decoded_pts):
                            decoded_pts.append(int(timing_match.group(2)))
                    progress_match = re.fullmatch(r'frame=(\d+)', line)
                    if progress_match:
                        current = min(int(progress_match.group(1)), expected)
                        if current > reported:
                            progress.update(current - reported)
                            reported = current
                returncode = process.wait()
                if returncode:
                    detail = '\n'.join(stderr_tail)
                    raise RuntimeError(f'{chunk}: FFmpeg failed:\n{detail}')

                decoded_files = sorted(
                    Path(temporary).glob(f'*.{args.format}'),
                    key=lambda path: int(path.stem))
                decoded = len(decoded_files)
                if decoded > expected:
                    raise RuntimeError(f'{chunk}: decoded more frames than expected')
                if len(decoded_pts) < decoded:
                    raise RuntimeError(
                        f'{chunk}: decoded {decoded} frames but captured only '
                        f'{len(decoded_pts)} original PTS values')
                decoded_indices = set()
                for decoded_file, pts in zip(decoded_files, decoded_pts):
                    if pts not in pts_to_index:
                        raise RuntimeError(
                            f'{chunk}: decoded frame has unknown packet PTS {pts}')
                    frame_index = pts_to_index[pts]
                    decoded_indices.add(frame_index)
                    with Image.open(decoded_file) as decoded_image:
                        width, height = decoded_image.size
                    timestamp = timestamps[frame_index]
                    relative = (Path('images') / sensor.lower() /
                                f'{timestamp}.{args.format}')
                    destination = args.output / relative
                    if destination.exists() and not args.overwrite:
                        raise FileExistsError(
                            f'{destination} exists; use --overwrite')
                    shutil.move(decoded_file, destination)
                    rows.append({
                        'sensor': sensor,
                        'timestamp_ns': timestamp,
                        'path': relative.as_posix(),
                        'chunk': chunk.name,
                        'frame_in_chunk': frame_index,
                        'source_pts': pts,
                        'width': width,
                        'height': height,
                    })
                if reported < expected:
                    progress.update(expected - reported)
                if args.limit is not None and len(rows) >= args.limit:
                    return rows
            missing = len(timestamps) - len(decoded_indices)
            if missing:
                message = (
                    f'{chunk}: skipped {missing} undecodable frame(s); decoded '
                    f'{decoded}/{len(timestamps)} with original timestamps preserved')
                if args.corrupt_policy == 'error':
                    raise RuntimeError(message)
                progress.write(f'warning: {message}', file=sys.stderr)
        return rows
    finally:
        progress.close()


def main() -> None:
    args = parse_args()
    args.output.mkdir(parents=True, exist_ok=True)
    all_rows: list[dict[str, object]] = []
    for sensor in args.cameras:
        rows = decode_camera(args, sensor)
        all_rows.extend(rows)
        print(f'{sensor}: wrote {len(rows)} images')
    manifest = args.output / 'camera_manifest.csv'
    with manifest.open('w', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(all_rows[0]))
        writer.writeheader()
        writer.writerows(all_rows)
    print(f'manifest: {manifest}')


if __name__ == '__main__':
    main()
