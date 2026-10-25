#!/usr/bin/env python3
"""Extract camera intrinsics and sensor-to-BASELINK transforms from a bag."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from bag_io import (bytes_value, field_map, integer, iter_record_payloads,
                    repeated_bytes, unpack_doubles)


FRAMES = [
    'ARC_F', 'ARC_R', 'ARC_SL', 'ARC_SR', 'USS_FCL', 'USS_FOL',
    'USS_FSL', 'USS_FCR', 'USS_FOR', 'USS_FSR', 'LRR', 'MFC',
    'USS_RCL', 'USS_ROL', 'USS_RSL', 'USS_RCR', 'USS_ROR', 'USS_RSR',
    'RSC_L', 'RSC_R', 'SRR_FL', 'SRR_FR', 'SRR_RL', 'SRR_RR', 'DCC',
    'RMC', 'IMC', 'CLIDAR', 'BASELINK', 'CAR_HW', 'BASE_FRAME_ROTATED',
    'IMU_EXT', 'IMU_INT',
]
CAMERAS = ('ARC_F', 'ARC_R', 'ARC_SL', 'ARC_SR')
# The bag's calibration message omits frame 27. This rounded transform was
# calculated from the static_transforms.json supplied in the shared predecessor
# conversation. The CLI can replace it when the original full-precision file is
# available.
SHARED_CLIDAR_TO_BASELINK = np.array([
    [0.99956, 0.00003, -0.02964, 0.84405],
    [0.00015, 0.99998, 0.00598, -0.00185],
    [0.02964, -0.00598, 0.99954, 1.41075],
    [0.0, 0.0, 0.0, 1.0],
])
EXTRINSICS_DIR = '__engineering__adcu__debug__ZMQ__Extrinsics'
INTRINSICS_DIR = '__engineering__adcu__debug__ZMQ__Intrinsics'


def rotation_x(angle: float) -> np.ndarray:
    """Rotation about the camera optical frame's image-x (pitch) axis."""
    cosine, sine = np.cos(angle), np.sin(angle)
    return np.array([
        [1.0, 0.0, 0.0],
        [0.0, cosine, -sine],
        [0.0, sine, cosine],
    ])


def newest_payload(bag: Path, directory: str) -> tuple[bytes, Path]:
    files = sorted((bag / directory).glob('*.bin'))
    if not files:
        raise FileNotFoundError(f'no calibration files in {bag / directory}')
    payloads = list(iter_record_payloads(files[-1]))
    if not payloads:
        raise ValueError(f'no messages in {files[-1]}')
    return payloads[-1], files[-1]


def parse_extrinsics(payload: bytes) -> dict[str, dict[str, object]]:
    result = {}
    for message in repeated_bytes(payload, 1):
        fields = field_map(message)
        source = FRAMES[integer(fields, 1)]
        target = FRAMES[integer(fields, 2)]
        matrix = np.asarray(unpack_doubles(bytes_value(fields, 3), 16)).reshape(4, 4)
        result[source] = {
            'from': source,
            'to': target,
            'matrix': matrix.tolist(),
        }
    return result


def parse_intrinsics(payload: bytes) -> dict[str, dict[str, object]]:
    result = {}
    for message in repeated_bytes(payload, 1):
        fields = field_map(message)
        frame = FRAMES[integer(fields, 1)]
        model = bytes_value(fields, 6).decode('utf-8')
        result[frame] = {
            'frame': frame,
            'camera_matrix': np.asarray(
                unpack_doubles(bytes_value(fields, 2), 9)).reshape(3, 3).tolist(),
            'distortion_coefficients': unpack_doubles(bytes_value(fields, 3)),
            'width': integer(fields, 4),
            'height': integer(fields, 5),
            'camera_model': model,
            'projection_for_validation': (
                'opencv_fisheye' if model == 'GENERIC_CAMERA_MODEL' else 'opencv_pinhole'),
        }
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag', type=Path)
    parser.add_argument('output', type=Path)
    parser.add_argument(
        '--lidar-to-base', type=float, nargs=16, metavar='VALUE',
        help='row-major CLIDAR-to-BASELINK matrix; overrides the shared transform')
    parser.add_argument(
        '--lidar-height-offset', type=float, default=0.0,
        help='add this many metres to CLIDAR height in BASELINK')
    parser.add_argument(
        '--camera-pitch', action='append', default=[], metavar='CAMERA=DEGREES',
        help='camera-local pitch correction; repeat for individual ARC cameras')
    args = parser.parse_args()

    extrinsics_payload, extrinsics_file = newest_payload(args.bag, EXTRINSICS_DIR)
    intrinsics_payload, intrinsics_file = newest_payload(args.bag, INTRINSICS_DIR)
    extrinsics = parse_extrinsics(extrinsics_payload)
    intrinsics = parse_intrinsics(intrinsics_payload)
    lidar_source = 'bag Extrinsics message'
    if args.lidar_to_base:
        matrix = np.asarray(args.lidar_to_base).reshape(4, 4)
        extrinsics['CLIDAR'] = {
            'from': 'CLIDAR', 'to': 'BASELINK', 'matrix': matrix.tolist()}
        lidar_source = '--lidar-to-base command-line override'
    elif 'CLIDAR' not in extrinsics:
        extrinsics['CLIDAR'] = {
            'from': 'CLIDAR', 'to': 'BASELINK',
            'matrix': SHARED_CLIDAR_TO_BASELINK.tolist()}
        lidar_source = (
            'rounded matrix recovered from the shared predecessor conversation; '
            'the bag Extrinsics message omits CLIDAR')
    if args.lidar_height_offset:
        matrix = np.asarray(extrinsics['CLIDAR']['matrix'], dtype=float)
        matrix[2, 3] += args.lidar_height_offset
        extrinsics['CLIDAR']['matrix'] = matrix.tolist()
        lidar_source += f'; CLIDAR height offset {args.lidar_height_offset:+.6f} m'
    camera_pitch = {}
    for specification in args.camera_pitch:
        try:
            camera, degrees_text = specification.split('=', 1)
            degrees = float(degrees_text)
        except ValueError as error:
            raise ValueError(
                f'invalid --camera-pitch {specification!r}; expected CAMERA=DEGREES') from error
        camera = camera.upper()
        if camera not in CAMERAS:
            raise ValueError(f'unknown camera {camera!r}; expected one of {CAMERAS}')
        matrix = np.asarray(extrinsics[camera]['matrix'], dtype=float)
        matrix[:3, :3] = matrix[:3, :3] @ rotation_x(np.deg2rad(degrees))
        extrinsics[camera]['matrix'] = matrix.tolist()
        camera_pitch[camera] = degrees
    lidar_to_camera = {}
    lidar_to_base = np.asarray(extrinsics['CLIDAR']['matrix'])
    for camera in CAMERAS:
        camera_to_base = np.asarray(extrinsics[camera]['matrix'])
        lidar_to_camera[camera] = (np.linalg.inv(camera_to_base) @ lidar_to_base).tolist()

    document = {
        'conventions': {
            'extrinsic_matrix': 'T_to_from; column vectors p_to = T_to_from @ p_from',
            'lidar_frame': 'CLIDAR',
            'common_frame': 'BASELINK',
            'matrix_storage': 'row-major JSON arrays',
            'generic_camera_assumption': (
                'ARC_* GENERIC_CAMERA_MODEL is validated initially with the '
                'OpenCV fisheye/equidistant equation; inspect overlays before model use.'),
        },
        'source': {
            'extrinsics': str(extrinsics_file),
            'intrinsics': str(intrinsics_file),
            'lidar_extrinsics': lidar_source,
            'camera_pitch_corrections_degrees': camera_pitch,
        },
        'extrinsics': extrinsics,
        'intrinsics': intrinsics,
        'lidar_to_camera': lidar_to_camera,
    }
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(document, indent=2) + '\n')
    missing = [camera for camera in CAMERAS
               if camera not in extrinsics or camera not in intrinsics]
    if missing:
        raise RuntimeError(f'missing calibration for {missing}')
    print(f'wrote {args.output} ({len(extrinsics)} extrinsics, '
          f'{len(intrinsics)} intrinsics)')


if __name__ == '__main__':
    main()
