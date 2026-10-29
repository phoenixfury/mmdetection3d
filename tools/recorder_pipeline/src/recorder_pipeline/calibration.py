"""Calibration extraction with explicit transform provenance."""
from __future__ import annotations
import argparse
import json
from pathlib import Path
import numpy as np
from .bag_io import bytes_value, field_map, integer, iter_record_payloads, repeated_bytes, unpack_doubles
from .config import CAMERAS, EXTRINSICS_DIR, INTRINSICS_DIR

FRAMES = ('ARC_F','ARC_R','ARC_SL','ARC_SR','USS_FCL','USS_FOL','USS_FSL',
          'USS_FCR','USS_FOR','USS_FSR','LRR','MFC','USS_RCL','USS_ROL',
          'USS_RSL','USS_RCR','USS_ROR','USS_RSR','RSC_L','RSC_R','SRR_FL',
          'SRR_FR','SRR_RL','SRR_RR','DCC','RMC','IMC','CLIDAR','BASELINK',
          'CAR_HW','BASE_FRAME_ROTATED','IMU_EXT','IMU_INT')


def newest_payload(bag: Path, topic: str) -> tuple[bytes, Path]:
    chunks = sorted((bag / topic).glob('*.bin'))
    if not chunks:
        raise FileNotFoundError(f'no calibration chunks under {bag / topic}')
    for chunk in reversed(chunks):
        payloads = list(iter_record_payloads(chunk))
        if payloads:
            return payloads[-1], chunk
    raise ValueError(f'no calibration messages for {topic}')


def parse_extrinsics(payload: bytes) -> dict:
    result = {}
    for message in repeated_bytes(payload, 1):
        fields = field_map(message)
        source, target = FRAMES[integer(fields, 1)], FRAMES[integer(fields, 2)]
        matrix = np.asarray(unpack_doubles(bytes_value(fields, 3), 16)).reshape(4, 4)
        result[source] = {'from': source, 'to': target, 'matrix': matrix.tolist()}
    return result


def parse_intrinsics(payload: bytes) -> dict:
    result = {}
    for message in repeated_bytes(payload, 1):
        f = field_map(message)
        frame = FRAMES[integer(f, 1)]
        model = bytes_value(f, 6).decode('utf-8')
        result[frame] = {'frame': frame,
                         'camera_matrix': np.asarray(unpack_doubles(bytes_value(f, 2), 9)).reshape(3,3).tolist(),
                         'distortion_coefficients': unpack_doubles(bytes_value(f, 3)),
                         'width': integer(f, 4), 'height': integer(f, 5),
                         'camera_model': model,
                         'projection_for_validation': 'opencv_fisheye' if model == 'GENERIC_CAMERA_MODEL' else 'opencv_pinhole'}
    return result


def rotation_x(degrees: float) -> np.ndarray:
    angle = np.deg2rad(degrees)
    c, s = np.cos(angle), np.sin(angle)
    return np.array([[1,0,0],[0,c,-s],[0,s,c]])


def extract(bag: Path, output: Path, lidar_to_base: list[float] | None = None,
            camera_pitch: list[str] = ()) -> dict:
    """Export camera and optional LiDAR calibration without guessing extrinsics."""
    e_payload, e_file = newest_payload(bag, EXTRINSICS_DIR)
    i_payload, i_file = newest_payload(bag, INTRINSICS_DIR)
    extrinsics, intrinsics = parse_extrinsics(e_payload), parse_intrinsics(i_payload)
    if lidar_to_base is not None:
        mat = np.asarray(lidar_to_base, dtype=float).reshape(4, 4)
        extrinsics['CLIDAR'] = {'from': 'CLIDAR', 'to': 'BASELINK', 'matrix': mat.tolist()}
    missing = [name for name in CAMERAS if name not in extrinsics or name not in intrinsics]
    if missing:
        raise ValueError(f'missing camera intrinsics/extrinsics: {missing}')
    corrections = {}
    for spec in camera_pitch:
        name, degrees = spec.split('=', 1)
        name = name.upper()
        if name not in CAMERAS:
            raise ValueError(f'unknown camera: {name}')
        mat = np.asarray(extrinsics[name]['matrix'], dtype=float)
        mat[:3, :3] = mat[:3, :3] @ rotation_x(float(degrees))
        extrinsics[name]['matrix'] = mat.tolist()
        corrections[name] = float(degrees)
    for name in (*CAMERAS, *(['CLIDAR'] if 'CLIDAR' in extrinsics else [])):
        if extrinsics[name]['to'] != 'BASELINK':
            raise ValueError(f'{name}: expected a direct transform into BASELINK, '
                             f"got {extrinsics[name]['to']}")
        matrix = np.asarray(extrinsics[name]['matrix'])
        if not (np.isfinite(matrix).all() and np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-4)):
            raise ValueError(f'{name}: invalid homogeneous transform matrix')
        if not np.allclose(matrix[:3,:3] @ matrix[:3,:3].T, np.eye(3), atol=.05):
            raise ValueError(f'{name}: extrinsic rotation is not orthonormal')
    lidar_to_camera = {}
    if 'CLIDAR' in extrinsics:
        lidar_to_base_matrix = np.asarray(extrinsics['CLIDAR']['matrix'])
        for name in CAMERAS:
            camera_to_base = np.asarray(extrinsics[name]['matrix'])
            lidar_to_camera[name] = (np.linalg.inv(camera_to_base) @ lidar_to_base_matrix).tolist()
    doc = {
        'conventions': {'extrinsic_matrix': 'T_target_source applied to column vectors',
                        'common_frame': 'BASELINK', 'lidar_frame': 'CLIDAR'},
        'source': {'extrinsics': str(e_file), 'intrinsics': str(i_file),
                   'camera_pitch_corrections_degrees': corrections,
                   'lidar_extrinsic': ('command-line override' if lidar_to_base is not None
                                       else 'bag value' if 'CLIDAR' in extrinsics
                                       else 'missing; no substitution performed')},
        'extrinsics': extrinsics, 'intrinsics': intrinsics,
        'lidar_to_camera': lidar_to_camera,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    tmp = output.with_suffix('.tmp.json')
    try:
        tmp.write_text(json.dumps(doc, indent=2) + '\n')
        tmp.replace(output)
    finally:
        tmp.unlink(missing_ok=True)
    return doc


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('bag',type=Path)
    parser.add_argument('output',type=Path)
    parser.add_argument('--lidar-to-base',type=float,nargs=16)
    parser.add_argument('--camera-pitch',action='append',default=[])
    args = parser.parse_args(argv)
    extract(args.bag,args.output,args.lidar_to_base,args.camera_pitch)
    print(f'Calibration: {args.output}')

if __name__ == '__main__':
    main()
