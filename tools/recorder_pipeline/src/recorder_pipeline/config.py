"""Centralized recorder topic configuration."""
from pathlib import Path

CAMERA_DIRS = {
    'ARC_F': '__engineering__adcu__debug__ZMQ__camera__arc_f',
    'ARC_R': '__engineering__adcu__debug__ZMQ__camera__arc_r',
    'ARC_SL': '__engineering__adcu__debug__ZMQ__camera__arc_sl',
    'ARC_SR': '__engineering__adcu__debug__ZMQ__camera__arc_sr',
}
CAMERAS = tuple(CAMERA_DIRS)
EXTRINSICS_DIR = '__engineering__adcu__debug__ZMQ__Extrinsics'
INTRINSICS_DIR = '__engineering__adcu__debug__ZMQ__Intrinsics'
LIDAR_DIR = 'ouster__points'


def topic_dir(bag: Path, topic: str) -> Path:
    return bag / topic
