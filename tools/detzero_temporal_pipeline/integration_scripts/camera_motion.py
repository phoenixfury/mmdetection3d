"""Ego-motion and planar-object compensation for asynchronous fisheye frames.

Boxes and LiDAR points are expressed at the LiDAR acquisition time in CLIDAR.
A camera detection is indexed by recorder packet time, NOT verified exposure
start. Only sensor-clock synchronization permits a conditional comparison.
Out-of-range or missing timestamps disable compensation (no extrapolation).
"""
from __future__ import annotations

import bisect
import math
import numpy as np


def quaternion_from_rotation(matrix):
    r = np.asarray(matrix, dtype=float)
    if r.shape != (3, 3):
        raise ValueError('rotation must be 3x3')
    trace = np.trace(r)
    if trace > 0:
        s = math.sqrt(trace + 1.0) * 2
        q = np.array([0.25 * s, (r[2, 1]-r[1, 2])/s,
                      (r[0, 2]-r[2, 0])/s, (r[1, 0]-r[0, 1])/s])
    else:
        i = int(np.argmax(np.diag(r)))
        if i == 0:
            s = math.sqrt(max(0., 1 + r[0,0]-r[1,1]-r[2,2])) * 2
            q = np.array([(r[2,1]-r[1,2])/s, .25*s,
                          (r[0,1]+r[1,0])/s, (r[0,2]+r[2,0])/s])
        elif i == 1:
            s = math.sqrt(max(0., 1 + r[1,1]-r[0,0]-r[2,2])) * 2
            q = np.array([(r[0,2]-r[2,0])/s, (r[0,1]+r[1,0])/s,
                          .25*s, (r[1,2]+r[2,1])/s])
        else:
            s = math.sqrt(max(0., 1 + r[2,2]-r[0,0]-r[1,1])) * 2
            q = np.array([(r[1,0]-r[0,1])/s, (r[0,2]+r[2,0])/s,
                          (r[1,2]+r[2,1])/s, .25*s])
    return q / np.linalg.norm(q)


def rotation_from_quaternion(q):
    w, x, y, z = np.asarray(q, float) / np.linalg.norm(q)
    return np.array([
        [1-2*(y*y+z*z), 2*(x*y-z*w), 2*(x*z+y*w)],
        [2*(x*y+z*w), 1-2*(x*x+z*z), 2*(y*z-x*w)],
        [2*(x*z-y*w), 2*(y*z+x*w), 1-2*(x*x+y*y)]])


def interpolate_rigid(first, second, fraction):
    """SE(3) translation interpolation and shortest-arc quaternion SLERP."""
    if not 0. <= fraction <= 1.:
        raise ValueError('pose interpolation must be bracketed')
    q0, q1 = quaternion_from_rotation(first[:3, :3]), quaternion_from_rotation(second[:3, :3])
    dot = float(np.dot(q0, q1))
    if dot < 0:
        q1, dot = -q1, -dot
    dot = np.clip(dot, -1., 1.)
    if dot > .9995:
        q = q0 + fraction * (q1 - q0)
        q /= np.linalg.norm(q)
    else:
        angle = math.acos(dot)
        q = (math.sin((1-fraction)*angle)*q0 + math.sin(fraction*angle)*q1) / math.sin(angle)
    result = np.eye(4)
    result[:3, :3] = rotation_from_quaternion(q)
    result[:3, 3] = (1-fraction)*first[:3,3] + fraction*second[:3,3]
    return result


class CameraMotion:
    """Interpolate ego LiDAR poses and project frames at camera packet time.

    No camera exposure timestamps are available: enable only when cameras and
    LiDAR sensor clocks are synchronized externally. No object yaw-rate model.
    """
    def __init__(self, rows, poses, *, enabled=True, offset_ms=0.,
                 max_offset_ms=200., max_pose_gap_s=4.):
        self.rows = rows
        self.poses = np.asarray(poses, dtype=float)
        self.enabled = enabled
        self.offset_ns = round(float(offset_ms) * 1e6)
        self.max_offset_ns = round(float(max_offset_ms) * 1e6)
        self.max_pose_gap_ns = round(float(max_pose_gap_s) * 1e9)
        self.times = [int(r.get('lidar_sensor_timestamp_ns') or 0) for r in rows]
        self.available = (len(rows) == len(poses) and all(t > 0 for t in self.times)
                          and all(a < b for a,b in zip(self.times, self.times[1:])))
        self._cache = {}
        self.status_counts = {}

    def _pose(self, stamp):
        idx = bisect.bisect_left(self.times, stamp)
        if idx < len(self.times) and self.times[idx] == stamp:
            return self.poses[idx]
        if idx == 0 or idx == len(self.times):
            return None
        previous, following = self.times[idx-1], self.times[idx]
        if following - previous > self.max_pose_gap_ns:
            return None
        return interpolate_rigid(self.poses[idx-1], self.poses[idx],
                                 (stamp-previous)/(following-previous))

    def geometry(self, frame_index, camera):
        key = frame_index, camera
        if key in self._cache:
            return self._cache[key]
        row = self.rows[frame_index]
        lidar_t = self.times[frame_index]
        camera_val = row.get(camera.lower() + '_timestamp_ns')
        camera_origin = 'manifest'
        if camera_val in ('', None):
            # Preserve support for pre-upgrade recorder exports which kept
            # only the synchronization target and camera delta.
            sync_target = row.get('camera_sync_target_ns')
            sync_delta = row.get(camera.lower() + '_delta_ns')
            if sync_target not in ('', None) and sync_delta not in ('', None):
                camera_val = int(sync_target) + int(sync_delta)
                camera_origin = 'reconstructed_from_sync_delta'
        camera_t = int(camera_val) + self.offset_ns if camera_val not in ('', None) else None
        reason = None
        if not self.enabled:
            reason = 'disabled_by_user'
        elif not self.available:
            reason = 'no_strictly_increasing_lidar_sensor_timestamps'
        elif row.get('camera_sync_clock') != 'sensor':
            reason = 'camera_and_lidar_clock_not_proven_comparable'
        elif camera_t is None:
            reason = 'camera_packet_timestamp_missing'
        elif abs(camera_t-lidar_t) > self.max_offset_ns:
            reason = 'camera_lidar_delta_outside_limit'
        else:
            target = self._pose(camera_t)
            if target is None:
                reason = 'camera_time_unbracketed_by_ego_poses'
        pose_camera = self.poses[frame_index] if reason else target
        output = {
            'applied': reason is None,
            'reason': reason,
            'camera_packet_timestamp_ns': camera_t - self.offset_ns if camera_t is not None else None,
            'camera_timestamp_origin': camera_origin if camera_t is not None else 'unavailable',
            'estimated_camera_timestamp_ns': camera_t,
            'camera_minus_lidar_seconds': ((camera_t-lidar_t)*1e-9 if camera_t is not None and lidar_t else None),
            'assumptions': ('camera packet timestamp approximates exposure; clocks share sensor timescale; '
                            'object velocity is constant; ego pose interpolated between LiDAR timestamps'),
            'T_camera_time_lidar_from_lidar_time_lidar':
                (np.linalg.inv(pose_camera) @ self.poses[frame_index]),
            'pose_camera': pose_camera,
        }
        self._cache[key] = output
        self.status_counts[reason or 'applied'] = self.status_counts.get(reason or 'applied', 0) + 1
        return output

    def box(self, box, frame_index, camera, velocity_world=None):
        """3D box at LiDAR instant -> CLIDAR coordinates of camera instant."""
        box = np.asarray(box, float)
        geometry = self.geometry(frame_index, camera)
        if not geometry['applied']:
            return box
        transform = geometry['T_camera_time_lidar_from_lidar_time_lidar']
        result = box.copy()
        result[:3] = (transform @ np.r_[box[:3], 1.])[:3]
        dt = geometry['camera_minus_lidar_seconds']
        if velocity_world is not None:
            vel = np.asarray(velocity_world, float).reshape(-1)
            if vel.shape == (2,) and np.isfinite(vel).all():
                delta_world = np.r_[vel * dt, 0.]
                result[:3] += geometry['pose_camera'][:3, :3].T @ delta_world
        # Yaw is defined in the local LiDAR xy plane. Roll/pitch of the pose
        # cannot be represented exactly by a yaw-only box; preserve dims.
        result[6] = ((box[6] + math.atan2(transform[1,0], transform[0,0]) + math.pi)
                     % (2*math.pi)) - math.pi
        return result

    def point_transform(self, frame_index, camera, lidar_to_camera):
        """LiDAR points at scan time -> camera coordinates at image time.

        Assumes stationary scene points; moving object points are NOT deskewed.
        """
        return (lidar_to_camera @ self.geometry(frame_index, camera)
                ['T_camera_time_lidar_from_lidar_time_lidar'])

    def details(self, frame_index, camera, has_velocity=False):
        record = self.geometry(frame_index, camera)
        details = {key: record[key] for key in (
            'applied', 'reason', 'camera_packet_timestamp_ns',
            'camera_timestamp_origin',
            'estimated_camera_timestamp_ns', 'camera_minus_lidar_seconds')}
        details['object_motion_velocity_used'] = bool(has_velocity)
        return details
