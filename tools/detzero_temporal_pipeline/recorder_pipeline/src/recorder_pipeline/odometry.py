"""Fused odometry protobuf adapter and deterministic interpolation.

`EgoPoseFused.timestamp` is assumed to be milliseconds by default, matching
historical datasets. Units are not defined in the shared .proto; configure
--pose-timestamp-unit if the recording uses micro/nanoseconds.
"""
from __future__ import annotations

import bisect
import math
import warnings
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .bag_io import BagFormatError, iter_bag_topic_samples
from .protos import localization_pb2

TIME_UNIT_TO_NS = {'ms': 1_000_000, 'us': 1_000, 'ns': 1}
FIELDS = ('x', 'y', 'z', 'roll', 'pitch', 'yaw', 'vx', 'vy', 'vz', 'yaw_rate')


def timestamp_ns(message_timestamp: int, unit: str) -> int:
    if unit not in TIME_UNIT_TO_NS:
        raise ValueError(f'unsupported odometry timestamp unit: {unit}')
    if message_timestamp <= 0:
        raise BagFormatError('EgoPoseFused has missing or nonpositive timestamp')
    return message_timestamp * TIME_UNIT_TO_NS[unit]


def load_fused_odometry(folder: Path, timestamp_unit: str = 'ms') -> list[dict]:
    """Read bag.TopicData chunks; validate index and decode sample.data.

    Raw `adas.common.EgoPose` fallback is intentionally unavailable because
    its separate `common.proto` definition has not been provided.
    """
    if timestamp_unit not in TIME_UNIT_TO_NS:
        raise ValueError(f'unsupported odometry timestamp unit {timestamp_unit}')
    samples: list[dict] = []
    for write_time, sample, chunk, n in iter_bag_topic_samples(folder):
        data = sample.data
        if not data:
            raise BagFormatError(f'{chunk}: sample #{n} has empty data')
        pose = localization_pb2.EgoPoseFused()
        try:
            pose.ParseFromString(data)
        except Exception as error:
            raise BagFormatError(f'{chunk}: invalid EgoPoseFused #{n}: {error}') from error
        if not (pose.HasField('position') and pose.HasField('orientation')):
            raise BagFormatError(f'{chunk}: sample #{n} missing position/orientation')
        values = dict(
            measurement_ns=timestamp_ns(pose.timestamp, timestamp_unit),
            message_timestamp=int(pose.timestamp),
            recorder_index_ns=write_time,
            sample_timestamp_ns=(
                sample.timestamp.seconds * 1_000_000_000 + sample.timestamp.nanos
                if sample.HasField('timestamp') else None),
            read_timestamp_ns=(
                sample.read_timestamp.seconds * 1_000_000_000 + sample.read_timestamp.nanos
                if sample.HasField('read_timestamp') else None),
            x=float(pose.position.x), y=float(pose.position.y), z=float(pose.position.z),
            roll=float(pose.orientation.roll), pitch=float(pose.orientation.pitch),
            yaw=float(pose.orientation.yaw),
            vx=float(pose.velocity.x), vy=float(pose.velocity.y), vz=float(pose.velocity.z),
            yaw_rate=float(pose.yaw_rate), status=int(pose.status),
        )
        if not all(math.isfinite(values[k]) for k in FIELDS):
            raise BagFormatError(f'{chunk}: nonfinite pose component in sample #{n}')
        samples.append(values)
    return deduplicate_samples(samples, 'fused')


def deduplicate_samples(samples: list[dict], label: str) -> list[dict]:
    # Multiple publications may carry the same localization measurement time.
    # Preserve exactly one sample for interpolation. In case of corrections at
    # the same measurement time, select the latest recorder/index timestamp;
    # ties are broken by sample read timestamp and then arrival order. This
    # is deterministic and prevents a zero-width interpolation interval.
    samples.sort(key=lambda row: row['measurement_ns'])
    deduplicated: list[dict] = []
    duplicate_count = 0
    conflict_count = 0
    start = 0
    while start < len(samples):
        end = start + 1
        while end < len(samples) and samples[end]['measurement_ns'] == samples[start]['measurement_ns']:
            end += 1
        group = samples[start:end]
        if len(group) > 1:
            duplicate_count += len(group) - 1
            if any(tuple(row[k] for k in FIELDS) + (row['status'],) !=
                   tuple(group[0][k] for k in FIELDS) + (group[0]['status'],)
                   for row in group[1:]):
                conflict_count += 1
        # max() is stable: the final item wins when both clocks tie.
        chosen = max(group, key=lambda row: (
            row['recorder_index_ns'],
            row['read_timestamp_ns'] if row['read_timestamp_ns'] is not None else -1))
        deduplicated.append(chosen)
        start = end
    if duplicate_count:
        warnings.warn(
            f'{label} odometry: collapsed {duplicate_count} repeated samples '
            f'at {len(samples) - len(deduplicated)} measurement timestamps; '
            f'{conflict_count} timestamp groups had conflicting pose/status '
            'data (selected latest recorder timestamp, then read timestamp)',
            RuntimeWarning, stacklevel=2)
    return deduplicated


def interpolate_angle(a: float, b: float, fraction: float) -> float:
    return a + fraction * ((b - a + math.pi) % (2 * math.pi) - math.pi)


@dataclass(frozen=True)
class FusedOdometrySeries:
    samples: list[dict]
    max_delta_ns: int = 200_000_000

    def __post_init__(self) -> None:
        if self.max_delta_ns <= 0:
            raise ValueError('max_delta_ns must be positive')
        if not self.samples:
            raise ValueError('no fused odometry samples')
        times = [row['measurement_ns'] for row in self.samples]
        if times != sorted(set(times)):
            raise ValueError('odometry samples must be sorted, with unique measurement timestamps')
        object.__setattr__(self, '_times', times)

    def at(self, target_ns: int) -> dict:
        """Interpolate only between sufficiently recent bracketing samples.

        At a precise existing measurement, return that pose. Never extrapolate
        outside coverage or across a recorder outage greater than tolerance.
        """
        times = self._times
        idx = bisect.bisect_left(times, target_ns)
        if idx < len(times) and times[idx] == target_ns:
            return self._row(target_ns, idx, idx, 0.)
        if idx == 0 or idx == len(times):
            raise ValueError(f'{target_ns}: outside fused odometry time coverage')
        lo, hi = idx - 1, idx
        before = target_ns - times[lo]
        after = times[hi] - target_ns
        if before > self.max_delta_ns or after > self.max_delta_ns:
            raise ValueError(
                f'{target_ns}: fused odometry gap ({before / 1e6:.1f}ms before, '
                f'{after / 1e6:.1f}ms after) exceeds {self.max_delta_ns / 1e6:.1f}ms')
        return self._row(target_ns, lo, hi, before / (times[hi] - times[lo]))

    def nearest_within_tolerance(self, target_ns: int) -> dict:
        """Hold the nearest measurement for a bounded one-sided gap.

        This is deliberately separate from :meth:`at`: normal interpolation
        still requires two valid brackets and never fills a gap.  The raw
        odometry fallback uses this only when one side of a recording segment
        is missing (for example, a stale sample followed by the real stream).
        """
        idx = bisect.bisect_left(self._times, target_ns)
        candidates = [index for index in (idx - 1, idx)
                      if 0 <= index < len(self._times)]
        if not candidates:
            raise ValueError(f'{target_ns}: no odometry measurements')
        nearest = min(candidates,
                      key=lambda index: abs(self._times[index] - target_ns))
        delta = abs(self._times[nearest] - target_ns)
        if delta > self.max_delta_ns:
            raise ValueError(
                f'{target_ns}: nearest odometry measurement is '
                f'{delta / 1e6:.1f}ms away, exceeding '
                f'{self.max_delta_ns / 1e6:.1f}ms')
        result = self._row(target_ns, nearest, nearest, 0.)
        result['source'] = 'bounded_boundary_hold'
        return result

    def _row(self, target_ns: int, lo: int, hi: int, fraction: float) -> dict:
        a, b = self.samples[lo], self.samples[hi]
        nearest = a if fraction <= .5 else b
        linear = ('x','y','z','vx','vy','vz','yaw_rate')
        result = {k: a[k] + fraction * (b[k] - a[k]) for k in linear}
        result.update({k: interpolate_angle(a[k], b[k], fraction)
                       for k in ('roll','pitch','yaw')})
        result.update(timestamp_ns=target_ns, source='fused',
                      source_measurement_ns=nearest['measurement_ns'],
                      source_index_ns=nearest['recorder_index_ns'],
                      source_read_timestamp_ns=nearest['read_timestamp_ns'],
                      delta_ns=nearest['measurement_ns'] - target_ns,
                      status=nearest['status'])
        return result


def pose_matrix(row: dict) -> np.ndarray:
    """SE(3) pose, T_world_base = Rz(yaw) @ Ry(pitch) @ Rx(roll), meters."""
    cr, sr = math.cos(row['roll']), math.sin(row['roll'])
    cp, sp = math.cos(row['pitch']), math.sin(row['pitch'])
    cy, sy = math.cos(row['yaw']), math.sin(row['yaw'])
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = [
        [cy*cp, cy*sp*sr-sy*cr, cy*sp*cr+sy*sr],
        [sy*cp, sy*sp*sr+cy*cr, sy*sp*cr-cy*sr],
        [-sp, cp*sr, cp*cr],
    ]
    pose[:3, 3] = row['x'], row['y'], row['z']
    return pose


class AlignedOdometry:
    """Prefer fused trajectory, rigidly align raw SE(2) odometry at a nearby boundary.

    Critical: raw odometry must share the same length units and yaw convention
    as fused. Raw is relative or in a potentially different global frame.
    """

    def __init__(self, fused: FusedOdometrySeries, raw: FusedOdometrySeries):
        self.fused = fused
        self.raw = raw

    def at(self, target_ns: int) -> dict:
        try:
            return self.fused.at(target_ns)
        except ValueError as fused_error:
            try:
                raw_pose = self.raw.at(target_ns)
            except ValueError as error:
                try:
                    raw_pose = self.raw.nearest_within_tolerance(target_ns)
                except ValueError as nearest_error:
                    raise ValueError(
                        f'{fused_error}; raw odometry unavailable: {error}; '
                        f'bounded hold unavailable: {nearest_error}') from nearest_error
            # Pick the closest valid fused boundary at which both streams overlap.
            times = self.fused._times
            idx = bisect.bisect_left(times, target_ns)
            candidates = sorted({max(0, idx-1), min(len(times)-1, idx)},
                                key=lambda i: abs(times[i] - target_ns))
            anchor = None
            for index in candidates:
                boundary_ns = times[index]
                try:
                    raw_anchor = self.raw.at(boundary_ns)
                    fused_anchor = self.fused.at(boundary_ns)
                except ValueError:
                    continue
                anchor = (raw_anchor, fused_anchor, boundary_ns)
                break
            if anchor is None:
                raise ValueError(f'{fused_error}; no shared raw/fused anchor near gap')
            raw_anchor, fused_anchor, boundary_ns = anchor
            delta = (fused_anchor['yaw'] - raw_anchor['yaw'] + math.pi) % (2*math.pi) - math.pi
            cosine, sine = math.cos(delta), math.sin(delta)
            dx = raw_pose['x'] - raw_anchor['x']
            dy = raw_pose['y'] - raw_anchor['y']
            aligned = dict(raw_pose)
            raw_was_held = raw_pose['source'] == 'bounded_boundary_hold'
            aligned.update(x=fused_anchor['x'] + cosine*dx - sine*dy,
                           y=fused_anchor['y'] + sine*dx + cosine*dy,
                           z=fused_anchor['z'] + raw_pose['z'] - raw_anchor['z'],
                           yaw=(raw_pose['yaw'] + delta + math.pi) % (2*math.pi) - math.pi,
                           roll=fused_anchor['roll'], pitch=fused_anchor['pitch'],
                           source=('aligned_raw_boundary_hold' if raw_was_held
                                   else 'aligned_raw'))
            return aligned
