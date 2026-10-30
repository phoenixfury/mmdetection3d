"""Timestamp-driven constant-velocity prediction shared by DetZero filters.

Timestamps must refer to the same monotonic acquisition clock. The nominal
DELTA_T is used only when a timestamp was not supplied (legacy Waymo input).
"""
import numpy as np


def elapsed_seconds(current_ns, last_ns, nominal_dt):
    if current_ns is None or last_ns is None:
        return float(nominal_dt)
    delta = (int(current_ns) - int(last_ns)) * 1e-9
    if not np.isfinite(delta) or delta == 0.0:
        raise ValueError('invalid repeated/nonfinite tracker acquisition timestamp')
    if np.sign(delta) != np.sign(nominal_dt):
        raise ValueError('tracker acquisition timestamps run opposite to tracking direction')
    return float(delta)


def cv_process_covariance(baseline, delta_t, reference_dt, pairs):
    """Scale baseline random-walk Q and add continuous white-acceleration Q.

    Variances grow with elapsed time; position/velocity cross-terms are
    induced by acceleration. Reverse filtering uses abs(dt) for noise.
    """
    dt = abs(float(delta_t))
    ref = abs(float(reference_dt))
    if not 0 < ref or not 0 < dt:
        raise ValueError('filter time step must be nonzero')
    q0 = np.asarray(baseline, dtype=np.float64)
    q = q0 * (dt / ref)
    for position, velocity in pairs:
        # A continuous acceleration spectral density with Q_vv=q0_vv*dt/ref
        spectral = float(q0[velocity, velocity]) / ref
        q[position, position] += spectral * dt**3 / 3.0
        cross = spectral * dt**2 / 2.0 * np.sign(delta_t)
        q[position, velocity] += cross
        q[velocity, position] += cross
    return q.astype(q0.dtype)
