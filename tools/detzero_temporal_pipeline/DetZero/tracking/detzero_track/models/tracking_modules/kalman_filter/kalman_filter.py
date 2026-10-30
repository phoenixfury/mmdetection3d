import copy

import numpy as np

from .motion_time import elapsed_seconds, cv_process_covariance


class BaseKalmanFilter:
    """
    Base Kalman Filter
    """
    def __init__(self, bbox, name, score, frame_id, track_id, num_points=0,
                 original_name=None, timestamp_ns=None,
                 x_dim=5, z_dim=3, delta_t=0.1, p=[1, 1], q=[1, 1], r=1):

        self.x_dim = x_dim
        self.z_dim = z_dim

        self.size = bbox[3:6]
        self.heading = bbox[6]
        self.name = name
        self.original_name = original_name if original_name is not None else name
        self.score = score
        self.update_score = score
        self.num_points = num_points

        self.frame_id = frame_id
        self.delta_t = float(delta_t)
        self.reference_dt = abs(float(delta_t))
        self.timestamp_ns = int(timestamp_ns) if timestamp_ns is not None else None

        self.track_id = track_id
        '''
        init filter parameters
        '''
        # state: [x, y, z, vx, vy]
        self.x = np.zeros((x_dim, 1), dtype=np.float32)
        self.x[:z_dim, :] = copy.deepcopy(bbox[:3].reshape(3, 1))

        self.bbox = np.zeros((bbox.shape[0]+2), dtype=np.float32)
        self.bbox[:bbox.shape[0]] = copy.deepcopy(bbox)

        self.F = np.eye(x_dim, dtype=np.float32)
        self.P = np.eye(x_dim, dtype=np.float32)
        self.Q = np.eye(x_dim, dtype=np.float32)

        self.H = np.eye(z_dim, x_dim, dtype=np.float32)
        self.R = np.eye(z_dim, dtype=np.float32)

        self.F[:2, 3:5] = np.eye(self.x_dim-self.z_dim, dtype=np.float32) * self.delta_t

        self.P[:self.z_dim, :self.z_dim] = self.P[:self.z_dim, :self.z_dim]*p[0]
        self.P[self.z_dim:, self.z_dim:] = self.P[self.z_dim:, self.z_dim:]*p[1]

        # meas: [x, y, z]
        self.Q[:3, :3] = self.Q[:3, :3]*q[0]
        self.Q[3:, 3:] = self.Q[3:, 3:]*q[1]
        self.R[:3, :3] = self.R[:3, :3]*r
        self.Q_baseline = self.Q.copy()

        # update count
        self.hit = 1
        self.miss = 0

    def state(self):
        return self.x

    def info(self):
        data = {
            'boxes_global': self.bbox,
            'name': self.name,
            'original_name': self.original_name,
            'score': self.score,
            'sample_idx': self.frame_id,
            'hit': self.hit,
            'num_points': self.num_points,
            'obj_ids': self.track_id,
            'timestamp_ns': -1 if self.timestamp_ns is None else self.timestamp_ns,
        }

        return {self.track_id: data}


class KalmanFilter(BaseKalmanFilter):
    """
    not update object center and size
    """
    def __init__(self, bbox, name, score, frame_id, track_id, num_points=0,
                 original_name=None, timestamp_ns=None, x_dim=5, z_dim=3, delta_t=0.1,
                 p=[1, 1], q=[1, 1], r=1, **kwargs):
        super().__init__(bbox, name, score, frame_id, track_id, num_points=num_points,
                 original_name=original_name, x_dim=x_dim, z_dim=z_dim,
                 timestamp_ns=timestamp_ns, delta_t=delta_t, p=p, q=q, r=r)


    def predict(self, frame_id, timestamp_ns=None):
        """
        Preict one step
        """
        dt = elapsed_seconds(timestamp_ns, self.timestamp_ns, self.delta_t)
        self.F[:2, 3:5] = np.eye(2, dtype=np.float32) * dt
        self.Q = cv_process_covariance(
            self.Q_baseline, dt, self.reference_dt, ((0, 3), (1, 4)))
        self.frame_id = frame_id
        self.timestamp_ns = int(timestamp_ns) if timestamp_ns is not None else None

        temp_x = copy.deepcopy(self.x)
        if self.name == "Vehicle":
            speed_norm = np.linalg.norm(temp_x[self.z_dim:])
            # The old threshold compared m/s with half the box length and
            # zeroed valid slow-vehicle motion (often up to ~2.5 m/s).  Only
            # suppress numerical drift that is genuinely close to zero.
            if speed_norm < 0.2:
                temp_x[self.z_dim:] = 0.

        self.x = self.F @ temp_x
        self.P = self.F @ self.P @ self.F.T + self.Q
        # self.Q = self.Q * 1.5

        self.miss += 1
        self.hit = 0

        bbox = np.concatenate((self.x.reshape(-1)[:3], self.size, self.heading.reshape(-1), 
                               self.x.reshape(-1)[3:5]), axis=0)
        self.bbox = bbox

        return self.bbox

    def update(self, bbox, name, score, num_points, two_stage=False,
               original_name=None):
        """
        Update state with measurement
        """
        self.hit = 1
        self.miss = 0

        self.score = score
        self.num_points = num_points
        if original_name is not None:
            self.original_name = original_name

        if two_stage:
            self.hit = 2
            return self.bbox 

        self.name = name
        self.update_score = max(score, 0.03)

        z = bbox[:3].reshape(3, 1)
        self.size = bbox[3:6]
        measured_heading = float(bbox[6])
        previous_heading = float(np.asarray(self.heading).reshape(-1)[0])
        speed_norm = float(np.linalg.norm(self.x[self.z_dim:]))
        if self.name == 'Vehicle' and speed_norm < 0.75:
            # Static and nearly-static vehicles provide no reliable motion
            # cue. Freeze the initialized yaw instead of following noisy
            # frame-level detector headings.
            measured_heading = previous_heading
        else:
            # 3D detectors often alternate between the two directions of the
            # same box axis. Keep the branch nearest to the existing track;
            # the offline trajectory pass later resolves motion direction.
            if abs((measured_heading - previous_heading + np.pi) %
                   (2 * np.pi) - np.pi) > np.pi / 2:
                measured_heading += np.pi
        self.heading = np.array(
            [(measured_heading + np.pi) % (2 * np.pi) - np.pi],
            dtype=np.float32)

        res_z = z - self.H @ self.x

        r = 1.
        S = self.H @ self.P @ self.H.T + self.R * r
        inv_S = np.linalg.inv(S)

        K = self.P @ self.H.T @ inv_S
        self.x = self.x + K @ res_z
        self.P = self.P - K @ self.H @ self.P

        self.x[:3]  = copy.deepcopy(z)
        bbox = np.concatenate((bbox[0:3], self.size, self.heading.reshape(-1), 
                               self.x.reshape(-1)[3:5]), axis=0)
        self.bbox = bbox

        return self.bbox
