from .kalman_filter import *
from .ab3dmot import AB3DMOT

__FILTER__ = {
    'KalmanFilter': KalmanFilter,
    'AB3DMOT': AB3DMOT
}

# try:
#     from .ab3dmot import AB3DMOT
#     __FILTER__['AB3DMOT'] = AB3DMOT
# except ImportError:
#     # AB3DMOT is optional; DetZero's default tracker uses KalmanFilter.
#     pass
