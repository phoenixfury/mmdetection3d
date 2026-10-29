"""Pack released fisheye samples into the Sparse4D MMEngine contract."""

from __future__ import annotations

import numpy as np
import torch

from mmdet3d.datasets.transforms import Pack3DDetInputs
from mmdet3d.registry import TRANSFORMS


@TRANSFORMS.register_module()
class PackSparse4DInputs(Pack3DDetInputs):
    META_KEYS = (
        "token",
        "sample_idx",
        "scene_token",
        "frame_idx",
        "timestamp",
        "img_path",
        "filename",
        "ori_shape",
        "img_shape",
        "pad_shape",
        "cam_names",
        "ego2global",
    )
    CALIBRATION_KEYS = (
        "projection_mat",
        "lidar2img",
        "lidar2cam",
        "cam2img",
        "cam2ego",
        "fisheye_params",
        "image_wh",
        "ocam_lens_fov_deg",
        "fisheye_lens_fov_deg",
        "image_affine",
        "camera_model",
        "mei_xi",
        "mei_distortion",
        "pinhole_intrinsic",
    )

    def __init__(self, ocam_lens_fov_deg: float = 220.0, **kwargs) -> None:
        self.ocam_lens_fov_deg = float(ocam_lens_fov_deg)
        super().__init__(
            keys=("img", "gt_bboxes_3d", "gt_labels_3d"),
            meta_keys=self.META_KEYS,
        )

    @staticmethod
    def _tensor(value):
        if torch.is_tensor(value):
            return value
        array = np.asarray(value)
        if array.dtype == np.float64:
            array = array.astype(np.float32)
        return torch.from_numpy(np.ascontiguousarray(array))

    @staticmethod
    def _pack_img(imgs) -> torch.Tensor:
        array = np.stack(imgs, axis=0) if isinstance(imgs, (list, tuple)) else imgs
        array = np.asarray(array)
        if array.ndim != 4:
            raise ValueError(f"multi-view image array must be 4D, got {array.shape}")
        if array.shape[-1] in (1, 3):
            array = np.transpose(array, (0, 3, 1, 2))
        return torch.from_numpy(np.ascontiguousarray(array))

    def pack_single_results(self, results: dict) -> dict:
        packed = super().pack_single_results(dict(results))
        inputs = packed["inputs"]
        inputs["img"] = self._pack_img(results["img"])

        camera_count = int(inputs["img"].shape[0])
        image_shape = results.get("img_shape", [(800, 800, 3)] * camera_count)
        image_wh = np.asarray(
            [[shape[1], shape[0]] for shape in image_shape], dtype=np.float32
        )
        inputs["image_wh"] = self._tensor(results.get("image_wh", image_wh))

        aliases = {
            "cam2img": "ocam_intrinsic",
            "fisheye_params": "ocam_pol",
        }
        for key in self.CALIBRATION_KEYS:
            if key not in results:
                continue
            output_key = aliases.get(key, key)
            value = results[key]
            if isinstance(value, (str, bytes)):
                continue
            if np.isscalar(value):
                inputs[output_key] = torch.tensor(value, dtype=torch.float32)
            else:
                inputs[output_key] = self._tensor(value)
        inputs.setdefault(
            "ocam_lens_fov_deg",
            torch.tensor(self.ocam_lens_fov_deg, dtype=torch.float32),
        )
        if "lidar2img" in inputs:
            inputs.setdefault("projection_mat", inputs["lidar2img"])
        inputs["timestamp"] = torch.tensor(
            float(results.get("timestamp", 0.0)), dtype=torch.float64
        )
        inputs["T_global"] = self._tensor(
            results.get("ego2global", np.eye(4, dtype=np.float32))
        ).float()
        inputs["T_global_inv"] = torch.linalg.inv(inputs["T_global"])
        return packed


__all__ = ["PackSparse4DInputs"]
