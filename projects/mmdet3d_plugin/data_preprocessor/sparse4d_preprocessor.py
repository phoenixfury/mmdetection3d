"""Multi-view image preprocessor for the MMEngine Sparse4D host."""

from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from mmdet3d.models.data_preprocessors import Det3DDataPreprocessor
from mmdet3d.registry import MODELS


@MODELS.register_module()
class Sparse4DDataPreprocessor(Det3DDataPreprocessor):
    """Normalize ``[B, camera, C, H, W]`` while retaining calibration inputs."""

    def _process_img(self, img: torch.Tensor) -> torch.Tensor:
        if img.ndim != 5 or img.shape[2] != 3:
            raise ValueError(
                "Sparse4D images must have shape [B,camera,3,H,W], "
                f"got {tuple(img.shape)}"
            )
        if self._channel_conversion:
            img = img[:, :, [2, 1, 0], ...]
        img = img.float()
        if self._enable_normalize:
            mean = self.mean.view(1, 1, 3, 1, 1)
            std = self.std.view(1, 1, 3, 1, 1)
            img = (img - mean) / std
        height, width = img.shape[-2:]
        target_h = math.ceil(height / self.pad_size_divisor) * self.pad_size_divisor
        target_w = math.ceil(width / self.pad_size_divisor) * self.pad_size_divisor
        if target_h != height or target_w != width:
            img = F.pad(
                img,
                (0, target_w - width, 0, target_h - height),
                value=float(self.pad_value),
            )
        return img

    def forward(self, data: dict, training: bool = False) -> dict:
        if isinstance(data, list):
            return [self.forward(item, training=training) for item in data]
        data = self.cast_data(data)
        inputs = dict(data["inputs"])
        image = inputs.pop("img", inputs.pop("imgs", None))
        if image is None:
            raise KeyError("Sparse4DDataPreprocessor requires image input")
        inputs["img"] = self._process_img(image)
        samples = data.get("data_samples")
        if samples is not None:
            shape = tuple(inputs["img"].shape[-2:])
            for sample in samples:
                sample.set_metainfo({"batch_input_shape": shape, "pad_shape": shape})
        return {"inputs": inputs, "data_samples": samples}


__all__ = ["Sparse4DDataPreprocessor"]
