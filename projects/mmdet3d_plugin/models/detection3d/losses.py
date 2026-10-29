"""Sparse4D box loss on the MMEngine model registry."""

from __future__ import annotations

import torch
from torch import nn

from mmdet3d.registry import MODELS

from ...core.box3d import CNS, COS_YAW, SIN_YAW, X, Y, YNS, Z


@MODELS.register_module()
class SparseBox3DLoss(nn.Module):
    def __init__(
        self,
        loss_box,
        loss_centerness=None,
        loss_yawness=None,
        cls_allow_reverse=None,
    ) -> None:
        super().__init__()
        self.loss_box = MODELS.build(loss_box)
        self.loss_cns = MODELS.build(loss_centerness) if loss_centerness else None
        self.loss_yns = MODELS.build(loss_yawness) if loss_yawness else None
        self.cls_allow_reverse = cls_allow_reverse

    def forward(
        self,
        box,
        box_target,
        weight=None,
        avg_factor=None,
        suffix="",
        quality=None,
        cls_target=None,
        **kwargs,
    ):
        del kwargs
        if self.cls_allow_reverse is not None and cls_target is not None:
            reverse = torch.nn.functional.cosine_similarity(
                box_target[..., [SIN_YAW, COS_YAW]],
                box[..., [SIN_YAW, COS_YAW]],
                dim=-1,
            ) < 0
            allowed = torch.isin(
                cls_target, cls_target.new_tensor(self.cls_allow_reverse)
            )
            box_target = box_target.clone()
            box_target[..., [SIN_YAW, COS_YAW]] = torch.where(
                (reverse & allowed)[..., None],
                -box_target[..., [SIN_YAW, COS_YAW]],
                box_target[..., [SIN_YAW, COS_YAW]],
            )
        output = {
            f"loss_box{suffix}": self.loss_box(
                box, box_target, weight=weight, avg_factor=avg_factor
            )
        }
        if quality is not None:
            centerness = quality[..., CNS]
            centerness_target = torch.exp(
                -torch.linalg.vector_norm(
                    box_target[..., [X, Y, Z]] - box[..., [X, Y, Z]], dim=-1
                )
            )
            output[f"loss_cns{suffix}"] = self.loss_cns(
                centerness, centerness_target, avg_factor=avg_factor
            )
            yawness = quality[..., YNS].sigmoid()
            yawness_target = (
                torch.nn.functional.cosine_similarity(
                    box_target[..., [SIN_YAW, COS_YAW]],
                    box[..., [SIN_YAW, COS_YAW]],
                    dim=-1,
                )
                > 0
            ).to(yawness.dtype)
            output[f"loss_yns{suffix}"] = self.loss_yns(
                yawness, yawness_target, avg_factor=avg_factor
            )
        return output


__all__ = ["SparseBox3DLoss"]
