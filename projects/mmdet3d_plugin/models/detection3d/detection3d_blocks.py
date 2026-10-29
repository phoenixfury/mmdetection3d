"""Sparse 3D box geometry components required by ROSA."""

from __future__ import annotations

import torch
from torch import nn

from mmcv.cnn import Linear
from mmengine.model import BaseModule
from mmdet3d.registry import MODELS

from ...core.box3d import COS_YAW, H, L, SIN_YAW, VX, W, X, Y, Z


@MODELS.register_module()
class SparseBox3DKeyPointsGenerator(BaseModule):
    def __init__(
        self,
        embed_dims: int = 256,
        num_learnable_pts: int = 0,
        fix_scale=None,
        anchor_z_is_bottom: bool = False,
        init_cfg=None,
    ) -> None:
        super().__init__(init_cfg=init_cfg)
        self.embed_dims = int(embed_dims)
        self.num_learnable_pts = int(num_learnable_pts)
        self.anchor_z_is_bottom = bool(anchor_z_is_bottom)
        if fix_scale is None:
            fix_scale = ((0.0, 0.0, 0.0),)
        self.fix_scale = nn.Parameter(torch.tensor(fix_scale), requires_grad=False)
        self.num_pts = len(fix_scale) + self.num_learnable_pts
        if self.num_learnable_pts:
            self.learnable_fc = Linear(self.embed_dims, self.num_learnable_pts * 3)

    def init_weight(self) -> None:
        if self.num_learnable_pts:
            nn.init.xavier_uniform_(self.learnable_fc.weight)
            if self.learnable_fc.bias is not None:
                nn.init.zeros_(self.learnable_fc.bias)

    def forward(self, anchor, instance_feature=None, **kwargs):
        del kwargs
        batch, count = anchor.shape[:2]
        size = anchor[..., None, [W, L, H]].exp()
        points = self.fix_scale.to(anchor) * size
        if self.num_learnable_pts and instance_feature is not None:
            learned = (
                self.learnable_fc(instance_feature)
                .reshape(batch, count, self.num_learnable_pts, 3)
                .sigmoid()
                - 0.5
            )
            points = torch.cat((points, learned * size), dim=-2)
        rotation = anchor.new_zeros(batch, count, 3, 3)
        rotation[..., 0, 0] = anchor[..., COS_YAW]
        rotation[..., 0, 1] = -anchor[..., SIN_YAW]
        rotation[..., 1, 0] = anchor[..., SIN_YAW]
        rotation[..., 1, 1] = anchor[..., COS_YAW]
        rotation[..., 2, 2] = 1
        points = torch.matmul(rotation[:, :, None], points[..., None]).squeeze(-1)
        center = anchor[..., None, [X, Y, Z]]
        if self.anchor_z_is_bottom:
            center = center.clone()
            center[..., 2] += 0.5 * size[..., 0, 2]
        return points + center

    @staticmethod
    def anchor_projection(anchor, T_src2dst_list, time_intervals=None, **kwargs):
        del kwargs
        outputs = []
        for index, transform in enumerate(T_src2dst_list):
            transform = transform.to(anchor).unsqueeze(1)
            center = anchor[..., [X, Y, Z]]
            velocity = anchor[..., VX:]
            if time_intervals is not None:
                center = center - velocity * time_intervals[index][:, None, None]
            center = (
                torch.matmul(transform[..., :3, :3], center[..., None]).squeeze(-1)
                + transform[..., :3, 3]
            )
            yaw = torch.matmul(
                transform[..., :2, :2],
                anchor[..., [COS_YAW, SIN_YAW], None],
            ).squeeze(-1)
            velocity = torch.matmul(
                transform[..., : velocity.shape[-1], : velocity.shape[-1]],
                velocity[..., None],
            ).squeeze(-1)
            outputs.append(
                torch.cat((center, anchor[..., [W, L, H]], yaw.flip(-1), velocity), -1)
            )
        return outputs

    @staticmethod
    def distance(anchor):
        return torch.linalg.vector_norm(anchor[..., :2], dim=-1)


__all__ = ["SparseBox3DKeyPointsGenerator"]
