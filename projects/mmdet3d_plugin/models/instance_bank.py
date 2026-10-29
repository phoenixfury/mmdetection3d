"""Single-frame Sparse4D instance bank ported to MMEngine."""

from __future__ import annotations

import numpy as np
import torch
from torch import nn

from mmdet3d.registry import MODELS


@MODELS.register_module()
class InstanceBank(nn.Module):
    def __init__(
        self,
        num_anchor: int,
        embed_dims: int,
        anchor,
        anchor_handler=None,
        num_temp_instances: int = 0,
        default_time_interval: float = 0.5,
        confidence_decay: float = 0.6,
        anchor_grad: bool = True,
        feat_grad: bool = True,
        max_time_interval: float = 2.0,
    ) -> None:
        super().__init__()
        self.embed_dims = int(embed_dims)
        self.num_temp_instances = int(num_temp_instances)
        self.default_time_interval = float(default_time_interval)
        self.confidence_decay = float(confidence_decay)
        self.max_time_interval = float(max_time_interval)
        self.anchor_handler = (
            MODELS.build(anchor_handler) if anchor_handler is not None else None
        )
        if isinstance(anchor, str):
            anchor = np.load(anchor)
        anchor = np.asarray(anchor, dtype=np.float32)[: int(num_anchor)]
        if anchor.ndim != 2:
            raise ValueError(f"InstanceBank anchors must be [Q,D], got {anchor.shape}")
        self.num_anchor = int(anchor.shape[0])
        self.register_buffer("anchor_init", torch.from_numpy(anchor).clone())
        self.anchor = nn.Parameter(self.anchor_init.clone(), requires_grad=anchor_grad)
        self.instance_feature = nn.Parameter(
            torch.zeros(self.num_anchor, self.embed_dims), requires_grad=feat_grad
        )
        self.reset()

    def init_weight(self) -> None:
        self.anchor.data.copy_(self.anchor_init)
        if self.instance_feature.requires_grad:
            nn.init.xavier_uniform_(self.instance_feature)

    def reset(self) -> None:
        self.cached_feature = None
        self.cached_anchor = None
        self.metas = None
        self.mask = None
        self.confidence = None
        self.temp_confidence = None
        self.instance_id = None
        self.prev_id = 0

    def get(self, batch_size: int, metas=None, dn_metas=None):
        del metas, dn_metas
        feature = self.instance_feature.unsqueeze(0).expand(batch_size, -1, -1)
        anchor = self.anchor.unsqueeze(0).expand(batch_size, -1, -1)
        interval = anchor.new_full((batch_size,), self.default_time_interval)
        if self.num_temp_instances != 0:
            raise NotImplementedError(
                "the local ROSA host currently supports its configured single-frame mode"
            )
        return feature, anchor, None, None, interval

    def update(self, instance_feature, anchor, confidence):
        del confidence
        return instance_feature, anchor

    def cache(self, *args, **kwargs) -> None:
        if self.num_temp_instances:
            raise NotImplementedError("temporal cache is not enabled by this ROSA config")

    def get_instance_id(self, confidence, anchor=None, threshold=None):
        del anchor
        scores = confidence.sigmoid().amax(dim=-1)
        ids = torch.arange(scores.shape[1], device=scores.device)[None].expand_as(scores)
        if threshold is not None:
            ids = torch.where(scores >= threshold, ids, ids.new_full((), -1))
        return ids


__all__ = ["InstanceBank"]
