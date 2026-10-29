"""Sparse4D task boundary used as the base of ROSA's fisheye head."""

from __future__ import annotations

import torch
from mmengine.model import BaseModule
from mmdet.utils import reduce_mean

from mmdet3d.registry import MODELS


@MODELS.register_module()
class Sparse4DHead(BaseModule):
    """MMEngine port of Sparse4D's assignment, loss and decode contract.

    ROSA's active fisheye head deliberately bypasses this constructor and
    supplies its own decoder.  Keeping the generic constructor unsupported
    prevents an accidental fallback to the retired native architecture.
    """

    def __init__(self, init_cfg=None, **kwargs) -> None:
        super().__init__(init_cfg=init_cfg)
        if kwargs:
            raise NotImplementedError(
                "the local host supports ROSA's FisheyeSparseBoxHead, not the "
                "retired generic Sparse4D architecture"
            )

    def _normalize_decoder_index(self, index: int, name: str) -> int:
        index = int(index)
        if index < 0:
            index += int(self.num_decoder)
        if not 0 <= index < int(self.num_decoder):
            raise ValueError(f"{name}={index} is outside the decoder range")
        return index

    def loss(self, model_outs, data, feature_maps=None):
        return self._loss_fp32(model_outs, data, feature_maps)

    def _loss_fp32(self, model_outs, data, feature_maps=None):
        del feature_maps
        classification = model_outs["classification"]
        prediction = model_outs["prediction"]
        quality = model_outs["quality"]
        output = {}
        for decoder_index, (cls, reg, qt) in enumerate(
            zip(classification, prediction, quality)
        ):
            cls = cls.float()
            reg = reg[..., : len(self.reg_weights)].float()
            qt = qt.float() if qt is not None else None
            cls_target, reg_target, reg_weight = self.sampler.sample(
                cls,
                reg,
                data[self.gt_cls_key],
                data[self.gt_reg_key],
            )
            reg_target = reg_target[..., : len(self.reg_weights)]
            matched = (cls_target >= 0) & (cls_target < cls.shape[-1])
            positive = reduce_mean(matched.sum().to(reg.dtype)).clamp_min(1.0)
            regression_mask = matched
            if self.cls_threshold_to_reg > 0:
                regression_mask = regression_mask & (
                    cls.sigmoid().amax(dim=-1) > self.cls_threshold_to_reg
                )
            flat_cls = cls.flatten(end_dim=1)
            flat_cls_target = cls_target.flatten(end_dim=1)
            output[f"loss_cls_{decoder_index}"] = self.loss_cls(
                flat_cls, flat_cls_target, avg_factor=positive
            )
            mask = regression_mask.flatten()
            weights = reg_weight * reg.new_tensor(self.reg_weights)
            selected_target = reg_target.flatten(end_dim=1)[mask]
            selected_reg = reg.flatten(end_dim=1)[mask]
            selected_weight = weights.flatten(end_dim=1)[mask]
            selected_target = torch.nan_to_num(selected_target)
            selected_cls = flat_cls_target[mask]
            selected_quality = qt.flatten(end_dim=1)[mask] if qt is not None else None
            output.update(
                self.loss_reg(
                    selected_reg,
                    selected_target,
                    weight=selected_weight,
                    avg_factor=positive,
                    suffix=f"_{decoder_index}",
                    quality=selected_quality,
                    cls_target=selected_cls,
                )
            )

        if "dn_prediction" not in model_outs:
            if getattr(self.sampler, 'num_dn_groups', 0) > 0:
                # Other DDP ranks may have DN queries even when this rank's
                # batch is empty. Match prepare_for_dn_loss's collective and
                # emit the same loss keys for MMEngine's distributed logging.
                reduce_mean(prediction[-1].new_zeros((), dtype=torch.float32))
                for decoder_index, (cls, reg) in enumerate(zip(classification, prediction)):
                    output[f'loss_cls_dn_{decoder_index}'] = cls.float().sum() * 0.
                    empty = reg.float().reshape(-1, reg.shape[-1])[:0, :len(self.reg_weights)]
                    losses = self.loss_reg(
                        empty, empty.detach(), weight=torch.zeros_like(empty),
                        avg_factor=1., suffix=f'_dn_{decoder_index}')
                    output.update({name: value * self.dn_loss_weight
                                   for name, value in losses.items()})
            return output
        (
            valid,
            dn_cls_target,
            dn_reg_target,
            dn_positive,
            dn_weight,
            positive_count,
        ) = self.prepare_for_dn_loss(model_outs)
        for decoder_index, (cls, reg) in enumerate(
            zip(model_outs["dn_classification"], model_outs["dn_prediction"])
        ):
            selected_cls = cls.float().flatten(end_dim=1)[valid]
            output[f"loss_cls_dn_{decoder_index}"] = self.loss_cls(
                selected_cls, dn_cls_target, avg_factor=positive_count
            ) * self.dn_loss_weight
            selected_reg = reg.float().flatten(end_dim=1)[valid][dn_positive][
                ..., : len(self.reg_weights)
            ]
            losses = self.loss_reg(
                selected_reg,
                dn_reg_target,
                weight=dn_weight,
                avg_factor=positive_count,
                suffix=f"_dn_{decoder_index}",
            )
            output.update(
                {name: value * self.dn_loss_weight for name, value in losses.items()}
            )
        return output

    def prepare_for_dn_loss(self, model_outs, prefix=""):
        valid = model_outs[f"{prefix}dn_valid_mask"].flatten(end_dim=1)
        cls_target = model_outs[f"{prefix}dn_cls_target"].flatten(end_dim=1)[valid]
        reg_target = model_outs[f"{prefix}dn_reg_target"].flatten(end_dim=1)[valid][
            ..., : len(self.reg_weights)
        ]
        # The fisheye head maps the -3 negative-DN sentinel to background C
        # for sigmoid focal loss. Background is a classification target only.
        positive = (cls_target >= 0) & (cls_target < self.num_classes)
        reg_target = reg_target[positive].float()
        weight = reg_target.new_tensor(self.reg_weights).expand(
            reg_target.shape[0], -1
        )
        positive_count = reduce_mean(positive.sum().to(weight.dtype)).clamp_min(1.0)
        return valid, cls_target, reg_target, positive, weight, positive_count

    def post_process(self, model_outs, output_idx=None):
        if output_idx is None:
            output_idx = getattr(self, "inference_output_decoder_index", -1)
        return self.decoder.decode(
            model_outs["classification"],
            model_outs["prediction"],
            model_outs.get("instance_id"),
            model_outs.get("quality"),
            output_idx=output_idx,
        )


__all__ = ["Sparse4DHead"]
