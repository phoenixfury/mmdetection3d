"""Hungarian matching and denoising targets for Sparse4D."""

from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import linear_sum_assignment

from mmdet3d.registry import TASK_UTILS

from ...core.box3d import COS_YAW, H, L, SIN_YAW, W, X, Y, Z


@TASK_UTILS.register_module()
class SparseBox3DTarget:
    def __init__(
        self,
        cls_weight=2.0,
        alpha=0.25,
        gamma=2.0,
        eps=1e-12,
        box_weight=0.25,
        reg_weights=None,
        cls_wise_reg_weights=None,
        num_dn_groups=0,
        dn_noise_scale=0.5,
        max_dn_gt=32,
        add_neg_dn=True,
        num_temp_dn_groups=0,
        dn_selection="random",
        dn_class_weights=None,
        dn_box_relative_noise=False,
        dn_ignore_gt_overlap=True,
    ) -> None:
        self.cls_weight = float(cls_weight)
        self.box_weight = float(box_weight)
        self.alpha = float(alpha)
        self.gamma = float(gamma)
        self.eps = float(eps)
        self.reg_weights = list(reg_weights or ([1.0] * 8 + [0.0] * 2))
        self.cls_wise_reg_weights = cls_wise_reg_weights
        self.num_dn_groups = int(num_dn_groups)
        self.num_temp_dn_groups = int(num_temp_dn_groups)
        self.dn_noise_scale = dn_noise_scale
        self.max_dn_gt = int(max_dn_gt)
        self.add_neg_dn = bool(add_neg_dn)
        if dn_selection not in ("random", "class_balanced"):
            raise ValueError("dn_selection must be random or class_balanced")
        self.dn_selection = dn_selection
        self.dn_class_weights = dn_class_weights
        self.dn_box_relative_noise = bool(dn_box_relative_noise)
        self.dn_ignore_gt_overlap = bool(dn_ignore_gt_overlap)
        self.dn_metas = None

    @staticmethod
    def encode_reg_target(box_target, device=None):
        outputs = []
        for box in box_target:
            tensor = box.tensor if hasattr(box, "tensor") else box
            tensor = tensor.to(device=device) if device is not None else tensor
            outputs.append(
                torch.cat(
                    (
                        tensor[..., [X, Y, Z]],
                        tensor[..., [W, L, H]].clamp_min(1e-5).log(),
                        torch.sin(tensor[..., 6:7]),
                        torch.cos(tensor[..., 6:7]),
                        tensor[..., 7:],
                    ),
                    dim=-1,
                )
            )
        return outputs

    def _cls_cost(self, prediction, targets):
        probabilities = prediction.sigmoid()
        output = []
        for batch_index, target in enumerate(targets):
            if not len(target):
                output.append(None)
                continue
            negative = (
                -(1 - probabilities[batch_index] + self.eps).log()
                * (1 - self.alpha)
                * probabilities[batch_index].pow(self.gamma)
            )
            positive = (
                -(probabilities[batch_index] + self.eps).log()
                * self.alpha
                * (1 - probabilities[batch_index]).pow(self.gamma)
            )
            output.append((positive[:, target] - negative[:, target]) * self.cls_weight)
        return output

    def _box_cost(self, prediction, targets, weights):
        coefficients = prediction.new_tensor(self.reg_weights)
        return [
            None
            if not len(target)
            else (
                torch.abs(prediction[index, :, None] - target[None])
                * weights[index][None]
                * coefficients
            ).sum(-1)
            * self.box_weight
            for index, target in enumerate(targets)
        ]

    def sample(self, cls_pred, box_pred, cls_target, box_target):
        batch, predictions, classes = cls_pred.shape
        cls_cost = self._cls_cost(cls_pred, cls_target)
        encoded = self.encode_reg_target(box_target, box_pred.device)
        weights = []
        for index, target in enumerate(encoded):
            value = (~target.isnan()).to(target.dtype)
            if self.cls_wise_reg_weights:
                for class_id, class_weights in self.cls_wise_reg_weights.items():
                    value = torch.where(
                        (cls_target[index] == int(class_id))[:, None],
                        value.new_tensor(class_weights),
                        value,
                    )
            weights.append(value)
        box_cost = self._box_cost(box_pred, encoded, weights)
        output_cls = cls_target[0].new_full((batch, predictions), classes)
        output_box = box_pred.new_zeros(box_pred.shape)
        output_weight = box_pred.new_zeros(box_pred.shape)
        for index in range(batch):
            if cls_cost[index] is None:
                continue
            cost = (cls_cost[index] + box_cost[index]).detach().cpu().numpy()
            cost = np.where(np.isfinite(cost), cost, 1e8)
            pred_index, target_index = linear_sum_assignment(cost)
            pred_index = torch.as_tensor(pred_index, device=box_pred.device)
            target_index = torch.as_tensor(target_index, device=box_pred.device)
            output_cls[index, pred_index] = cls_target[index][target_index]
            output_box[index, pred_index] = encoded[index][target_index]
            output_weight[index, pred_index] = weights[index][target_index]
        return output_cls, output_box, output_weight

    def _select_dn_gt(self, labels):
        count = len(labels)
        limit = min(count, self.max_dn_gt) if self.max_dn_gt > 0 else count
        if limit == 0:
            return labels.new_empty((0,), dtype=torch.long)
        if self.dn_selection == "random":
            return torch.randperm(count, device=labels.device)[:limit]
        categories = labels.unique()
        weights = torch.ones(len(categories), device=labels.device)
        if self.dn_class_weights is not None:
            weights = weights.new_tensor(self.dn_class_weights)[categories]
        order = torch.multinomial(weights, len(categories), replacement=False)
        # One randomized representative per class before filling remaining
        # slots. A Van can no longer be excluded merely by annotation order.
        representatives = []
        for category in categories[order[:limit]]:
            candidates = torch.nonzero(labels == category, as_tuple=True)[0]
            representatives.append(candidates[torch.randint(len(candidates), (), device=labels.device)])
        chosen = torch.stack(representatives) if representatives else labels.new_empty((0,))
        if len(chosen) < limit:
            remaining = torch.ones(count, dtype=torch.bool, device=labels.device)
            remaining[chosen] = False
            available = torch.nonzero(remaining, as_tuple=True)[0]
            chosen = torch.cat((chosen, available[torch.randperm(len(available), device=labels.device)[:limit - len(chosen)]]))
        return chosen

    def _dn_noise(self, boxes, negative=False):
        scale = boxes.new_tensor(self.dn_noise_scale)
        magnitude = torch.rand_like(boxes) + (1.0 if negative else 0.0)
        signs = torch.where(torch.rand_like(boxes) < .5, -1., 1.)
        noise = magnitude * signs * scale
        if self.dn_box_relative_noise:
            noise[..., :3] *= boxes[..., 3:6].exp()
            # Translate along the object's length/width axes, not world axes.
            yaw = torch.atan2(boxes[..., SIN_YAW], boxes[..., COS_YAW])
            x, y = noise[..., X].clone(), noise[..., Y].clone()
            noise[..., X] = x * yaw.cos() - y * yaw.sin()
            noise[..., Y] = x * yaw.sin() + y * yaw.cos()
        anchors = boxes + noise
        yaw_vector = anchors[..., [SIN_YAW, COS_YAW]]
        anchors[..., [SIN_YAW, COS_YAW]] = F.normalize(yaw_vector, dim=-1, eps=1e-6)
        return anchors

    @staticmethod
    def _inside_gt(anchors, gt):
        if len(gt) == 0:
            return torch.zeros(anchors.shape[:-1], dtype=torch.bool, device=anchors.device)
        delta = anchors[..., None, :3] - gt[:, :3]
        yaw = torch.atan2(gt[:, SIN_YAW], gt[:, COS_YAW])
        local_x = delta[..., X] * yaw.cos() + delta[..., Y] * yaw.sin()
        local_y = -delta[..., X] * yaw.sin() + delta[..., Y] * yaw.cos()
        local = torch.stack((local_x, local_y, delta[..., Z]), dim=-1)
        return (local.abs() < gt[:, 3:6].exp() * .5).all(-1).any(-1)

    def get_dn_anchors(self, cls_target, box_target, gt_instance_id=None):
        del gt_instance_id
        if self.num_dn_groups <= 0:
            return None
        all_encoded = self.encode_reg_target(box_target, cls_target[0].device)
        selected = [self._select_dn_gt(labels) for labels in cls_target]
        cls_target = [target[index] for target, index in zip(cls_target, selected)]
        encoded = [target[index] for target, index in zip(all_encoded, selected)]
        maximum = max((len(target) for target in cls_target), default=0)
        if maximum == 0:
            return None
        classes = torch.stack(
            [F.pad(target, (0, maximum - len(target)), value=-1) for target in cls_target]
        )
        boxes = torch.stack(
            [F.pad(target, (0, 0, 0, maximum - len(target))) for target in encoded]
        )
        boxes = torch.where(classes[..., None] >= 0, boxes, boxes.new_zeros(()))
        # Build [batch, group, positive/negative, GT, state] before flattening
        # so every attention block contains its own positive/negative pair.
        classes = classes[:, None].expand(-1, self.num_dn_groups, -1)
        boxes = boxes[:, None].expand(-1, self.num_dn_groups, -1, -1)
        anchors = self._dn_noise(boxes)
        targets = boxes.clone()
        labels = classes.clone()
        valid = classes >= 0
        group_width = maximum
        if self.add_neg_dn:
            negatives = self._dn_noise(boxes, negative=True)
            negative_valid = valid.clone()
            if self.dn_ignore_gt_overlap:
                for index, gt in enumerate(all_encoded):
                    negative_valid[index] &= ~self._inside_gt(negatives[index], gt)
            anchors = torch.cat((anchors, negatives), dim=2)
            targets = torch.cat((targets, torch.zeros_like(boxes)), dim=2)
            labels = torch.cat((labels, labels.new_full(labels.shape, -3)), dim=2)
            valid = torch.cat((valid, negative_valid), dim=2)
            group_width *= 2
        anchors = anchors.flatten(1, 2)
        targets = targets.flatten(1, 2)
        labels = labels.flatten(1, 2)
        valid = valid.flatten(1, 2)
        total = anchors.shape[1]
        attention = torch.ones(total, total, dtype=torch.bool, device=anchors.device)
        for group in range(self.num_dn_groups):
            start = group * group_width
            attention[start : start + group_width, start : start + group_width] = False
        return anchors, targets, labels.long(), attention, valid, None

    def cache_dn(self, *args, **kwargs) -> None:
        self.dn_metas = None


__all__ = ["SparseBox3DTarget"]
