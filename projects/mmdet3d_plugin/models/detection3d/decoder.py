"""Sparse4D box decoder registered as an MMEngine task utility."""

from __future__ import annotations

import torch

from mmdet3d.registry import TASK_UTILS

from ...core.box3d import CNS, COS_YAW, H, L, SIN_YAW, VX, W, X, Y, Z


@TASK_UTILS.register_module()
class SparseBox3DDecoder:
    def __init__(self, num_output=300, score_threshold=None, sorted=True,
                 single_label=False, quality_before_topk=False,
                 nms_iou_threshold=None, nms_class_groups=()) -> None:
        self.num_output = int(num_output)
        self.score_threshold = score_threshold
        self.sorted = bool(sorted)
        self.single_label = bool(single_label)
        self.quality_before_topk = bool(quality_before_topk)
        self.nms_iou_threshold = nms_iou_threshold
        self.nms_class_groups = tuple(tuple(group) for group in nms_class_groups)

    @staticmethod
    def decode_box(box):
        yaw = torch.atan2(box[:, SIN_YAW], box[:, COS_YAW])
        return torch.cat(
            (box[:, [X, Y, Z]], box[:, [W, L, H]].exp(), yaw[:, None], box[:, VX:]),
            dim=-1,
        )

    def decode(
        self,
        cls_scores,
        box_preds,
        instance_id=None,
        quality=None,
        output_idx=-1,
        **kwargs,
    ):
        del kwargs
        scores_all = cls_scores[output_idx].sigmoid()
        class_count = scores_all.shape[-1]
        if quality is not None and self.quality_before_topk:
            scores_all = scores_all * quality[output_idx][..., CNS, None].sigmoid()
        squeeze_cls = instance_id is not None or self.single_label
        if squeeze_cls:
            scores_all, class_ids_all = scores_all.max(dim=-1)
            scores_all = scores_all.unsqueeze(-1)
        boxes_all = box_preds[output_idx]
        batch, _, classes = scores_all.shape
        count = min(self.num_output, scores_all.shape[1] * classes)
        scores, indices = scores_all.flatten(1).topk(
            count, dim=1, sorted=self.sorted
        )
        if quality is not None and not self.quality_before_topk:
            centerness = quality[output_idx][..., CNS]
            scores = scores * torch.gather(centerness, 1, indices // classes).sigmoid()
            scores, order = torch.sort(scores, dim=1, descending=True)
            indices = torch.gather(indices, 1, order)
        output = []
        for batch_index in range(batch):
            selected_scores = scores[batch_index]
            selected_indices = indices[batch_index]
            selected_classes = (
                class_ids_all[batch_index, selected_indices]
                if squeeze_cls
                else selected_indices % classes
            )
            query_indices = selected_indices // classes
            selected_boxes = boxes_all[batch_index, query_indices]
            if self.score_threshold is not None:
                keep = selected_scores >= self.score_threshold
                selected_scores = selected_scores[keep]
                selected_classes = selected_classes[keep]
                selected_boxes = selected_boxes[keep]
                query_indices = query_indices[keep]
            decoded_boxes = self.decode_box(selected_boxes)
            if self.nms_iou_threshold is not None and len(selected_scores):
                from mmcv.ops import nms_rotated
                group_labels = selected_classes.clone()
                for group_index, group in enumerate(self.nms_class_groups):
                    for class_id in group:
                        group_labels[selected_classes == class_id] = class_count + group_index
                # Some MMCV CPU builds ignore the optional labels argument.
                # Explicit groups prevent cross-class suppression on all hosts.
                group_keeps = []
                for group_id in group_labels.unique():
                    members = torch.nonzero(group_labels == group_id, as_tuple=True)[0]
                    _, group_keep = nms_rotated(
                        decoded_boxes[members][:, [0, 1, 3, 4, 6]].float().contiguous(),
                        selected_scores[members].float().contiguous(),
                        float(self.nms_iou_threshold), clockwise=False)
                    group_keeps.append(members[group_keep])
                keep = torch.cat(group_keeps)
                keep = keep[selected_scores[keep].argsort(descending=True)]
                decoded_boxes = decoded_boxes[keep]
                selected_scores = selected_scores[keep]
                selected_classes = selected_classes[keep]
                query_indices = query_indices[keep]
            result = {
                "boxes_3d": decoded_boxes,
                "scores_3d": selected_scores,
                "labels_3d": selected_classes,
            }
            if instance_id is not None:
                result["instance_ids"] = instance_id[
                    batch_index, query_indices
                ]
            output.append(result)
        return output


__all__ = ["SparseBox3DDecoder"]
