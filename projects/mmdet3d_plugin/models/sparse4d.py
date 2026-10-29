"""MMEngine detector wrapper for ROSA's Sparse4D-compatible head."""

from __future__ import annotations

from inspect import signature

import torch
from mmengine.structures import InstanceData

from mmdet3d.models.detectors.base import Base3DDetector
from mmdet3d.registry import MODELS


@MODELS.register_module()
class Sparse4D(Base3DDetector):
    def __init__(
        self,
        img_backbone,
        head,
        img_neck=None,
        data_preprocessor=None,
        init_cfg=None,
        train_cfg=None,
        test_cfg=None,
        pretrained=None,
        use_grid_mask=False,
        use_deformable_func=False,
        depth_branch=None,
        **kwargs,
    ) -> None:
        del train_cfg, test_cfg, kwargs
        super().__init__(data_preprocessor=data_preprocessor, init_cfg=init_cfg)
        if pretrained is not None:
            img_backbone = dict(img_backbone)
            img_backbone.setdefault("init_cfg", dict(type="Pretrained", checkpoint=pretrained))
        self.img_backbone = MODELS.build(img_backbone)
        self.img_neck = MODELS.build(img_neck) if img_neck is not None else None
        self.head = MODELS.build(head)
        self.use_grid_mask = bool(use_grid_mask)
        self.use_deformable_func = bool(use_deformable_func)
        self.depth_branch = MODELS.build(depth_branch) if depth_branch else None
        if self.use_grid_mask:
            raise NotImplementedError("the released ROSA fisheye config disables grid mask")
        if self.use_deformable_func:
            raise NotImplementedError("build the optional CUDA host before enabling fused DAF")

    def extract_feat(self, img, return_depth=False, metas=None):
        batch, cameras = img.shape[:2]
        flat = img.flatten(0, 1)
        if "metas" in signature(self.img_backbone.forward).parameters:
            features = self.img_backbone(flat, cameras, metas=metas)
        else:
            features = self.img_backbone(flat)
        if self.img_neck is not None:
            features = self.img_neck(features)
        features = [
            feature.reshape(batch, cameras, *feature.shape[1:]) for feature in features
        ]
        if return_depth:
            depths = self.depth_branch(features, metas.get("focal")) if self.depth_branch else None
            return features, depths
        return features

    @staticmethod
    def _build_metas(inputs, data_samples=None):
        metas = {key: value for key, value in inputs.items() if key not in ("img", "imgs")}
        samples = list(data_samples or [])
        metas["img_metas"] = [sample.metainfo for sample in samples]
        has_ground_truth = bool(samples) and all(
            hasattr(sample, "gt_instances_3d")
            and "bboxes_3d" in sample.gt_instances_3d
            and "labels_3d" in sample.gt_instances_3d
            for sample in samples
        )
        if has_ground_truth:
            metas["gt_bboxes_3d"] = [
                sample.gt_instances_3d.bboxes_3d.tensor
                for sample in samples
            ]
            metas["gt_labels_3d"] = [
                sample.gt_instances_3d.labels_3d for sample in samples
            ]
            instance_ids = []
            for sample in samples:
                instances = sample.gt_instances_3d
                instance_ids.append(
                    instances.instance_inds
                    if "instance_inds" in instances
                    else torch.arange(len(instances.labels_3d), device=instances.labels_3d.device)
                )
            metas["instance_id"] = instance_ids
        return metas

    def loss(self, inputs, data_samples, **kwargs):
        del kwargs
        metas = self._build_metas(inputs, data_samples)
        features, depths = self.extract_feat(inputs["img"], True, metas)
        outputs = self.head(features, metas)
        losses = self.head.loss(outputs, metas, features)
        if depths is not None and "gt_depth" in metas:
            losses["loss_dense_depth"] = self.depth_branch.loss(depths, metas["gt_depth"])
        return losses

    def _forward(self, inputs, data_samples=None, **kwargs):
        del kwargs
        metas = self._build_metas(inputs, data_samples)
        features = self.extract_feat(inputs["img"], metas=metas)
        return self.head(features, metas)

    def simple_test(self, data):
        inputs = data["inputs"]
        samples = data.get("data_samples")
        metas = self._build_metas(inputs, samples)
        features = self.extract_feat(inputs["img"], metas=metas)
        outputs = self.head(features, metas)
        results = self.head.post_process(outputs)
        return [{"img_bbox": result} for result in results]

    def predict(self, inputs, data_samples, **kwargs):
        del kwargs
        image = inputs.get("img", inputs.get("imgs"))
        # AmpOptimWrapper only autocasts train_step; MMEngine validation calls
        # predict outside that context. The four 800x800 V99 views exceed a
        # 6 GiB GPU in FP32, while this fused path is AMP-safe in training.
        with torch.autocast(
            device_type="cuda",
            dtype=torch.float16,
            enabled=image is not None and image.is_cuda,
        ):
            outputs = self.simple_test(
                {"inputs": inputs, "data_samples": data_samples}
            )
        for sample, output in zip(data_samples, outputs):
            prediction = output["img_bbox"]
            instances = InstanceData()
            instances.bboxes_3d = prediction["boxes_3d"]
            instances.scores_3d = prediction["scores_3d"]
            instances.labels_3d = prediction["labels_3d"]
            if "instance_ids" in prediction:
                instances.instance_ids = prediction["instance_ids"]
            sample.pred_instances_3d = instances
        return data_samples


__all__ = ["Sparse4D"]
