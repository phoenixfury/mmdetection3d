"""ROSA CUDA operator bridge.

The enhanced ROSA checkout owns the maintained deformable-aggregation source
and extension.  Expose that implementation through Sparse4D's conventional
``projects.mmdet3d_plugin.ops`` import path so both projects use one binary.
"""

from ops import (  # noqa: F401
    deformable_aggregation_function,
    deformable_aggregation_per_camera_deterministic_function,
    deformable_aggregation_per_camera_function,
    deformable_aggregation_rfl_rosa_inference,
    feature_maps_format,
    v49_build_keypoints_project_geometry_inference,
    v49_hybrid_state_update_inference,
    v49_point_attention_inference,
    v49_project_geometry_inference,
    v49_state_geometry_inputs_inference,
)

__all__ = [
    "deformable_aggregation_function",
    "deformable_aggregation_per_camera_deterministic_function",
    "deformable_aggregation_per_camera_function",
    "deformable_aggregation_rfl_rosa_inference",
    "feature_maps_format",
    "v49_build_keypoints_project_geometry_inference",
    "v49_hybrid_state_update_inference",
    "v49_point_attention_inference",
    "v49_project_geometry_inference",
    "v49_state_geometry_inputs_inference",
]
