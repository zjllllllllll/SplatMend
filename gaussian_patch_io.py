"""PLY schema conversion and rigid camera-to-scene transformation helpers.

This module contains only the two operations used by the current append-only
pipeline.  It deliberately contains no depth alignment, ICP, scale fitting, or
post-generation position correction.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation


def transform_patch_to_scene(
    patch: np.ndarray,
    scene_to_camera: np.ndarray,
) -> dict[str, np.ndarray | float]:
    """Transform Gaussian centers and covariance frames into scene coordinates."""

    camera_to_scene = np.linalg.inv(scene_to_camera)
    rotation_camera_to_scene = camera_to_scene[:3, :3]
    determinant = float(np.linalg.det(rotation_camera_to_scene))
    orthogonality_error = float(
        np.max(
            np.abs(
                rotation_camera_to_scene @ rotation_camera_to_scene.T
                - np.eye(3)
            )
        )
    )
    if abs(determinant - 1.0) > 1.0e-3 or orthogonality_error > 1.0e-3:
        raise ValueError(
            "Camera transform is not a proper rigid rotation: "
            f"det={determinant}, orthogonality_error={orthogonality_error}"
        )

    xyz_camera = np.column_stack((patch["x"], patch["y"], patch["z"])).astype(
        np.float64
    )
    xyz_homogeneous = np.concatenate(
        (xyz_camera, np.ones((len(patch), 1), dtype=np.float64)), axis=1
    )
    xyz_scene = (camera_to_scene @ xyz_homogeneous.T).T[:, :3].astype(np.float32)

    quaternion_wxyz = np.column_stack(
        (patch["rot_0"], patch["rot_1"], patch["rot_2"], patch["rot_3"])
    ).astype(np.float64)
    quaternion_wxyz /= np.maximum(
        np.linalg.norm(quaternion_wxyz, axis=1, keepdims=True), 1.0e-12
    )
    quaternion_xyzw = np.column_stack(
        (
            quaternion_wxyz[:, 1],
            quaternion_wxyz[:, 2],
            quaternion_wxyz[:, 3],
            quaternion_wxyz[:, 0],
        )
    )
    rotation_camera = Rotation.from_quat(quaternion_xyzw).as_matrix()
    rotation_scene = rotation_camera_to_scene[None, :, :] @ rotation_camera
    quaternion_scene_xyzw = Rotation.from_matrix(rotation_scene).as_quat()
    quaternion_scene_wxyz = np.column_stack(
        (
            quaternion_scene_xyzw[:, 3],
            quaternion_scene_xyzw[:, 0],
            quaternion_scene_xyzw[:, 1],
            quaternion_scene_xyzw[:, 2],
        )
    ).astype(np.float32)

    scale_logs = np.column_stack(
        (patch["scale_0"], patch["scale_1"], patch["scale_2"])
    ).astype(np.float32)

    return {
        "xyz": xyz_scene,
        "quaternion_wxyz": quaternion_scene_wxyz,
        "scale_logs": scale_logs,
        "camera_to_scene": camera_to_scene,
        "rotation_det": determinant,
        "rotation_orth_error": orthogonality_error,
    }


def append_patch_to_base(
    base_vertex: np.ndarray,
    sharp_patch: np.ndarray,
    transformed: dict[str, np.ndarray | float],
    output_ply: Path,
) -> int:
    """Convert SHARP fields to the base schema and append without changing base rows."""

    base_names = set(base_vertex.dtype.names or ())
    required_base = {
        "x",
        "y",
        "z",
        "f_dc_0",
        "f_dc_1",
        "f_dc_2",
        "opacity",
        "scale_0",
        "scale_1",
        "scale_2",
        "rot_0",
        "rot_1",
        "rot_2",
        "rot_3",
    }
    missing = required_base.difference(base_names)
    if missing:
        raise ValueError(f"Base PLY is missing required fields: {sorted(missing)}")

    patch_out = np.zeros(len(sharp_patch), dtype=base_vertex.dtype)
    xyz = np.asarray(transformed["xyz"])
    quaternion = np.asarray(transformed["quaternion_wxyz"])
    scale_logs = np.asarray(transformed["scale_logs"])
    for index, name in enumerate(("x", "y", "z")):
        patch_out[name] = xyz[:, index]
    for name in ("f_dc_0", "f_dc_1", "f_dc_2", "opacity"):
        patch_out[name] = sharp_patch[name]
    for index, name in enumerate(("scale_0", "scale_1", "scale_2")):
        patch_out[name] = scale_logs[:, index]
    for index, name in enumerate(("rot_0", "rot_1", "rot_2", "rot_3")):
        patch_out[name] = quaternion[:, index]
    # SHARP exports degree-0 color.  Higher-order f_rest_* fields remain zero.

    merged = np.empty(len(base_vertex) + len(patch_out), dtype=base_vertex.dtype)
    merged[: len(base_vertex)] = base_vertex
    merged[len(base_vertex) :] = patch_out
    output_ply.parent.mkdir(parents=True, exist_ok=True)
    PlyData([PlyElement.describe(merged, "vertex")], text=False, byte_order="<").write(
        output_ply
    )
    return int(len(merged))
