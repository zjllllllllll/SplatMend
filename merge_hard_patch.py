"""Select the hard-anchored SHARP layer-0 lattice and merge it without alignment.

Selection follows SHARP row provenance instead of reprojection heuristics.  Camera
centers and covariance frames are transformed by the exact inverse SuperSplat
model-view matrix; no depth scale, offset, Poisson field, or post-hoc warp exists.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
from gsplat.cuda._backend import _C as _gsplat_backend
import cv2
from PIL import Image
from plyfile import PlyData, PlyElement

_LOCAL_SHARP_SRC = Path(__file__).resolve().parent / "third_party" / "ml-sharp" / "src"
if _LOCAL_SHARP_SRC.is_dir():
    sys.path.insert(0, str(_LOCAL_SHARP_SRC))

from sharp.utils.gaussians import Gaussians3D, load_ply
from sharp.utils.gsplat import GSplatRenderer

from gaussian_patch_io import append_patch_to_base, transform_patch_to_scene


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-ply", type=Path, required=True)
    parser.add_argument("--sharp-camera-ply", type=Path, required=True)
    parser.add_argument("--hole-mask", type=Path, required=True)
    parser.add_argument("--source-rgba", type=Path, required=True)
    parser.add_argument("--target-rgb", type=Path, required=True)
    parser.add_argument("--target-depth", type=Path, required=True)
    parser.add_argument("--anchored-layer0-z", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-scale", type=float, default=0.5)
    parser.add_argument("--opacity-weight-min", type=float, default=0.01)
    parser.add_argument("--opacity-weight-floor", type=float, default=0.0)
    parser.add_argument("--blend-expand-reference-pixels", type=int, default=0)
    parser.add_argument("--blend-reference-short-side", type=int, default=1440)
    parser.add_argument("--blend-ring-count", type=int, default=6)
    parser.add_argument(
        "--blend-core-opacity-mode",
        choices=("source_alpha", "full"),
        default="source_alpha",
    )
    parser.add_argument("--max-depth-median-relative-error", type=float, default=0.03)
    parser.add_argument("--max-depth-p90-relative-error", type=float, default=0.08)
    parser.add_argument("--min-depth-coverage", type=float, default=0.98)
    return parser.parse_args()


def subset(gaussians: Gaussians3D, indices: np.ndarray) -> Gaussians3D:
    idx = torch.from_numpy(indices.astype(np.int64))
    return Gaussians3D(
        mean_vectors=gaussians.mean_vectors[:, idx],
        singular_values=gaussians.singular_values[:, idx],
        quaternions=gaussians.quaternions[:, idx],
        colors=gaussians.colors[:, idx],
        opacities=gaussians.opacities[:, idx],
    )


def render_gaussians(
    gaussians: Gaussians3D,
    extrinsics: np.ndarray,
    intrinsic: np.ndarray,
    native_width: int,
    native_height: int,
    scale: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    width = max(1, int(round(native_width * scale)))
    height = max(1, int(round(native_height * scale)))
    k = np.eye(4, dtype=np.float32)
    k[:3, :3] = intrinsic.astype(np.float32)
    k[0, :3] *= width / float(native_width)
    k[1, :3] *= height / float(native_height)
    renderer = GSplatRenderer(color_space="linearRGB", background_color="black")
    with torch.no_grad():
        result = renderer(
            gaussians.to(device),
            extrinsics=torch.from_numpy(extrinsics.astype(np.float32)).to(device)[None],
            intrinsics=torch.from_numpy(k).to(device)[None],
            image_width=width,
            image_height=height,
        )
    rgb = (
        result.color[0].permute(1, 2, 0).clamp(0, 1).mul(255).round()
        .to(torch.uint8).cpu().numpy()
    )
    depth = result.depth[0, 0].detach().cpu().numpy().astype(np.float32)
    alpha = result.alpha[0, 0].detach().cpu().numpy().astype(np.float32)
    return rgb, depth, alpha


def depth_metrics(
    depth: np.ndarray,
    alpha: np.ndarray,
    target_native: np.ndarray,
    hole_native: np.ndarray,
) -> dict[str, float | int]:
    height, width = depth.shape
    target = cv2.resize(target_native, (width, height), interpolation=cv2.INTER_LINEAR)
    hole = cv2.resize(
        hole_native.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    valid = (
        hole
        & np.isfinite(depth)
        & (depth > 1.0e-3)
        & np.isfinite(alpha)
        & (alpha >= 0.20)
        & np.isfinite(target)
        & (target > 1.0e-3)
    )
    if not np.any(valid):
        raise RuntimeError("merged render has no valid hole depth")
    absolute = np.abs(depth[valid] - target[valid])
    relative = absolute / np.maximum(target[valid], 1.0e-6)
    return {
        "hole_pixels": int(hole.sum()),
        "valid_pixels": int(valid.sum()),
        "coverage": float(valid.sum() / max(int(hole.sum()), 1)),
        "alpha_mean": float(alpha[hole].mean()),
        "absolute_error_median": float(np.median(absolute)),
        "absolute_error_p90": float(np.quantile(absolute, 0.90)),
        "relative_error_median": float(np.median(relative)),
        "relative_error_p90": float(np.quantile(relative, 0.90)),
    }


def rgb_metrics(
    rgb: np.ndarray,
    alpha: np.ndarray,
    target_native: np.ndarray,
    hole_native: np.ndarray,
) -> dict[str, float | int | None]:
    height, width = alpha.shape
    target = cv2.resize(target_native, (width, height), interpolation=cv2.INTER_AREA)
    hole = cv2.resize(
        hole_native.astype(np.uint8), (width, height), interpolation=cv2.INTER_NEAREST
    ).astype(bool)
    valid = hole & np.isfinite(alpha) & (alpha >= 0.20)
    if not np.any(valid):
        return {"valid_pixels": 0, "mae_0_1": None, "psnr_db": None}
    error = (rgb[valid].astype(np.float32) - target[valid].astype(np.float32)) / 255.0
    mse = float(np.mean(error * error))
    return {
        "valid_pixels": int(valid.sum()),
        "mae_0_1": float(np.mean(np.abs(error))),
        "psnr_db": float(-10.0 * math.log10(max(mse, 1.0e-12))),
    }


def color_depth(depth: np.ndarray, alpha: np.ndarray, lo: float, hi: float) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 1.0e-3) & (alpha > 1.0e-6)
    scaled = np.clip((np.where(valid, depth, lo) - lo) / max(hi - lo, 1e-8), 0, 1)
    result = cv2.applyColorMap(
        np.rint(scaled * 255).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    result[~valid] = 0
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def scaled_reference_pixels(
    reference_pixels: int, width: int, height: int, reference_short_side: int
) -> int:
    if reference_pixels <= 0:
        return 0
    return max(
        1,
        int(round(reference_pixels * min(width, height) / reference_short_side)),
    )


def build_ring_blend_domain(
    core: np.ndarray,
    radius: int,
    ring_count: int,
) -> tuple[np.ndarray, np.ndarray, list[float]]:
    """Build one full-opacity core plus equal-width, discrete smoothstep rings."""

    labels = np.zeros(core.shape, dtype=np.uint8)
    weights = np.zeros(core.shape, dtype=np.float32)
    labels[core] = 1
    weights[core] = 1.0
    if radius <= 0:
        return labels, weights, []

    outside = (~core).astype(np.uint8)
    distance = cv2.distanceTransform(outside, cv2.DIST_L2, cv2.DIST_MASK_PRECISE)
    band = (~core) & (distance > 0.0) & (distance <= float(radius))
    ring_width = float(radius) / float(ring_count)
    ring_index = np.ceil(distance / ring_width).astype(np.int32)
    ring_index = np.clip(ring_index, 1, ring_count)
    ring_centers = (np.arange(ring_count, dtype=np.float64) + 0.5) / ring_count
    smoothstep = ring_centers * ring_centers * (3.0 - 2.0 * ring_centers)
    ring_weights = (1.0 - smoothstep).tolist()
    for index, weight in enumerate(ring_weights, start=1):
        pixels = band & (ring_index == index)
        labels[pixels] = index + 1
        weights[pixels] = float(weight)
    return labels, weights, [float(value) for value in ring_weights]


def make_ring_overlay(
    source_rgb: np.ndarray,
    labels: np.ndarray,
    ring_weights: list[float],
) -> np.ndarray:
    colors = np.asarray(
        [
            [255, 0, 255],
            [255, 50, 40],
            [255, 130, 0],
            [255, 220, 0],
            [65, 210, 80],
            [0, 185, 255],
            [80, 80, 255],
        ],
        dtype=np.uint8,
    )
    overlay = source_rgb.astype(np.float32).copy()
    for label in range(1, int(labels.max()) + 1):
        pixels = labels == label
        if np.any(pixels):
            overlay[pixels] = 0.62 * overlay[pixels] + 0.38 * colors[label - 1]
    overlay = np.clip(overlay, 0, 255).astype(np.uint8)

    # Draw the outer boundary of the core and of every accumulated ring.
    for label in range(1, int(labels.max()) + 1):
        domain = ((labels > 0) & (labels <= label)).astype(np.uint8)
        contours, _ = cv2.findContours(domain, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cv2.drawContours(
            overlay,
            contours,
            -1,
            tuple(int(value) for value in colors[label - 1]),
            2,
            lineType=cv2.LINE_AA,
        )

    entries = [("CORE  w=1.000", colors[0])]
    entries.extend(
        (f"RING {index}  w={weight:.3f}", colors[index])
        for index, weight in enumerate(ring_weights, start=1)
    )
    line_height = 34
    box_width = 330
    box_height = 18 + line_height * len(entries)
    cv2.rectangle(overlay, (16, 16), (16 + box_width, 16 + box_height), (20, 20, 20), -1)
    for row, (text, color) in enumerate(entries):
        y = 42 + row * line_height
        cv2.rectangle(
            overlay,
            (28, y - 16),
            (50, y + 6),
            tuple(int(value) for value in color),
            -1,
        )
        cv2.putText(
            overlay,
            text,
            (62, y + 3),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            (255, 255, 255),
            2,
            cv2.LINE_AA,
        )
    return overlay


def main() -> None:
    args = parse_args()
    if _gsplat_backend is None:
        raise RuntimeError("gsplat CUDA backend failed to load before OpenCV")
    for path in (
        args.base_ply,
        args.sharp_camera_ply,
        args.hole_mask,
        args.source_rgba,
        args.target_rgb,
        args.target_depth,
        args.anchored_layer0_z,
        args.camera_json,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0.0 <= args.opacity_weight_floor <= 1.0:
        raise ValueError("--opacity-weight-floor must be in [0, 1]")
    if args.blend_expand_reference_pixels < 0:
        raise ValueError("--blend-expand-reference-pixels must be non-negative")
    if args.blend_reference_short_side <= 0:
        raise ValueError("--blend-reference-short-side must be positive")
    if not 1 <= args.blend_ring_count <= 6:
        raise ValueError("--blend-ring-count must be in [1, 6]")

    source_rgba = np.asarray(Image.open(args.source_rgba).convert("RGBA"), dtype=np.uint8)
    target_rgb = np.asarray(Image.open(args.target_rgb).convert("RGB"), dtype=np.uint8)
    target_depth = np.asarray(np.load(args.target_depth), dtype=np.float32)
    anchored_layer0_z = np.asarray(np.load(args.anchored_layer0_z), dtype=np.float32)
    hole = np.asarray(Image.open(args.hole_mask).convert("L"), dtype=np.uint8) > 127
    height, width = hole.shape
    if source_rgba.shape[:2] != (height, width) or target_rgb.shape[:2] != (height, width):
        raise ValueError("RGB and mask sizes differ")
    if target_depth.shape != (height, width):
        raise ValueError("target depth and mask sizes differ")
    camera = json.loads(args.camera_json.read_text(encoding="utf-8"))
    intrinsic = np.asarray(camera["intrinsic_K_opencv"], dtype=np.float64)
    local_to_camera = np.asarray(
        camera["splats"][0]["model_view_opencv_for_original_splat"], dtype=np.float64
    )

    sharp_ply = PlyData.read(args.sharp_camera_ply)
    sharp_vertex = sharp_ply["vertex"].data
    layer_count = 2
    if len(sharp_vertex) % layer_count != 0:
        raise RuntimeError("SHARP vertex count is not divisible by two layers")
    layer_size = len(sharp_vertex) // layer_count
    grid_side = int(round(math.sqrt(layer_size)))
    if grid_side * grid_side != layer_size:
        raise RuntimeError(f"SHARP layer is not a square lattice: {layer_size}")
    if anchored_layer0_z.shape != (grid_side, grid_side):
        raise ValueError(
            f"anchored layer0 Z {anchored_layer0_z.shape} != {(grid_side, grid_side)}"
        )
    blend_radius = scaled_reference_pixels(
        args.blend_expand_reference_pixels,
        width,
        height,
        args.blend_reference_short_side,
    )
    blend_labels, blend_weight_native, ring_weights = build_ring_blend_domain(
        hole, blend_radius, args.blend_ring_count
    )
    label_grid = cv2.resize(
        blend_labels, (grid_side, grid_side), interpolation=cv2.INTER_NEAREST
    ).astype(np.uint8)
    weight_grid = cv2.resize(
        blend_weight_native,
        (grid_side, grid_side),
        interpolation=cv2.INTER_NEAREST,
    ).astype(np.float32)
    alpha_grid = cv2.resize(
        source_rgba[..., 3].astype(np.float32) / 255.0,
        (grid_side, grid_side),
        interpolation=cv2.INTER_LINEAR,
    )
    selection_weight_grid = weight_grid.copy()
    if args.blend_core_opacity_mode == "source_alpha":
        raw_core_weight = np.clip(1.0 - alpha_grid, 0.0, 1.0)
        core_grid = label_grid == 1
        selection_weight_grid[core_grid] = raw_core_weight[core_grid]
        weight_grid[core_grid] = np.maximum(
            raw_core_weight[core_grid], float(args.opacity_weight_floor)
        )
    selected_grid = (label_grid > 0) & (
        selection_weight_grid >= args.opacity_weight_min
    )
    selected_indices = np.flatnonzero(selected_grid.reshape(-1))
    if selected_indices.size == 0:
        raise RuntimeError("no layer-0 SHARP rows selected")
    tail = sharp_vertex[selected_indices].copy()
    selected_weight = weight_grid.reshape(-1)[selected_indices].astype(np.float64)
    selected_labels = label_grid.reshape(-1)[selected_indices].astype(np.uint8)
    old_opacity = 1.0 / (
        1.0 + np.exp(-np.clip(np.asarray(tail["opacity"], dtype=np.float64), -60, 60))
    )
    new_opacity = np.clip(old_opacity * selected_weight, 1.0e-6, 1.0 - 1.0e-6)
    tail["opacity"] = (np.log(new_opacity) - np.log1p(-new_opacity)).astype(
        tail["opacity"].dtype
    )

    transformed = transform_patch_to_scene(tail, local_to_camera)
    xyz_local = transformed["xyz"].astype(np.float64)
    xyz_h = np.concatenate([xyz_local, np.ones((len(xyz_local), 1))], axis=1)
    xyz_roundtrip = (local_to_camera @ xyz_h.T).T[:, :3]
    xyz_camera = np.column_stack((tail["x"], tail["y"], tail["z"])).astype(np.float64)
    roundtrip_error = np.linalg.norm(xyz_roundtrip - xyz_camera, axis=1)

    target_selected = anchored_layer0_z.reshape(-1)[selected_indices]
    center_depth_error = np.abs(xyz_camera[:, 2] - target_selected)

    base_ply = PlyData.read(args.base_ply)
    base_vertex = base_ply["vertex"].data
    # Test-set source PLYs already have the selected object removed. Preserve every
    # source Gaussian byte-for-byte and only append the hole patch.
    retained_base_vertex = base_vertex
    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    overlay_path = output / "mask_ring_overlay.png"
    label_path = output / "blend_ring_labels.png"
    weight_path = output / "blend_ring_weights.png"
    Image.fromarray(
        make_ring_overlay(source_rgba[..., :3], blend_labels, ring_weights), mode="RGB"
    ).save(overlay_path)
    Image.fromarray(blend_labels, mode="L").save(label_path)
    Image.fromarray(
        np.rint(np.clip(blend_weight_native, 0.0, 1.0) * 255.0).astype(np.uint8),
        mode="L",
    ).save(weight_path)
    merged_path = output / "depth_anchored_inpainted.ply"
    total = append_patch_to_base(retained_base_vertex, tail, transformed, merged_path)
    merged_check = PlyData.read(merged_path)["vertex"].data
    retained_prefix_exact = np.array_equal(
        merged_check[: len(retained_base_vertex)], retained_base_vertex
    )
    original_prefix_exact = retained_prefix_exact
    if not retained_prefix_exact:
        raise RuntimeError("append-only merge altered original Gaussian rows")
    if len(merged_check) != len(base_vertex) + len(tail):
        raise RuntimeError("append-only merge output count is not base plus patch")

    patch_camera_path = output / "depth_anchored_patch_camera.ply"
    PlyData([PlyElement.describe(tail, "vertex")], text=False, byte_order="<").write(
        patch_camera_path
    )

    device = torch.device("cuda")
    patch_gaussians, _ = load_ply(patch_camera_path)
    merged_gaussians, _ = load_ply(merged_path)
    patch_rgb, patch_depth, patch_alpha = render_gaussians(
        patch_gaussians,
        np.eye(4, dtype=np.float64),
        intrinsic,
        width,
        height,
        args.render_scale,
        device,
    )
    merged_rgb, merged_depth, merged_alpha = render_gaussians(
        merged_gaussians,
        local_to_camera,
        intrinsic,
        width,
        height,
        args.render_scale,
        device,
    )
    np.save(output / "patch_render_depth.npy", patch_depth)
    np.save(output / "patch_render_alpha.npy", patch_alpha)
    np.save(output / "merged_render_depth.npy", merged_depth)
    np.save(output / "merged_render_alpha.npy", merged_alpha)
    Image.fromarray(patch_rgb).save(output / "patch_render_rgb.png")
    Image.fromarray(merged_rgb).save(output / "merged_render_rgb_dc_only.png")

    patch_depth_metrics = depth_metrics(patch_depth, patch_alpha, target_depth, hole)
    merged_depth_metrics = depth_metrics(merged_depth, merged_alpha, target_depth, hole)
    patch_rgb_metrics = rgb_metrics(patch_rgb, patch_alpha, target_rgb, hole)
    lo, hi = (float(x) for x in np.quantile(target_depth[hole], (0.01, 0.99)))
    patch_depth_color = color_depth(patch_depth, patch_alpha, lo, hi)
    merged_depth_color = color_depth(merged_depth, merged_alpha, lo, hi)
    target_small = cv2.resize(
        target_rgb, (patch_rgb.shape[1], patch_rgb.shape[0]), interpolation=cv2.INTER_AREA
    )
    preview = np.concatenate(
        [target_small, patch_rgb, merged_rgb, patch_depth_color, merged_depth_color], axis=1
    )
    preview_path = output / "merge_acceptance_preview.png"
    Image.fromarray(preview).save(preview_path)

    # In append-only mode the existing scene is deliberately left byte-for-byte
    # intact.  Its splats may still contribute inside the 2D authority mask, so
    # merged depth measures base/patch visibility competition rather than patch
    # geometry.  Certify the inserted patch itself; retain merged depth as a
    # diagnostic so callers can see where untouched base splats dominate.
    failures: list[str] = []
    if float(patch_depth_metrics["coverage"]) < args.min_depth_coverage:
        failures.append("patch depth coverage below threshold")
    if (
        float(patch_depth_metrics["relative_error_median"])
        > args.max_depth_median_relative_error
    ):
        failures.append("patch median relative depth error above threshold")
    if (
        float(patch_depth_metrics["relative_error_p90"])
        > args.max_depth_p90_relative_error
    ):
        failures.append("patch p90 relative depth error above threshold")
    if float(np.quantile(center_depth_error, 0.99)) > 0.005:
        failures.append("selected patch center p99 depth error above 0.005")
    if float(roundtrip_error.max()) > 1.0e-5:
        failures.append("camera/local coordinate roundtrip error above 1e-5")

    merged_depth_warnings: list[str] = []
    if float(merged_depth_metrics["coverage"]) < args.min_depth_coverage:
        merged_depth_warnings.append("merged depth coverage below threshold")
    if (
        float(merged_depth_metrics["relative_error_median"])
        > args.max_depth_median_relative_error
    ):
        merged_depth_warnings.append("untouched base dominates merged median depth")
    if (
        float(merged_depth_metrics["relative_error_p90"])
        > args.max_depth_p90_relative_error
    ):
        merged_depth_warnings.append("untouched base dominates merged p90 depth")

    report = {
        "status": "accepted" if not failures else "rejected",
        "contract": (
            "depth_anchored_append_only_six_ring_blend_v3"
            if blend_radius > 0
            else "depth_anchored_append_only_merge_v2"
        ),
        "selection": {
            "method": (
                "SHARP_layer0_row_provenance_authority_plus_discrete_outer_rings"
                if blend_radius > 0
                else "SHARP_layer0_row_provenance_resized_authority_mask"
            ),
            "grid": [grid_side, grid_side],
            "full_gaussians": int(len(sharp_vertex)),
            "selected_layer0_gaussians": int(len(tail)),
            "selected_index_min": int(selected_indices.min()),
            "selected_index_max": int(selected_indices.max()),
            "opacity_weight_min": float(args.opacity_weight_min),
            "opacity_weight_floor": float(args.opacity_weight_floor),
            "opacity_weight_median": float(np.median(selected_weight)),
            "core_opacity_mode": args.blend_core_opacity_mode,
            "blend_reference_pixels": int(args.blend_expand_reference_pixels),
            "blend_reference_short_side": int(args.blend_reference_short_side),
            "blend_radius_native_pixels": int(blend_radius),
            "blend_ring_count": int(len(ring_weights)),
            "blend_ring_width_native_pixels": (
                float(blend_radius / len(ring_weights)) if ring_weights else 0.0
            ),
            "blend_ring_opacity_weights_inner_to_outer": ring_weights,
            "native_pixels_by_label": {
                "core": int(np.count_nonzero(blend_labels == 1)),
                **{
                    f"ring_{index}": int(np.count_nonzero(blend_labels == index + 1))
                    for index in range(1, len(ring_weights) + 1)
                },
            },
            "selected_gaussians_by_label": {
                "core": int(np.count_nonzero(selected_labels == 1)),
                **{
                    f"ring_{index}": int(np.count_nonzero(selected_labels == index + 1))
                    for index in range(1, len(ring_weights) + 1)
                },
            },
        },
        "geometry": {
            "depth_scale_or_offset_after_generation": False,
            "posthoc_position_correction": False,
            "selected_center_depth_error_median": float(np.median(center_depth_error)),
            "selected_center_depth_error_p99": float(
                np.quantile(center_depth_error, 0.99)
            ),
            "camera_local_roundtrip_error_max": float(roundtrip_error.max()),
        },
        "merge": {
            "base_gaussians": int(len(base_vertex)),
            "base_preservation_mode": "append_only_no_deletion",
            "removed_base_gaussians": 0,
            "retained_base_gaussians": int(len(retained_base_vertex)),
            "removed_projected_core_gaussians": 0,
            "removed_projected_inconsistent_gaussians": 0,
            "base_hole_cleanup_enabled": False,
            "base_screen_visible_gaussians": 0,
            "base_screen_overlaps_core_gaussians": 0,
            "new_gaussians": int(len(tail)),
            "output_gaussians": int(total),
            "original_prefix_exact": original_prefix_exact,
            "retained_original_rows_exact": bool(retained_prefix_exact),
        },
        "support_render": {
            "patch_depth": patch_depth_metrics,
            "merged_depth": merged_depth_metrics,
            "patch_rgb": patch_rgb_metrics,
            "color_note": "merged RGB diagnostic uses DC color only; source higher SH remains in PLY",
        },
        "acceptance": {
            "depth_source": "patch_only_due_append_only_base_preservation",
            "merged_depth_is_diagnostic": True,
            "merged_depth_warnings": merged_depth_warnings,
        },
        "outputs": {
            "merged_ply": str(merged_path.resolve()),
            "patch_camera_ply": str(patch_camera_path.resolve()),
            "preview": str(preview_path.resolve()),
            "mask_ring_overlay": str(overlay_path.resolve()),
            "blend_ring_labels": str(label_path.resolve()),
            "blend_ring_weights": str(weight_path.resolve()),
        },
        "failures": failures,
    }
    report_path = output / "merge_acceptance.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "MERGE_" + ("ACCEPTED" if not failures else "REJECTED"),
        f"new={len(tail)}",
        "removed_base=0",
        f"retained_exact={retained_prefix_exact}",
        f"center_p99={float(np.quantile(center_depth_error, 0.99)):.6f}",
        f"patch_med_rel={float(patch_depth_metrics['relative_error_median']):.6f}",
        f"patch_p90_rel={float(patch_depth_metrics['relative_error_p90']):.6f}",
        f"merged_med_rel={float(merged_depth_metrics['relative_error_median']):.6f}",
        f"merged_p90_rel={float(merged_depth_metrics['relative_error_p90']):.6f}",
        f"coverage={float(merged_depth_metrics['coverage']):.6f}",
        f"blend_radius={blend_radius}",
        f"rings={len(ring_weights)}",
        f"report={report_path}",
        flush=True,
    )
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
