"""Generate SHARP Gaussians with an exact first-surface camera-Z anchor.

The learned Alignment UNet is bypassed.  Completed camera-Z supplies the first
depth channel directly; SHARP's monocular relative gap supplies a conservative
second channel to keep the five-channel decoder contract.  SHARP still predicts
appearance, covariance, rotation and opacity.  Layer-0 keeps the learned lateral
ray, moves to exact target camera-Z, transports its covariance by that motion, and
optionally applies a target-surface Jacobian floor to tangent scale and rotation.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from gsplat.cuda._backend import _C as _gsplat_backend
from PIL import Image

if _gsplat_backend is None:
    raise RuntimeError(
        "gsplat CUDA backend is unavailable; run through run_cuda_python.cmd "
        "and preserve the verified CUDA extension cache"
    )

_LOCAL_SHARP_SRC = Path(__file__).resolve().parent / "third_party" / "ml-sharp" / "src"
if _LOCAL_SHARP_SRC.is_dir():
    sys.path.insert(0, str(_LOCAL_SHARP_SRC))

from sharp.utils import gaussians as sharp_gaussians
from sharp.utils.gaussians import (
    Gaussians3D,
    get_unprojection_matrix,
    save_ply,
    unproject_gaussians,
)

from sharp_runtime import (
    color_depth,
    depth_metrics,
    load_predictor,
    patch_gpu_svd,
    render,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", type=Path, required=True)
    parser.add_argument("--depth", type=Path, required=True)
    parser.add_argument("--hole-mask", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--render-scale", type=float, default=0.5)
    parser.add_argument("--alpha-min", type=float, default=0.20)
    parser.add_argument("--max-median-relative-error", type=float, default=0.03)
    parser.add_argument("--max-p90-relative-error", type=float, default=0.08)
    parser.add_argument("--min-hole-coverage", type=float, default=0.98)
    parser.add_argument(
        "--covariance-mode",
        choices=("ray_scale", "surface_jacobian"),
        default="surface_jacobian",
    )
    parser.add_argument("--surface-tangent-sigma", type=float, default=0.75)
    parser.add_argument("--surface-normal-min-ratio", type=float, default=0.05)
    parser.add_argument("--surface-normal-max-ratio", type=float, default=0.35)
    parser.add_argument("--surface-max-stretch", type=float, default=4.0)
    parser.add_argument("--min-hole-alpha-mean", type=float, default=0.95)
    parser.add_argument("--min-hole-alpha-p10", type=float, default=0.85)
    return parser.parse_args()


def flatten_gaussians(gaussians: Gaussians3D) -> Gaussians3D:
    def vector(value: torch.Tensor) -> torch.Tensor:
        return value.permute(0, 2, 3, 4, 1).flatten(1, 3).contiguous()

    return Gaussians3D(
        mean_vectors=vector(gaussians.mean_vectors),
        singular_values=vector(gaussians.singular_values),
        quaternions=vector(gaussians.quaternions),
        colors=vector(gaussians.colors),
        opacities=gaussians.opacities.flatten(1, 3).contiguous(),
    )


def tensor_quantiles(value: torch.Tensor) -> dict[str, float]:
    flat = value.detach().float().reshape(-1)
    q = torch.quantile(
        flat,
        torch.tensor([0.01, 0.10, 0.50, 0.90, 0.99], device=flat.device),
    ).cpu()
    return {
        "p01": float(q[0]),
        "p10": float(q[1]),
        "median": float(q[2]),
        "p90": float(q[3]),
        "p99": float(q[4]),
    }


def robust_grid_tangent(
    points: torch.Tensor,
    axis: int,
    baseline: torch.Tensor,
    max_stretch: float,
) -> torch.Tensor:
    """Finite-difference a surface without bridging a strong depth discontinuity."""

    if axis == 2:
        edges = points[:, :, 1:] - points[:, :, :-1]
        forward = torch.cat([edges, edges[:, :, -1:]], dim=2)
        backward = torch.cat([edges[:, :, :1], edges], dim=2)
        fallback_axis = 0
    elif axis == 1:
        edges = points[:, 1:] - points[:, :-1]
        forward = torch.cat([edges, edges[:, -1:]], dim=1)
        backward = torch.cat([edges[:, :1], edges], dim=1)
        fallback_axis = 1
    else:
        raise ValueError(f"unsupported grid axis: {axis}")

    forward_norm = torch.linalg.vector_norm(forward, dim=-1).clamp_min(1.0e-12)
    backward_norm = torch.linalg.vector_norm(backward, dim=-1).clamp_min(1.0e-12)
    cosine = (forward * backward).sum(dim=-1) / (forward_norm * backward_norm)
    norm_ratio = torch.maximum(forward_norm, backward_norm) / torch.minimum(
        forward_norm, backward_norm
    )
    consistent = (cosine > 0.25) & (norm_ratio < 3.0)
    shorter = torch.where(
        (forward_norm <= backward_norm)[..., None], forward, backward
    )
    tangent = torch.where(consistent[..., None], 0.5 * (forward + backward), shorter)

    norm = torch.linalg.vector_norm(tangent, dim=-1)
    fallback = torch.zeros_like(tangent)
    fallback[..., fallback_axis] = 1.0
    direction = torch.where(
        (norm > 1.0e-10)[..., None], tangent / norm.clamp_min(1.0e-12)[..., None], fallback
    )
    minimum = 0.50 * baseline
    maximum = float(max_stretch) * baseline
    clamped_norm = torch.maximum(torch.minimum(norm, maximum), minimum)
    return direction * clamped_norm[..., None]


def surface_jacobian_covariance(
    gaussians: Gaussians3D,
    surface_points: torch.Tensor,
    layer_size: int,
    focal: float,
    native_width: int,
    native_height: int,
    grid_width: int,
    grid_height: int,
    tangent_sigma: float,
    normal_min_ratio: float,
    normal_max_ratio: float,
    max_stretch: float,
) -> tuple[Gaussians3D, dict[str, object]]:
    """Floor layer-0 covariance from the target surface's image-space Jacobian."""

    z = surface_points[..., 2].clamp_min(1.0e-6)
    base_u = z * (float(native_width) / float(grid_width)) / float(focal)
    base_v = z * (float(native_height) / float(grid_height)) / float(focal)
    tangent_u = robust_grid_tangent(surface_points, 2, base_u, max_stretch)
    tangent_v = robust_grid_tangent(surface_points, 1, base_v, max_stretch)

    length_u = torch.linalg.vector_norm(tangent_u, dim=-1).clamp_min(1.0e-10)
    e_u = tangent_u / length_u[..., None]
    tangent_v_orthogonal = tangent_v - (tangent_v * e_u).sum(dim=-1)[..., None] * e_u
    length_v = torch.linalg.vector_norm(tangent_v_orthogonal, dim=-1)
    fallback_v = torch.zeros_like(e_u)
    fallback_v[..., 1] = 1.0
    fallback_v = fallback_v - (fallback_v * e_u).sum(dim=-1)[..., None] * e_u
    fallback_v_norm = torch.linalg.vector_norm(fallback_v, dim=-1)
    fallback_z = torch.zeros_like(e_u)
    fallback_z[..., 2] = 1.0
    fallback_z = fallback_z - (fallback_z * e_u).sum(dim=-1)[..., None] * e_u
    fallback_v = torch.where(
        (fallback_v_norm > 1.0e-8)[..., None], fallback_v, fallback_z
    )
    e_v = torch.where(
        (length_v > 1.0e-10)[..., None],
        tangent_v_orthogonal / length_v.clamp_min(1.0e-12)[..., None],
        fallback_v / torch.linalg.vector_norm(fallback_v, dim=-1).clamp_min(1.0e-12)[
            ..., None
        ],
    )
    normal = torch.linalg.cross(e_u, e_v, dim=-1)
    normal = normal / torch.linalg.vector_norm(normal, dim=-1).clamp_min(1.0e-12)[
        ..., None
    ]
    e_v = torch.linalg.cross(normal, e_u, dim=-1)

    old_covariance = sharp_gaussians.compose_covariance_matrices(
        gaussians.quaternions[:, :layer_size],
        gaussians.singular_values[:, :layer_size],
    ).reshape(surface_points.shape[0], grid_height, grid_width, 3, 3)

    def directional_sigma(direction: torch.Tensor) -> torch.Tensor:
        return torch.sqrt(
            torch.einsum("...i,...ij,...j->...", direction, old_covariance, direction)
            .clamp_min(1.0e-16)
        )

    old_sigma_u = directional_sigma(e_u)
    old_sigma_v = directional_sigma(e_v)
    old_sigma_n = directional_sigma(normal)
    target_sigma_u = float(tangent_sigma) * length_u
    effective_length_v = torch.maximum(length_v, 0.50 * base_v)
    target_sigma_v = float(tangent_sigma) * effective_length_v
    sigma_u = torch.maximum(old_sigma_u, target_sigma_u)
    sigma_v = torch.maximum(old_sigma_v, target_sigma_v)
    tangent_min = torch.minimum(sigma_u, sigma_v)
    sigma_n = torch.maximum(old_sigma_n, float(normal_min_ratio) * tangent_min)
    sigma_n = torch.minimum(sigma_n, float(normal_max_ratio) * tangent_min)

    covariance = (
        sigma_u.square()[..., None, None] * e_u[..., :, None] * e_u[..., None, :]
        + sigma_v.square()[..., None, None] * e_v[..., :, None] * e_v[..., None, :]
        + sigma_n.square()[..., None, None]
        * normal[..., :, None]
        * normal[..., None, :]
    ).reshape(surface_points.shape[0], layer_size, 3, 3)
    quaternions, singular_values = sharp_gaussians.decompose_covariance_matrices(
        covariance
    )
    updated_scales = gaussians.singular_values.clone()
    updated_quaternions = gaussians.quaternions.clone()
    updated_scales[:, :layer_size] = singular_values
    updated_quaternions[:, :layer_size] = quaternions

    diagnostics: dict[str, object] = {
        "tangent_sigma_factor": float(tangent_sigma),
        "normal_min_ratio": float(normal_min_ratio),
        "normal_max_ratio": float(normal_max_ratio),
        "max_surface_stretch": float(max_stretch),
        "old_sigma_u_over_cell": tensor_quantiles(old_sigma_u / length_u),
        "old_sigma_v_over_cell": tensor_quantiles(
            old_sigma_v / effective_length_v
        ),
        "new_sigma_u_over_cell": tensor_quantiles(sigma_u / length_u),
        "new_sigma_v_over_cell": tensor_quantiles(
            sigma_v / effective_length_v
        ),
        "normal_over_min_tangent": tensor_quantiles(sigma_n / tangent_min),
    }
    return (
        Gaussians3D(
            mean_vectors=gaussians.mean_vectors,
            singular_values=updated_scales,
            quaternions=updated_quaternions,
            colors=gaussians.colors,
            opacities=gaussians.opacities,
        ),
        diagnostics,
    )


def alpha_quality(
    alpha: np.ndarray, hole_native: np.ndarray
) -> dict[str, float | int]:
    height, width = alpha.shape
    hole = F.interpolate(
        torch.from_numpy(hole_native.astype(np.float32))[None, None],
        size=(height, width),
        mode="nearest",
    )[0, 0].numpy().astype(bool)
    values = np.asarray(alpha[hole], dtype=np.float32)
    return {
        "pixels": int(values.size),
        "mean": float(values.mean()),
        "p10": float(np.quantile(values, 0.10)),
        "median": float(np.median(values)),
        "p90": float(np.quantile(values, 0.90)),
        "fraction_alpha_ge_0_80": float(np.mean(values >= 0.80)),
        "fraction_alpha_ge_0_95": float(np.mean(values >= 0.95)),
    }


def hard_predict(
    predictor,
    image: np.ndarray,
    target_depth: np.ndarray,
    focal: float,
    device: torch.device,
    covariance_mode: str,
    surface_tangent_sigma: float,
    surface_normal_min_ratio: float,
    surface_normal_max_ratio: float,
    surface_max_stretch: float,
) -> tuple[Gaussians3D, np.ndarray, dict[str, object], float]:
    internal = (1536, 1536)
    height, width = image.shape[:2]
    image_tensor = (
        torch.from_numpy(image.copy()).to(device=device, dtype=torch.float32)
        .permute(2, 0, 1)
        / 255.0
    )
    image_resized = F.interpolate(
        image_tensor[None], size=internal, mode="bilinear", align_corners=True
    )
    target_resized = F.interpolate(
        torch.from_numpy(target_depth.copy()).to(device=device, dtype=torch.float32)[
            None, None
        ],
        size=internal,
        mode="bilinear",
        align_corners=True,
    )
    disparity_factor = torch.tensor(
        [float(focal) / float(width)], dtype=torch.float32, device=device
    )

    started = time.perf_counter()
    with torch.no_grad():
        monodepth_output = predictor.monodepth_model(image_resized)
        monodepth = disparity_factor[:, None, None, None] / monodepth_output.disparity.clamp(
            min=1.0e-4, max=1.0e4
        )
        if monodepth.shape[1] == 1:
            injected_depth = target_resized
        else:
            # Preserve only a bounded relative occlusion gap; the first channel is
            # the externally accepted surface and is never estimated by SHARP.
            ratio = monodepth[:, 1:2] / monodepth[:, 0:1].clamp_min(1.0e-4)
            ratio = ratio.clamp(min=1.01, max=2.0)
            injected_depth = torch.cat([target_resized, target_resized * ratio], dim=1)

        init_output = predictor.init_model(image_resized, injected_depth)
        features = predictor.feature_model(
            init_output.feature_input, encodings=monodepth_output.output_features
        )
        delta = predictor.prediction_head(features)
        gaussians_grid = predictor.gaussian_composer(
            delta=delta,
            base_values=init_output.gaussian_base_values,
            global_scale=init_output.global_scale,
            flatten_output=False,
        )

        layers = int(predictor.init_model.num_layers)
        grid_h, grid_w = gaussians_grid.opacities.shape[-2:]
        if layers < 1 or gaussians_grid.mean_vectors.shape[2] != layers:
            raise RuntimeError("unexpected SHARP Gaussian layer layout")
        target_grid = F.interpolate(
            target_resized,
            size=(grid_h, grid_w),
            mode="bilinear",
            align_corners=False,
        )[:, 0]
        base_x = init_output.gaussian_base_values.mean_x_ndc[:, 0, 0]
        base_y = init_output.gaussian_base_values.mean_y_ndc[:, 0, 0]
        old_means = gaussians_grid.mean_vectors[:, :, 0]
        old_z = old_means[:, 2].clamp_min(1.0e-6)
        depth_ratio = target_grid / old_z
        means = gaussians_grid.mean_vectors.clone()
        # Keep SHARP's learned lateral ray (including delta-x/y), and move the
        # Gaussian along that same ray to the exact target camera-Z.
        means[:, :, 0] = old_means * depth_ratio[:, None]
        means[:, 2, 0] = target_grid

        # T = ratio * I is the exact local covariance transport for same-ray
        # radial motion.  A target-surface Jacobian floor is optionally applied
        # after camera unprojection to account for non-uniform neighboring motion.
        singular_values = gaussians_grid.singular_values.clone()
        singular_values[:, :, 0] *= depth_ratio[:, None]

        # A conventional depth map provides no supervision for layer 1.  Disable it
        # in the first accepted route instead of allowing an unverified back layer to
        # bias the rendered expected depth or create floaters.
        opacities = gaussians_grid.opacities.clone()
        if layers > 1:
            opacities[:, 1:] = 0.0
        anchored_grid = Gaussians3D(
            mean_vectors=means,
            singular_values=singular_values,
            quaternions=gaussians_grid.quaternions,
            colors=gaussians_grid.colors,
            opacities=opacities,
        )
        gaussians_ndc = flatten_gaussians(anchored_grid)

        intrinsic = torch.tensor(
            [
                [focal, 0.0, width / 2.0, 0.0],
                [0.0, focal, height / 2.0, 0.0],
                [0.0, 0.0, 1.0, 0.0],
                [0.0, 0.0, 0.0, 1.0],
            ],
            dtype=torch.float32,
            device=device,
        )
        intrinsic_resized = intrinsic.clone()
        intrinsic_resized[0] *= internal[1] / float(width)
        intrinsic_resized[1] *= internal[0] / float(height)
        gaussians = unproject_gaussians(
            gaussians_ndc,
            torch.eye(4, dtype=torch.float32, device=device),
            intrinsic_resized,
            internal,
        )
        surface_ndc = torch.stack(
            (target_grid * base_x, target_grid * base_y, target_grid), dim=-1
        )
        unprojection = get_unprojection_matrix(
            torch.eye(4, dtype=torch.float32, device=device),
            intrinsic_resized,
            internal,
        )[:3]
        surface_camera = (
            surface_ndc @ unprojection[:, :3].T + unprojection[:, 3]
        )
        surface_diagnostics: dict[str, object] = {
            "mode": covariance_mode,
            "applied": False,
        }
        if covariance_mode == "surface_jacobian":
            gaussians, surface_diagnostics = surface_jacobian_covariance(
                gaussians,
                surface_camera,
                grid_h * grid_w,
                focal,
                width,
                height,
                grid_w,
                grid_h,
                surface_tangent_sigma,
                surface_normal_min_ratio,
                surface_normal_max_ratio,
                surface_max_stretch,
            )
            surface_diagnostics = {
                "mode": covariance_mode,
                "applied": True,
                **surface_diagnostics,
            }
    torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - started

    camera_grid = gaussians.mean_vectors.reshape(
        1, layers, grid_h, grid_w, 3
    )[0, 0, ..., 2]
    target_grid_cpu = target_grid[0].detach().cpu().numpy().astype(np.float32)
    camera_grid_cpu = camera_grid.detach().cpu().numpy().astype(np.float32)
    error = np.abs(camera_grid_cpu - target_grid_cpu)
    old_ray_x = old_means[:, 0] / old_z
    old_ray_y = old_means[:, 1] / old_z
    new_ray_x = means[:, 0, 0] / means[:, 2, 0].clamp_min(1.0e-6)
    new_ray_y = means[:, 1, 0] / means[:, 2, 0].clamp_min(1.0e-6)
    base_ray_offset = torch.sqrt(
        (old_ray_x - base_x).square() + (old_ray_y - base_y).square()
    )
    preserved_ray_error = torch.maximum(
        torch.abs(new_ray_x - old_ray_x), torch.abs(new_ray_y - old_ray_y)
    )
    diagnostics: dict[str, object] = {
        "layers": layers,
        "grid_width": grid_w,
        "grid_height": grid_h,
        "layer0_center_absolute_error_median": float(np.median(error)),
        "layer0_center_absolute_error_p99": float(np.quantile(error, 0.99)),
        "layer0_center_absolute_error_max": float(error.max()),
        "disabled_unsupervised_layers": max(layers - 1, 0),
        "center_update": "preserve_learned_ray_then_set_exact_camera_z",
        "depth_ratio": tensor_quantiles(depth_ratio),
        "learned_ray_offset_magnitude": tensor_quantiles(base_ray_offset),
        "preserved_ray_error_max": float(preserved_ray_error.max()),
        "surface_covariance": surface_diagnostics,
    }
    return gaussians, camera_grid_cpu, diagnostics, inference_seconds


def main() -> None:
    args = parse_args()
    for path in (args.image, args.depth, args.hole_mask, args.camera_json, args.checkpoint):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required")
    if not 0.0 < args.surface_tangent_sigma <= 2.0:
        raise ValueError("--surface-tangent-sigma must be in (0, 2]")
    if not 0.0 < args.surface_normal_min_ratio <= args.surface_normal_max_ratio <= 1.0:
        raise ValueError("surface normal ratios must satisfy 0 < min <= max <= 1")
    if args.surface_max_stretch < 1.0:
        raise ValueError("--surface-max-stretch must be >= 1")
    image = np.asarray(Image.open(args.image).convert("RGB"), dtype=np.uint8)
    depth = np.asarray(np.load(args.depth), dtype=np.float32)
    hole = np.asarray(Image.open(args.hole_mask).convert("L"), dtype=np.uint8) > 127
    height, width = image.shape[:2]
    if depth.shape != (height, width) or hole.shape != (height, width):
        raise ValueError("image, depth and mask are not registered")
    if not np.all(np.isfinite(depth) & (depth > 1.0e-3)):
        raise ValueError("hard SHARP depth must be dense, finite and positive")
    camera = json.loads(args.camera_json.read_text(encoding="utf-8"))
    intrinsic = np.asarray(camera["intrinsic_K_opencv"], dtype=np.float64)
    focal = float(intrinsic[0, 0])

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda")
    patch_gpu_svd()
    predictor = load_predictor(args.checkpoint, device)
    torch.cuda.reset_peak_memory_stats(device)
    gaussians, layer0_z, center_diagnostics, inference_seconds = hard_predict(
        predictor,
        image,
        depth,
        focal,
        device,
        args.covariance_mode,
        args.surface_tangent_sigma,
        args.surface_normal_min_ratio,
        args.surface_normal_max_ratio,
        args.surface_max_stretch,
    )
    ply_path = output / "sharp_hard_layer0_camera.ply"
    save_ply(gaussians, focal, (height, width), ply_path)
    rgb_render, depth_render, alpha_render, render_seconds = render(
        gaussians, focal, width, height, args.render_scale, device
    )
    metrics, target_render, hole_render = depth_metrics(
        depth_render, alpha_render, depth, hole, args.alpha_min
    )
    alpha_metrics = alpha_quality(alpha_render, hole)
    np.save(output / "sharp_hard_render_depth.npy", depth_render)
    np.save(output / "sharp_hard_render_alpha.npy", alpha_render)
    np.save(output / "sharp_hard_layer0_z.npy", layer0_z)
    Image.fromarray(rgb_render).save(output / "sharp_hard_render_rgb.png")
    lo, hi = (float(x) for x in np.quantile(depth[hole], (0.01, 0.99)))
    depth_color = color_depth(depth_render, alpha_render, lo, hi)
    target_color = color_depth(
        target_render, np.ones_like(target_render, dtype=np.float32), lo, hi
    )
    target_color[~hole_render] = (
        target_color[~hole_render].astype(np.float32) * 0.25
    ).astype(np.uint8)
    Image.fromarray(depth_color).save(output / "sharp_hard_render_depth.png")
    preview = np.concatenate([rgb_render, target_color, depth_color], axis=1)
    preview_path = output / "sharp_hard_acceptance_preview.png"
    Image.fromarray(preview).save(preview_path)

    failures: list[str] = []
    if float(metrics["coverage"]) < args.min_hole_coverage:
        failures.append(
            f"coverage {float(metrics['coverage']):.6f} < {args.min_hole_coverage:.6f}"
        )
    if float(metrics["relative_error_median"]) > args.max_median_relative_error:
        failures.append(
            f"median relative error {float(metrics['relative_error_median']):.6f} > "
            f"{args.max_median_relative_error:.6f}"
        )
    if float(metrics["relative_error_p90"]) > args.max_p90_relative_error:
        failures.append(
            f"p90 relative error {float(metrics['relative_error_p90']):.6f} > "
            f"{args.max_p90_relative_error:.6f}"
        )
    if float(center_diagnostics["layer0_center_absolute_error_p99"]) > 0.005:
        failures.append(
            "layer0 center p99 absolute error exceeds 0.005 scene units"
        )
    if float(alpha_metrics["mean"]) < args.min_hole_alpha_mean:
        failures.append(
            f"hole alpha mean {float(alpha_metrics['mean']):.6f} < "
            f"{args.min_hole_alpha_mean:.6f}"
        )
    if float(alpha_metrics["p10"]) < args.min_hole_alpha_p10:
        failures.append(
            f"hole alpha p10 {float(alpha_metrics['p10']):.6f} < "
            f"{args.min_hole_alpha_p10:.6f}"
        )

    report = {
        "status": "accepted" if not failures else "rejected",
        "contract": "sharp_hard_camera_z_surface_covariance_v2",
        "alignment_unet": "bypassed",
        "first_layer_geometry": (
            "preserved_learned_ray_exact_camera_z_with_surface_jacobian_covariance"
        ),
        "second_layer": "opacity_disabled_no_depth_supervision",
        "inference_seconds": inference_seconds,
        "render_seconds": render_seconds,
        "peak_cuda_memory_bytes": int(torch.cuda.max_memory_allocated(device)),
        "gaussians": int(gaussians.mean_vectors.shape[1]),
        "center_diagnostics": center_diagnostics,
        "render_depth_metrics_in_hole": metrics,
        "render_alpha_quality_in_hole": alpha_metrics,
        "thresholds": {
            "min_hole_coverage": float(args.min_hole_coverage),
            "max_median_relative_error": float(args.max_median_relative_error),
            "max_p90_relative_error": float(args.max_p90_relative_error),
            "max_layer0_center_p99_absolute_error": 0.005,
            "min_hole_alpha_mean": float(args.min_hole_alpha_mean),
            "min_hole_alpha_p10": float(args.min_hole_alpha_p10),
        },
        "ply": str(ply_path.resolve()),
        "preview": str(preview_path.resolve()),
        "failures": failures,
    }
    report_path = output / "sharp_hard_acceptance.json"
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "SHARP_HARD_" + ("ACCEPTED" if not failures else "REJECTED"),
        f"center_p99={float(center_diagnostics['layer0_center_absolute_error_p99']):.6f}",
        f"render_med_rel={float(metrics['relative_error_median']):.6f}",
        f"render_p90_rel={float(metrics['relative_error_p90']):.6f}",
        f"coverage={float(metrics['coverage']):.6f}",
        f"alpha_mean={float(alpha_metrics['mean']):.6f}",
        f"alpha_p10={float(alpha_metrics['p10']):.6f}",
        f"report={report_path}",
        flush=True,
    )
    if failures:
        raise RuntimeError("; ".join(failures))


if __name__ == "__main__":
    main()
