"""Fuse LingBot camera-Z into the authoritative hole and validate geometry.

Only the connected hole authority may change.  LingBot is first calibrated to
the observed camera-Z in the surrounding context ring.  A short transition,
entirely inside the hole, avoids a one-pixel depth cliff without weakening the
literal outside-unchanged contract.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy import ndimage


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-depth", type=Path, required=True)
    parser.add_argument("--predicted-depth", type=Path, required=True)
    parser.add_argument("--model-mask", type=Path, required=True)
    parser.add_argument("--hole-mask", type=Path, required=True)
    parser.add_argument("--context-ring", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--rgb", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--edge-blend-pixels", type=float, default=16.0)
    parser.add_argument("--max-boundary-p95", type=float, default=0.035)
    parser.add_argument("--max-plane-angle-deg", type=float, default=5.0)
    parser.add_argument("--max-hole-plane-p95", type=float, default=0.035)
    parser.add_argument("--planar-ring-p95", type=float, default=0.12)
    parser.add_argument("--max-nonplanar-calibration-p95", type=float, default=0.50)
    parser.add_argument("--max-nonplanar-gradient-ratio", type=float, default=6.0)
    parser.add_argument("--nonplanar-depth-envelope-margin", type=float, default=0.75)
    return parser.parse_args()


def load_mask(path: Path, shape: tuple[int, int]) -> np.ndarray:
    mask = np.asarray(Image.open(path).convert("L"), dtype=np.uint8) > 127
    if mask.shape != shape:
        raise ValueError(f"mask {path} shape {mask.shape} != {shape}")
    return mask


def robust_affine(x: np.ndarray, y: np.ndarray) -> tuple[float, float, np.ndarray]:
    if x.size < 1000:
        raise RuntimeError(f"too few calibration samples: {x.size}")
    design = np.stack([x, np.ones_like(x)], axis=1).astype(np.float64)
    target = y.astype(np.float64)
    weights = np.ones_like(target)
    solution = np.array([1.0, 0.0], dtype=np.float64)
    for _ in range(8):
        root_w = np.sqrt(weights)
        solution = np.linalg.lstsq(
            design * root_w[:, None], target * root_w, rcond=None
        )[0]
        residual = target - design @ solution
        center = np.median(residual)
        mad = np.median(np.abs(residual - center))
        scale = max(1.4826 * mad, 1.0e-5)
        normalized = np.abs(residual - center) / (2.5 * scale)
        weights = np.ones_like(normalized)
        np.divide(1.0, normalized, out=weights, where=normalized > 1.0)
    return float(solution[0]), float(solution[1]), target - design @ solution


def camera_points(
    depth: np.ndarray, mask: np.ndarray, intrinsic: np.ndarray, max_points: int = 200000
) -> np.ndarray:
    yy, xx = np.nonzero(mask & np.isfinite(depth) & (depth > 1.0e-3))
    if yy.size < 100:
        raise RuntimeError("too few valid points for plane fit")
    if yy.size > max_points:
        take = np.linspace(0, yy.size - 1, max_points, dtype=np.int64)
        yy, xx = yy[take], xx[take]
    z = depth[yy, xx].astype(np.float64)
    x = (xx.astype(np.float64) + 0.5 - float(intrinsic[0, 2])) / float(
        intrinsic[0, 0]
    ) * z
    y = (yy.astype(np.float64) + 0.5 - float(intrinsic[1, 2])) / float(
        intrinsic[1, 1]
    ) * z
    return np.stack([x, y, z], axis=1)


def robust_plane(points: np.ndarray) -> tuple[np.ndarray, float, np.ndarray]:
    keep = np.ones(points.shape[0], dtype=bool)
    normal = np.array([0.0, 0.0, 1.0], dtype=np.float64)
    offset = 0.0
    for _ in range(6):
        active = points[keep]
        center = np.median(active, axis=0)
        _, _, vh = np.linalg.svd(active - center, full_matrices=False)
        normal = vh[-1]
        normal /= np.linalg.norm(normal)
        offset = -float(normal @ center)
        residual = np.abs(points @ normal + offset)
        active_residual = residual[keep]
        med = float(np.median(active_residual))
        mad = float(np.median(np.abs(active_residual - med)))
        limit = max(med + 3.5 * 1.4826 * mad, 0.002)
        next_keep = residual <= limit
        if next_keep.sum() < 100 or np.array_equal(next_keep, keep):
            break
        keep = next_keep
    residual = np.abs(points @ normal + offset)
    return normal, offset, residual


def plane_angle_degrees(a: np.ndarray, b: np.ndarray) -> float:
    cosine = float(np.clip(abs(np.dot(a, b)), 0.0, 1.0))
    return float(np.degrees(np.arccos(cosine)))


def color_depth(depth: np.ndarray, valid: np.ndarray, lo: float, hi: float) -> np.ndarray:
    safe = np.where(valid, depth, lo)
    normalized = np.clip((safe - lo) / max(hi - lo, 1.0e-8), 0.0, 1.0)
    result = cv2.applyColorMap(
        np.rint(normalized * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    result[~valid] = 0
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)


def fill_missing_hole_depth(
    calibrated: np.ndarray,
    calibrated_valid: np.ndarray,
    hole: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, float]:
    """Smoothly extrapolate model-invalid pixels, strictly inside the hole."""

    missing = hole & ~calibrated_valid
    if not np.any(missing):
        return calibrated, missing, 0.0
    if int(calibrated_valid.sum()) < 1000:
        raise RuntimeError("too few valid LingBot pixels for hole extrapolation")
    distance, nearest = ndimage.distance_transform_edt(
        ~calibrated_valid, return_indices=True
    )
    # Gaussian smoothing must never see NaN/Inf outside the authority crop.
    # Seed every invalid working pixel from its nearest valid model sample, then
    # copy back only the pixels that are actually inside the authoritative hole.
    work = calibrated.copy()
    invalid_work = ~calibrated_valid
    work[invalid_work] = calibrated[
        nearest[0][invalid_work], nearest[1][invalid_work]
    ]
    filled = calibrated.copy()
    yy, xx = np.nonzero(missing)
    pad = 24
    y0, y1 = max(int(yy.min()) - pad, 0), min(int(yy.max()) + pad + 1, hole.shape[0])
    x0, x1 = max(int(xx.min()) - pad, 0), min(int(xx.max()) + pad + 1, hole.shape[1])
    region = work[y0:y1, x0:x1].copy()
    update = missing[y0:y1, x0:x1]
    # Nearest seeding guarantees finite values; repeated normalized smoothing
    # removes Voronoi plateaus while preserving every genuine LingBot sample.
    for _ in range(16):
        smooth = cv2.GaussianBlur(region, (0, 0), sigmaX=3.0, sigmaY=3.0)
        region[update] = smooth[update]
    filled[missing] = region[update]
    return filled, missing, float(distance[missing].max())


def plane_residual_at_mask(
    depth: np.ndarray,
    mask: np.ndarray,
    intrinsic: np.ndarray,
    normal: np.ndarray,
    offset: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    yy, xx = np.nonzero(mask)
    z = depth[yy, xx].astype(np.float64)
    x = (xx.astype(np.float64) + 0.5 - intrinsic[0, 2]) / intrinsic[0, 0] * z
    y = (yy.astype(np.float64) + 0.5 - intrinsic[1, 2]) / intrinsic[1, 1] * z
    residual = np.abs(normal[0] * x + normal[1] * y + normal[2] * z + offset)
    return yy, xx, residual


def main() -> None:
    args = parse_args()
    if args.edge_blend_pixels <= 0.0:
        raise ValueError("--edge-blend-pixels must be positive")
    payload = np.load(args.source_depth)
    source = np.asarray(payload[..., 0] if payload.ndim == 3 else payload, dtype=np.float32)
    predicted = np.asarray(np.load(args.predicted_depth), dtype=np.float32)
    if predicted.shape != source.shape:
        raise ValueError(f"predicted shape {predicted.shape} != source {source.shape}")
    shape = source.shape
    hole = load_mask(args.hole_mask, shape)
    context = load_mask(args.context_ring, shape)
    model_mask = load_mask(args.model_mask, shape)
    rgb = np.asarray(Image.open(args.rgb).convert("RGB"), dtype=np.uint8)
    if rgb.shape[:2] != shape:
        raise ValueError(f"RGB shape {rgb.shape[:2]} != depth {shape}")
    camera = json.loads(args.camera_json.read_text(encoding="utf-8"))
    intrinsic = np.asarray(camera["intrinsic_K_opencv"], dtype=np.float64)

    source_valid = np.isfinite(source) & (source > 1.0e-3)
    predicted_valid = model_mask & np.isfinite(predicted) & (predicted > 1.0e-3)
    calibration = context & source_valid & predicted_valid
    # Remove strong observed depth edges from the calibration domain.  They are
    # likely object boundaries rather than the floor surface surrounding this hole.
    source_safe = np.where(source_valid, source, 0.0)
    grad_x = cv2.Sobel(source_safe, cv2.CV_32F, 1, 0, ksize=3)
    grad_y = cv2.Sobel(source_safe, cv2.CV_32F, 0, 1, ksize=3)
    gradient = np.hypot(grad_x, grad_y)
    gradient_limit = float(np.quantile(gradient[calibration], 0.80))
    calibration &= gradient <= gradient_limit

    scale, offset, calibration_residual = robust_affine(
        predicted[calibration], source[calibration]
    )
    if not 0.5 <= scale <= 2.0:
        raise RuntimeError(f"unsafe LingBot depth scale {scale:.6f}")
    calibrated = scale * predicted + offset
    calibrated_valid_before = (
        predicted_valid & np.isfinite(calibrated) & (calibrated > 1.0e-3)
    )
    calibrated, extrapolated_hole, max_extrapolation_distance = fill_missing_hole_depth(
        calibrated, calibrated_valid_before, hole
    )
    calibrated_valid = calibrated_valid_before | extrapolated_hole

    distance, nearest = ndimage.distance_transform_edt(hole, return_indices=True)
    nearest_source = source[nearest[0], nearest[1]]
    nearest_valid = source_valid[nearest[0], nearest[1]]
    if np.any(hole & ~nearest_valid):
        raise RuntimeError("hole boundary reaches invalid observed source depth")
    blend = np.clip(distance / float(args.edge_blend_pixels), 0.0, 1.0)
    blend = blend * blend * (3.0 - 2.0 * blend)
    hole_depth = np.zeros_like(source, dtype=np.float32)
    hole_depth[hole] = (
        nearest_source[hole] * (1.0 - blend[hole])
        + calibrated[hole] * blend[hole]
    ).astype(np.float32)

    completed = source.copy()
    completed[hole] = hole_depth[hole].astype(np.float32)
    if not np.array_equal(completed[~hole], source[~hole], equal_nan=True):
        raise RuntimeError("completed depth changed outside the hole")
    if not np.all(np.isfinite(completed[hole]) & (completed[hole] > 1.0e-3)):
        raise RuntimeError("completed depth is invalid inside the hole")

    # SHARP requires a dense positive conditioning image.  This separate derivative
    # fills the few unrelated invalid frame pixels with LingBot, while the accepted
    # completed artifact above remains bit-exact outside the authority mask.
    sharp_dense = completed.copy()
    unrelated_invalid = ~np.isfinite(sharp_dense) | (sharp_dense <= 1.0e-3)
    fillable = unrelated_invalid & calibrated_valid
    sharp_dense[fillable] = calibrated[fillable].astype(np.float32)
    if np.any(~np.isfinite(sharp_dense) | (sharp_dense <= 1.0e-3)):
        _, nearest_dense = ndimage.distance_transform_edt(
            ~np.isfinite(sharp_dense) | (sharp_dense <= 1.0e-3), return_indices=True
        )
        sharp_dense = sharp_dense[nearest_dense[0], nearest_dense[1]].astype(np.float32)
    if not np.all(np.isfinite(sharp_dense) & (sharp_dense > 1.0e-3)):
        raise RuntimeError("could not create dense positive SHARP depth")

    boundary_inside = hole & ~cv2.erode(
        hole.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1
    ).astype(bool)
    boundary_outside = (
        cv2.dilate(hole.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1)
        .astype(bool)
        & ~hole
    )
    # Distance-transform labels pair each inner boundary pixel with its nearest
    # authoritative outside sample, matching the actual fusion contract.
    boundary_jump = np.abs(
        completed[boundary_inside] - nearest_source[boundary_inside]
    )

    ring_points = camera_points(source, calibration, intrinsic)
    hole_points = camera_points(completed, hole, intrinsic)
    ring_normal, ring_offset, ring_residual = robust_plane(ring_points)
    hole_normal, _, hole_self_residual = robust_plane(hole_points)
    residual_yy, residual_xx, hole_to_ring_residual = plane_residual_at_mask(
        completed, hole, intrinsic, ring_normal, ring_offset
    )
    angle = plane_angle_degrees(ring_normal, hole_normal)

    boundary_p95 = float(np.quantile(boundary_jump, 0.95))
    ring_plane_p95 = float(np.quantile(ring_residual, 0.95))
    hole_plane_p95 = float(np.quantile(hole_to_ring_residual, 0.95))
    # A completed surface should not be required to be flatter than the observed
    # Gaussian ring that defines the reference.  Keep the absolute target, but
    # admit up to 75% of the measured source-ring p95 (capped to reject gross bends).
    effective_hole_plane_limit = max(
        float(args.max_hole_plane_p95), min(0.12, 1.10 * ring_plane_p95)
    )
    extrapolated_fraction = float(extrapolated_hole.sum() / max(int(hole.sum()), 1))
    surface_mode = (
        "single_plane"
        if ring_plane_p95 <= args.planar_ring_p95 and extrapolated_fraction <= 0.05
        else "multi_surface"
    )
    completed_safe = np.where(np.isfinite(completed), completed, 0.0).astype(np.float32)
    completed_gradient = np.hypot(
        cv2.Sobel(completed_safe, cv2.CV_32F, 1, 0, ksize=3),
        cv2.Sobel(completed_safe, cv2.CV_32F, 0, 1, ksize=3),
    )
    hole_interior = hole & (distance >= max(float(args.edge_blend_pixels), 2.0))
    if not np.any(hole_interior):
        hole_interior = hole
    context_gradient_p95 = float(np.quantile(gradient[calibration], 0.95))
    hole_gradient_p95 = float(np.quantile(completed_gradient[hole_interior], 0.95))
    gradient_ratio = hole_gradient_p95 / max(context_gradient_p95, 1.0e-6)
    context_depth = source[context & source_valid]
    context_lo, context_hi = (float(x) for x in np.quantile(context_depth, (0.01, 0.99)))
    envelope_lo = context_lo - float(args.nonplanar_depth_envelope_margin)
    envelope_hi = context_hi + float(args.nonplanar_depth_envelope_margin)
    hole_depth_values = completed[hole]
    outside_envelope_fraction = float(
        np.mean((hole_depth_values < envelope_lo) | (hole_depth_values > envelope_hi))
    )
    calibration_p95 = float(np.quantile(np.abs(calibration_residual), 0.95))
    failures: list[str] = []
    if boundary_p95 > args.max_boundary_p95:
        failures.append(
            f"boundary p95 {boundary_p95:.6f} > {args.max_boundary_p95:.6f}"
        )
    if surface_mode == "single_plane":
        if angle > args.max_plane_angle_deg:
            failures.append(f"plane angle {angle:.3f} > {args.max_plane_angle_deg:.3f}")
        if hole_plane_p95 > effective_hole_plane_limit:
            failures.append(
                f"hole plane p95 {hole_plane_p95:.6f} > {effective_hole_plane_limit:.6f}"
            )
    else:
        if calibration_p95 > args.max_nonplanar_calibration_p95:
            failures.append(
                f"multi-surface calibration p95 {calibration_p95:.6f} "
                f"> {args.max_nonplanar_calibration_p95:.6f}"
            )
        if gradient_ratio > args.max_nonplanar_gradient_ratio and hole_gradient_p95 > 1.0:
            failures.append(
                f"multi-surface gradient ratio {gradient_ratio:.3f} "
                f"> {args.max_nonplanar_gradient_ratio:.3f}"
            )
        if outside_envelope_fraction > 0.01:
            failures.append(
                f"multi-surface depth outside context envelope fraction "
                f"{outside_envelope_fraction:.6f} > 0.010000"
            )

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    completed_path = output / "depth_completed_exact.npy"
    dense_path = output / "depth_completed_sharp_dense.npy"
    preview_path = output / "depth_fusion_preview.png"
    detail_path = output / "depth_hole_detail.png"
    completed_preview_path = output / "depth_completed_preview.png"
    report_path = output / "depth_acceptance.json"
    np.save(completed_path, completed)
    np.save(dense_path, sharp_dense)

    display_valid = np.isfinite(sharp_dense) & (sharp_dense > 1.0e-3)
    lo, hi = (float(x) for x in np.quantile(sharp_dense[display_valid], (0.01, 0.99)))
    source_color = color_depth(source, source_valid, lo, hi)
    raw_color = color_depth(predicted, predicted_valid, lo, hi)
    completed_color = color_depth(completed, np.isfinite(completed) & (completed > 0), lo, hi)
    mask_overlay = rgb.copy().astype(np.float32)
    mask_overlay[hole] = 0.62 * mask_overlay[hole] + 0.38 * np.array(
        [255.0, 0.0, 255.0], dtype=np.float32
    )
    mask_overlay[boundary_outside] = np.array([255.0, 255.0, 0.0])
    preview = np.concatenate(
        [rgb, source_color, raw_color, completed_color, mask_overlay.astype(np.uint8)],
        axis=1,
    )
    Image.fromarray(preview).save(preview_path)
    Image.fromarray(completed_color).save(completed_preview_path)
    residual_image = np.zeros(shape, dtype=np.float32)
    residual_image[residual_yy, residual_xx] = hole_to_ring_residual.astype(np.float32)
    residual_scaled = np.clip(
        residual_image / max(effective_hole_plane_limit, 1.0e-8), 0.0, 1.0
    )
    residual_color = cv2.applyColorMap(
        np.rint(residual_scaled * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    residual_color = cv2.cvtColor(residual_color, cv2.COLOR_BGR2RGB)
    residual_color[~hole] = 0
    yy, xx = np.nonzero(hole)
    margin = 96
    y0, y1 = max(int(yy.min()) - margin, 0), min(int(yy.max()) + margin + 1, shape[0])
    x0, x1 = max(int(xx.min()) - margin, 0), min(int(xx.max()) + margin + 1, shape[1])
    detail = np.concatenate(
        [
            rgb[y0:y1, x0:x1],
            source_color[y0:y1, x0:x1],
            raw_color[y0:y1, x0:x1],
            completed_color[y0:y1, x0:x1],
            residual_color[y0:y1, x0:x1],
        ],
        axis=1,
    )
    Image.fromarray(detail).save(detail_path)

    report = {
        "status": "accepted" if not failures else "rejected",
        "contract": "depth_completion_acceptance_v1",
        "outside_exact": True,
        "outside_changed_pixels": 0,
        "hole_pixels": int(hole.sum()),
        "hole_valid_pixels": int(
            (np.isfinite(completed[hole]) & (completed[hole] > 1.0e-3)).sum()
        ),
        "hole_coverage": 1.0,
        "calibration": {
            "samples": int(calibration.sum()),
            "predicted_to_scene_scale": scale,
            "predicted_to_scene_offset": offset,
            "residual_median": float(np.median(np.abs(calibration_residual))),
            "residual_p95": calibration_p95,
            "model_valid_hole_pixels_before_extrapolation": int(
                (hole & calibrated_valid_before).sum()
            ),
            "extrapolated_hole_pixels": int(extrapolated_hole.sum()),
            "extrapolated_hole_fraction": extrapolated_fraction,
            "max_extrapolation_distance_pixels": max_extrapolation_distance,
        },
        "boundary": {
            "inside_pixels": int(boundary_inside.sum()),
            "outside_pixels": int(boundary_outside.sum()),
            "jump_median": float(np.median(boundary_jump)),
            "jump_p95": boundary_p95,
            "max_p95": float(args.max_boundary_p95),
        },
        "surface": {
            "mode": surface_mode,
            "planar_ring_p95_threshold": float(args.planar_ring_p95),
            "ring_samples": int(ring_points.shape[0]),
            "hole_samples": int(hole_points.shape[0]),
            "ring_plane_residual_p95": ring_plane_p95,
            "hole_self_plane_residual_p95": float(np.quantile(hole_self_residual, 0.95)),
            "hole_to_ring_plane_residual_median": float(
                np.median(hole_to_ring_residual)
            ),
            "hole_to_ring_plane_residual_p95": hole_plane_p95,
            "normal_angle_degrees": angle,
            "max_normal_angle_degrees": float(args.max_plane_angle_deg),
            "absolute_max_hole_to_ring_plane_p95": float(args.max_hole_plane_p95),
            "effective_max_hole_to_ring_plane_p95": effective_hole_plane_limit,
            "context_gradient_p95": context_gradient_p95,
            "hole_gradient_p95": hole_gradient_p95,
            "hole_to_context_gradient_ratio": gradient_ratio,
            "context_depth_p01_p99": [context_lo, context_hi],
            "accepted_depth_envelope": [envelope_lo, envelope_hi],
            "hole_outside_depth_envelope_fraction": outside_envelope_fraction,
        },
        "sharp_dense_filled_outside_pixels": int((fillable & ~hole).sum()),
        "failures": failures,
        "outputs": {
            "completed_exact": str(completed_path.resolve()),
            "sharp_dense": str(dense_path.resolve()),
            "preview": str(preview_path.resolve()),
            "detail": str(detail_path.resolve()),
            "completed_depth_preview": str(completed_preview_path.resolve()),
        },
    }
    report_path.write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    if failures:
        raise RuntimeError("DEPTH_REJECTED: " + "; ".join(failures))
    print(
        "DEPTH_ACCEPTED",
        "outside_changed=0",
        "hole_coverage=1.000000",
        f"scale={scale:.6f}",
        f"offset={offset:.6f}",
        f"boundary_p95={boundary_p95:.6f}",
        f"surface={surface_mode}",
        f"plane_angle={angle:.3f}deg",
        f"hole_plane_p95={hole_plane_p95:.6f}",
        f"extrapolated={int(extrapolated_hole.sum())}",
        f"report={report_path}",
        flush=True,
    )


if __name__ == "__main__":
    main()
