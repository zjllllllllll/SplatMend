"""Minimal SHARP loading, rendering, and acceptance helpers for this pipeline."""

from __future__ import annotations

import sys
import time
from pathlib import Path

import cv2
import numpy as np
import torch

_LOCAL_SHARP_SRC = Path(__file__).resolve().parent / "third_party" / "ml-sharp" / "src"
if _LOCAL_SHARP_SRC.is_dir():
    sys.path.insert(0, str(_LOCAL_SHARP_SRC))

from sharp.models import PredictorParams, create_predictor
from sharp.utils import gaussians as sharp_gaussians
from sharp.utils import linalg as sharp_linalg
from sharp.utils.gaussians import Gaussians3D
from sharp.utils.gsplat import GSplatRenderer


def patch_gpu_svd() -> None:
    """Use a CUDA-safe covariance decomposition for the installed SHARP build."""

    original = sharp_gaussians.decompose_covariance_matrices

    def decompose(covariance_matrices: torch.Tensor):
        if covariance_matrices.device.type != "cuda":
            return original(covariance_matrices)
        dtype = covariance_matrices.dtype
        matrices = covariance_matrices.detach().to(dtype=torch.float32)
        rotations, singular_values_2, _ = torch.linalg.svd(matrices)
        bad = torch.linalg.det(rotations) < 0
        if bool(bad.any()):
            rotations[bad, :, -1] *= -1
        quaternions = sharp_linalg.quaternions_from_rotation_matrices(rotations)
        return (
            quaternions.to(dtype=dtype),
            singular_values_2.clamp_min(1.0e-12).sqrt().to(dtype=dtype),
        )

    sharp_gaussians.decompose_covariance_matrices = decompose


def load_predictor(checkpoint: Path, device: torch.device):
    state = torch.load(checkpoint, weights_only=True, map_location="cpu", mmap=True)
    predictor = None
    try:
        with torch.device("meta"):
            predictor = create_predictor(PredictorParams())
        predictor.load_state_dict(state, assign=True)
    except (AttributeError, NotImplementedError, RuntimeError, TypeError):
        del predictor
        predictor = create_predictor(PredictorParams())
        try:
            predictor.load_state_dict(state, assign=True)
        except TypeError:
            predictor.load_state_dict(state)
    del state
    return predictor.eval().to(device)


def render(
    gaussians: Gaussians3D,
    focal: float,
    width: int,
    height: int,
    render_scale: float,
    device: torch.device,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, float]:
    out_width = max(1, int(round(width * render_scale)))
    out_height = max(1, int(round(height * render_scale)))
    intrinsic = torch.eye(4, dtype=torch.float32, device=device)
    intrinsic[0, 0] = float(focal) * out_width / float(width)
    intrinsic[1, 1] = float(focal) * out_height / float(height)
    intrinsic[0, 2] = out_width / 2.0
    intrinsic[1, 2] = out_height / 2.0
    renderer = GSplatRenderer(color_space="linearRGB", background_color="black")
    started = time.perf_counter()
    with torch.no_grad():
        result = renderer(
            gaussians,
            extrinsics=torch.eye(4, dtype=torch.float32, device=device)[None],
            intrinsics=intrinsic[None],
            image_width=out_width,
            image_height=out_height,
        )
    torch.cuda.synchronize(device)
    seconds = time.perf_counter() - started
    rgb = (
        result.color[0]
        .permute(1, 2, 0)
        .clamp(0.0, 1.0)
        .mul(255.0)
        .round()
        .to(torch.uint8)
        .cpu()
        .numpy()
    )
    depth = result.depth[0, 0].detach().cpu().numpy().astype(np.float32)
    alpha = result.alpha[0, 0].detach().cpu().numpy().astype(np.float32)
    return rgb, depth, alpha, seconds


def depth_metrics(
    rendered: np.ndarray,
    alpha: np.ndarray,
    target_native: np.ndarray,
    hole_native: np.ndarray,
    alpha_min: float,
) -> tuple[dict[str, float | int], np.ndarray, np.ndarray]:
    height, width = rendered.shape
    target = cv2.resize(target_native, (width, height), interpolation=cv2.INTER_LINEAR)
    hole = cv2.resize(
        hole_native.astype(np.uint8),
        (width, height),
        interpolation=cv2.INTER_NEAREST,
    ).astype(bool)
    valid = (
        hole
        & np.isfinite(rendered)
        & (rendered > 1.0e-3)
        & np.isfinite(target)
        & (target > 1.0e-3)
        & np.isfinite(alpha)
        & (alpha >= alpha_min)
    )
    total = int(hole.sum())
    count = int(valid.sum())
    if count == 0:
        raise RuntimeError("SHARP render has no valid depth inside the hole")
    absolute = np.abs(rendered[valid] - target[valid])
    relative = absolute / np.maximum(target[valid], 1.0e-6)
    signed = rendered[valid] - target[valid]
    metrics: dict[str, float | int] = {
        "hole_pixels": total,
        "valid_pixels": count,
        "coverage": float(count / max(total, 1)),
        "alpha_mean_in_hole": float(np.mean(alpha[hole])),
        "absolute_error_median": float(np.median(absolute)),
        "absolute_error_p90": float(np.quantile(absolute, 0.90)),
        "absolute_error_p95": float(np.quantile(absolute, 0.95)),
        "relative_error_median": float(np.median(relative)),
        "relative_error_p90": float(np.quantile(relative, 0.90)),
        "relative_error_p95": float(np.quantile(relative, 0.95)),
        "signed_error_median": float(np.median(signed)),
    }
    return metrics, target, hole


def color_depth(
    depth: np.ndarray,
    alpha: np.ndarray,
    lower: float,
    upper: float,
) -> np.ndarray:
    valid = np.isfinite(depth) & (depth > 1.0e-3) & (alpha > 1.0e-6)
    safe = np.where(valid, depth, lower)
    scaled = np.clip((safe - lower) / max(upper - lower, 1.0e-8), 0.0, 1.0)
    result = cv2.applyColorMap(
        np.rint(scaled * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    result[~valid] = 0
    return cv2.cvtColor(result, cv2.COLOR_BGR2RGB)
