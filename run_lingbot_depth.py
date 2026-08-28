#!/usr/bin/env python3
"""Run the official LingBot-Depth RGB-D inference contract on one job view.

This adapter intentionally contains no depth post-processing.  It only converts
the project's native RGB, camera-Z depth and pixel intrinsics into the tensors
documented by LingBot-Depth, calls ``MDMModel.infer``, and persists the raw model
outputs plus reproducibility evidence.  Geometry ownership and backprojection
remain responsibilities of ``sharp_sd2_inpaint_pipeline.py``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

import cv2
import numpy as np
import torch


def _sha256(path: str) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        while True:
            block = handle.read(1024 * 1024)
            if not block:
                break
            digest.update(block)
    return digest.hexdigest()


def _imread_unicode(path: str, flags: int) -> np.ndarray | None:
    """Read an image from a Windows path that may contain non-ASCII text."""

    try:
        encoded = np.fromfile(os.fspath(path), dtype=np.uint8)
    except OSError:
        return None
    if encoded.size == 0:
        return None
    return cv2.imdecode(encoded, flags)


def _imwrite_unicode(path: str, image: np.ndarray) -> None:
    """Write an image without passing a Unicode path through OpenCV's fopen."""

    extension = os.path.splitext(os.fspath(path))[1]
    if not extension:
        raise ValueError(f"image output path has no extension: {path}")
    ok, encoded = cv2.imencode(extension, image)
    if not ok:
        raise ValueError(f"failed to encode image output: {path}")
    try:
        encoded.tofile(os.fspath(path))
    except OSError as exc:
        raise ValueError(f"failed to write image output: {path}") from exc


def _depth_color(depth: np.ndarray, valid: np.ndarray, vmin: float, vmax: float) -> np.ndarray:
    safe = np.where(valid, depth, vmin)
    scaled = np.clip((safe - vmin) / max(vmax - vmin, 1.0e-8), 0.0, 1.0)
    colored = cv2.applyColorMap((scaled * 255.0 + 0.5).astype(np.uint8), cv2.COLORMAP_TURBO)
    colored[~valid] = 0
    return colored


def _use_fp16_for_device(device: torch.device) -> bool:
    """LingBot half precision is supported only by the CUDA inference route."""

    return device.type == "cuda"


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Official LingBot-Depth inference adapter for a native Click-GS job view"
    )
    parser.add_argument("--lingbot_root", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--rgb", required=True)
    parser.add_argument("--depth", required=True)
    parser.add_argument("--intrinsics", required=True)
    parser.add_argument("--hole_mask", required=True)
    parser.add_argument("--out_depth", required=True)
    parser.add_argument("--out_mask", required=True)
    parser.add_argument("--out_preview", required=True)
    parser.add_argument("--out_manifest", required=True)
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--resolution_level", type=int, choices=range(10), default=9)
    args = parser.parse_args()

    root = os.path.abspath(os.path.expanduser(args.lingbot_root))
    checkpoint = os.path.abspath(os.path.expanduser(args.checkpoint))
    if not os.path.isfile(os.path.join(root, "mdm", "model", "v2.py")):
        raise FileNotFoundError(f"LingBot source tree is incomplete: {root}")
    if not os.path.isfile(checkpoint):
        raise FileNotFoundError(f"LingBot checkpoint not found: {checkpoint}")
    sys.path.insert(0, root)
    from mdm.model.v2 import MDMModel  # noqa: E402

    rgb_bgr = _imread_unicode(args.rgb, cv2.IMREAD_COLOR)
    if rgb_bgr is None:
        raise ValueError(f"failed to read RGB input: {args.rgb}")
    rgb = cv2.cvtColor(rgb_bgr, cv2.COLOR_BGR2RGB)
    depth = np.load(args.depth).astype(np.float32)
    intrinsics_px = np.loadtxt(args.intrinsics, dtype=np.float32).reshape(3, 3)
    hole_mask_raw = _imread_unicode(args.hole_mask, cv2.IMREAD_GRAYSCALE)
    if hole_mask_raw is None:
        raise ValueError(f"failed to read hole mask: {args.hole_mask}")
    hole_mask = hole_mask_raw > 127
    height, width = rgb.shape[:2]
    if depth.shape != (height, width) or hole_mask.shape != (height, width):
        raise ValueError(
            "LingBot native inputs are not pixel registered: "
            f"rgb={(height, width)} depth={depth.shape} mask={hole_mask.shape}"
        )
    if not np.isfinite(intrinsics_px).all():
        raise ValueError("LingBot intrinsics contain non-finite values")
    if float(intrinsics_px[0, 0]) <= 0.0 or float(intrinsics_px[1, 1]) <= 0.0:
        raise ValueError("LingBot focal lengths must be positive")
    depth = np.nan_to_num(depth, nan=0.0, posinf=0.0, neginf=0.0)
    depth[depth <= 0.0] = 0.0
    if np.any(depth[hole_mask] != 0.0):
        raise ValueError("LingBot hole-mask pixels must be zero in the input camera-Z depth")

    intrinsics_normalized = intrinsics_px.copy()
    intrinsics_normalized[0, 0] /= float(width)
    intrinsics_normalized[0, 2] /= float(width)
    intrinsics_normalized[1, 1] /= float(height)
    intrinsics_normalized[1, 2] /= float(height)

    device = torch.device(args.device)
    if device.type == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("LingBot was asked to use CUDA, but CUDA is unavailable")
    use_fp16 = _use_fp16_for_device(device)
    image_tensor = torch.tensor(
        rgb.astype(np.float32) / 255.0, dtype=torch.float32, device=device
    ).permute(2, 0, 1).unsqueeze(0)
    depth_tensor = torch.tensor(depth, dtype=torch.float32, device=device)
    intrinsics_tensor = torch.tensor(
        intrinsics_normalized, dtype=torch.float32, device=device
    ).unsqueeze(0)

    if device.type == "cuda":
        torch.cuda.reset_peak_memory_stats(device)
    load_started = time.perf_counter()
    model = MDMModel.from_pretrained(checkpoint).eval().to(device)
    load_seconds = time.perf_counter() - load_started
    min_tokens, max_tokens = model.num_tokens_range
    num_tokens = int(
        min_tokens
        + (int(args.resolution_level) / 9.0) * (max_tokens - min_tokens)
    )
    inference_started = time.perf_counter()
    with torch.no_grad():
        output = model.infer(
            image_tensor,
            depth_in=depth_tensor,
            num_tokens=num_tokens,
            apply_mask=True,
            use_fp16=use_fp16,
            intrinsics=intrinsics_tensor,
        )
    if device.type == "cuda":
        torch.cuda.synchronize(device)
    inference_seconds = time.perf_counter() - inference_started

    predicted = output["depth"].squeeze().detach().cpu().numpy().astype(np.float32)
    model_mask_tensor = output.get("mask")
    model_mask = (
        model_mask_tensor.squeeze().detach().cpu().numpy().astype(bool)
        if model_mask_tensor is not None
        else np.isfinite(predicted) & (predicted > 0.0)
    )
    if predicted.shape != (height, width) or model_mask.shape != (height, width):
        raise RuntimeError(
            f"LingBot output shape mismatch: depth={predicted.shape} mask={model_mask.shape}"
        )
    finite_positive = np.isfinite(predicted) & (predicted > 1.0e-3)
    valid_output = model_mask & finite_positive
    hole_valid = valid_output & hole_mask

    for path in (args.out_depth, args.out_mask, args.out_preview, args.out_manifest):
        parent = os.path.dirname(os.path.abspath(path))
        os.makedirs(parent, exist_ok=True)
    np.save(args.out_depth, predicted)
    _imwrite_unicode(args.out_mask, model_mask.astype(np.uint8) * 255)

    range_values = predicted[valid_output]
    if range_values.size:
        vmin, vmax = (float(x) for x in np.quantile(range_values, [0.01, 0.99]))
    else:
        vmin, vmax = 0.0, 1.0
    input_valid = depth > 1.0e-3
    input_color = _depth_color(depth, input_valid, vmin, vmax)
    output_color = _depth_color(predicted, valid_output, vmin, vmax)
    preview = np.concatenate([rgb_bgr, input_color, output_color], axis=1)
    _imwrite_unicode(args.out_preview, preview)

    peak_memory = (
        int(torch.cuda.max_memory_allocated(device)) if device.type == "cuda" else None
    )
    manifest = {
        "adapter": "official_lingbot_depth_rgbd_v1",
        "lingbot_root": root,
        "checkpoint": checkpoint,
        "checkpoint_size_bytes": int(os.path.getsize(checkpoint)),
        "checkpoint_sha256": _sha256(checkpoint),
        "rgb": os.path.abspath(args.rgb),
        "rgb_sha256": _sha256(args.rgb),
        "depth_input": os.path.abspath(args.depth),
        "depth_input_sha256": _sha256(args.depth),
        "intrinsics_input": os.path.abspath(args.intrinsics),
        "intrinsics_pixels": intrinsics_px.tolist(),
        "intrinsics_normalized": intrinsics_normalized.tolist(),
        "hole_mask": os.path.abspath(args.hole_mask),
        "hole_mask_sha256": _sha256(args.hole_mask),
        "width": int(width),
        "height": int(height),
        "depth_semantics": "camera_z",
        "inference": {
            "api": "MDMModel.infer",
            "resolution_level": int(args.resolution_level),
            "num_tokens": int(num_tokens),
            "apply_mask": True,
            "use_fp16": bool(use_fp16),
            "device": str(device),
            "load_seconds": float(load_seconds),
            "inference_seconds": float(inference_seconds),
            "peak_cuda_memory_bytes": peak_memory,
        },
        "input": {
            "hole_pixels": int(hole_mask.sum()),
            "nonzero_depth_pixels": int(input_valid.sum()),
        },
        "output": {
            "depth_output": os.path.abspath(args.out_depth),
            "depth_output_sha256": _sha256(args.out_depth),
            "model_mask_output": os.path.abspath(args.out_mask),
            "model_mask_output_sha256": _sha256(args.out_mask),
            "model_mask_pixels": int(model_mask.sum()),
            "finite_positive_pixels": int(finite_positive.sum()),
            "valid_pixels": int(valid_output.sum()),
            "hole_valid_pixels": int(hole_valid.sum()),
            "hole_coverage": float(hole_valid.sum() / max(int(hole_mask.sum()), 1)),
            "depth_min": float(range_values.min()) if range_values.size else None,
            "depth_median": float(np.median(range_values)) if range_values.size else None,
            "depth_max": float(range_values.max()) if range_values.size else None,
        },
    }
    with open(args.out_manifest, "w", encoding="utf-8") as handle:
        json.dump(manifest, handle, indent=2, ensure_ascii=False)
    print(
        "[LingBot] official inference complete: "
        f"shape={width}x{height} hole={int(hole_mask.sum())} "
        f"coverage={manifest['output']['hole_coverage']:.6f} "
        f"load={load_seconds:.2f}s infer={inference_seconds:.2f}s",
        flush=True,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
