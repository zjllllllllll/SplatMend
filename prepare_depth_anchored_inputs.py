"""Prepare deterministic RGB-D inputs for depth-anchored SHARP completion.

The source render is authoritative outside the expanded repair region.  The
repaired RGB and missing depth share the alpha hole plus its blend band, before
LingBot or SHARP consumes either image.  This script writes the
pixel-registered inputs consumed by LingBot and a machine-readable acceptance
report; it intentionally performs no learned inference.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rgba", type=Path, required=True)
    parser.add_argument("--repaired-rgb", type=Path, required=True)
    parser.add_argument("--source-depth", type=Path, required=True)
    parser.add_argument("--camera-json", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--min-transparency", type=float, default=0.20)
    parser.add_argument("--context-radius", type=int, default=48)
    return parser.parse_args()


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while block := handle.read(1024 * 1024):
            digest.update(block)
    return digest.hexdigest()


def largest_component(mask: np.ndarray) -> np.ndarray:
    count, labels, stats, _ = cv2.connectedComponentsWithStats(
        mask.astype(np.uint8), connectivity=8
    )
    if count <= 1:
        raise RuntimeError("alpha threshold produced no connected hole component")
    component = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
    return labels == component


def components_touching_seed(mask: np.ndarray, seed: np.ndarray) -> np.ndarray:
    count, labels = cv2.connectedComponents(mask.astype(np.uint8), connectivity=8)
    if count <= 1:
        return seed.copy()
    touched = np.unique(labels[seed])
    touched = touched[touched != 0]
    if touched.size == 0:
        raise RuntimeError("soft alpha region does not touch the authoritative hole")
    return np.isin(labels, touched)


MASK_EXPAND_REFERENCE_PIXELS = 150
MASK_REFERENCE_SHORT_SIDE = 1440


def scaled_reference_pixels(
    reference_pixels: int, width: int, height: int, reference_short_side: int
) -> int:
    if reference_pixels <= 0:
        return 0
    return max(
        1,
        int(round(reference_pixels * min(width, height) / reference_short_side)),
    )


def repair_masks_from_alpha(
    alpha: np.ndarray, min_transparency: float = 0.20
) -> tuple[np.ndarray, np.ndarray, np.ndarray, int]:
    """Share the original hole and its resolution-scaled blend domain."""
    alpha_limit = int(round(255.0 * (1.0 - min_transparency)))
    core = largest_component(alpha <= alpha_limit)
    alpha_hole = components_touching_seed(alpha < 255, core) | core
    height, width = alpha.shape
    radius = scaled_reference_pixels(
        MASK_EXPAND_REFERENCE_PIXELS, width, height, MASK_REFERENCE_SHORT_SIDE
    )
    distance = cv2.distanceTransform(
        (~alpha_hole).astype(np.uint8), cv2.DIST_L2, cv2.DIST_MASK_PRECISE
    )
    authority = alpha_hole | (distance <= float(radius))
    return core, alpha_hole, authority, radius


def depth_preview(depth: np.ndarray, valid: np.ndarray) -> np.ndarray:
    values = depth[valid]
    if values.size == 0:
        raise RuntimeError("source depth has no valid pixels")
    lo, hi = (float(x) for x in np.quantile(values, (0.01, 0.99)))
    safe = np.where(valid, depth, lo)
    scaled = np.clip((safe - lo) / max(hi - lo, 1.0e-8), 0.0, 1.0)
    preview = cv2.applyColorMap(
        np.rint(scaled * 255.0).astype(np.uint8), cv2.COLORMAP_TURBO
    )
    preview[~valid] = 0
    return cv2.cvtColor(preview, cv2.COLOR_BGR2RGB)


def save_image(path: Path, array: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(array).save(path)


def main() -> None:
    args = parse_args()
    for path in (
        args.source_rgba,
        args.repaired_rgb,
        args.source_depth,
        args.camera_json,
    ):
        if not path.is_file():
            raise FileNotFoundError(path)
    if not 0.0 < args.min_transparency < 1.0:
        raise ValueError("--min-transparency must be in (0, 1)")
    if args.context_radius < 1:
        raise ValueError("--context-radius must be positive")

    source = np.asarray(Image.open(args.source_rgba).convert("RGBA"), dtype=np.uint8)
    repaired = np.asarray(Image.open(args.repaired_rgb).convert("RGB"), dtype=np.uint8)
    if repaired.shape != source.shape[:2] + (3,):
        raise ValueError(
            f"RGB shape mismatch: source={source.shape} repaired={repaired.shape}"
        )
    height, width = source.shape[:2]
    source_rgb = source[..., :3]
    alpha_u8 = source[..., 3]

    depth_payload = np.load(args.source_depth)
    if depth_payload.shape == (height, width, 2):
        source_depth = np.asarray(depth_payload[..., 0], dtype=np.float32)
        source_depth_alpha = np.asarray(depth_payload[..., 1], dtype=np.float32)
    elif depth_payload.shape == (height, width):
        source_depth = np.asarray(depth_payload, dtype=np.float32)
        source_depth_alpha = np.isfinite(source_depth).astype(np.float32)
    else:
        raise ValueError(
            f"depth shape {depth_payload.shape} does not match image {(height, width)}"
        )

    camera = json.loads(args.camera_json.read_text(encoding="utf-8"))
    intrinsic = np.asarray(camera["intrinsic_K_opencv"], dtype=np.float32)
    if intrinsic.shape != (3, 3) or not np.isfinite(intrinsic).all():
        raise ValueError("camera intrinsic_K_opencv must be a finite 3x3 matrix")
    declared = camera.get("image", {})
    if (declared.get("height"), declared.get("width")) != (height, width):
        raise ValueError(
            "camera/image size mismatch: "
            f"camera={(declared.get('height'), declared.get('width'))} "
            f"image={(height, width)}"
        )

    alpha_limit = int(round(255.0 * (1.0 - args.min_transparency)))
    core_seed, alpha_hole, authority, expand_radius = repair_masks_from_alpha(
        alpha_u8, args.min_transparency
    )
    if not np.any(authority):
        raise RuntimeError("empty edit authority")

    kernel_size = 2 * args.context_radius + 1
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    context = cv2.dilate(authority.astype(np.uint8), kernel, iterations=1).astype(bool)
    context_ring = context & ~authority

    # Fully opaque pixels beside the alpha hole can still contain damaged color
    # and depth. Replace the complete expanded domain before either model runs;
    # the context ring above starts outside that same domain.
    rgb_exact = source_rgb.copy()
    rgb_exact[authority] = repaired[authority]
    rgb_exact[~authority] = source_rgb[~authority]

    outside_delta = np.abs(
        rgb_exact[~authority].astype(np.int16) - source_rgb[~authority].astype(np.int16)
    )
    if outside_delta.size and int(outside_delta.max()) != 0:
        raise RuntimeError("RGB outside-authority invariant failed")

    boundary_inside = authority & ~cv2.erode(
        authority.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1
    ).astype(bool)
    boundary_outside = (
        cv2.dilate(
            authority.astype(np.uint8), np.ones((3, 3), dtype=np.uint8), iterations=1
        ).astype(bool)
        & ~authority
    )
    repaired_source_difference = np.abs(
        repaired.astype(np.int16) - source_rgb.astype(np.int16)
    )

    source_valid = np.isfinite(source_depth) & (source_depth > 1.0e-3)
    depth_input = np.where(source_valid, source_depth, 0.0).astype(np.float32)
    depth_before_hole_zero = depth_input.copy()
    depth_input[authority] = 0.0
    if np.any(depth_input[authority] != 0.0):
        raise RuntimeError("depth input is nonzero inside the hole")
    if not np.array_equal(
        depth_input[~authority], depth_before_hole_zero[~authority], equal_nan=True
    ):
        raise RuntimeError("depth input changed outside the hole")

    output = args.output_dir
    output.mkdir(parents=True, exist_ok=True)
    paths = {
        "rgb": output / "rgb_completed_exact.png",
        "hole": output / "hole_authority.png",
        "alpha_hole": output / "hole_alpha.png",
        "core": output / "hole_core.png",
        "context": output / "context_ring.png",
        "depth_input": output / "depth_observed_with_hole.npy",
        "intrinsics": output / "intrinsics.txt",
        "preview": output / "preparation_preview.png",
        "report": output / "preparation_report.json",
    }
    save_image(paths["rgb"], rgb_exact)
    save_image(paths["hole"], authority.astype(np.uint8) * 255)
    save_image(paths["alpha_hole"], alpha_hole.astype(np.uint8) * 255)
    save_image(paths["core"], core_seed.astype(np.uint8) * 255)
    save_image(paths["context"], context_ring.astype(np.uint8) * 255)
    np.save(paths["depth_input"], depth_input)
    np.savetxt(paths["intrinsics"], intrinsic, fmt="%.9g")

    overlay = rgb_exact.copy().astype(np.float32)
    overlay[authority] = 0.62 * overlay[authority] + 0.38 * np.array(
        [255.0, 0.0, 255.0], dtype=np.float32
    )
    overlay[context_ring] = 0.72 * overlay[context_ring] + 0.28 * np.array(
        [0.0, 210.0, 255.0], dtype=np.float32
    )
    depth_rgb = depth_preview(source_depth, source_valid)
    preview = np.concatenate(
        [source_rgb, repaired, rgb_exact, np.clip(overlay, 0, 255).astype(np.uint8), depth_rgb],
        axis=1,
    )
    save_image(paths["preview"], preview)

    inside_delta = np.abs(
        rgb_exact[authority].astype(np.int16) - source_rgb[authority].astype(np.int16)
    )
    ys, xs = np.nonzero(authority)
    report = {
        "status": "accepted",
        "contract": "depth_anchored_input_v1",
        "inputs": {
            "source_rgba": str(args.source_rgba.resolve()),
            "source_rgba_sha256": sha256(args.source_rgba),
            "repaired_rgb": str(args.repaired_rgb.resolve()),
            "repaired_rgb_sha256": sha256(args.repaired_rgb),
            "source_depth": str(args.source_depth.resolve()),
            "source_depth_sha256": sha256(args.source_depth),
            "camera_json": str(args.camera_json.resolve()),
            "camera_json_sha256": sha256(args.camera_json),
        },
        "image": {"width": width, "height": height},
        "mask": {
            "alpha_limit": alpha_limit,
            "core_pixels": int(core_seed.sum()),
            "alpha_hole_pixels": int(alpha_hole.sum()),
            "authority_pixels": int(authority.sum()),
            "expand_reference_pixels": MASK_EXPAND_REFERENCE_PIXELS,
            "expand_reference_short_side": MASK_REFERENCE_SHORT_SIDE,
            "expand_radius_pixels": expand_radius,
            "context_ring_pixels": int(context_ring.sum()),
            "bbox_xyxy": [int(xs.min()), int(ys.min()), int(xs.max()), int(ys.max())],
        },
        "rgb_acceptance": {
            "outside_changed_pixels": int(
                np.any(rgb_exact != source_rgb, axis=2)[~authority].sum()
            ),
            "outside_max_abs_channel_error": int(outside_delta.max())
            if outside_delta.size
            else 0,
            "inside_changed_pixels": int(
                np.any(rgb_exact != source_rgb, axis=2)[authority].sum()
            ),
            "inside_mean_abs_channel_change": float(inside_delta.mean()),
            "boundary_inside_mean_abs_repaired_source_difference": float(
                repaired_source_difference[boundary_inside].mean()
            ),
            "boundary_outside_mean_abs_repaired_source_difference": float(
                repaired_source_difference[boundary_outside].mean()
            ),
            "boundary_outside_p95_abs_repaired_source_difference": float(
                np.quantile(repaired_source_difference[boundary_outside], 0.95)
            ),
            "outside_exact": True,
        },
        "depth_input_acceptance": {
            "source_valid_pixels": int(source_valid.sum()),
            "hole_pixels_zero": int((depth_input[authority] == 0.0).sum()),
            "hole_all_zero": True,
            "outside_unchanged": True,
            "source_depth_alpha_min": float(np.nanmin(source_depth_alpha)),
            "source_depth_alpha_max": float(np.nanmax(source_depth_alpha)),
        },
        "outputs": {name: str(path.resolve()) for name, path in paths.items()},
    }
    paths["report"].write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    print(
        "PREPARE_ACCEPTED",
        f"image={width}x{height}",
        f"core={int(core_seed.sum())}",
        f"authority={int(authority.sum())}",
        "rgb_outside_changed=0",
        "depth_outside_changed=0",
        "depth_hole_nonzero=0",
        f"report={paths['report']}",
        flush=True,
    )


if __name__ == "__main__":
    main()
