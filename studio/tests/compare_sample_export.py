"""Compare a UI-exported bundled sample with the pre-existing manual export.

Run from the checkout root with run_cuda_python.cmd. This does not call any API
or model and does not alter the original sample.
"""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image

from studio.jobs import atomic_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    reference = Path(__file__).resolve().parents[2] / "assets"
    source = args.job / "input"
    a = json.loads((source / "point_cloud.camera.json").read_text(encoding="utf-8"))
    b = json.loads((reference / "point_cloud.camera.json").read_text(encoding="utf-8"))
    ka, kb = (np.asarray(c["intrinsic_K_opencv"]) for c in (a, b))
    ma, mb = (np.asarray(c["splats"][0]["model_view_opencv_for_original_splat"]) for c in (a, b))
    rgba = np.asarray(Image.open(source / "point_cloud.png").convert("RGBA"))
    expected = np.asarray(Image.open(reference / "point_cloud.png").convert("RGBA"))
    depth = np.load(source / "point_cloud.depth.npy", allow_pickle=False)
    expected_depth = np.load(reference / "point_cloud.depth.npy", allow_pickle=False)
    assert rgba.shape == expected.shape
    assert depth.shape == expected_depth.shape
    outside = (expected[..., 3] == 255) & (rgba[..., 3] == 255)
    delta = np.abs(rgba[..., :3].astype(float) - expected[..., :3].astype(float))
    valid = outside & np.isfinite(depth[..., 0]) & np.isfinite(expected_depth[..., 0]) & (expected_depth[..., 0] > 1e-3)
    relative = np.abs(depth[..., 0][valid] - expected_depth[..., 0][valid]) / expected_depth[..., 0][valid]
    report = {
        "image_shape": list(rgba.shape), "depth_shape": list(depth.shape),
        "K_max_abs_error": float(np.max(np.abs(ka - kb))),
        "model_view_max_abs_error": float(np.max(np.abs(ma - mb))),
        "rgb_outside_mean_abs_error": float(delta[outside].mean()),
        "rgb_outside_p95_abs_error": float(np.quantile(delta[outside], .95)),
        "alpha_mean_abs_error": float(np.abs(rgba[..., 3].astype(float) - expected[..., 3]).mean()),
        "depth_outside_median_relative_error": float(np.median(relative)),
        "depth_outside_p95_relative_error": float(np.quantile(relative, .95)),
        "valid_depth_pixels": int(valid.sum()),
    }
    atomic_json(args.job / "export_comparison.json", report)
    print(json.dumps(report, indent=2))
    assert report["K_max_abs_error"] < 1e-3
    assert report["model_view_max_abs_error"] < 1e-5


if __name__ == "__main__":
    main()
