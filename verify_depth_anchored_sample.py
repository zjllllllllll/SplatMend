"""Consolidate every acceptance gate for one current pipeline sample."""

from __future__ import annotations

import argparse
import json
from pathlib import Path


SHARP_CONTRACT = "sharp_hard_camera_z_surface_covariance_v2"
MERGE_CONTRACT = "depth_anchored_append_only_six_ring_blend_v3"
EXPECTED_SHARP_GAUSSIANS = 1_179_648


def load_report(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def ply_vertex_count(path: Path) -> int:
    with path.open("rb") as handle:
        for _ in range(200):
            line = handle.readline()
            if not line:
                break
            text = line.decode("ascii", errors="strict").strip()
            if text.startswith("element vertex "):
                return int(text.split()[-1])
            if text == "end_header":
                break
    raise RuntimeError(f"PLY vertex count not found: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--sample-id", type=int, required=True)
    parser.add_argument("--work-root", type=Path, required=True)
    args = parser.parse_args()

    report_paths = {
        "preparation": args.work_root / "prepared" / "preparation_report.json",
        "lingbot": args.work_root / "depth" / "lingbot_manifest.json",
        "depth": args.work_root / "depth" / "depth_acceptance.json",
        "sharp": args.work_root / "sharp_hard" / "sharp_hard_acceptance.json",
        "merge": args.work_root / "fusion" / "merge_acceptance.json",
    }
    reports = {name: load_report(path) for name, path in report_paths.items()}
    preparation = reports["preparation"]
    lingbot = reports["lingbot"]
    depth = reports["depth"]
    sharp = reports["sharp"]
    merge = reports["merge"]
    failures: list[str] = []

    if preparation.get("status") != "accepted":
        failures.append("input preparation rejected")
    if int(preparation.get("rgb_acceptance", {}).get("outside_changed_pixels", -1)) != 0:
        failures.append("RGB outside the authoritative hole changed")
    if preparation.get("depth_input_acceptance", {}).get("outside_unchanged") is not True:
        failures.append("observed depth outside the hole changed before completion")

    if lingbot.get("adapter") != "official_lingbot_depth_rgbd_v1":
        failures.append("unexpected LingBot adapter contract")
    lingbot_hole_coverage = float(lingbot.get("output", {}).get("hole_coverage", 0.0))
    if lingbot_hole_coverage < 0.999:
        failures.append("LingBot raw hole coverage is incomplete")

    if depth.get("status") != "accepted":
        failures.append("depth completion rejected")
    if depth.get("outside_exact") is not True:
        failures.append("completed depth changed outside the authoritative hole")
    authority_pixels = int(preparation.get("mask", {}).get("authority_pixels", -1))
    if int(depth.get("hole_valid_pixels", -2)) != authority_pixels:
        failures.append("completed depth does not cover the full authoritative hole")

    if sharp.get("status") != "accepted":
        failures.append("SHARP generation rejected")
    if sharp.get("contract") != SHARP_CONTRACT:
        failures.append("wrong SHARP geometry contract")
    if sharp.get("alignment_unet") != "bypassed":
        failures.append("SHARP Alignment UNet was not bypassed")
    if int(sharp.get("gaussians", -1)) != EXPECTED_SHARP_GAUSSIANS:
        failures.append("unexpected full SHARP Gaussian count")
    surface = sharp.get("center_diagnostics", {}).get("surface_covariance", {})
    if surface.get("mode") != "surface_jacobian" or surface.get("applied") is not True:
        failures.append("surface-Jacobian covariance correction was not applied")

    if merge.get("status") != "accepted":
        failures.append("patch merge rejected")
    if merge.get("contract") != MERGE_CONTRACT:
        failures.append("wrong six-ring append-only merge contract")

    selection = merge.get("selection", {})
    if int(selection.get("blend_reference_pixels", -1)) != 150:
        failures.append("blend width is not 150 reference pixels")
    if int(selection.get("blend_ring_count", -1)) != 6:
        failures.append("blend region does not contain six rings")
    if selection.get("core_opacity_mode") != "full":
        failures.append("authoritative hole core is not full-opacity weighted")
    weights = [
        float(value)
        for value in selection.get("blend_ring_opacity_weights_inner_to_outer", [])
    ]
    if len(weights) != 6 or not all(
        1.0 > weights[index] > weights[index + 1] > 0.0 for index in range(5)
    ):
        failures.append("six-ring opacity weights are missing or not strictly decreasing")
    elif weights[-1] >= 0.05:
        failures.append("outermost ring opacity is not close to zero")

    native_counts = selection.get("native_pixels_by_label", {})
    selected_counts = selection.get("selected_gaussians_by_label", {})
    if int(native_counts.get("core", -2)) != authority_pixels:
        failures.append("blend core differs from the authoritative hole mask")
    for ring_index in range(1, 7):
        if int(native_counts.get(f"ring_{ring_index}", 0)) <= 0:
            failures.append(f"native blend ring {ring_index} is empty")
        if int(selected_counts.get(f"ring_{ring_index}", 0)) <= 0:
            failures.append(f"SHARP blend ring {ring_index} is empty")
    selected_sum = int(selected_counts.get("core", 0)) + sum(
        int(selected_counts.get(f"ring_{ring_index}", 0))
        for ring_index in range(1, 7)
    )
    if selected_sum != int(selection.get("selected_layer0_gaussians", -1)):
        failures.append("per-region Gaussian counts do not sum to the patch size")

    geometry = merge.get("geometry", {})
    if geometry.get("depth_scale_or_offset_after_generation") is not False:
        failures.append("post-generation depth scale or offset was used")
    if geometry.get("posthoc_position_correction") is not False:
        failures.append("post-generation position correction was used")

    merge_info = merge.get("merge", {})
    if merge_info.get("base_preservation_mode") != "append_only_no_deletion":
        failures.append("base scene is not preserved in append-only mode")
    if int(merge_info.get("removed_base_gaussians", -1)) != 0:
        failures.append("base Gaussians were removed")
    if merge_info.get("original_prefix_exact") is not True:
        failures.append("original Gaussian prefix changed")
    if merge_info.get("retained_original_rows_exact") is not True:
        failures.append("retained original Gaussian rows changed")
    expected_final_count = int(merge_info.get("base_gaussians", 0)) + int(
        merge_info.get("new_gaussians", 0)
    )
    if int(merge_info.get("output_gaussians", -1)) != expected_final_count:
        failures.append("final count is not base count plus patch count")

    outputs = merge.get("outputs", {})
    required_outputs = {
        "final_ply": Path(outputs.get("merged_ply", "")),
        "patch_camera_ply": Path(outputs.get("patch_camera_ply", "")),
        "mask_ring_overlay": Path(outputs.get("mask_ring_overlay", "")),
        "blend_ring_labels": Path(outputs.get("blend_ring_labels", "")),
        "blend_ring_weights": Path(outputs.get("blend_ring_weights", "")),
    }
    for name, path in required_outputs.items():
        if not path.is_file():
            failures.append(f"required output is missing: {name}")
    final_ply = required_outputs["final_ply"]
    if final_ply.is_file() and ply_vertex_count(final_ply) != expected_final_count:
        failures.append("final PLY header count differs from the accepted merge count")
    sharp_ply = args.work_root / "sharp_hard" / "sharp_hard_layer0_camera.ply"
    if not sharp_ply.is_file() or ply_vertex_count(sharp_ply) != EXPECTED_SHARP_GAUSSIANS:
        failures.append("full SHARP PLY is missing or has the wrong vertex count")

    summary = {
        "status": "accepted" if not failures else "rejected",
        "sample_id": args.sample_id,
        "contract": "depth_anchored_surface_jacobian_six_ring_sample_v1",
        "key_metrics": {
            "hole_pixels": authority_pixels,
            "lingbot_hole_coverage": lingbot_hole_coverage,
            "completed_depth_hole_coverage": depth.get("hole_coverage"),
            "sharp_render_depth": sharp.get("render_depth_metrics_in_hole"),
            "sharp_render_alpha": sharp.get("render_alpha_quality_in_hole"),
            "patch_render_depth": merge.get("support_render", {}).get("patch_depth"),
            "base_gaussians": merge_info.get("base_gaussians"),
            "removed_base_gaussians": merge_info.get("removed_base_gaussians"),
            "new_gaussians": merge_info.get("new_gaussians"),
            "output_gaussians": merge_info.get("output_gaussians"),
            "blend_ring_weights": weights,
        },
        "final_ply": str(final_ply.resolve()) if final_ply.is_file() else str(final_ply),
        "reports": {name: str(path.resolve()) for name, path in report_paths.items()},
        "failures": failures,
    }
    summary_path = args.work_root / "sample_acceptance.json"
    summary_path.write_text(
        json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    if failures:
        raise RuntimeError("SAMPLE_REJECTED " + "; ".join(failures))
    print(
        f"SAMPLE_ACCEPTED id={args.sample_id:02d} "
        f"new={merge_info.get('new_gaussians')} final={final_ply}"
    )


if __name__ == "__main__":
    main()
