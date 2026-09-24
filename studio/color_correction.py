"""Match an AI repair to the original image across the authoritative hole edge."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import cv2
import numpy as np
from PIL import Image
from scipy.ndimage import distance_transform_edt
from scipy.sparse import coo_matrix
from scipy.sparse.linalg import splu

from prepare_depth_anchored_inputs import components_touching_seed, largest_component


def authority_from_alpha(alpha: np.ndarray) -> np.ndarray:
    core = largest_component(alpha <= 204)
    return components_touching_seed(alpha < 255, core) | core


def _solve_laplace(mask: np.ndarray, center: np.ndarray, boundary: np.ndarray) -> tuple[np.ndarray, dict]:
    """Solve the three color offsets together with fixed outer and zero center values."""
    unknown = mask & ~center
    index = np.full(mask.shape, -1, dtype=np.int32)
    count = int(unknown.sum())
    index[unknown] = np.arange(count)
    fixed = boundary.copy()
    fixed[center] = 0
    if count == 0:
        return fixed, {"unknowns": 0, "relative_solver_residual": 0.0}
    diagonal = np.zeros(count, dtype=np.float64)
    rhs = np.zeros((count, 3), dtype=np.float64)
    rows, cols, values = [], [], []
    for first, second in [
        (np.s_[:, :-1], np.s_[:, 1:]),
        (np.s_[:-1, :], np.s_[1:, :]),
    ]:
        i, j = index[first].ravel(), index[second].ravel()
        both = (i >= 0) & (j >= 0)
        for a, b, neighbor in [(i, j, fixed[second]), (j, i, fixed[first])]:
            active = a >= 0
            np.add.at(diagonal, a[active], 1.0)
            other_fixed = active & (b < 0)
            np.add.at(rhs, a[other_fixed], neighbor.reshape(-1, 3)[other_fixed])
        rows.extend([i[both], j[both]])
        cols.extend([j[both], i[both]])
        values.extend([-np.ones(int(both.sum())), -np.ones(int(both.sum()))])
    rows.append(np.arange(count))
    cols.append(np.arange(count))
    values.append(diagonal)
    matrix = coo_matrix((np.concatenate(values), (np.concatenate(rows), np.concatenate(cols))),
                        shape=(count, count)).tocsc()
    solution = splu(matrix).solve(rhs)
    residual = float(np.linalg.norm(matrix @ solution - rhs) / max(np.linalg.norm(rhs), 1e-12))
    if not np.all(np.isfinite(solution)) or not np.isfinite(residual) or residual > 1e-8:
        raise ValueError("The Laplace color-correction solve did not converge.")
    fixed[unknown] = solution.astype(np.float32)
    return fixed, {"unknowns": count, "relative_solver_residual": residual}


def _laplace_correction_field(authority: np.ndarray, boundary: np.ndarray, inside: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    h, w = authority.shape
    # Match the approved preview's low-frequency solve. Refine only when a tiny
    # hole or its zero-correction center would disappear on the reduced grid.
    for step in (4, 2, 1):
        size = (int(np.ceil(w / step)), int(np.ceil(h / step)))
        mask = cv2.resize(authority.astype(np.uint8), size, interpolation=cv2.INTER_NEAREST).astype(bool)
        distance = cv2.resize(inside.astype(np.float32), size, interpolation=cv2.INTER_LINEAR)
        center = mask & (distance >= 0.95 * inside.max())
        if np.any(center):
            break
    boundary_small = cv2.resize(boundary, size, interpolation=cv2.INTER_AREA)
    small_field, report = _solve_laplace(mask, center, boundary_small)
    field = cv2.resize(small_field, (w, h), interpolation=cv2.INTER_LINEAR)
    # Smooth only the color offset, retaining full-resolution image texture.
    field = cv2.GaussianBlur(field, (0, 0), 2)
    exact_center = authority & (inside >= 0.98 * inside.max())
    field[exact_center] = 0
    report.update({"solve_grid_scale": step, "equation": "laplacian(C) = 0",
                   "center_zero_distance_fraction": 0.95,
                   "exact_center_pixels": int(exact_center.sum())})
    return field, exact_center, report


def correct_rgb(
    source_rgba: np.ndarray,
    repaired_rgb: np.ndarray,
    *,
    radius: int = 150,
    reference_offset: int = 30,
    rings: int = 6,
    blur_sigma: float = 24.0,
) -> tuple[np.ndarray, dict]:
    """Interpolate reference-band color differences harmonically to a zero center."""
    if source_rgba.dtype != np.uint8 or repaired_rgb.dtype != np.uint8:
        raise ValueError("Color correction requires uint8 images.")
    if source_rgba.ndim != 3 or source_rgba.shape[2] != 4 or repaired_rgb.shape != source_rgba.shape[:2] + (3,):
        raise ValueError("Source RGBA and repaired RGB dimensions differ.")
    if radius < 1 or reference_offset < 0 or not 1 <= rings <= radius or blur_sigma <= 0:
        raise ValueError("Invalid reference width, offset, ring count, or blur width.")

    authority = authority_from_alpha(source_rgba[..., 3])
    outside_distance = distance_transform_edt(~authority)
    # Skip the near-hole fringe; keep the accepted preview's 150-pixel band width.
    reference = (~authority) & (outside_distance >= reference_offset) & (outside_distance <= reference_offset + radius)
    if not np.any(reference):
        raise ValueError("No original-image pixels surround the hole.")

    source_lab = cv2.cvtColor(source_rgba[..., :3].astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    repaired_lab = cv2.cvtColor(repaired_rgb.astype(np.float32) / 255.0, cv2.COLOR_RGB2LAB)
    # Use all six outside rings, with nearby original pixels carrying more weight.
    reference_weight = np.where(reference, np.exp(-outside_distance / (radius / rings * 2.0)), 0.0).astype(np.float32)
    denominator = cv2.GaussianBlur(reference_weight, (0, 0), blur_sigma, borderType=cv2.BORDER_REFLECT)
    delta = source_lab - repaired_lab
    local_delta = np.empty_like(delta)
    for channel in range(3):
        numerator = cv2.GaussianBlur(delta[..., channel] * reference_weight, (0, 0), blur_sigma, borderType=cv2.BORDER_REFLECT)
        local_delta[..., channel] = numerator / np.maximum(denominator, 1e-6)
    # Bound unusual API changes outside the edit area before applying them inside.
    local_delta[..., 0] = np.clip(local_delta[..., 0], -30.0, 30.0)
    local_delta[..., 1:] = np.clip(local_delta[..., 1:], -20.0, 20.0)

    inside_distance = distance_transform_edt(authority)
    center_distance = float(inside_distance[authority].max())
    correction, exact_center, solver_report = _laplace_correction_field(authority, local_delta, inside_distance)
    corrected_lab = repaired_lab + correction
    corrected_rgb = np.rint(np.clip(cv2.cvtColor(corrected_lab, cv2.COLOR_LAB2RGB), 0.0, 1.0) * 255.0).astype(np.uint8)
    corrected_rgb[exact_center] = repaired_rgb[exact_center]
    # The source render remains authoritative everywhere outside the hole.
    composite = source_rgba[..., :3].copy()
    composite[authority] = corrected_rgb[authority]

    boundary = authority & (inside_distance <= 3)
    boundary_reference = (~authority) & (outside_distance <= 3)
    boundary_difference_before = np.mean(np.abs(delta[boundary_reference])) if np.any(boundary_reference) else 0.0
    report = {
        "method": "local_lab_laplace_outward_v4",
        **solver_report,
        "radius_pixels": radius,
        "reference_inner_distance_pixels": reference_offset,
        "reference_outer_distance_pixels": reference_offset + radius,
        "center_distance_pixels": center_distance,
        "ring_count": rings,
        "ring_width_pixels": radius / rings,
        "blur_sigma_pixels": blur_sigma,
        "authority_pixels": int(authority.sum()),
        "reference_pixels": int(reference.sum()),
        "corrected_pixels": int(np.any(composite != repaired_rgb, axis=2)[authority].sum()),
        "outside_changed_pixels": int(np.any(composite != source_rgba[..., :3], axis=2)[~authority].sum()),
        "center_changed_pixels": int(np.any(composite != repaired_rgb, axis=2)[exact_center].sum()),
        "boundary_lab_delta_before": float(boundary_difference_before),
        "boundary_pixels": int(boundary.sum()),
    }
    return composite, report


def correct_file(source_path: Path, repaired_path: Path, output_path: Path, *, radius: int = 150, reference_offset: int = 30) -> Path:
    source = np.asarray(Image.open(source_path).convert("RGBA"), dtype=np.uint8)
    repaired = np.asarray(Image.open(repaired_path).convert("RGB"), dtype=np.uint8)
    corrected, report = correct_rgb(source, repaired, radius=radius, reference_offset=reference_offset)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray(corrected).save(output_path)
    report.update({
        "source_sha256": hashlib.sha256(source_path.read_bytes()).hexdigest(),
        "repaired_sha256": hashlib.sha256(repaired_path.read_bytes()).hexdigest(),
        "corrected_sha256": hashlib.sha256(output_path.read_bytes()).hexdigest(),
    })
    output_path.with_suffix(".json").write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding="utf-8")
    return output_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-rgba", type=Path, required=True)
    parser.add_argument("--repaired-rgb", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--radius", type=int, default=150)
    parser.add_argument("--reference-offset", type=int, default=30)
    args = parser.parse_args()
    output = correct_file(args.source_rgba, args.repaired_rgb, args.output, radius=args.radius, reference_offset=args.reference_offset)
    print(f"COLOR_CORRECTED {output}", flush=True)


if __name__ == "__main__":
    main()
