"""Bounded, read-only explanations of original pipeline acceptance failures."""

import json
import math
from pathlib import Path


def depth_rejection_message(folder: Path) -> str | None:
    """Never expose arbitrary report strings, paths, credentials or URLs."""
    try:
        with (folder / "pipeline" / "depth" / "depth_acceptance.json").open("rb") as stream:
            raw = stream.read(65537)
        if len(raw) > 65536:
            return None
        report = json.loads(raw)
        if not isinstance(report, dict) or report.get("status") != "rejected":
            return None
        message = "Image generation succeeded, but original step 3 rejected the completed depth."
        surface = report.get("surface")
        if isinstance(surface, dict) and surface.get("mode") == "single_plane":
            value = surface.get("hole_to_ring_plane_residual_p95")
            limit = surface.get("effective_max_hole_to_ring_plane_p95")
            if (all(type(number) in (int, float) and math.isfinite(number)
                    and 0 <= number < 1e9 for number in (value, limit)) and value > limit):
                message += f" Hole plane p95 {value:.6f} exceeds {limit:.6f} (scene units)."
        return (message + " The API image is saved; no accepted 3D result was loaded."
                " Retrying unchanged inputs may fail again. See pipeline.log; original thresholds were not changed.")
    except (OSError, ValueError, TypeError, OverflowError, RecursionError):
        return None
