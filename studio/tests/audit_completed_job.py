"""Independent byte/array checks for a real completed Studio task (no API/model)."""
import argparse
import json
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData

from studio.jobs import BASELINE_FILES, atomic_json, digest


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("job", type=Path)
    args = parser.parse_args()
    folder = args.job.resolve()
    root = Path(__file__).resolve().parents[2]
    read_json = lambda path: json.loads(path.read_text(encoding="utf-8"))
    state = read_json(folder / "status.json")
    marker = read_json(folder / "studio_success.json")
    request = read_json(folder / "request.json")
    baseline = read_json(folder / "pipeline_baseline.json")
    output = folder / "pipeline"
    assert state["status"] == "succeeded"
    assert read_json(output / "sample_acceptance.json")["status"] == "accepted"
    assert baseline == {name: digest(root / name) for name in BASELINE_FILES}
    result_path = output / "fusion" / "depth_anchored_inpainted.ply"
    assert digest(result_path) == marker["result"]
    assert digest(output / "sample_acceptance.json") == marker["acceptance"]

    original = PlyData.read(root / "outputs" / "studio" / "scenes" / request["scene_id"] / "scene.ply")["vertex"].data
    deleted = np.fromfile(folder / "input" / "deleted.bin", dtype=np.uint8)
    base = PlyData.read(folder / "input" / "point_cloud.ply")["vertex"].data
    final = PlyData.read(result_path)["vertex"].data
    kept = np.flatnonzero(deleted == 0)
    assert len(kept) == len(base) and len(final) > len(base)
    assert original.dtype == base.dtype == final.dtype
    for start in range(0, len(base), 100_000):
        stop = min(start + 100_000, len(base))
        assert original[kept[start:stop]].tobytes() == base[start:stop].tobytes()
        assert base[start:stop].tobytes() == final[start:stop].tobytes()

    image = np.asarray(Image.open(folder / "input" / "point_cloud.png").convert("RGB"))
    rgb = np.asarray(Image.open(output / "prepared" / "rgb_completed_exact.png").convert("RGB"))
    hole = np.asarray(Image.open(output / "prepared" / "hole_authority.png")) > 0
    assert np.array_equal(image[~hole], rgb[~hole])
    source_depth = np.load(folder / "input" / "point_cloud.depth.npy", allow_pickle=False)[..., 0]
    final_depth = np.load(output / "depth" / "depth_completed_exact.npy", allow_pickle=False)
    # The pipeline treats unobserved invalid samples separately; compare all
    # finite positive observed samples and require complete valid hole depth.
    observed = ~hole & np.isfinite(source_depth) & (source_depth > 0)
    assert np.array_equal(source_depth[observed], final_depth[observed])
    assert np.all(np.isfinite(final_depth[hole]) & (final_depth[hole] > 0))

    report = {"status": "accepted", "job_id": state["id"], "model": state["model"],
              "source_gaussians": len(original), "deleted_gaussians": int(deleted.sum()),
              "retained_gaussians": len(base), "new_gaussians": len(final) - len(base),
              "final_gaussians": len(final), "retained_rows_byte_exact": True,
              "final_original_prefix_byte_exact": True, "rgb_outside_exact": True,
              "observed_depth_outside_exact": True, "hole_depth_complete": True,
              "original_pipeline_sources_unchanged": True, "result": marker["result"]}
    atomic_json(folder / "independent_audit.json", report)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
