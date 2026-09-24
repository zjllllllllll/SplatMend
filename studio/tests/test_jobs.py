import io
import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from studio.jobs import BASELINE_FILES, JobManager, atomic_json, digest, load_ply, validate_camera, validate_viewer_camera


def camera():
    return {"image": {"width": 256, "height": 256}, "camera": {"projection_type": "perspective"},
            "intrinsic_K_opencv": [[128, 0, 128], [0, 128, 128], [0, 0, 1]],
            "splats": [{"model_view_opencv_for_original_splat": np.eye(4).tolist()}]}


class JobTests(unittest.TestCase):
    def setUp(self):
        output = Path(__file__).resolve().parents[2] / "outputs"
        output.mkdir(exist_ok=True)
        self.temporary = tempfile.TemporaryDirectory(prefix="studio_test_", dir=output)
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.manager = JobManager(self.root)
        self.key = patch.dict("os.environ", {"GRSAI_API_KEY": "TEST_SECRET_DO_NOT_PERSIST"})
        self.key.start()
        self.addCleanup(self.key.stop)
        ready = patch("studio.jobs.model_preflight")
        ready.start()
        self.addCleanup(ready.stop)
        for name in BASELINE_FILES:
            (self.root / name).write_text("unchanged baseline", encoding="utf-8")
        self.scene_id = "a" * 32
        directory = self.manager.scenes / self.scene_id
        directory.mkdir()
        names = ["x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2", "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"]
        self.vertex = np.zeros(3, dtype=[(name, "f4") for name in names])
        self.vertex["x"] = [1, 2, 3]
        self.vertex["z"] = 2
        self.vertex["rot_0"] = 1
        PlyData([PlyElement.describe(self.vertex, "vertex")], text=False).write(directory / "upload.ply")
        self.manager.register_scene(directory, "sample.ply")

    def new_job(self):
        with patch.object(self.manager, "preflight", return_value={"missing_files": [], "disk_free_gb": 100}):
            job = self.manager.create({"scene_id": self.scene_id, "model": "gpt-image-2", "prompt": "修复孔洞", "camera": camera()})
        folder = self.manager.jobs / job["id"]
        image = np.full((256, 256, 4), 255, dtype=np.uint8)
        image[120:136, 120:136, 3] = 0
        Image.fromarray(image).save(folder / "input" / "point_cloud.png")
        np.save(folder / "input" / "point_cloud.depth.npy", np.ones((256, 256, 2), dtype=np.float32))
        np.array([0, 1, 0], dtype=np.uint8).tofile(folder / "input" / "deleted.bin")
        return job["id"], folder

    def test_registered_inputs_keep_base_rows_exact(self):
        job_id, folder = self.new_job()
        request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
        self.manager._validate_inputs(job_id, request)
        base = load_ply(folder / "input" / "point_cloud.ply")
        np.testing.assert_array_equal(base, self.vertex[[0, 2]])
        self.assertEqual(self.manager.state(job_id)["deleted_count"], 1)

    def test_retired_models_cannot_submit_or_retry_but_history_is_preserved(self):
        from studio.image_api import ImageAPIError, model_preflight
        job_id, folder = self.new_job()
        self.manager.update(job_id, model="retired-image-model")
        with patch("studio.jobs.model_preflight", side_effect=model_preflight), patch("studio.jobs.threading.Thread") as thread:
            with self.assertRaises(ImageAPIError):
                self.manager.submit(job_id)
            self.manager.update(job_id, status="failed", image_ready=True, can_retry_pipeline=True, can_retry_download=True)
            with self.assertRaises(ValueError):
                self.manager.retry_pipeline(job_id)
            with self.assertRaises(ValueError):
                self.manager.retry_download(job_id)
            thread.assert_not_called()
        self.assertFalse(self.manager.state(job_id)["can_retry_pipeline"])
        self.assertFalse(self.manager.state(job_id)["can_retry_download"])
        self.assertTrue((folder / "input" / "point_cloud.png").is_file())

    def test_failure_records_exact_phase_for_progress_without_launching_pipeline(self):
        from studio.image_api import ImageAPIError
        for failed_phase in ("validating", "image_api"):
            job_id, _ = self.new_job()
            self.manager.update(job_id, status="validating")
            target = "studio.jobs.repair_image" if failed_phase == "image_api" else "studio.jobs.JobManager._validate_inputs"
            with patch(target, side_effect=ImageAPIError("mock failure")), patch("studio.jobs.subprocess.Popen") as process:
                self.manager._run(job_id)
                process.assert_not_called()
            state = self.manager.state(job_id)
            self.assertEqual(state["status"], "failed")
            self.assertEqual(state["failed_phase"], failed_phase)

    def test_camera_contract(self):
        self.assertEqual(validate_camera(camera()), (256, 256))
        bad = camera()
        bad["intrinsic_K_opencv"][1][1] = 129
        with self.assertRaises(ValueError):
            validate_camera(bad)

    def test_supersplat_camera_roundtrip_uses_string_tonemapping(self):
        view = {"focalPoint": [1, 2, 3], "azim": 30, "elev": -10, "distance": .25, "fov": 90.5, "tonemapping": "linear"}
        self.assertEqual(validate_viewer_camera(view), view)
        for field, value in [("fov", float("nan")), ("distance", -1), ("tonemapping", "invalid")]:
            with self.subTest(field=field), self.assertRaises(ValueError):
                validate_viewer_camera({**view, field: value})
        bad = camera()
        bad["splats"][0]["model_view_opencv_for_original_splat"][0][0] = 2
        with self.assertRaises(ValueError):
            validate_camera(bad)

    def test_invalid_export_fails_before_paid_api(self):
        job_id, folder = self.new_job()
        np.save(folder / "input" / "point_cloud.depth.npy", np.ones((2, 2), dtype=np.float32))
        with patch("studio.jobs.repair_image") as repair:
            self.manager._run(job_id)
        repair.assert_not_called()
        self.assertEqual(self.manager.state(job_id)["status"], "failed")

    def test_missing_key_creates_no_job(self):
        before = list(self.manager.jobs.iterdir())
        with patch.dict("os.environ", {"GRSAI_API_KEY": ""}), patch.object(self.manager, "preflight", return_value={"missing_files": [], "disk_free_gb": 100}):
            with self.assertRaises(RuntimeError):
                self.manager.create({"scene_id": self.scene_id, "model": "gpt-image-2", "prompt": "repair", "camera": camera()})
        self.assertEqual(before, list(self.manager.jobs.iterdir()))

    def test_missing_oss_creates_no_job(self):
        from studio.image_api import ImageAPIError
        before = list(self.manager.jobs.iterdir())
        with patch("studio.jobs.model_preflight", side_effect=ImageAPIError("provider not configured")):
            with self.assertRaisesRegex(ImageAPIError, "provider"):
                self.new_job()
        self.assertEqual(before, list(self.manager.jobs.iterdir()))

    def test_expired_download_retry_never_generates(self):
        job_id, _ = self.new_job()
        self.manager.image_results[job_id] = {"result": {"data": []}, "received_at": 0}
        with patch("studio.jobs.repair_image") as repair:
            self.manager._run(job_id, download_only=True)
        repair.assert_not_called()
        self.assertEqual(self.manager.state(job_id)["status"], "failed")
        self.assertFalse(self.manager.state(job_id)["can_retry_download"])

    def test_interrupted_image_api_never_reposts_on_service_restart(self):
        job_id, _ = self.new_job()
        self.manager.update(job_id, status="image_api")
        with patch("studio.jobs.matching_job_processes", return_value=([], False)), \
             patch("studio.jobs.repair_image") as repair, patch("studio.jobs.subprocess.Popen") as spawn:
            self.manager.recover_interrupted()
        self.assertEqual(self.manager.state(job_id)["status"], "failed")
        self.assertIn("may have been billed", self.manager.state(job_id)["message"])
        self.assertIsNone(self.manager.active)
        repair.assert_not_called()
        spawn.assert_not_called()

    def test_recovery_waits_for_live_or_unobservable_process(self):
        for alive in (True, None):
            job_id, _ = self.new_job()
            self.manager.update(job_id, status="pipeline", pipeline_process={"pid": 123, "created": 1})
            with patch("studio.jobs.process_alive", return_value=alive), patch("studio.jobs.threading.Thread"):
                self.manager.recover_interrupted()
            self.assertEqual(self.manager.active, job_id)
            self.assertEqual(self.manager.state(job_id)["status"], "recovering")
            # An observation failure must not be interpreted as process death.
            self.manager.update(job_id, status="failed")
            self.manager.active = None
            self.manager.idle.set()

    def test_only_original_launcher_and_verified_result(self):
        job_id, folder = self.new_job()
        output = folder / "pipeline"
        (output / "fusion").mkdir(parents=True)
        PlyData([PlyElement.describe(self.vertex, "vertex")], text=False).write(output / "fusion" / "depth_anchored_inpainted.ply")
        atomic_json(output / "sample_acceptance.json", {"status": "accepted"})
        process = Mock(stdout=io.StringIO("[1/6] Prepare registered inputs.\n[6/6] Verify.\n"))
        process.wait.return_value = 0
        with patch("studio.jobs.repair_image", return_value=folder / "repaired_rgb.png"), patch("studio.jobs.subprocess.Popen", return_value=process) as spawn:
            self.manager._run(job_id)
        command = spawn.call_args.args[0]
        self.assertEqual(command, [str(self.root / "run_sample.cmd"), str(int(job_id, 16)),
                                   str(folder / "input"), str(folder / "repaired_rgb.png"), str(output)])
        self.assertEqual(self.manager.state(job_id)["status"], "succeeded")
        self.assertTrue(self.manager.result(job_id).is_file())
        for path in folder.rglob("*.json"):
            self.assertNotIn("TEST_SECRET", path.read_text(encoding="utf-8"))
        self.assertNotIn("TEST_SECRET", str(command))
        (output / "sample_acceptance.json").write_text('{"status":"changed"}')
        with self.assertRaisesRegex(ValueError, "hashes"):
            self.manager.result(job_id)

    def test_pipeline_retry_preserves_failure_and_never_calls_image_api(self):
        job_id, folder = self.new_job()
        appearance = folder / "appearance"
        appearance.mkdir()
        Image.new("RGB", (256, 256)).save(appearance / "repaired_rgb.png")
        atomic_json(appearance / "image_manifest.json", {
            "model": "gpt-image-2", "prompt": "修复孔洞",
            "source_sha256": digest(folder / "input" / "point_cloud.png")["sha256"],
            "repaired_sha256": digest(appearance / "repaired_rgb.png")["sha256"]})
        output = folder / "pipeline"
        output.mkdir()
        (output / "failed.txt").write_text("retained diagnostic")
        (folder / "pipeline.log").write_text("old failed log")
        process = Mock(stdout=io.StringIO("[6/6] Verify\n"))
        process.wait.return_value = 0

        def launch(*args, **kwargs):
            (output / "fusion").mkdir(parents=True)
            PlyData([PlyElement.describe(self.vertex, "vertex")], text=False).write(output / "fusion" / "depth_anchored_inpainted.ply")
            atomic_json(output / "sample_acceptance.json", {"status": "accepted"})
            self.assertTrue(args[0][1].isdigit())
            return process

        with patch("studio.jobs.repair_image") as repair, patch("studio.jobs.subprocess.Popen", side_effect=launch):
            self.manager._run(job_id, pipeline_only=True)
        repair.assert_not_called()
        self.assertEqual(self.manager.state(job_id)["status"], "succeeded")
        self.assertEqual(len(list((folder / "failed_attempts").glob("*/pipeline/failed.txt"))), 1)
        self.assertEqual(len(list((folder / "failed_attempts").glob("*/pipeline.log"))), 1)

    def test_modified_locked_input_cannot_be_retried(self):
        job_id, folder = self.new_job()
        atomic_json(folder / "locked_input_manifest.json", {"invalid": True})
        with patch("studio.jobs.repair_image") as repair, patch("studio.jobs.subprocess.Popen") as spawn:
            self.manager._run(job_id)
        repair.assert_not_called()
        spawn.assert_not_called()
        self.assertEqual(self.manager.state(job_id)["status"], "failed")

    def test_partial_output_never_loads(self):
        for exit_code, accepted in [(1, True), (0, False)]:
            job_id, folder = self.new_job()
            output = folder / "pipeline"
            (output / "fusion").mkdir(parents=True)
            (output / "fusion" / "depth_anchored_inpainted.ply").write_bytes(b"partial")
            if accepted:
                atomic_json(output / "sample_acceptance.json", {"status": "accepted"})
            process = Mock(stdout=io.StringIO("[5/6] writing PLY\n"))
            process.wait.return_value = exit_code
            with patch("studio.jobs.repair_image", return_value=folder / "repaired.png"), patch("studio.jobs.subprocess.Popen", return_value=process):
                self.manager._run(job_id)
            self.assertEqual(self.manager.state(job_id)["status"], "failed")
            self.assertFalse((folder / "studio_success.json").exists())
            with self.assertRaises(ValueError):
                self.manager.result(job_id)

    def test_reading_depth_rejection_explains_saved_image_without_retry(self):
        job_id, folder = self.new_job()
        depth = folder / "pipeline" / "depth"
        depth.mkdir(parents=True)
        atomic_json(depth / "depth_acceptance.json", {"status": "rejected"})
        self.manager.update(job_id, status="failed", stage=3, image_ready=True,
                            message="Existing pipeline failed (exit 1). See pipeline.log.")
        before = (folder / "status.json").read_bytes()
        with patch("studio.jobs.repair_image") as repair, patch("studio.jobs.subprocess.Popen") as spawn:
            state = self.manager.state(job_id)
        self.assertIn("Image generation succeeded", state["message"])
        self.assertEqual(state["status"], "failed")
        self.assertEqual(before, (folder / "status.json").read_bytes())
        repair.assert_not_called()
        spawn.assert_not_called()
        self.manager.update(job_id, status="succeeded", message="Accepted.")
        self.assertEqual(self.manager.state(job_id)["message"], "Accepted.")


if __name__ == "__main__":
    unittest.main()
