"""Local files and background scheduling around the UNMODIFIED six-step launcher."""

from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import subprocess
import threading
import time
import uuid
from pathlib import Path

import numpy as np
from PIL import Image
from plyfile import PlyData, PlyElement

from studio.image_api import ImageAPIError, ImageDownloadError, MODELS, decode_image, model_size, model_preflight, read_key, repair_image, validate_prompt
from studio.color_correction import correct_file
from studio.diagnostics import depth_rejection_message
from studio.lifecycle import matching_job_processes, process_alive, process_identity

REQUIRED_FIELDS = {"x", "y", "z", "opacity", "f_dc_0", "f_dc_1", "f_dc_2",
                   "scale_0", "scale_1", "scale_2", "rot_0", "rot_1", "rot_2", "rot_3"}
BASELINE_FILES = (
    "run_sample.cmd", "run_cuda_python.cmd", "prepare_depth_anchored_inputs.py",
    "run_lingbot_depth.py", "fuse_and_validate_depth.py", "run_sharp_hard_depth.py",
    "merge_hard_patch.py", "verify_depth_anchored_sample.py", "gaussian_patch_io.py", "sharp_runtime.py",
    "studio/color_correction.py",
)
TERMINAL = {"succeeded", "failed"}


def digest(path: Path) -> dict:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        while block := stream.read(1024 * 1024):
            h.update(block)
    return {"bytes": path.stat().st_size, "sha256": h.hexdigest()}


def atomic_json(path: Path, data: dict) -> None:
    temporary = path.with_suffix(".json.tmp")
    temporary.write_text(json.dumps(data, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")
    temporary.replace(path)


def identifier(value: object) -> str:
    if not isinstance(value, str) or not re.fullmatch(r"[a-f0-9]{32}", value):
        raise ValueError("Invalid local identifier.")
    return value


def load_ply(path: Path):
    # PlyData does not execute content; reject lists and non-Gaussian schemas.
    data = PlyData.read(path)
    if len(data.elements) != 1 or data.elements[0].name != "vertex":
        raise ValueError("Use an uncompressed Gaussian PLY with one vertex element.")
    vertex = data["vertex"].data
    if not 0 < len(vertex) <= 10_000_000:
        raise ValueError("PLY must contain between 1 and 10 million Gaussians.")
    if not REQUIRED_FIELDS.issubset(vertex.dtype.names or ()):
        raise ValueError("PLY is missing standard Gaussian position, color, opacity, scale or quaternion fields.")
    for name in vertex.dtype.names:
        if vertex[name].dtype.kind not in "fiu" or not np.isfinite(vertex[name]).all():
            raise ValueError("PLY contains non-numeric, non-finite, or list properties.")
    return vertex


def link_or_copy(source: Path, destination: Path):
    try:
        os.link(source, destination)
    except OSError:
        shutil.copyfile(source, destination)


def validate_camera(camera: object) -> tuple[int, int]:
    if not isinstance(camera, dict):
        raise ValueError("Missing camera metadata.")
    width, height = camera.get("image", {}).get("width"), camera.get("image", {}).get("height")
    if (type(width) is not int or type(height) is not int or min(width, height) < 256
            or max(width, height) > 4096 or width * height > 8_500_000):
        raise ValueError("Camera render size must be 256–4096 per side and at most 8.5M pixels.")
    if camera.get("camera", {}).get("projection_type") != "perspective":
        raise ValueError("The existing pipeline requires a perspective camera.")
    k = np.asarray(camera.get("intrinsic_K_opencv"), dtype=np.float64)
    if k.shape != (3, 3) or not np.isfinite(k).all():
        raise ValueError("Camera intrinsics must be a finite 3x3 matrix.")
    expected = np.array([[k[0, 0], 0, width / 2], [0, k[0, 0], height / 2], [0, 0, 1]])
    if k[0, 0] <= 0 or not np.allclose(k, expected, atol=1e-3, rtol=1e-5):
        raise ValueError("The unchanged pipeline requires square pixels, zero skew and a centered principal point.")
    splats = camera.get("splats")
    if not isinstance(splats, list) or len(splats) != 1:
        raise ValueError("Repair one Gaussian scene at a time.")
    matrix = np.asarray(splats[0].get("model_view_opencv_for_original_splat"), dtype=np.float64)
    if matrix.shape != (4, 4) or not np.isfinite(matrix).all():
        raise ValueError("Missing registered original-PLY camera transform.")
    r = matrix[:3, :3]
    if (not np.allclose(matrix[3], [0, 0, 0, 1], atol=1e-6)
            or not np.allclose(r @ r.T, np.eye(3), atol=1e-3)
            or abs(np.linalg.det(r) - 1) > 1e-3
            or splats[0].get("has_per_splat_palette_transforms", False)):
        raise ValueError("Camera transform must be rigid and match the original PLY.")
    return width, height


def validate_viewer_camera(value: object) -> dict | None:
    if value is None:
        return None
    if not isinstance(value, dict):
        raise ValueError("Invalid saved viewer camera.")
    focal = value.get("focalPoint")
    fields = ("azim", "elev", "distance", "fov")
    if not isinstance(focal, list) or len(focal) != 3:
        raise ValueError("Invalid saved viewer camera focus.")
    numbers = focal + [value.get(field) for field in fields]
    if any(type(number) not in (float, int) or not np.isfinite(number) for number in numbers):
        raise ValueError("Saved viewer camera must contain finite numbers.")
    if not 0 < value["fov"] < 180 or value["distance"] <= 0:
        raise ValueError("Invalid saved viewer camera range.")
    if value.get("tonemapping") not in ("linear", "neutral", "aces", "aces2", "filmic", "hejl"):
        raise ValueError("Invalid saved viewer tone mapping.")
    return {"focalPoint": focal, "tonemapping": value["tonemapping"], **{field: value[field] for field in fields}}


class JobManager:
    def __init__(self, root: Path):
        self.root = root.resolve()
        self.storage = self.root / "outputs" / "studio"
        self.scenes = self.storage / "scenes"
        self.jobs = self.storage / "jobs"
        self.scenes.mkdir(parents=True, exist_ok=True)
        self.jobs.mkdir(parents=True, exist_ok=True)
        self.lock = threading.RLock()
        self.active: str | None = None
        self.process: subprocess.Popen | None = None
        self.image_results: dict[str, dict] = {}
        self.owner = process_identity(os.getpid())
        self.stop_recovery = threading.Event()
        self.idle = threading.Event()
        self.idle.set()

    def recover_interrupted(self):
        """Called only after the new server exclusively owns this checkout."""
        pending = []
        for folder in self.jobs.iterdir():
            if not (folder / "status.json").is_file():
                continue
            state = self.state(folder.name)
            if state["status"] in TERMINAL or state["status"] == "uploading":
                continue
            if self._previous_process_pending(folder.name, state):
                pending.append(folder.name)
                self.update(folder.name, status="recovering", can_retry_download=False,
                            message="Previous service stopped. Waiting for the existing local process; no job is being restarted.")
            else:
                self._mark_interrupted(folder.name)
        if pending:
            self.active = pending[0]
            self.idle.clear()
            threading.Thread(target=self._watch_interrupted, args=(pending,), daemon=True).start()

    def _previous_process_pending(self, job_id: str, state: dict) -> bool:
        identity = state.get("pipeline_process")
        if identity is not None and process_alive(identity) is not False:
            return True
        processes, uncertain = matching_job_processes(self.jobs / job_id)
        return bool(processes) or uncertain

    def _mark_interrupted(self, job_id: str):
        image_ready = False
        try:
            self._verified_repaired_image(job_id)
            image_ready = True
        except (OSError, ValueError, KeyError):
            pass
        message = ("Previous service stopped; the old local process has ended. Its exit code was not confirmed. "
                   "Reuse the saved repaired image to rerun the original pipeline." if image_ready else
                   "Previous service stopped. No job was resubmitted. The image request may have been billed; "
                   "review this task before starting a new generation.")
        self.update(job_id, status="failed", image_ready=image_ready, can_retry_download=False,
                    can_retry_pipeline=image_ready, message=message)

    def _watch_interrupted(self, pending: list[str]):
        while pending and not self.stop_recovery.wait(2):
            for job_id in list(pending):
                if not self._previous_process_pending(job_id, self.state(job_id)):
                    self._mark_interrupted(job_id)
                    pending.remove(job_id)
            with self.lock:
                self.active = pending[0] if pending else None
                if not pending:
                    self.idle.set()

    def preflight(self) -> dict:
        required = [*BASELINE_FILES, "third_party/ml-sharp/ckpt/sharp_2572gikvuh.pt",
                    "third_party/lingbot-depth/model/lingbot-depth/model.pt"]
        missing = [name for name in required if not (self.root / name).is_file()]
        keys_ready = {}
        for model in MODELS:
            try:
                read_key(self.root, model)
                keys_ready[model] = True
            except ImageAPIError:
                keys_ready[model] = False
        return {"missing_files": missing, "key_ready": any(keys_ready.values()),
                "keys_ready": keys_ready,
                "disk_free_gb": round(shutil.disk_usage(self.storage).free / 1e9, 1)}

    def register_scene(self, directory: Path, name: str) -> dict:
        vertex = load_ply(directory / "upload.ply")
        # Same row order and scalar properties; canonical binary PLY for the viewer.
        with (directory / "upload.ply").open("rb") as stream:
            stream.readline(1024)
            binary_little_endian = stream.readline(1024).strip() == b"format binary_little_endian 1.0"
        if binary_little_endian:
            link_or_copy(directory / "upload.ply", directory / "scene.ply")
        else:
            PlyData([PlyElement.describe(vertex, "vertex")], text=False, byte_order="<").write(directory / "scene.ply")
        result = {"id": directory.name, "name": name[:200], "count": len(vertex),
                  "file": f"/api/scenes/{directory.name}/scene.ply"}
        atomic_json(directory / "scene.json", result)
        return result

    def create(self, request: dict) -> dict:
        with self.lock:
            if self.active:
                raise ValueError("Another repair job is running. Wait for it to finish.")
            checks = self.preflight()
            if checks["missing_files"]:
                raise ValueError("Required local pipeline files or model weights are missing.")
            if checks["disk_free_gb"] < 8:
                raise ValueError("At least 8 GB of free disk space is required for a repair job.")
            scene_id = identifier(request.get("scene_id"))
            if not (self.scenes / scene_id / "scene.json").is_file():
                raise ValueError("Load the scene again before starting repair.")
            model = request.get("model")
            if not isinstance(model, str) or model not in MODELS:
                raise ValueError("Choose a supported image model.")
            prompt = validate_prompt(request.get("prompt"))
            width, height = validate_camera(request.get("camera"))
            viewer_camera = validate_viewer_camera(request.get("viewer_camera"))
            model_size(model, width, height)
            read_key(self.root, model)  # fail before creating a job or uploading the view
            model_preflight(model)
            job_id = uuid.uuid4().hex
            folder = self.jobs / job_id
            (folder / "input").mkdir(parents=True)
            atomic_json(folder / "input" / "point_cloud.camera.json", request["camera"])
            atomic_json(folder / "request.json", {"scene_id": scene_id, "model": model, "prompt": prompt,
                                                  "viewer_camera": viewer_camera})
            state = {"id": job_id, "status": "uploading", "stage": 0, "message": "Saving the locked view.",
                     "created_at": time.time(), "model": model, "directory": str(folder)}
            atomic_json(folder / "status.json", state)
            return state

    def view(self, job_id: str) -> dict:
        folder = self.jobs / identifier(job_id)
        request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
        scene = json.loads((self.scenes / identifier(request["scene_id"]) / "scene.json").read_text(encoding="utf-8"))
        return {"scene": scene, "model": request["model"], "prompt": request["prompt"],
                "viewer_camera": request.get("viewer_camera"),
                "camera": json.loads((folder / "input" / "point_cloud.camera.json").read_text(encoding="utf-8"))}

    def state(self, job_id: str) -> dict:
        path = self.jobs / identifier(job_id) / "status.json"
        if not path.is_file():
            raise ValueError("Job was not found.")
        state = json.loads(path.read_text(encoding="utf-8"))
        cache = self.image_results.get(job_id, {})
        available = "result" in cache and time.time() - cache.get("received_at", 0) < 1200
        supported = state.get("model") in MODELS
        state["can_retry_download"] = bool(state.get("can_retry_download") and available and supported)
        state["can_retry_pipeline"] = bool(state.get("can_retry_pipeline") and supported)
        # Explain a known original-algorithm gate without changing its verdict,
        # editing historical artifacts, or rerunning the API/pipeline.
        if (state.get("status") == "failed" and state.get("stage") == 3
                and state.get("image_ready")
                and str(state.get("message", "")).startswith("Existing pipeline failed (exit ")):
            detail = depth_rejection_message(path.parent)
            if detail:
                state["message"] = detail
        return state

    def update(self, job_id: str, **changes) -> dict:
        with self.lock:
            state = self.state(job_id)
            state.update(changes, updated_at=time.time())
            atomic_json(self.jobs / job_id / "status.json", state)
            return state

    def submit(self, job_id: str) -> dict:
        with self.lock:
            if self.active:
                raise ValueError("Another repair job is already running.")
            state = self.state(job_id)
            if state["status"] != "uploading":
                raise ValueError("This job was already submitted; create a new job to retry.")
            model_preflight(state.get("model"))  # also reject retired models in old unsubmitted jobs
            folder = self.jobs / job_id / "input"
            for name in ("point_cloud.png", "point_cloud.depth.npy", "deleted.bin"):
                if not (folder / name).is_file():
                    raise ValueError("The locked-view upload is incomplete.")
            self.active = job_id
            self.idle.clear()
            self.update(job_id, status="validating", server_process=self.owner,
                        message="Checking registered inputs before the API call.")
            threading.Thread(target=self._run, args=(job_id,), daemon=True).start()
            return self.state(job_id)

    def retry_download(self, job_id: str) -> dict:
        with self.lock:
            if self.active or self.state(job_id)["status"] != "failed":
                raise ValueError("Wait for the current task to finish.")
            if not self.state(job_id).get("can_retry_download"):
                raise ValueError("The in-memory API result is no longer available. A new job requires a new image request.")
            model_preflight(self.state(job_id).get("model"))
            self.active = job_id
            self.idle.clear()
            self.update(job_id, status="validating", can_retry_download=False, message="Retrying the existing image download; no new generation request.")
            threading.Thread(target=self._run, args=(job_id, True), daemon=True).start()
            return self.state(job_id)

    def retry_pipeline(self, job_id: str) -> dict:
        with self.lock:
            state = self.state(job_id)
            if self.active or state["status"] != "failed" or not state.get("image_ready"):
                raise ValueError("Only a failed pipeline with a completed image can be retried.")
            if state.get("model") not in MODELS:
                raise ValueError("This historical model is no longer supported. Open a scene and choose a supported model.")
            self._verified_repaired_image(job_id)
            self.active = job_id
            self.idle.clear()
            self.update(job_id, status="validating", stage=0, can_retry_pipeline=False,
                        message="Reusing the saved repaired image; no new image API request.")
            threading.Thread(target=self._run, args=(job_id, False, True), daemon=True).start()
            return self.state(job_id)

    def _verified_repaired_image(self, job_id: str) -> Path:
        folder = self.jobs / identifier(job_id)
        report = json.loads((folder / "appearance" / "image_manifest.json").read_text(encoding="utf-8"))
        request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
        repaired = folder / "appearance" / "repaired_rgb.png"
        if (digest(repaired)["sha256"] != report.get("repaired_sha256")
                or digest(folder / "input" / "point_cloud.png")["sha256"] != report.get("source_sha256")):
            raise ValueError("Saved input or repaired image changed; refusing to reuse it.")
        if report.get("model") != request["model"] or report.get("prompt") != request["prompt"]:
            raise ValueError("Saved image does not match this task's model and prompt.")
        return repaired

    def _validate_inputs(self, job_id: str, request: dict):
        folder = self.jobs / job_id / "input"
        camera = json.loads((folder / "point_cloud.camera.json").read_text(encoding="utf-8"))
        width, height = validate_camera(camera)
        image = decode_image((folder / "point_cloud.png").read_bytes())
        if image.size != (width, height):
            raise ValueError("RGB dimensions do not match the locked camera.")
        depth = np.load(folder / "point_cloud.depth.npy", allow_pickle=False, mmap_mode="r")
        if depth.dtype != np.dtype("float32") or depth.shape != (height, width, 2):
            raise ValueError("Depth must be float32 [height, width, 2], camera-Z and alpha.")
        if np.isinf(depth).any() or not np.isfinite(depth[..., 1]).all():
            raise ValueError("Depth contains invalid infinite values or alpha.")
        if np.any((depth[..., 1] < 0) | (depth[..., 1] > 1)):
            raise ValueError("Depth alpha must be between 0 and 1.")
        # Reuse the exact original mask definition as a no-cost preflight.
        from prepare_depth_anchored_inputs import largest_component, components_touching_seed
        alpha = np.asarray(image)[..., 3]
        core = largest_component(alpha <= 204)
        authority = components_touching_seed(alpha < 255, core) | core
        if not np.any(~authority & np.isfinite(depth[..., 0]) & (depth[..., 0] > 1e-3)):
            raise ValueError("No valid observed depth outside the hole; choose a view with more surrounding context.")
        Image.fromarray(authority.astype(np.uint8) * 255).save(folder / "hole_preview.png")
        vertex = load_ply(self.scenes / request["scene_id"] / "scene.ply")
        deleted = np.fromfile(folder / "deleted.bin", dtype=np.uint8)
        if len(deleted) != len(vertex) or np.any(deleted > 1) or np.all(deleted):
            raise ValueError("Deleted-Gaussian mask does not match the loaded scene.")
        base = vertex[deleted == 0]
        PlyData([PlyElement.describe(base, "vertex")], text=False, byte_order="<").write(folder / "point_cloud.ply")
        self.update(job_id, input_count=len(vertex), deleted_count=int(deleted.sum()), base_count=len(base))

    def _run(self, job_id: str, download_only: bool = False, pipeline_only: bool = False):
        folder = self.jobs / job_id
        try:
            request = json.loads((folder / "request.json").read_text(encoding="utf-8"))
            locked_inputs = {name: digest(folder / "input" / name) for name in (
                "point_cloud.png", "point_cloud.depth.npy", "point_cloud.camera.json", "deleted.bin")}
            input_manifest = folder / "locked_input_manifest.json"
            if input_manifest.is_file():
                if json.loads(input_manifest.read_text(encoding="utf-8")) != locked_inputs:
                    raise ValueError("The saved locked-view inputs changed; refusing to retry this task.")
            else:
                atomic_json(input_manifest, locked_inputs)
            if pipeline_only:
                repaired = self._verified_repaired_image(job_id)
                # Preserve every failed attempt in this exact task directory.
                attempt = folder / "failed_attempts" / uuid.uuid4().hex
                attempt.mkdir(parents=True)
                for name in ("pipeline", "pipeline.log", "pipeline_baseline.json", "studio_success.json"):
                    previous = folder / name
                    if previous.exists():
                        previous.rename(attempt / name)
            self._validate_inputs(job_id, request)
            if not pipeline_only:
                self.update(job_id, status="image_api", message="Repairing the view with the selected image model.")
                # Only the current failed generation needs a retry cache. Keep signed
                # URLs/base64 in memory, bounded to one result and twenty minutes.
                self.image_results = {key: value for key, value in self.image_results.items()
                                      if key == job_id and time.time() - value.get("received_at", 0) < 1200}
                cached = self.image_results.setdefault(job_id, {})
                if download_only and "result" not in cached:
                    raise ImageAPIError("The cached image link expired. No new generation request was sent.")
                repaired = repair_image(folder / "input" / "point_cloud.png", folder / "appearance",
                                        request["model"], request["prompt"], read_key(self.root, request["model"]), result_cache=cached)
            self.image_results.pop(job_id, None)
            self.update(job_id, status="color_correction", image_ready=True,
                        message="Color correction: Laplacian interpolation using a 30–180 px band outside the mask; zero correction at the patch center.")
            corrected = correct_file(folder / "input" / "point_cloud.png", repaired,
                                     folder / "appearance" / "color_corrected_rgb.png")
            self.update(job_id, status="pipeline", image_ready=True, color_corrected=True,
                        message="Running the existing six-step depth-anchored pipeline.")
            fingerprints = {name: digest(self.root / name) for name in BASELINE_FILES}
            atomic_json(folder / "pipeline_baseline.json", fingerprints)
            output = folder / "pipeline"
            # The only numerical entry point. No alternate commands, flags, or model fallbacks.
            # The existing verifier's --sample-id is an integer, not a UUID.
            sample_id = str(int(job_id, 16))
            command = [str(self.root / "run_sample.cmd"), sample_id, str(folder / "input"), str(corrected), str(output)]
            with (folder / "pipeline.log").open("w", encoding="utf-8") as log:
                self.update(job_id, log_ready=True)
                process = subprocess.Popen(command, cwd=self.root, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                           text=True, encoding="utf-8", errors="replace",
                                           creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0))
                self.process = process
                self.update(job_id, pipeline_process=process_identity(process.pid), server_process=self.owner)
                for line in process.stdout:
                    log.write(line)
                    log.flush()
                    matched = re.match(r"\[(\d)/6\] (.*)", line)
                    if matched:
                        self.update(job_id, stage=int(matched[1]), message=matched[2].strip())
                code = process.wait()
            self.process = None
            if code != 0:
                raise ValueError(f"Existing pipeline failed (exit {code}). See pipeline.log; the original scene was not replaced.")
            acceptance = json.loads((output / "sample_acceptance.json").read_text(encoding="utf-8"))
            if acceptance.get("status") != "accepted":
                raise ValueError("The original pipeline did not accept this repair.")
            if fingerprints != {name: digest(self.root / name) for name in BASELINE_FILES}:
                raise ValueError("Pipeline source files changed while the job was running.")
            result_path = output / "fusion" / "depth_anchored_inpainted.ply"
            artifact = digest(result_path)
            if artifact["bytes"] < 100:
                raise ValueError("Pipeline result is missing or empty.")
            atomic_json(folder / "studio_success.json", {"result": artifact, "acceptance": digest(output / "sample_acceptance.json")})
            result_scene_id = uuid.uuid4().hex
            result_scene_dir = self.scenes / result_scene_id
            result_scene_dir.mkdir()
            link_or_copy(result_path, result_scene_dir / "scene.ply")
            result_count = PlyData.read(result_path)["vertex"].count
            result_scene = {"id": result_scene_id, "name": "repaired_scene.ply", "count": result_count,
                            "file": f"/api/scenes/{result_scene_id}/scene.ply"}
            atomic_json(result_scene_dir / "scene.json", result_scene)
            self.update(job_id, status="succeeded", stage=6, message="Repair accepted. Loading the completed scene.",
                        result=f"/api/jobs/{job_id}/result.ply", result_scene=result_scene)
        except Exception as error:
            # Errors from parsers and libraries can contain user-controlled or remote strings.
            safe = str(error) if isinstance(error, (ImageAPIError, ValueError)) else f"Local job failed ({type(error).__name__})."
            retryable = isinstance(error, ImageDownloadError) and "result" in self.image_results.get(job_id, {})
            if not retryable:
                self.image_results.pop(job_id, None)
            self.update(job_id, status="failed", failed_phase=self.state(job_id).get("status"),
                        message=safe[:1000], log=f"/api/jobs/{job_id}/pipeline.log",
                        can_retry_download=retryable, can_retry_pipeline=bool(self.state(job_id).get("image_ready")))
        finally:
            with self.lock:
                self.active = None
                self.idle.set()

    def result(self, job_id: str) -> Path:
        state = self.state(job_id)
        folder = self.jobs / job_id
        if state["status"] != "succeeded" or not (folder / "studio_success.json").is_file():
            raise ValueError("No accepted result exists for this job.")
        result = folder / "pipeline" / "fusion" / "depth_anchored_inpainted.ply"
        marker = json.loads((folder / "studio_success.json").read_text(encoding="utf-8"))
        if (digest(result) != marker.get("result")
                or digest(folder / "pipeline" / "sample_acceptance.json") != marker.get("acceptance")):
            raise ValueError("Published result no longer matches its acceptance hashes.")
        return result
