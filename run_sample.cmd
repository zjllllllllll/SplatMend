@echo off
chcp 65001 >nul
setlocal EnableExtensions DisableDelayedExpansion

if "%~4"=="" (
  echo Usage: run_sample.cmd SAMPLE_ID SAMPLE_DIR REPAIRED_RGB OUTPUT_DIR
  exit /b 2
)

set "PIPELINE_ROOT=%~dp0"
if exist "%PIPELINE_ROOT%config.local.cmd" call "%PIPELINE_ROOT%config.local.cmd"
if not defined PIPELINE_PYTHON_LAUNCHER set "PIPELINE_PYTHON_LAUNCHER=%PIPELINE_ROOT%run_cuda_python.cmd"

set "SAMPLE_ID=%~1"
set "SAMPLE_DIR=%~f2"
set "REPAIRED_RGB=%~f3"
set "OUTPUT_DIR=%~f4"
set "SHARP_ROOT=%PIPELINE_ROOT%third_party\ml-sharp"
set "SHARP_CHECKPOINT=%SHARP_ROOT%\ckpt\sharp_2572gikvuh.pt"
set "LINGBOT_ROOT=%PIPELINE_ROOT%third_party\lingbot-depth"
set "LINGBOT_CHECKPOINT=%LINGBOT_ROOT%\model\lingbot-depth\model.pt"

call :require "%PIPELINE_PYTHON_LAUNCHER%" || goto :missing
call :require "%SAMPLE_DIR%\point_cloud.png" || goto :missing
call :require "%SAMPLE_DIR%\point_cloud.depth.npy" || goto :missing
call :require "%SAMPLE_DIR%\point_cloud.camera.json" || goto :missing
call :require "%SAMPLE_DIR%\point_cloud.ply" || goto :missing
call :require "%REPAIRED_RGB%" || goto :missing
call :require "%SHARP_ROOT%\src\sharp\models\predictor.py" || goto :missing
call :require "%SHARP_CHECKPOINT%" || goto :missing
call :require "%LINGBOT_ROOT%\mdm\model\v2.py" || goto :missing
call :require "%LINGBOT_CHECKPOINT%" || goto :missing

if not exist "%OUTPUT_DIR%" mkdir "%OUTPUT_DIR%"

echo [1/6] Prepare registered RGB-D inputs and masks.
call "%PIPELINE_PYTHON_LAUNCHER%" "%PIPELINE_ROOT%prepare_depth_anchored_inputs.py" ^
  --source-rgba "%SAMPLE_DIR%\point_cloud.png" ^
  --repaired-rgb "%REPAIRED_RGB%" ^
  --source-depth "%SAMPLE_DIR%\point_cloud.depth.npy" ^
  --camera-json "%SAMPLE_DIR%\point_cloud.camera.json" ^
  --output-dir "%OUTPUT_DIR%\prepared"
if errorlevel 1 goto :fail

echo [2/6] Complete missing camera-Z depth with LingBot.
call "%PIPELINE_PYTHON_LAUNCHER%" "%PIPELINE_ROOT%run_lingbot_depth.py" ^
  --lingbot_root "%LINGBOT_ROOT%" ^
  --checkpoint "%LINGBOT_CHECKPOINT%" ^
  --rgb "%OUTPUT_DIR%\prepared\rgb_completed_exact.png" ^
  --depth "%OUTPUT_DIR%\prepared\depth_observed_with_hole.npy" ^
  --intrinsics "%OUTPUT_DIR%\prepared\intrinsics.txt" ^
  --hole_mask "%OUTPUT_DIR%\prepared\hole_authority.png" ^
  --out_depth "%OUTPUT_DIR%\depth\lingbot_raw_depth.npy" ^
  --out_mask "%OUTPUT_DIR%\depth\lingbot_model_mask.png" ^
  --out_preview "%OUTPUT_DIR%\depth\lingbot_preview.png" ^
  --out_manifest "%OUTPUT_DIR%\depth\lingbot_manifest.json"
if errorlevel 1 goto :fail

echo [3/6] Calibrate depth to the source scene and keep observed depth exact outside.
call "%PIPELINE_PYTHON_LAUNCHER%" "%PIPELINE_ROOT%fuse_and_validate_depth.py" ^
  --source-depth "%SAMPLE_DIR%\point_cloud.depth.npy" ^
  --predicted-depth "%OUTPUT_DIR%\depth\lingbot_raw_depth.npy" ^
  --model-mask "%OUTPUT_DIR%\depth\lingbot_model_mask.png" ^
  --hole-mask "%OUTPUT_DIR%\prepared\hole_authority.png" ^
  --context-ring "%OUTPUT_DIR%\prepared\context_ring.png" ^
  --camera-json "%SAMPLE_DIR%\point_cloud.camera.json" ^
  --rgb "%OUTPUT_DIR%\prepared\rgb_completed_exact.png" ^
  --edge-blend-pixels 16 ^
  --output-dir "%OUTPUT_DIR%\depth"
if errorlevel 1 goto :fail

echo [4/6] Generate full-frame SHARP Gaussians with exact depth and surface Jacobian.
call "%PIPELINE_PYTHON_LAUNCHER%" "%PIPELINE_ROOT%run_sharp_hard_depth.py" ^
  --image "%OUTPUT_DIR%\prepared\rgb_completed_exact.png" ^
  --depth "%OUTPUT_DIR%\depth\depth_completed_sharp_dense.npy" ^
  --hole-mask "%OUTPUT_DIR%\prepared\hole_authority.png" ^
  --camera-json "%SAMPLE_DIR%\point_cloud.camera.json" ^
  --checkpoint "%SHARP_CHECKPOINT%" ^
  --covariance-mode surface_jacobian ^
  --output-dir "%OUTPUT_DIR%\sharp_hard"
if errorlevel 1 goto :fail

echo [5/6] Select the authority core plus six rings and append in scene coordinates.
call "%PIPELINE_PYTHON_LAUNCHER%" "%PIPELINE_ROOT%merge_hard_patch.py" ^
  --base-ply "%SAMPLE_DIR%\point_cloud.ply" ^
  --sharp-camera-ply "%OUTPUT_DIR%\sharp_hard\sharp_hard_layer0_camera.ply" ^
  --hole-mask "%OUTPUT_DIR%\prepared\hole_authority.png" ^
  --source-rgba "%SAMPLE_DIR%\point_cloud.png" ^
  --target-rgb "%OUTPUT_DIR%\prepared\rgb_completed_exact.png" ^
  --target-depth "%OUTPUT_DIR%\depth\depth_completed_exact.npy" ^
  --anchored-layer0-z "%OUTPUT_DIR%\sharp_hard\sharp_hard_layer0_z.npy" ^
  --camera-json "%SAMPLE_DIR%\point_cloud.camera.json" ^
  --opacity-weight-min 0.0 ^
  --opacity-weight-floor 0.25 ^
  --blend-expand-reference-pixels 150 ^
  --blend-reference-short-side 1440 ^
  --blend-ring-count 6 ^
  --blend-core-opacity-mode full ^
  --output-dir "%OUTPUT_DIR%\fusion"
if errorlevel 1 goto :fail

echo [6/6] Verify all algorithm contracts and output invariants.
call "%PIPELINE_PYTHON_LAUNCHER%" "%PIPELINE_ROOT%verify_depth_anchored_sample.py" ^
  --sample-id "%SAMPLE_ID%" ^
  --work-root "%OUTPUT_DIR%"
if errorlevel 1 goto :fail

echo PIPELINE_ACCEPTED
echo Final PLY: %OUTPUT_DIR%\fusion\depth_anchored_inpainted.ply
echo Report:    %OUTPUT_DIR%\sample_acceptance.json
exit /b 0

:require
if exist "%~1" exit /b 0
echo ERROR: required file not found: "%~1"
exit /b 1

:missing
echo ERROR: sample, model, source, or launcher dependency is missing.
exit /b 2

:fail
echo ERROR: pipeline stopped at a failed acceptance gate.
exit /b 1
