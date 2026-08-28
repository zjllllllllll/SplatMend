@echo off
chcp 65001 >nul
setlocal EnableExtensions DisableDelayedExpansion

set "PIPELINE_ROOT=%~dp0"
if "%~1"=="" (
  set "OUTPUT_DIR=%PIPELINE_ROOT%outputs\demo"
) else (
  set "OUTPUT_DIR=%~f1"
)

call "%PIPELINE_ROOT%run_sample.cmd" 1 ^
  "%PIPELINE_ROOT%assets" ^
  "%PIPELINE_ROOT%assets\repaired_rgb.png" ^
  "%OUTPUT_DIR%"
exit /b %ERRORLEVEL%
