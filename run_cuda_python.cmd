@echo off
setlocal EnableExtensions DisableDelayedExpansion

set "PIPELINE_ROOT=%~dp0"
if exist "%PIPELINE_ROOT%config.local.cmd" call "%PIPELINE_ROOT%config.local.cmd"
if not defined GORIS_ROOT set "GORIS_ROOT=%USERPROFILE%\AppData\Local\miniforge3\envs\goris"
if not defined VCVARS64 set "VCVARS64=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
if not defined CUDA_HOME set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"

set "GORIS_PYTHON=%GORIS_ROOT%\python.exe"
set "GORIS_NINJA=%GORIS_ROOT%\Scripts\ninja.exe"
if not exist "%GORIS_PYTHON%" goto :missing
if not exist "%GORIS_NINJA%" goto :missing
if not exist "%VCVARS64%" goto :missing
if not exist "%CUDA_HOME%\bin\nvcc.exe" goto :missing

call "%VCVARS64%" >nul
if errorlevel 1 exit /b 2

set "PATH=%GORIS_ROOT%\Scripts;%CUDA_HOME%\bin;%PATH%"
set "PYTHONPATH=%PIPELINE_ROOT%;%PIPELINE_ROOT%third_party\ml-sharp\src;%PYTHONPATH%"
set "TORCH_CUDA_ARCH_LIST=12.0"
set "MAX_JOBS=4"
set "DISTUTILS_USE_SDK=1"
set "PYTHONUTF8=1"
set "PYTHONIOENCODING=utf-8"
set "PYTHONUNBUFFERED=1"
set "HF_HUB_OFFLINE=1"

where cl.exe >nul 2>&1
if errorlevel 1 exit /b 2
where ninja.exe >nul 2>&1
if errorlevel 1 exit /b 2
where nvcc.exe >nul 2>&1
if errorlevel 1 exit /b 2

if "%~1"=="" (
  echo ENVIRONMENT_OK
  exit /b 0
)

"%GORIS_PYTHON%" %*
exit /b %ERRORLEVEL%

:missing
echo ERROR: Python, Ninja, VS2022 x64, or CUDA Toolkit is missing.
exit /b 2
