@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
if not exist "studio\web\dist\index.html" (
  echo Frontend is not built. Run build_studio.cmd first.
  exit /b 2
)
call "%~dp0download_models.cmd"
if errorlevel 1 exit /b 1
call "%~dp0run_cuda_python.cmd" -m studio.server %*
exit /b %ERRORLEVEL%
