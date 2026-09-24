@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0"
where curl.exe >nul 2>&1
if errorlevel 1 (
  echo ERROR: curl.exe is required to download model weights.
  exit /b 2
)
powershell.exe -NoProfile -ExecutionPolicy Bypass -File "%~dp0download_models.ps1" %*
exit /b %ERRORLEVEL%
