@echo off
setlocal EnableExtensions DisableDelayedExpansion
cd /d "%~dp0studio\web"
where npm.cmd >nul 2>&1
if errorlevel 1 (
  echo Node.js 20.19 or newer is required to build the local viewer.
  exit /b 2
)
if not exist "package-lock.json" (
  echo Missing package-lock.json. Restore the version-controlled lock file.
  exit /b 2
)
call npm.cmd ci --no-audit --no-fund
if errorlevel 1 exit /b 1
call npm.cmd run build
exit /b %ERRORLEVEL%
