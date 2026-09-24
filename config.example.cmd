@echo off
rem Copy this file to config.local.cmd and adapt paths for the target machine.
rem Verified baseline: Python 3.13.13, VS 2022 x64 C++, CUDA Toolkit 12.8.
rem run_cuda_python.cmd sets Ninja/PYTHONPATH, sm_120 and MAX_JOBS automatically.
rem Do not put API keys or model weights in this file.
set "GORIS_ROOT=%USERPROFILE%\AppData\Local\miniforge3\envs\goris"
set "VCVARS64=C:\Program Files (x86)\Microsoft Visual Studio\2022\BuildTools\VC\Auxiliary\Build\vcvars64.bat"
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
rem GrsAI prefers grsaiapi.com, then tries the China node if connection fails before sending.
rem To prefer the China node, uncomment the next line (global remains the backup):
rem set "GRSAI_API_HOST=grsai.dakka.com.cn"
