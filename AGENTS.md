# Repository execution rules

## CUDA entry point

Never invoke an environment Python directly for commands that import, or may transitively import, `torch`, `gsplat`, `sharp`, or LingBot. From this repository root always use:

```bat
run_cuda_python.cmd <python arguments>
```

Read the verified environment and compatibility notes below before changing GPU dependencies. Do not reinstall or upgrade torch, torchvision, gsplat, SHARP, LingBot, or Ninja merely to bypass a launcher or extension error. Fix the exact failed precondition and preserve the normal compiled Torch extension cache.

## Local sources

The authoritative SHARP and LingBot sources and weights are under `third_party`. Do not add fallbacks to sibling repositories or stale editable installs. The launcher and scripts place `third_party/ml-sharp/src` before site-packages.

## Generated data

Runtime output belongs under ignored `outputs/`. The local demo sample belongs under ignored `assets/`; neither directory may be committed. Model weights and large binary inputs must not be committed. If future files are intentionally tracked, preserve the PLY/NPY/weight Git LFS rules in `.gitattributes`.

## Verified Windows GPU environment

The working baseline is Windows with Miniforge `goris` Python 3.13.13, PyTorch 2.11.0+cu128, torchvision 0.26.0+cu128, gsplat 1.5.3, CUDA Toolkit 12.8, VS 2022 Build Tools x64, Ninja 1.13.0, and RTX 5070 Ti (`sm_120`). `run_cuda_python.cmd` calls `vcvars64.bat`, prepends Ninja/CUDA and checkout-local SHARP to the environment, sets `TORCH_CUDA_ARCH_LIST=12.0`, `MAX_JOBS=4`, and Hugging Face offline mode. Machine-specific paths belong in ignored `config.local.cmd`, copied from `config.example.cmd`.

Installed packages include required Windows fixes: gsplat `cuda/_backend.py` uses MSVC `/O2`; PyTorch `utils/cpp_extension.py` decodes compiler output as UTF-8; PyTorch `include/c10/cuda/CUDACachingAllocator.h` avoids the Windows SDK `small` macro. Source fixes are in `third_party/ml-sharp/src/sharp/utils/gsplat.py` and `third_party/lingbot-depth/mdm/model/dinov2_rgbd/layers/block.py`. Package upgrades can erase the installed fixes.

Preserve the standard compiled Torch extension cache, including `gsplat_cuda.pyd`; never set a temporary `TORCH_EXTENSIONS_DIR`, clear the cache, or bypass the launcher. `run_cuda_python.cmd` with no arguments must print `ENVIRONMENT_OK`. A failed preflight means fix the missing Python/Ninja/VS/CUDA prerequisite. For a backend diagnostic, use `run_cuda_python.cmd -c "from gsplat.cuda._backend import _C; print(_C.__file__)"`; it should resolve to `gsplat_cuda.pyd`, not `None`.
