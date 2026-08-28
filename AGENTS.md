# Repository execution rules

## CUDA entry point

Never invoke an environment Python directly for commands that import, or may transitively import, `torch`, `gsplat`, `sharp`, or LingBot. From this repository root always use:

```bat
run_cuda_python.cmd <python arguments>
```

Read `WINDOWS_ENVIRONMENT.md` before changing GPU dependencies. Do not reinstall or upgrade torch, torchvision, gsplat, SHARP, LingBot, or Ninja merely to bypass a launcher or extension error. Fix the exact failed precondition and preserve the normal compiled Torch extension cache.

## Local sources

The authoritative SHARP and LingBot sources and weights are under `third_party`. Do not add fallbacks to sibling repositories or stale editable installs. The launcher and scripts place `third_party/ml-sharp/src` before site-packages.

## Generated data

Runtime output belongs under `outputs/` and must not be committed. Keep only the single documented sample under `assets/`. Large binary inputs, weights and PLY/NPY files must remain under Git LFS rules from `.gitattributes`.
