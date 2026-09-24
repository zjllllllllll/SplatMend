# SplatMend

**Training-Free Hole Filling for Existing 3DGS Scenes**

[English](#english) · [简体中文](#chinese)

<a id="english"></a>
## English

SplatMend is a local Windows tool for repairing a hole in a 3D Gaussian Splatting (3DGS) scene. Open a PLY, select and delete an area, lock a camera view, generate a repaired RGB image with an image API, complete depth with LingBot-Depth, and generate an append-only SHARP Gaussian patch. The accepted result loads in the viewer and can be downloaded as a PLY.

**Platform and limits.** The verified setup is Windows, an NVIDIA RTX 5070 Ti (`sm_120`), CUDA Toolkit 12.8, Visual Studio 2022 Build Tools, and Python 3.13.13. Other GPUs/OSes have not been verified. Image generation uses a third-party provider and may incur charges. Model weights, API keys, local scenes and outputs are not included in Git.

### 1. Install the Windows/Python environment

Install [Miniforge](https://github.com/conda-forge/miniforge), [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/) with the C++ desktop workload, and the [CUDA 12.8 Toolkit](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-microsoft-windows/index.html). In **Miniforge Prompt**, create a new environment and install the verified package versions:

```bat
conda create -n goris python=3.13.13 -y
conda activate goris
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install gsplat==1.5.3
python -m pip install -r requirements.txt
```

Run these installation commands only for a **new** environment; do not reinstall over an already working one. The verified Windows build needs three compatibility adjustments in installed packages (MSVC flags in gsplat, UTF-8 compiler-output decoding in PyTorch, and the Windows `small` macro conflict). Their exact files and checks are in [AGENTS.md](AGENTS.md#verified-windows-gpu-environment). A fresh environment may need those adjustments before gsplat can compile. For the verified versions, inspect these installed files under the new environment's `Lib/site-packages/` before the first GPU run:

- `gsplat/cuda/_backend.py`: on Windows, use `extra_cflags = ["/O2"] if os.name == "nt" else [opt_level, "-Wno-attributes"]`; do not pass GCC's `-Wno-attributes` to MSVC.
- `torch/utils/cpp_extension.py`: set `SUBPROCESS_DECODE_ARGS = ('utf-8',) if IS_WINDOWS else ()` and use `.decode(*SUBPROCESS_DECODE_ARGS)` for compiler output.
- `torch/include/c10/cuda/CUDACachingAllocator.h`: rename the `StreamSegmentSize` boolean parameter/member using `small` to `small_pool`/`is_small_pool` to avoid the Windows SDK macro.

These are compatibility edits to a **new** installation, not commands to run against an already working environment. Preserve the normal Torch extension cache after a successful compile.

From the repository root, copy the path template and edit its three values if your machine differs:

```bat
copy config.example.cmd config.local.cmd
run_cuda_python.cmd
```

`config.local.cmd` is ignored by Git. `GORIS_ROOT` is the Miniforge environment directory, `VCVARS64` is the VS x64 compiler setup batch file, and `CUDA_HOME` is the toolkit directory. The launcher must print `ENVIRONMENT_OK`. **Always use `run_cuda_python.cmd` for Python commands that may import torch, gsplat, SHARP or LingBot**; it sets the verified x64 compiler, CUDA, Ninja, `sm_120`, `PYTHONPATH`, and offline model mode. To check that gsplat's compiled backend loads:

```bat
run_cuda_python.cmd -c "from gsplat.cuda._backend import _C; print(_C.__file__)"
```

The result should be a `gsplat_cuda.pyd` path, not `None`. The launcher and [AGENTS.md](AGENTS.md) describe the exact preflight and compatibility boundaries.

### 2. Install the web viewer

Install [Node.js](https://nodejs.org/en/download) **20.19 or newer**. From the repository root, run:

```bat
build_studio.cmd
```

This runs `npm ci --no-audit --no-fund` against `studio/web/package-lock.json` and builds `studio/web/dist/`. It does not install Python/CUDA packages. After changing front-end source, run it again and refresh the browser. Equivalent manual commands in `studio/web/`: `npm.cmd ci --no-audit --no-fund`, then `npm.cmd run build`.

### 3. Download model weights

From the repository root, run `download_models.cmd`. `run_studio.cmd` and `run_sample.cmd` call it automatically. The script downloads, resumes interrupted transfers, checks exact size and SHA-256, and skips already verified files. `download_models.cmd -VerifyOnly` checks local files without downloading.

| Model | Official manual download | Save **exactly** here (relative to repo root) | Bytes | SHA-256 |
|---|---|---|---:|---|
| Apple SHARP | [Apple checkpoint](https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt) | `third_party/ml-sharp/ckpt/sharp_2572gikvuh.pt` | 2,809,738,232 | `94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d` |
| LingBot-Depth v0.5 | [Hugging Face model.pt](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5/blob/79204ed6b837f4fdd192cf563e59481fecfa0295/model.pt) | `third_party/lingbot-depth/model/lingbot-depth/model.pt` | 1,284,837,952 | `b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23` |

If the script cannot access either host, use the links above in a browser, create the destination directories, save each file with the exact name, and run `download_models.cmd -VerifyOnly`. A Hugging Face login/network route may be needed. Do **not** save the small Git/Xet pointer instead of the 1.28 GB LingBot file. Weights are ignored by Git. SHARP weights carry a **non-commercial research-only** model license; read `third_party/ml-sharp/LICENSE_MODEL` before downloading or using them.

### 4. Obtain an image API key

To use **GrsAI GPT Image 2**, visit [GrsAI](https://grsai.ai/), register or sign in, open the [API Keys dashboard](https://grsai.ai/zh/dashboard/api-keys), create/copy an API key, and put only that key in `grs-key.txt` at the repository root. The site may require credits for calls. The project sends requests to `https://grsaiapi.com/v1/api/generate` using model `gpt-image-2`. To use GrsAI's China node, set `GRSAI_API_HOST=grsai.dakka.com.cn` in `config.local.cmd`; generation and result queries then use that same node. If the preferred node cannot connect before the generation request is sent, the tool tries the other official node once; it never resubmits after sending request bytes. GrsAI is a third-party service; its key is not an OpenAI API key.

Alternatively, for **Volcengine Seedream 5.0**, provide a key in root-level `ark-key.txt`. The app uses the `ARK_API_KEY` or `GRSAI_API_KEY` environment variable when the corresponding file is absent. Both key files are ignored by Git. Configure at least the key for the image model you intend to use. The rendered RGB image is sent to the selected provider; the PLY and local outputs stay on your machine.

### 5. Start and use the tool

```bat
run_studio.cmd
```

Leave that command window open, then use `http://127.0.0.1:8765/`. The service listens on localhost. If port 8765 is occupied, run `run_studio.cmd --port 8766` and open `http://127.0.0.1:8766/` instead. The web UI defaults to **English**; choose **中文** in the header to switch languages. The browser remembers your choice.

1. Click **Open PLY file** and load a standard, uncompressed Gaussian PLY. The source file is not modified.
2. Choose the camera view. Orbit with left drag, pan with right drag, zoom with the wheel; use WASD and Q/E to move. Use **Lasso** or **Rectangle** to select the area to remove, then **Delete selected**. Selection reaches occluded Gaussians under the selected screen region. Shift adds and Ctrl removes from the selection; Undo restores deletions.
3. Select the image model and locked-view output size, review the prompt (default `prompt.txt`), then click **Start repair**. The app saves the locked RGB, depth, camera, deletion mask and PLY with the hole. It calls the image API, color-corrects the result, and runs the original six-step depth-anchored pipeline.
4. Follow the 11-step progress panel. Only an accepted result is loaded. Use **Download result PLY**; generated files and logs stay under ignored `outputs/studio/`. A task URL `http://127.0.0.1:8765/?job=<job-id>` reopens that local task. If possible, retry an image download or rerun the 3D pipeline without generating another image.

The **Open local sample** button appears only when local `assets/point_cloud.ply` and `assets/point_cloud.camera.json` exist. `assets/` is ignored and is **not distributed** with the public repository. You can use your own PLY without it. `run_demo.cmd` needs a complete local sample; `run_sample.cmd SAMPLE_ID SAMPLE_DIR REPAIRED_RGB OUTPUT_DIR` runs the six-step numeric pipeline on prepared inputs.

### Validation and files

Run backend tests from the repository root with `run_cuda_python.cmd -m unittest discover -s studio/tests -v`. In `studio/web`, run `npm.cmd test`, `npm.cmd run check`, and `npm.cmd run build`. `requirements.txt` pins non-GPU Python packages; `config.example.cmd` provides machine paths; `package-lock.json` pins browser dependencies. CUDA/PyTorch/gsplat are intentionally separate because Windows fixes are required.

The numeric input directory for `run_sample.cmd` needs `point_cloud.png` (RGBA hole), `point_cloud.depth.npy` (aligned camera-Z depth), `point_cloud.camera.json` (intrinsics/extrinsics), and `point_cloud.ply` (base Gaussians), plus a same-view repaired RGB image. The six stages prepare a strict mask and RGB-D input, complete depth with LingBot, align/validate depth, generate depth-anchored SHARP Gaussians, crop and blend a six-ring patch, then append and validate the final PLY. Base Gaussians are retained.

### Licensing and third-party notices

This repository currently has **no project-level LICENSE** for its own code. Choose one before describing the project itself as open source. Vendored source retains its own terms: Apple SHARP source in `third_party/ml-sharp/LICENSE`, SHARP weights in `third_party/ml-sharp/LICENSE_MODEL`, LingBot-Depth in `third_party/lingbot-depth/LICENSE` and `LEGAL.md`, and the SuperSplat viewer in `studio/web/vendor/supersplat/LICENSE`. SHARP model use is restricted to non-commercial research/academic development and requires the attribution stated in its model license: `Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.` See `studio/web/vendor/UPSTREAM.md` and `studio/web/vendor/supersplat/LOCAL_CHANGES.md` for viewer provenance and local changes. Do not publish local `assets/`, API keys or weights without the necessary rights.

---

<a id="chinese"></a>
## Chinese

**简体中文**

SplatMend 是在 Windows 本机运行的 3D Gaussian Splatting（3DGS）补洞工具：打开 PLY，圈选/框选并删除目标区域，锁定视角，通过图片 API 修复 RGB，再用 LingBot-Depth 补深度、SHARP 生成高斯补丁。通过验收的结果会加载到网页，并可下载 PLY。

**已验证平台与边界：**Windows、RTX 5070 Ti（`sm_120`）、CUDA Toolkit 12.8、VS 2022 Build Tools、Python 3.13.13。其他 GPU/系统尚未验证。图片 API 由第三方提供，可能产生费用。模型权重、密钥、本地场景与输出均不随 Git 分发。

### 1. 安装 Windows / Python 环境

安装 [Miniforge](https://github.com/conda-forge/miniforge)、含“使用 C++ 的桌面开发”工作负载的 [Visual Studio 2022 Build Tools](https://visualstudio.microsoft.com/visual-cpp-build-tools/)，以及 [CUDA 12.8 Toolkit](https://docs.nvidia.com/cuda/archive/12.8.0/cuda-installation-guide-microsoft-windows/index.html)。在 **Miniforge Prompt** 中为新机器创建环境：

```bat
conda create -n goris python=3.13.13 -y
conda activate goris
python -m pip install torch==2.11.0+cu128 torchvision==0.26.0+cu128 --index-url https://download.pytorch.org/whl/cu128
python -m pip install gsplat==1.5.3
python -m pip install -r requirements.txt
```

这些安装命令只用于**新环境**，不要覆盖已经能运行的环境。已验证的 Windows 环境还需要修改已安装包的三个兼容问题：gsplat 的 MSVC 编译参数、PyTorch 编译输出的 UTF-8 解码、Windows `small` 宏冲突。具体位置与检查见 [AGENTS.md](AGENTS.md#verified-windows-gpu-environment)。新环境首次编译 gsplat 前可能需要完成这些修改。针对上述已验证版本，可在新环境的 `Lib/site-packages/` 中核对：`gsplat/cuda/_backend.py` 的 Windows 编译参数用 `/O2` 而不是 GCC 的 `-Wno-attributes`；`torch/utils/cpp_extension.py` 使用 `SUBPROCESS_DECODE_ARGS = ('utf-8',) if IS_WINDOWS else ()` 解码编译器输出；`torch/include/c10/cuda/CUDACachingAllocator.h` 的 `StreamSegmentSize` 布尔参数/成员避免使用 Windows SDK 的 `small` 宏名，改为 `small_pool` / `is_small_pool`。只在**新环境**中处理这些兼容问题，已跑通的环境不要重装或覆盖；成功后保留标准 Torch 扩展缓存。

在仓库根目录复制路径模板，按本机安装位置修改 `GORIS_ROOT`、`VCVARS64`、`CUDA_HOME`：

```bat
copy config.example.cmd config.local.cmd
run_cuda_python.cmd
```

`config.local.cmd` 被 Git 忽略。启动器应输出 `ENVIRONMENT_OK`。任何可能导入 torch、gsplat、SHARP、LingBot 的 Python 命令都要从仓库根目录通过 `run_cuda_python.cmd` 执行。检查 gsplat：

```bat
run_cuda_python.cmd -c "from gsplat.cuda._backend import _C; print(_C.__file__)"
```

输出应为 `gsplat_cuda.pyd` 路径，而不是 `None`。

### 2. 安装网页依赖

安装 [Node.js](https://nodejs.org/en/download) **20.19 或更新版本**，在仓库根目录运行 `build_studio.cmd`。它根据 `studio/web/package-lock.json` 执行 `npm ci` 并生成 `studio/web/dist/`，不会安装 Python/CUDA 包。修改前端源码后重新构建并刷新网页。手动等价命令是在 `studio/web` 中执行 `npm.cmd ci --no-audit --no-fund`、`npm.cmd run build`。

### 3. 下载两个模型权重

在仓库根目录运行 `download_models.cmd`。`run_studio.cmd` 与 `run_sample.cmd` 启动时也会自动调用；脚本支持断点续传、文件大小和 SHA-256 校验。已有文件验证通过就跳过。只检查不下载：`download_models.cmd -VerifyOnly`。

| 模型 | 脚本失效时的官方下载页 | 必须放到仓库内的准确路径 | 字节数 | SHA-256 |
|---|---|---|---:|---|
| Apple SHARP | [Apple 权重直链](https://ml-site.cdn-apple.com/models/sharp/sharp_2572gikvuh.pt) | `third_party/ml-sharp/ckpt/sharp_2572gikvuh.pt` | 2,809,738,232 | `94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d` |
| LingBot-Depth v0.5 | [Hugging Face model.pt](https://huggingface.co/robbyant/lingbot-depth-pretrain-vitl-14-v0.5/blob/79204ed6b837f4fdd192cf563e59481fecfa0295/model.pt) | `third_party/lingbot-depth/model/lingbot-depth/model.pt` | 1,284,837,952 | `b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23` |

如果脚本无法访问下载站，就在浏览器打开上面的链接，创建对应目录，按表中**完全一致**的文件名保存，再运行 `download_models.cmd -VerifyOnly`。Hugging Face 可能需要登录或合适的网络线路；不要把很小的 Git/Xet 指针文件误当成 1.28 GB 权重。权重被 Git 忽略。SHARP 权重仅限**非商业科研用途**，下载或使用前请阅读 `third_party/ml-sharp/LICENSE_MODEL`。

### 4. 申请图片 API

使用 **GrsAI GPT Image 2**：访问 [GrsAI 官网](https://grsai.ai/)，注册并登录，打开 [API Key 控制台](https://grsai.ai/zh/dashboard/api-keys)，创建/复制密钥，把密钥单独写入仓库根目录 `grs-key.txt`。平台可能按积分收费。本项目用模型 `gpt-image-2` 请求 `https://grsaiapi.com/v1/api/generate`。如果国内节点在本机可达，可在 `config.local.cmd` 设置 `GRSAI_API_HOST=grsai.dakka.com.cn`；提交和结果查询会使用同一节点。如果首选节点在发送生图请求前无法连接，工具会尝试另一个官方节点；请求一旦发出就不会跨节点重复提交，以免重复扣费。GrsAI 是第三方服务，它的密钥与 OpenAI 官方 API Key 不通用。

也可以使用**火山引擎 Seedream 5.0**，把相应密钥写入根目录 `ark-key.txt`。没有密钥文件时可分别用 `GRSAI_API_KEY` / `ARK_API_KEY` 环境变量；文件优先。两个密钥文件均被 Git 忽略。至少配置你要选择的模型对应的密钥。锁定视角的 RGB 会发给所选服务；PLY 和本机输出留在本地。

### 5. 启动并使用网页

在仓库根目录运行 `run_studio.cmd`，保持命令窗口开启，再访问 `http://127.0.0.1:8765/`。服务只监听本机。如果 8765 被占用，可运行 `run_studio.cmd --port 8766`，改为访问 `http://127.0.0.1:8766/`。网页**默认英文**，右上角选择 **中文** 即可切换，浏览器会记住选择。

1. 点击“打开 PLY 文件”，加载标准、未压缩的 Gaussian PLY；原文件不会修改。
2. 选好视角：左键旋转、右键平移、滚轮缩放，WASD 与 Q/E 移动。使用“圈选”或“框选”选中需要移除的区域，再点击“删除选中”。区域选择会穿透遮挡层；Shift 添加、Ctrl 移除，误删可撤销。
3. 选择图片模型和锁定视图尺寸，检查提示词（默认来自 `prompt.txt`），点击“开始补洞”。工具保存 RGB、深度、相机、删除记录和带洞 PLY，调用图片 API、做颜色校正，再运行原六步深度锚定管线。
4. 查看 11 步进度；只有通过验收的结果才会自动加载。点击“下载结果 PLY”。产物与日志在被 Git 忽略的 `outputs/studio/`。任务链接 `http://127.0.0.1:8765/?job=<任务ID>` 可恢复本机旧任务。符合条件时可只重试下载修复图，或复用已有图片重跑三维管线，不会重新生图。

只有本机同时存在 `assets/point_cloud.ply` 和 `assets/point_cloud.camera.json` 时，网页才显示“打开本地示例”。**公开仓库不提供 `assets/`**，直接打开自己的 PLY 即可。`run_demo.cmd` 需要完整本地示例；`run_sample.cmd SAMPLE_ID SAMPLE_DIR REPAIRED_RGB OUTPUT_DIR` 可对准备好的输入单独运行六步数值管线。

### 检查、输入与许可证

后端测试：在根目录运行 `run_cuda_python.cmd -m unittest discover -s studio/tests -v`。前端测试/检查：在 `studio/web` 运行 `npm.cmd test`、`npm.cmd run check`、`npm.cmd run build`。`requirements.txt` 锁定非 GPU Python 依赖，`config.example.cmd` 给出本机路径模板，`package-lock.json` 锁定网页依赖；CUDA/PyTorch/gsplat 因 Windows 兼容修改而单独安装。

`run_sample.cmd` 的输入目录需有同视角配准的 `point_cloud.png`（RGBA 洞图）、`point_cloud.depth.npy`（相机 Z 深度）、`point_cloud.camera.json`（内外参）、`point_cloud.ply`（基础高斯），另需一张修复后的 RGB。六步依次构造严格 Mask 与 RGB-D、LingBot 补深度、深度标定和验收、SHARP 深度锚定生成、六环裁剪与合并、最终不变量验收；基础高斯保持不变。

本仓库的**自有代码目前没有项目级 LICENSE**；若要声明整个项目开源，应先选择并添加许可证。第三方原始条款分别在 `third_party/ml-sharp/LICENSE`、`third_party/ml-sharp/LICENSE_MODEL`、`third_party/lingbot-depth/LICENSE`、`third_party/lingbot-depth/LEGAL.md`、`studio/web/vendor/supersplat/LICENSE`。SHARP 模型许可限制为非商业科研/学术开发，要求保留归属声明：`Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.` Viewer 上游来源与本地修改见 `studio/web/vendor/UPSTREAM.md`、`studio/web/vendor/supersplat/LOCAL_CHANGES.md`。未确认权利前不要公开本机 `assets/`、密钥或权重。