# 已验证的 Windows / CUDA 环境

本文件记录 2026-08-28 实际跑通本仓库示例的环境。它是复现基线，不代表可以无条件升级到相近版本。

## 版本基线

| 组件 | 已验证版本/配置 |
|---|---|
| Windows Python | 3.13.13，Miniforge 环境 `goris` |
| PyTorch | 2.11.0+cu128 |
| torchvision | 0.26.0+cu128 |
| PyTorch CUDA runtime | 12.8 |
| CUDA Toolkit | 12.8，`nvcc.exe` 可用 |
| GPU | NVIDIA GeForce RTX 5070 Ti |
| Compute Capability | 12.0 / sm_120 |
| gsplat | 1.5.3 |
| Visual Studio | 2022 Build Tools，x64 C++ 环境 |
| Ninja | 1.13.0 |
| `TORCH_CUDA_ARCH_LIST` | 12.0 |
| `MAX_JOBS` | 4 |

其余直接依赖见 `requirements.txt`。

## 唯一支持的启动方式

任何可能直接或间接导入 `torch`、`gsplat`、`sharp` 或 LingBot 的命令，都从仓库根目录通过以下入口运行：

```bat
run_cuda_python.cmd <python arguments>
```

该脚本负责：

- 调用 VS 2022 `vcvars64.bat`；
- 把环境内 Ninja 与 CUDA 12.8 放进 PATH；
- 设置 sm_120、并发编译数和 UTF-8；
- 把仓库内 SHARP 源码放在 PYTHONPATH 最前面；
- 启用 Hugging Face 离线模式，避免运行时下载模型。

本机路径不同则复制 `config.example.cmd` 为 `config.local.cmd` 后修改。不要编辑公共入口来写某个同事的用户名路径。

## 安装边界

`requirements.txt` 不包含 torch、torchvision 和 gsplat。这三项不是普通纯 Python 依赖：它们必须与 CUDA、GPU 架构和下述 Windows 修改一致。已有可运行环境时，不要执行会覆盖它们的 `pip install -U` 或强制重装。

新机器建议先由环境负责人复现以下精确 GPU 栈，再安装 `requirements.txt`。模型源码已经在仓库里，不需要 `pip install sharp`；两个模型权重不随 Git 分发，须按 README 放到指定路径。

## 当前环境依赖的 Windows 兼容修改

当前已验证环境包含三处安装级修改，升级包可能覆盖：

1. `site-packages/gsplat/cuda/_backend.py`
   - Windows 编译使用 MSVC `/O2`，不传 GCC 专用参数；
   - 保留标准 Torch extension 缓存，不在失败后删除整个缓存；
2. `site-packages/torch/utils/cpp_extension.py`
   - 以 UTF-8 解码 Windows 编译器输出，避免中文系统代码页触发 `UnicodeDecodeError`；
3. `site-packages/torch/include/c10/cuda/CUDACachingAllocator.h`
   - 避免参数名 `small` 与 Windows SDK 宏冲突。

仓库内还保留两处模型源码级修改：

- `third_party/ml-sharp/src/sharp/utils/gsplat.py` 同时兼容 gsplat rasterization 返回 3 项或 4 项；
- `third_party/lingbot-depth/mdm/model/dinov2_rgbd/layers/block.py` 在 Windows 无 xFormers 时使用普通 Block 路径处理列表输入。

## gsplat 扩展缓存

RTX 5070 Ti 需要为 sm_120 编译 gsplat CUDA 扩展。首次编译可能需要一分钟以上；之后应复用用户目录下的标准 Torch extension 缓存。不要设置临时 `TORCH_EXTENSIONS_DIR`，不要删除或替换已经成功生成的 `gsplat_cuda.pyd`。

检查后端：

```bat
run_cuda_python.cmd -c "from gsplat.cuda._backend import _C; print(_C.__file__)"
```

输出应指向 `gsplat_cuda.pyd`，不能是 `None`。

## 常见错误

| 日志 | 含义 |
|---|---|
| `gsplat: No CUDA toolkit found` | 绕过了启动器，或 CUDA/VS 初始化顺序错误 |
| `_C is None` / `CameraModelType` 属性错误 | gsplat CUDA 后端未成功加载 |
| `Ninja is required` | 环境内 Ninja 未进入 PATH |
| GCC 参数或 `-Wno-attributes` 错误 | gsplat Windows 修改被覆盖 |
| 编译器输出 `UnicodeDecodeError` | PyTorch `cpp_extension.py` UTF-8 修改被覆盖 |
| nvcc 报 `bool char` | Windows SDK 的 `small` 宏冲突修改被覆盖 |

遇到这些问题时，先修复对应前提；不要通过重装 torch/gsplat 或创建新扩展缓存绕过去。
