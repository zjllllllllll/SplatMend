# 单视角深度锚定的 3D Gaussian 补洞 Demo

本仓库是当前已验证的补洞链路最小工程版：在一个带洞视角中合成修复 RGB，用 LingBot 补齐相机深度，再让 SHARP 在生成阶段直接接受这张完整深度；最后按二维洞区与六级边缘带裁出 SHARP 高斯，并原位追加到原场景。链路不再做 ICP、生成后深度缩放、整体平移或 Poisson 后校正。

2026-08-28 已在随仓库提供的 `assets` 示例上，用 `run_demo.cmd` 从第 1 步连续跑到第 6 步，最终状态为 `PIPELINE_ACCEPTED`。

## SHARP 是否在仓库里

在。为了避免算法逻辑依赖另一份本地 checkout，仓库已经包含运行所需的两套模型源码；模型权重体积较大，不上传 GitLab，需要使用者自行放到指定位置：

- `third_party/ml-sharp`：SHARP 源码、Windows/gsplat 兼容修改和原始许可证；
- `third_party/lingbot-depth`：LingBot-Depth 源码、无 xFormers 的 Windows fallback 和许可证。

运行脚本会把仓库内的 `third_party/ml-sharp/src` 放在 `PYTHONPATH` 最前面，不会退回已安装的 `sharp`，也不会引用旧的 `sharp_sd2_test`、`real-gauss-hole` 或其他工程目录。

注意：SHARP 模型权重只允许用于非商业科研用途，具体条款见 `third_party/ml-sharp/LICENSE_MODEL`。示例 PLY 和深度文件通过 Git LFS 管理，两个模型权重不在仓库中。

## 目录结构

```text
.
├─ assets/                         唯一保留的完整示例输入
├─ third_party/
│  ├─ ml-sharp/                    SHARP 源码、许可证；权重需自行放置
│  └─ lingbot-depth/               LingBot 源码、许可证；权重需自行放置
├─ prepare_depth_anchored_inputs.py  构造 Mask、精确合成 RGB、构造带洞深度
├─ run_lingbot_depth.py            LingBot RGB-D 深度补全适配器
├─ fuse_and_validate_depth.py      深度标定、边界融合与验收
├─ run_sharp_hard_depth.py         深度锚定 SHARP 与表面 Jacobian 协方差
├─ sharp_runtime.py                SHARP 加载、渲染及深度指标
├─ merge_hard_patch.py             六级补丁裁剪、坐标还原、场景合并
├─ gaussian_patch_io.py            PLY 和高斯中心/协方差坐标变换
├─ verify_depth_anchored_sample.py 整链路不变量验收
├─ run_sample.cmd                  通用样本入口
├─ run_demo.cmd                    assets 示例入口
├─ run_cuda_python.cmd             Windows CUDA/VS/Ninja 统一启动器
├─ config.example.cmd              本机环境配置模板
├─ requirements.txt                非 CUDA Python 依赖
└─ WINDOWS_ENVIRONMENT.md          GPU 环境和兼容补丁说明
```

仓库中没有历史运行记录、批处理结果、旧数据集、缓存或旧路线启动脚本。运行输出统一写入 `outputs/`，并被 `.gitignore` 排除。

## 快速运行

### 1. 配置本机环境

当前验证环境见 `WINDOWS_ENVIRONMENT.md`。如果路径与示例机器不同：

```bat
copy config.example.cmd config.local.cmd
```

然后编辑 `config.local.cmd` 中的 Python 环境、VS 2022 和 CUDA 12.8 路径。`config.local.cmd` 已被 Git 忽略。

先检查启动环境：

```bat
run_cuda_python.cmd
```

成功时输出 `ENVIRONMENT_OK`。

### 2. 放置模型权重

运行前必须准备下面两个文件，路径和文件名必须完全一致：

| 模型 | 放置位置（仓库根目录相对路径） | 大小 | SHA256 |
|---|---|---:|---|
| SHARP | `third_party/ml-sharp/ckpt/sharp_2572gikvuh.pt` | 2,809,738,232 bytes | `94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d` |
| LingBot-Depth | `third_party/lingbot-depth/model/lingbot-depth/model.pt` | 1,284,837,952 bytes | `b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23` |

两个路径已写入 `.gitignore`，本地放入后不会被误提交。权重来源和使用权限应分别遵循 SHARP 与 LingBot-Depth 的原始许可证；不要使用来源不明或哈希不一致的文件。

### 3. 跑随仓库示例

```bat
run_demo.cmd
```

默认输出到 `outputs/demo`。如需指定临时目录：

```bat
run_demo.cmd D:\temp\gaussian-hole-demo
```

若权重未放到上述路径，入口会在启动前明确报出缺失文件并停止。

### 4. 跑自定义样本

```bat
run_sample.cmd SAMPLE_ID SAMPLE_DIR REPAIRED_RGB OUTPUT_DIR
```

例如：

```bat
run_sample.cmd 7 D:\data\sample_07 D:\data\sample_07\repaired_rgb.png D:\results\sample_07
```

只有全部六个阶段及最终不变量检查通过，入口才会输出 `PIPELINE_ACCEPTED` 并返回退出码 0。

## 输入数据契约

每个 `SAMPLE_DIR` 必须含以下四个同视角、同分辨率文件：

| 文件 | 含义 | 强约束 |
|---|---|---|
| `point_cloud.png` | 删除物体后的场景 RGBA 渲染 | Alpha 表示洞区；RGB 与深度逐像素配准 |
| `point_cloud.depth.npy` | 原场景相机 Z 深度 | 前两维为 H×W；洞外有效深度必须保留 |
| `point_cloud.camera.json` | 相机和导出元数据 | 必须含 `intrinsic_K_opencv` 及相机外参/矩阵 |
| `point_cloud.ply` | 已经删除目标物体的基础高斯场景 | 最终采用 append-only，基础高斯不会再被删除或修改 |

另需一张 `REPAIRED_RGB`：外部二维修复模型生成的完整 RGB。它必须和 `point_cloud.png` 宽高一致。该模型可以不知道深度，但应尽量保证洞内内容合理；代码只接受它在权威洞区内的像素，洞外强制换回原图。

`assets/README.md` 给出了仓库示例的文件大小与 SHA256。

## 当前算法链路

```text
带洞 RGBA + 原深度 + 相机 + Base PLY + 修复 RGB
        │
        ├─ 1. 构造 Core / Authority / 48 px Context
        │     洞内取修复 RGB，洞外逐像素保留原 RGB；Authority 内深度清零
        │
        ├─ 2. LingBot 根据完整 RGB、带洞深度、Mask、K 预测缺失 camera-Z
        │
        ├─ 3. 在 Context 上标定尺度/偏移，洞内边缘 16 px 过渡
        │     得到洞外严格等于原值的完整深度
        │
        ├─ 4. SHARP 生成全图高斯；Layer-0 中心直接锚定完整深度
        │     再按目标深度表面 Jacobian 修正高斯协方差；Layer-1 关闭
        │
        ├─ 5. 从 Layer-0 取 Authority + 外扩 150 px 六环
        │     六环仅衰减 opacity，随后用相机逆变换还原到场景坐标
        │
        └─ 6. Final PLY = Base PLY + Patch，并检查全部算法不变量
```

### 第 1 步：Mask 与严格 RGB-D 输入

代码：`prepare_depth_anchored_inputs.py`

输入：带洞 RGBA、外部修复 RGB、原深度、相机 JSON。

Mask 定义：

1. `Core`：取 `alpha <= 204`（即透明度至少 20%）中的最大连通域，排除零散透明噪声；
2. `Authority`：取所有 `alpha < 255` 且与 Core 连通的区域。这是正式洞区，也是 RGB 与深度允许被替换的唯一内部区域；
3. `Context`：Authority 向外膨胀 48 个原图像素后减去 Authority，仅用于把 LingBot 深度标定到原场景尺度；
4. 150 px 的边缘 Blend 不在本步生成，它在第 5 步从 Authority 向外建立。

RGB 合成是严格的：

```text
rgb_completed_exact[p] = repaired_rgb[p]，p ∈ Authority
rgb_completed_exact[p] = source_rgb[p]，  p ∉ Authority
```

因此 SHARP 看到的是一张完整 RGB；洞外像素与原图逐通道完全一致。深度输入则把 Authority 内置 0，Authority 外逐值保留原深度。

主要输出：`rgb_completed_exact.png`、`hole_core.png`、`hole_authority.png`、`context_ring.png`、`depth_observed_with_hole.npy`、`intrinsics.txt`、`preparation_report.json`。

### 第 2 步：LingBot 补相机深度

代码：`run_lingbot_depth.py`，模型代码位于 `third_party/lingbot-depth`。

LingBot 接收完整 RGB、Authority 内为 0 的观测深度、Authority Mask 和相机内参，预测一张稠密深度。这里使用 LingBot 而不是豆包补深度；豆包/其他外部二维模型只负责 `repaired_rgb.png`。

本步只提供候选几何，不能直接作为最终场景深度，因为模型输出仍可能存在尺度与偏移误差。输出为 `lingbot_raw_depth.npy`、模型有效 Mask、预览和 `lingbot_manifest.json`。

### 第 3 步：深度标定与边界融合

代码：`fuse_and_validate_depth.py`

在 48 px Context 中，用同时有效的“LingBot 预测深度 x”和“原场景深度 y”稳健拟合仿射关系：

```text
z_scene = scale × z_lingbot + offset
```

然后：

- Authority 内使用标定后的补全深度；
- 洞内靠边 16 px 根据距边界距离，在补全深度和邻近原深度之间平滑过渡；
- Authority 外强制复制原深度，不允许模型改动；
- 对残余无效洞像素才做受 Mask 约束的补值；
- 用边界跳变、平面法向/残差、深度范围和洞内覆盖率做验收。

输出两张完整深度：`depth_completed_exact.npy` 用于最终补丁检查，`depth_completed_sharp_dense.npy` 用于 SHARP；当前示例中二者都没有改动 Authority 外像素。

### 第 4 步：深度锚定的 SHARP 高斯生成

代码：`run_sharp_hard_depth.py`、`sharp_runtime.py`，SHARP 本体位于 `third_party/ml-sharp`。

输入 RGB 与深度先缩放到 1536×1536，SHARP 生成两个 768×768 高斯层，每层 589,824 个。当前链路绕过 SHARP 的 Alignment UNet，避免生成后再估计整体位置。完整深度直接作为 Layer-0 的第一表面；SHARP 单目深度只用于构造受限的第二层相对间隔。

SHARP 仍负责预测颜色、opacity、scale、rotation 和学习到的横向射线偏移。对 Layer-0 的每个高斯（不只是洞内）执行：

```text
r = z_target / z_old
P_new = r × P_old
P_new.z = z_target
Scale_new = r × Scale_old
```

这里沿 SHARP 已学习到的射线移动中心，所以保留其横向结构，同时让相机 Z 精确等于补好的深度。Layer-1 没有第二表面深度监督，故保留行结构但把 opacity 设为 0，防止背层产生漂浮点。

仅缩放中心和 scale 仍可能让平面覆盖不足，所以随后用完整深度构造目标表面：

```text
P(u,v) = z(u,v) K⁻¹ [u,v,1]ᵀ
J = [∂P/∂u, ∂P/∂v]
```

代码对相邻三维点做稳健有限差分，在深度断层处选择较短方向，得到两个切向量和表面法向。沿两个切向的标准差至少覆盖 0.75 个采样单元，法向厚度被限制为最小切向尺度的 0.05～0.35；最后把新协方差重新分解为 SHARP PLY 所需的 scale 与 quaternion。这个 Jacobian 影响 Layer-0 全图高斯，不只影响洞内。

输出完整的两层 SHARP PLY、Layer-0 Z、渲染 RGB/Depth/Alpha 和 `sharp_hard_acceptance.json`。

### 第 5 步：裁出补丁并融合边缘

代码：`merge_hard_patch.py`、`gaussian_patch_io.py`

补丁不是按三维包围盒或重新投影裁切，而是利用 SHARP 的固定像素行来源：

1. 从 Authority 向外扩 150 个参考像素；当前参考短边为 1440，因此示例原图实际也是 150 px；
2. 把外扩带均分为 6 个约 25 px 的环；
3. 将标签图最近邻缩放到 SHARP 的 768×768 网格；
4. 按像素行号直接选出对应的 Layer-0 高斯；Layer-1 不进入补丁；
5. Authority 核心 opacity 保持 SHARP 原值，六环 opacity 依次乘：

```text
0.980324 / 0.843750 / 0.623843 / 0.376157 / 0.156250 / 0.019676
```

这 150 px 六环就是当前的边缘融合带。它不会移动高斯中心，只让越远离洞区的补丁高斯越透明，使补丁外缘逐渐交给原场景。

选出的高斯仍处于 SHARP 相机坐标。代码使用相机 Model-View 的精确逆矩阵变换中心，并用同一个旋转变换协方差，再分解为 scene-space scale/quaternion。

### 第 6 步：Append-only 合并和验收

最终合并严格为：

```text
Final PLY = Base PLY + transformed Patch
```

基础 PLY 删除 0 个高斯、修改 0 个属性；基础高斯必须构成最终 PLY 的逐字段精确前缀。当前链路没有任何生成后对齐：无 ICP、无深度缩放/偏移、无位置平移、无 Poisson 修正。

`verify_depth_anchored_sample.py` 会联合读取前五步报告和实际 PLY，检查：Mask/RGB/深度的洞外精确性、LingBot 覆盖、SHARP 深度锚定与 Jacobian 已执行、两层数量、六环非空且权重递减、补丁来自 Layer-0、中心深度误差、坐标往返误差、基础前缀不变以及最终数量守恒。

## 输出目录

```text
OUTPUT_DIR/
├─ prepared/           RGB、Core/Authority/Context、带洞深度、准备报告
├─ depth/              LingBot 原始深度、融合深度、预览、深度验收报告
├─ sharp_hard/         完整 SHARP PLY、Layer-0 Z、渲染与 SHARP 验收报告
├─ fusion/             Patch PLY、Final PLY、六环可视化、合并验收报告
└─ sample_acceptance.json
```

正式结果是 `fusion/depth_anchored_inpainted.ply`；整链路总报告是 `sample_acceptance.json`。

## 随仓库示例的实测验收结果

2026-08-28，2560×1440 示例：

| 指标 | 结果 |
|---|---:|
| Authority 洞区 | 146,144 px |
| LingBot 洞内有效覆盖率 | 100% |
| 最终完整深度洞内覆盖率 | 100% |
| 洞外 RGB / 深度改动像素 | 0 / 0 |
| SHARP Layer-0 中心 Z p99 绝对误差 | 0 |
| SHARP 洞内渲染深度中位相对误差 | 0.0618% |
| SHARP 洞内渲染深度 p90 相对误差 | 0.2450% |
| SHARP 洞内 Alpha 均值 | 0.997683 |
| Base 高斯 | 2,252,716 |
| 新增补丁高斯 | 76,253 |
| 最终高斯 | 2,328,969 |
| 删除/修改 Base 高斯 | 0 / 0 |

Smoke test 的临时输出在验收后已删除；上表来自删除前的最终报告。

## 依赖与 Windows 注意事项

- 非 CUDA 依赖列在 `requirements.txt`；
- GPU 栈必须匹配 `WINDOWS_ENVIRONMENT.md` 中的已验证版本；
- 所有可能导入 torch、gsplat 或 SHARP 的 Python 都必须经 `run_cuda_python.cmd` 启动；
- 不要随意升级/重装 torch 或 gsplat，Windows 兼容修改和已编译的 sm_120 扩展缓存会被破坏；
- `run_sharp_hard_depth.py` 会在加载完整 SHARP 模型树前预加载 gsplat CUDA 后端，防止后端被错误初始化为 `None`。

## 上传 GitLab

两个模型权重不进入 Git；仓库只用 Git LFS 管理示例 PLY 和深度文件。克隆后按“放置模型权重”一节补齐两个 `.pt` 文件即可运行。

```bat
git init
git lfs install
git add .gitattributes
git add .
git lfs ls-files
git commit -m "Add depth-anchored SHARP Gaussian inpainting demo"
git branch -M main
git remote add origin <gitlab-repository-url>
git push -u origin main
```

提交前确认 `git lfs ls-files` 包含 `assets/point_cloud.ply` 和 `assets/point_cloud.depth.npy`，且不包含两个模型权重。团队内部可把权重放到受控模型存储，部署时下载到 README 指定位置并校验 SHA256。

## 已知边界

- 当前方法是单视角补丁生成，二维修复 RGB 的内容质量仍直接影响颜色和局部语义；
- 只约束第一可见表面，第二层 opacity 被关闭；
- SHARP 网格固定为 768×768，两层共 1,179,648 行；补丁裁剪依赖这个行来源约定；
- 相机模型按 OpenCV pinhole K 与导出矩阵解释，当前实现以 `fx` 作为 SHARP 焦距，并要求输入数据彼此严格配准；
- Append-only 设计不会清理基础 PLY 中残余的半透明洞边高斯；边缘遮盖主要由 Authority 核心和 150 px 六环补丁完成。

## 第三方声明

详见 `THIRD_PARTY_NOTICES.md` 及各 `third_party` 子目录中的原始许可证。本仓库未替用户或所属组织指定项目级许可证；对外发布前还应由仓库所有者补充适合自有代码和示例数据的 LICENSE。
