# 第三方代码与模型声明

本仓库为了可复现性，随代码保存了运行所需的第三方源码和模型权重。以下只是索引，法律效力以子目录中的原始许可证全文为准。

## Apple SHARP

- 路径：`third_party/ml-sharp`
- 内容：SHARP 源码；模型权重不在仓库中，运行前需放到 `ckpt/sharp_2572gikvuh.pt`
- 源码许可证：`third_party/ml-sharp/LICENSE`
- 模型许可证：`third_party/ml-sharp/LICENSE_MODEL`
- 其他依赖声明：`third_party/ml-sharp/ACKNOWLEDGEMENTS`
- 权重 SHA256：`94211a75198c47f61fca7d739ba08a215418d8d398d48fddf023baccc24f073d`

重要：SHARP 模型许可把用途限制为非商业科研和学术开发，不允许商业利用、产品开发或商业产品/服务使用。再分发必须同时提供模型许可，并保留许可要求的归属声明：`Apple Machine Learning Research Model is licensed under the Apple Machine Learning Research Model License Agreement.`

仓库对 SHARP 源码的本地兼容修改：

- `src/sharp/utils/gsplat.py`：适配 gsplat 1.5.x 在不同构建中返回 3 项或 4 项 rasterization 输出。

没有对 SHARP 权重做微调、再训练或参数修改。

## LingBot-Depth

- 路径：`third_party/lingbot-depth`
- 内容：LingBot-Depth 源码；模型权重不在仓库中，运行前需放到 `model/lingbot-depth/model.pt`
- 许可证：Apache License 2.0，见 `third_party/lingbot-depth/LICENSE`
- 其他声明：`third_party/lingbot-depth/LEGAL.md`
- 权重 SHA256：`b60cf27ddbd0e51e9b59b03475c0d39d02d2e48ecf8dbb5866f04d46802b3c23`

仓库对 LingBot 源码的本地兼容修改：

- `mdm/model/dinov2_rgbd/layers/block.py`：在 Windows 未安装 xFormers 时，用普通 Block 顺序处理列表样本。

## 项目代码与示例数据

本仓库当前没有项目级 LICENSE。上传者应在扩大分发范围前，确认自己有权分发 `assets` 中的场景与修复图，并为自有代码选择许可证。第三方许可证不会自动给自有代码或示例数据授权。
