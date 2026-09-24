// English is the source language. A saved choice affects only this browser.
const chinese = {
  'Local workspace': '本地工作区',
  'Volcengine': '火山引擎',
  'Select and remove an area in a local Gaussian scene, then repair it with depth anchoring.': '本地高斯场景圈选、删除和深度锚定补洞工具',
  'Gaussian scene': '高斯场景',
  'No scene open': '未打开场景',
  '3D Gaussian view': '高斯三维视图',
  'Locked view for the current job': '当前任务的锁定视图',
  'View locked · repairing': '视角已锁定 · 正在补洞',
  'Open a Gaussian scene': '打开一个高斯场景',
  'Select and remove the area to repair, then start the repair.': '圈选并删除需要移除的区域，然后一键补洞。',
  'Open PLY file': '打开 PLY 文件',
  'Open local sample': '打开本地示例',
  'The scene stays on this computer. The rendered view is sent to the selected Volcengine or GrsAI image API.': '场景留在本机；当前渲染图会发送至所选的火山引擎或 GrsAI 图片 API。',
  'Orbit': '浏览', 'Lasso': '圈选', 'Rectangle': '框选', 'Delete selected': '删除选中',
  'Undo': '撤销', 'Clear selection': '清除选择', 'Frame view': '适配视图',
  'View: WASD move · Q/E down/up · Shift faster · V returns to orbit': '浏览模式：WASD 移动 · Q/E 升降 · Shift 加速 · V 返回旋转',
  'Left drag orbit/look · right drag pan · wheel zoom': '鼠标左键旋转/转向 · 右键平移 · 滚轮缩放',
  'Lasso/rectangle select through occlusion': '穿透圈选/框选（含遮挡层）',
  'Shift add · Ctrl remove': 'Shift 添加 · Ctrl 移除',
  'Scene repair': '场景补洞', 'Open file': '打开文件', 'Repair progress': '补洞进度',
  'Completed steps': '已完成步骤', 'All repair steps': '完整补洞步骤',
  'Steps show actual progress, not elapsed-time percentages.': '按实际步骤更新；单步耗时不同，并非时间百分比。',
  'Image model': '图片模型', 'Loading configuration…': '正在读取配置…',
  'Locked-view output': '锁定视图输出', 'Repair prompt': '补洞提示词',
  'Default: prompt.txt in the repository.': '默认读取仓库中的 prompt.txt。',
  'Starting locks the view and saves RGB, depth, camera and the PLY with a hole. The rendered image goes to the selected Volcengine or GrsAI API. Laplacian color correction uses pixels 30–180 px outside the mask, reaches zero at the patch center, then runs the six-step pipeline.': '开始后锁定当前视角，自动保存 RGB、深度、相机和带洞 PLY。完整渲染图会发送给所选的火山引擎或 GrsAI 图片 API。颜色校正采用拉普拉斯插值，参考 Mask 外 30–180 像素区域，补丁中心校正量为 0，再运行原有六步补洞链路。',
  'Start repair': '开始补洞', 'Starting local tool…': '正在启动本地工具…',
  'Retry image download (no new generation)': '重试图片下载（不重新生图）',
  'Rerun 3D repair (reuse image)': '重跑三维补洞（复用修复图）',
  'Job page': '任务页面', 'Source with hole': '带洞图', 'API repaired image': 'API 修复图',
  'Color-corrected image': '颜色校正图', 'Pipeline log': '运行日志', 'Download result PLY': '下载结果 PLY',
  'Depth anchored · original pipeline preserved': '深度锚定 · 原链路保留',
  'LingBot → SHARP → six-ring edge blending': 'LingBot → SHARP → 六级边缘融合',
  '{count} splats · {selected} selected': '{count} 个高斯 · 选中 {selected}',
  'Reading Gaussian scene…': '正在读取高斯场景…',
  'Scene ready. Adjust the view, then select and remove the area to repair.': '场景已就绪。调整视角，圈选并删除需要移除的区域。',
  'Opening local sample and restoring its saved camera view…': '正在打开本地示例，并恢复其已记录的相机视角…',
  'Sample ready. Its saved camera view has been restored; the hole is already present.': '示例已就绪，视角与仓库相机记录一致。示例已包含空洞，可直接开始补洞。',
  'Checking configuration, then exporting locked RGB, depth and camera data.': '检查配置，然后导出锁定视角的 RGB、深度和相机参数。',
  'API key missing. Configure ark-key.txt or grs-key.txt.': 'API 密钥缺失，请配置 ark-key.txt 或 grs-key.txt。',
  'View locked. Exporting the image with a hole, depth and camera data…': '已锁定视角，正在导出带洞图片、深度和相机参数…',
  'Saving the image with a hole, depth, deletion mask and camera data…': '正在保存带洞图片、深度、删除记录与相机参数…',
  'Reading this job; no new image generation request…': '正在读取同一个任务的状态，不会重复生图…',
  'Confirming this job submission; no new image generation request…': '正在确认同一个任务的提交状态，不会重复生图…',
  'Local connection interrupted. Rechecking this job without calling the image API again.': '本地连接暂时中断，正在重新读取同一个任务；不会重复调用图片 API。',
  'Connection interrupted. Rechecking job progress.': '连接暂时中断，正在重新确认同一个任务的进度。',
  'Job has not started. Inputs are saved; no image API call was made.': '任务尚未启动；未调用图片 API。输入已保留，可以重新开始。',
  'Job directory: {path}': '任务目录：{path}',
  'The six-step pipeline accepted the result. Scene loaded; PLY is ready to download.': '六步原链路验收通过，结果已加载，可以下载 PLY。',
  'Repair complete. The accepted result is loaded and ready to download.': '补洞完成，六步原链路验收通过。结果已加载，也可以下载 PLY。',
  'The original scene is still saved locally. Please reopen it.': '原场景文件仍保存在本地，请重新打开。',
  'API key configured': 'API 密钥已配置', 'API key not found': '未找到 API 密钥',
  'Free disk: {size} GB': '剩余磁盘 {size} GB',
  'Model or pipeline files missing': '模型或管线文件缺失',
  'Open a PLY file to begin.': '打开 PLY 文件开始。',
  'Check model files and API key, then refresh this page.': '请检查模型文件和 API 密钥，然后刷新页面。',
  'Startup failed: {message}': '启动失败：{message}',
  'Choose a supported image model. Historical models are not selected automatically.': '请选择当前支持的图片模型。不会自动替换旧模型。',
  'Invalid locked-view dimensions.': '锁定视图尺寸无效。',
  'The selected model does not support this aspect ratio. Change the locked-view size.': '所选模型不支持这个视图比例，请调整锁定视图尺寸。',
  'Source {width} × {height} → API {outputWidth} × {outputHeight} · {provider}': '原图 {width} × {height} → API 输出 {outputWidth} × {outputHeight} · {provider}',
  'The returned image is resized after aspect-ratio validation.': '返回图检查比例后恢复原尺寸。',
  'Configure the selected model’s API key.': '请配置所选模型的 API 密钥。',
  'Lock view and export': '锁定视角并导出', 'Save and verify inputs': '保存与校验输入',
  'AI image repair': 'AI 图片修复', 'Color correction (Laplacian)': '颜色校正（拉普拉斯）',
  'Prepare RGB-D and masks': '准备 RGB-D 与掩码', 'LingBot depth completion': 'LingBot 深度补全',
  'Depth alignment and validation': '深度标定与验收', 'SHARP patch generation': 'SHARP 生成补丁',
  'Merge with original scene': '合并原始场景', 'Validate final result': '最终结果验收',
  'Load repaired result': '加载补洞结果',
  'Reading job status…': '正在读取任务状态…',
  '3D result accepted. Downloading and loading the scene…': '三维结果已验收，正在下载并加载场景…',
  'Recovering the job without resubmitting. {message}': '正在恢复任务状态；不会重复提交。{message}',
  'Confirming job status…': '正在确认任务状态…',
  'Running': '进行中', 'Awaiting confirmation': '等待确认', 'Failed': '失败', 'Complete': '完成',
  'Repair complete · result loaded': '补洞完成 · 结果已加载',
  '{completed} / {total} steps complete': '已完成 {completed} / {total} 步',
  'Saving the locked view.': '正在保存锁定视图。',
  'Checking registered inputs before the API call.': '调用图片 API 前正在校验输入。',
  'Retrying the existing image download; no new generation request.': '重试现有图片下载，不重新生图。',
  'Reusing the saved repaired image; no new image API request.': '复用已保存的修复图，不重新请求图片 API。',
  'Repairing the view with the selected image model.': '正在使用所选图片模型修复视图。',
  'Color correction: Laplacian interpolation using a 30–180 px band outside the mask; zero correction at the patch center.': '颜色校正：使用拉普拉斯插值，参考 Mask 外 30–180 像素区域，补丁中心校正量为 0。',
  'Running the existing six-step depth-anchored pipeline.': '正在运行原六步深度锚定管线。',
  'Repair accepted. Loading the completed scene.': '补洞结果通过验收，正在加载场景。',
  'Local job link is invalid. Reopen the studio.': '本地任务链接无效，请重新打开工作台。',
  'Local sample files are unavailable.': '本地示例文件不可用。',
  'Sample · point_cloud.ply': '示例 · point_cloud.ply',
  'Language': '语言',
  'Editing tools': '编辑工具',
  'Unable to open this Gaussian scene.': '无法打开此高斯场景。',
  'Saved selection no longer matches the scene.': '已保存的选区与场景不匹配。',
  'Saved deletion mask is invalid.': '已保存的删除记录无效。',
  'Unable to retrieve the imported scene.': '无法读取导入的场景。',
  'Unable to retrieve the sample scene.': '无法读取本地示例。',
  'Repair succeeded, but the result could not be downloaded. Use the result link to retry.': '补洞已成功，但结果下载失败。请使用结果链接重试。',
  'Original scene download failed.': '原场景下载失败。',
  'Saved deletion mask download failed.': '删除记录下载失败。',
  'Prepare registered RGB-D inputs and masks.': '准备已配准的 RGB-D 输入与掩码。',
  'Complete missing camera-Z depth with LingBot.': '用 LingBot 补全缺失的相机 Z 深度。',
  'Calibrate depth to the source scene and keep observed depth exact outside.': '将深度标定到原场景，并严格保留洞外原始深度。',
  'Generate full-frame SHARP Gaussians with exact depth and surface Jacobian.': '使用精确深度和表面 Jacobian 生成全图 SHARP 高斯。',
  'Select the authority core plus six rings and append in scene coordinates.': '裁出授权核心和六环，在场景坐标系中追加。',
  'Verify all algorithm contracts and output invariants.': '校验算法契约与最终输出不变量。'
};
let language = 'en';
export const getLanguage = () => language;
export function setLanguage(next) { language = next === 'zh' ? 'zh' : 'en'; return language; }
export function t(source, values = {}) {
  let result = language === 'zh' ? (chinese[source] ?? source) : source;
  for (const [key, value] of Object.entries(values)) result = result.replaceAll(`{${key}}`, String(value));
  return result;
}
export function translateMessage(message) {
  // Provider and diagnostic errors are shown verbatim when no safe translation is known.
  if (message?.startsWith('Recovering the job without resubmitting. ')) {
    return t('Recovering the job without resubmitting. {message}', {
      message: translateMessage(message.slice('Recovering the job without resubmitting. '.length))
    });
  }
  return t(message ?? '');
}
export function translateDom(doc) {
  doc.documentElement.lang = language === 'zh' ? 'zh-CN' : 'en';
  for (const item of doc.querySelectorAll('[data-i18n]')) item.textContent = t(item.dataset.i18n);
  for (const item of doc.querySelectorAll('[data-i18n-aria]')) item.setAttribute('aria-label', t(item.dataset.i18nAria));
  for (const item of doc.querySelectorAll('[data-i18n-alt]')) item.setAttribute('alt', t(item.dataset.i18nAlt));
  for (const item of doc.querySelectorAll('[data-i18n-content]')) item.setAttribute('content', t(item.dataset.i18nContent));
}