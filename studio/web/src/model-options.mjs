// Public server-provided model profiles are authoritative; no duplicate size table.
export function requestOptions(profile, width, height, ossReady) {
    if (!profile) return { ready: false, message: '请选择当前支持的图片模型。不会自动替换旧模型。' };
    if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
        return { ready: false, message: '锁定视图尺寸无效。' };
    }
    const sizes = [...profile.sizes].sort((a, b) =>
        Math.abs(a[0] / a[1] - width / height) - Math.abs(b[0] / b[1] - width / height));
    const [w, h] = sizes[0];
    if (Math.abs(w * height - width * h) * 100 > width * h * 2) {
        return { ready: false, message: '所选模型不支持这个视图比例，请调整锁定视图尺寸。' };
    }
    const oss = profile.input_transport === 'oss_url';
    const summary = `原图 ${width} × ${height} → API 输出 ${w} × ${h} · ${oss ? 'OSS 私有链接' : 'Base64 输入'}`;
    return { ready: !oss || ossReady, size: `${w}x${h}`,
        message: summary + (oss && !ossReady ? '。请先配置 OSS 上传凭据。' : '。返回图检查比例后恢复原尺寸。') };
}
