import { t } from './i18n.mjs';

// Public server-provided model profiles are authoritative; no duplicate size table.
export function requestOptions(profile, width, height, keyReady) {
    if (!profile) return { ready: false, message: t('Choose a supported image model. Historical models are not selected automatically.') };
    if (!Number.isInteger(width) || !Number.isInteger(height) || width <= 0 || height <= 0) {
        return { ready: false, message: t('Invalid locked-view dimensions.') };
    }
    const sizes = [...profile.sizes].sort((a, b) =>
        Math.abs(a[0] / a[1] - width / height) - Math.abs(b[0] / b[1] - width / height));
    const [w, h] = sizes[0];
    if (Math.abs(w * height - width * h) * 100 > width * h * 2) {
        return { ready: false, message: t('The selected model does not support this aspect ratio. Change the locked-view size.') };
    }
    const provider = profile.id === 'gpt-image-2' ? 'GrsAI' : (t('Volcengine'));
    const summary = t('Source {width} × {height} → API {outputWidth} × {outputHeight} · {provider}',
        { width, height, outputWidth: w, outputHeight: h, provider });
    return { ready: keyReady, size: `${w}x${h}`,
        message: summary + ' ' + (keyReady ? t('The returned image is resized after aspect-ratio validation.') : t('Configure the selected model’s API key.')) };
}