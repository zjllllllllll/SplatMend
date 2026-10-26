import { t, translateMessage } from './i18n.mjs';

export const progressSteps = [
    'Lock view and export', 'Save and verify inputs', 'AI image repair',
    'Color correction (Laplacian)', 'Prepare RGB-D and masks', 'LingBot depth completion',
    'Depth alignment and validation', 'SHARP patch generation', 'Merge with original scene',
    'Validate final result', 'Load repaired result'
];

export function phase(index, message, mode = 'running', colorCorrection = true) {
    return { index, message, mode, colorCorrection };
}

export function jobProgress(state, previous = phase(1, 'Reading job status…')) {
    const next = (index, message, mode = 'running') => phase(index, message, mode, state.color_correction ?? state.color_corrected ?? true);
    const stage = Number.isInteger(state.stage) && state.stage >= 1 && state.stage <= 6 ? state.stage : 0;
    const pipelineIndex = stage ? stage + 3 : 4;
    switch (state.status) {
        case 'uploading': return next(1, state.message);
        case 'validating': return next(state.image_ready ? 3 : 1, state.message);
        case 'image_api': return next(2, state.message);
        case 'color_correction': return next(3, state.message);
        case 'pipeline': return next(pipelineIndex, state.message);
        // Accepted output still needs to download and load before UI completion.
        case 'succeeded': return next(10, '3D result accepted. Downloading and loading the scene…');
        case 'failed': {
            let index = stage || state.image_ready ? pipelineIndex : 1;
            if (state.failed_phase === 'image_api' || state.can_retry_download) index = 2;
            else if (state.failed_phase === 'color_correction') index = 3;
            else if (state.failed_phase === 'validating' && !state.image_ready) index = 1;
            else if (!stage && !state.image_ready && previous.index === 2) index = 2;
            return next(index, state.message, 'failed');
        }
        case 'recovering': return next(stage || state.image_ready ? pipelineIndex : previous.index,
            `Recovering the job without resubmitting. ${state.message || ''}`, 'waiting');
        default: return next(previous.index, 'Confirming job status…', 'waiting');
    }
}

export function progressDisplay(current) {
    const complete = current.mode === 'complete';
    const completed = complete ? progressSteps.length : current.index;
    const stateLabel = { running: t('Running'), waiting: t('Awaiting confirmation'), failed: t('Failed'), complete: t('Complete') }[current.mode];
    return {
        completed,
        headline: complete ? t('Repair complete · result loaded') : `${String(current.index + 1).padStart(2, '0')} ${t(progressSteps[current.index])} · ${stateLabel}`,
        count: t('{completed} / {total} steps complete', { completed, total: progressSteps.length }),
        rows: progressSteps.map((label, index) => ({ label: t(index === 3 && !current.colorCorrection ? 'Color correction (skipped)' : label),
            state: index < completed ? 'done' : index === current.index ? current.mode : 'pending',
            marker: index < completed ? (index === 3 && !current.colorCorrection ? '–' : '✓') : index === current.index && current.mode === 'failed' ? '!' : String(index + 1).padStart(2, '0')
        }))
    };
}

export function renderProgress(doc, current) {
    const card = doc.getElementById('job-progress');
    card.hidden = !current;
    if (!current) return;
    const display = progressDisplay(current);
    card.dataset.state = current.mode;
    doc.getElementById('progress-heading').textContent = display.headline;
    doc.getElementById('progress-count').textContent = display.count;
    doc.getElementById('progress-detail').textContent = translateMessage(current.message);
    const meter = doc.getElementById('progress');
    meter.max = progressSteps.length;
    meter.value = display.completed;
    meter.setAttribute('aria-valuetext', display.count);
    doc.getElementById('progress-steps').replaceChildren(...display.rows.map(row => {
        const item = doc.createElement('li');
        item.dataset.state = row.state;
        if (row.state === 'running' || row.state === 'waiting' || row.state === 'failed') item.setAttribute('aria-current', 'step');
        const marker = doc.createElement('span');
        marker.className = 'step-marker';
        marker.textContent = row.marker;
        const label = doc.createElement('span');
        label.textContent = row.label;
        item.append(marker, label);
        return item;
    }));
}