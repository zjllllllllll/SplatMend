export const progressSteps = [
    '锁定视角并导出', '保存与校验输入', 'AI 图片修复',
    '准备 RGB-D 与掩码', 'LingBot 深度补全', '深度标定与验收',
    'SHARP 生成补丁', '合并原始场景', '最终结果验收', '加载补洞结果'
];

export function phase(index, message, mode = 'running') {
    return { index, message, mode };
}

export function jobProgress(state, previous = phase(1, '正在读取任务状态…')) {
    const stage = Number.isInteger(state.stage) && state.stage >= 1 && state.stage <= 6 ? state.stage : 0;
    const pipelineIndex = stage ? stage + 2 : 3;
    switch (state.status) {
        case 'uploading': return phase(1, state.message);
        case 'validating': return phase(state.image_ready ? 3 : 1, state.message);
        case 'image_api': return phase(2, state.message);
        case 'pipeline': return phase(pipelineIndex, state.message);
        // Accepted output still needs to download and load before UI completion.
        case 'succeeded': return phase(9, '三维结果已验收，正在下载并加载场景…');
        case 'failed': {
            let index = stage || state.image_ready ? pipelineIndex : 1;
            if (state.failed_phase === 'image_api' || state.can_retry_download) index = 2;
            else if (state.failed_phase === 'validating' && !state.image_ready) index = 1;
            else if (!stage && !state.image_ready && previous.index === 2) index = 2;
            return phase(index, state.message, 'failed');
        }
        case 'recovering': return phase(stage || state.image_ready ? pipelineIndex : previous.index,
            `正在恢复任务状态；不会重复提交。${state.message || ''}`, 'waiting');
        default: return phase(previous.index, '正在确认任务状态…', 'waiting');
    }
}

export function progressDisplay(current) {
    const complete = current.mode === 'complete';
    const completed = complete ? progressSteps.length : current.index;
    const stateLabel = { running: '进行中', waiting: '等待确认', failed: '失败', complete: '完成' }[current.mode];
    return {
        completed,
        headline: complete ? '补洞完成 · 结果已加载' : `${String(current.index + 1).padStart(2, '0')} ${progressSteps[current.index]} · ${stateLabel}`,
        count: `已完成 ${completed} / ${progressSteps.length} 步`,
        rows: progressSteps.map((label, index) => ({ label,
            state: index < completed ? 'done' : index === current.index ? current.mode : 'pending',
            marker: index < completed ? '✓' : index === current.index && current.mode === 'failed' ? '!' : String(index + 1).padStart(2, '0')
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
    doc.getElementById('progress-detail').textContent = current.message;
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
