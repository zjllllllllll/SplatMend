const directions = {
    KeyW: 'forward', KeyS: 'backward', KeyA: 'left', KeyD: 'right', KeyQ: 'down', KeyE: 'up'
};

export function isEditing(target) {
    return Boolean(target?.isContentEditable || target?.closest?.('input, textarea, select, [contenteditable]:not([contenteditable="false"])'));
}

// Bridge to the renderer's frame-time movement, not a second movement engine.
export function keyboardNavigation({ win, doc, fire, canMove }) {
    const held = new Set();
    const clear = () => {
        for (const code of held) fire(`camera.fly.${directions[code]}`, false);
        held.clear();
        fire('camera.modifier.fast', false);
    };
    const down = event => {
        if (event.isComposing || event.ctrlKey || event.metaKey || event.altKey ||
            isEditing(event.target) || !canMove()) {
            clear();
            return;
        }
        if (event.key === 'Shift') fire('camera.modifier.fast', true);
        if (!directions[event.code]) return;
        event.preventDefault();
        fire('camera.modifier.fast', event.shiftKey);
        // A key held across a job/focus change must be released and pressed again.
        if (event.repeat || held.has(event.code)) return;
        held.add(event.code);
        fire(`camera.fly.${directions[event.code]}`, true);
    };
    const up = event => {
        if (held.delete(event.code)) fire(`camera.fly.${directions[event.code]}`, false);
        if (event.key === 'Shift') fire('camera.modifier.fast', false);
    };
    const focus = event => { if (isEditing(event.target)) clear(); };
    const visibility = () => { if (doc.hidden) clear(); };
    doc.addEventListener('keydown', down);
    doc.addEventListener('keyup', up);
    doc.addEventListener('focusin', focus);
    doc.addEventListener('visibilitychange', visibility);
    win.addEventListener('blur', clear);
    return { clear, destroy() {
        clear();
        doc.removeEventListener('keydown', down);
        doc.removeEventListener('keyup', up);
        doc.removeEventListener('focusin', focus);
        doc.removeEventListener('visibilitychange', visibility);
        win.removeEventListener('blur', clear);
    } };
}
