import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';

// Unit-test the actual bundled gesture controller, not a rewritten algorithm.
// This is a Node DOM stub, not browser automation or hidden page-state access.
function loadController(document) {
    const options = { module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022 };
    const compile = path => ts.transpileModule(readFileSync(new URL(path, import.meta.url), 'utf8'), { compilerOptions: options }).outputText;
    const modifiers = {};
    new Function('exports', compile('../vendor/supersplat/src/select-op.ts'))(modifiers);
    const exports = {};
    const require = name => {
        assert.equal(name, '../select-op');
        return modifiers;
    };
    new Function('require', 'exports', 'document', compile('../vendor/supersplat/src/tools/lasso-selection.ts'))(require, exports, document);
    return exports.LassoSelection;
}

function fixture() {
    const elements = [];
    const document = { createElementNS: () => {
        const attrs = {};
        const element = { classList: { add() {}, remove() {} }, attrs,
            namespaceURI: 'http://www.w3.org/2000/svg', appendChild() {},
            setAttribute: (name, value) => { attrs[name] = value; } };
        elements.push(element);
        return element;
    } };
    const listeners = new Map();
    const captured = new Set();
    const parent = { clientWidth: 320, clientHeight: 180, style: {}, appendChild() {},
        addEventListener: (type, callback) => listeners.set(type, callback),
        removeEventListener: type => listeners.delete(type),
        setPointerCapture: id => captured.add(id),
        hasPointerCapture: id => captured.has(id),
        releasePointerCapture: id => captured.delete(id) };
    const path = [], calls = [];
    const canvas = { width: 0, height: 0 };
    const context = { clearRect() {}, beginPath() { path.length = 0; },
        moveTo: (x, y) => path.push([x, y]), lineTo: (x, y) => path.push([x, y]), closePath() {}, fill() {} };
    const events = { invoke: async (...args) => { calls.push({ args, points: path.map(p => [...p]) }); } };
    const Controller = loadController(document);
    const tool = new Controller(events, parent, { canvas, context });
    const send = async (type, x, y, modifiers = {}) => {
        const handler = listeners.get(type);
        if (handler) await handler({ pointerType: 'mouse', button: 0, pointerId: 1,
            offsetX: x, offsetY: y, isPrimary: true, shiftKey: false, ctrlKey: false,
            preventDefault() {}, stopPropagation() {}, ...modifiers });
    };
    return { tool, canvas, context, calls, captured, elements, send, events };
}

async function square(f, offset = 10, modifiers = {}) {
    await f.send('pointerdown', offset, offset);
    await f.send('pointermove', offset + 30, offset);
    await f.send('pointermove', offset + 30, offset + 30);
    await f.send('pointermove', offset, offset + 30);
    await f.send('pointerup', offset, offset, modifiers);
}

test('freehand stroke sends the actual polygon mask with modifier semantics', async () => {
    for (const [modifiers, operation] of [[{}, 'set'], [{ shiftKey: true }, 'add'], [{ ctrlKey: true }, 'remove'], [{ shiftKey: true, ctrlKey: true }, 'intersect']]) {
        const f = fixture();
        f.tool.activate();
        await square(f, 10, modifiers);
        assert.equal(f.calls.length, 1);
        assert.deepEqual(f.calls[0].points, [[10, 10], [40, 10], [40, 40], [10, 40]]);
        assert.deepEqual(f.calls[0].args, ['select.byMask', operation, f.canvas, f.context]);
        assert.deepEqual([f.canvas.width, f.canvas.height], [320, 180]);
        assert.equal(f.captured.size, 0);
    }
});

test('switching away mid-stroke cannot contaminate the next lasso', async () => {
    const f = fixture();
    f.tool.activate();
    await f.send('pointerdown', 10, 10);
    await f.send('pointermove', 40, 10);
    f.tool.deactivate();
    f.tool.activate();
    await square(f, 100);
    assert.deepEqual(f.calls[0].points, [[100, 100], [130, 100], [130, 130], [100, 130]]);
});

test('cancelled pointer stroke is discarded without selecting anything', async () => {
    const f = fixture();
    f.tool.activate();
    await f.send('pointerdown', 10, 10);
    await f.send('pointermove', 40, 10);
    await f.send('pointercancel', 40, 10);
    assert.equal(f.captured.size, 0);
    assert.equal(f.calls.length, 0);
    await square(f, 100);
    assert.deepEqual(f.calls[0].points, [[100, 100], [130, 100], [130, 130], [100, 130]]);
});

test('old asynchronous completion cannot release a newer gesture', async () => {
    const f = fixture();
    let finish;
    f.events.invoke = () => new Promise(resolve => { finish = resolve; });
    f.tool.activate();
    await f.send('pointerdown', 10, 10);
    await f.send('pointermove', 40, 10);
    const committing = f.send('pointerup', 40, 10);
    f.tool.deactivate();
    f.tool.activate();
    await f.send('pointerdown', 100, 100);
    finish();
    await committing;
    assert.equal(f.captured.size, 1);
    await f.send('pointercancel', 100, 100);
    assert.equal(f.captured.size, 0);
});
