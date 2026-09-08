import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import test from 'node:test';
import ts from 'typescript';
import * as playcanvas from 'playcanvas';

// Actual editor, history, state and delete operations. Only GPU readback and
// unrelated editor services are stubs; this is not a browser/runtime hook.
function modules() {
    const cache = new Map();
    const actual = new Set(['editor', 'events', 'command-queue', 'edit-history',
        'edit-ops', 'index-ranges', 'splat-state']);
    class MaskTexture {
        setSource(canvas) { this.width = canvas.width; this.height = canvas.height; }
        destroy() {}
    }
    function load(name) {
        if (cache.has(name)) return cache.get(name);
        const exports = {};
        cache.set(name, exports);
        const source = readFileSync(new URL(`../vendor/supersplat/src/${name}.ts`, import.meta.url), 'utf8');
        const compiled = ts.transpileModule(source, { compilerOptions: {
            module: ts.ModuleKind.CommonJS, target: ts.ScriptTarget.ES2022
        } }).outputText;
        const require = path => {
            if (path === 'playcanvas') return { ...playcanvas, Texture: MaskTexture };
            const relative = path.replace(/^\.\//, '');
            if (actual.has(relative)) return load(relative);
            if (relative === 'element') return { ElementType: { splat: 'splat' } };
            return {};
        };
        new Function('require', 'exports', 'window', compiled)(require, exports, { addEventListener() {} });
        return exports;
    }
    return load;
}

const load = modules();
const { Events } = load('events');
const { CommandQueue } = load('command-queue');
const { EditHistory } = load('edit-history');
const { State, SplatState } = load('splat-state');
const { registerEditorEvents } = load('editor');

function fixture(areaSelection = 'through') {
    const events = new Events();
    events.function('tool.active', () => null);
    events.function('polygonSelection.removeLastPoint', () => false);
    const queue = new CommandQueue();
    const history = new EditHistory(events, queue);
    // 0 = front hit, 1 = occluded hit, 2/5 = outside. 3/4 cannot be edited.
    const data = new Uint8Array([0, 0, 0, State.locked, State.deleted, 0]);
    const state = new SplatState(data, { lock: () => new Uint8Array(data.length), unlock() {} });
    state.flush();
    const splat = { visible: true, state,
        splatData: { numSplats: data.length, getProp: name => name === 'state' ? data : null },
        updateState: async () => state.flush() };
    events.function('selection', () => splat);
    const calls = [];
    let hits = new Uint8Array([255, 255, 0, 255, 255, 0]);
    const scene = { commandQueue: queue, config: { show: {}, camera: {} },
        grid: { visible: false }, camera: { fov: 75, pickPrep() { calls.push('visible-prep'); },
            async pickRect() { calls.push('visible-pick'); return new Uint32Array([0, 0, 0, 0]); } },
        graphicsDevice: {}, targetSize: { width: 2, height: 2 },
        dataProcessor: { async intersect(options) { calls.push(options); return hits.slice(); },
            releaseMask(mask) { mask.fill(0); } } };
    registerEditorEvents(events, history, scene, { areaSelection });
    events.fire('camera.setMode', 'rings');
    events.fire('view.setOutlineSelection', true);
    const canvas = { width: 2, height: 2 };
    const context = { getImageData: () => ({ width: 2, height: 2, data: new Uint8Array(16).fill(255) }) };
    async function select(kind, op = 'set') {
        if (kind === 'mask') await events.invoke('select.byMask', op, canvas, context);
        else await events.invoke('select.rect', op, { start: { x: 0, y: 0 }, end: { x: 1, y: 1 } });
        await queue.enqueue(() => {});
    }
    return { events, queue, history, data, state, calls, select, setHits: value => { hits = new Uint8Array(value); } };
}

for (const kind of ['mask', 'rect']) {
    test(`${kind}: rings display must still select and delete occluded layers`, async () => {
        const f = fixture();
        await f.select(kind);
        assert.equal(f.events.invoke('camera.mode'), 'rings');
        assert.equal(f.events.invoke('view.outlineSelection'), true);
        assert.equal(f.calls.some(call => typeof call === 'string'), false, 'must not use visible-only picking');
        assert.equal(f.state.numSelected, 2, 'both front and occluded Gaussian must be selected');
        assert.deepEqual([...f.data], [1, 1, 0, 2, 4, 0]);
        f.events.fire('select.delete');
        await f.queue.enqueue(() => {});
        assert.deepEqual([...f.data], [5, 5, 0, 2, 4, 0]);
        await f.history.undo();
        assert.deepEqual([...f.data], [1, 1, 0, 2, 4, 0]);
        await f.history.undo();
        assert.deepEqual([...f.data], [0, 0, 0, 2, 4, 0]);
        await f.history.redo();
        await f.history.redo();
        assert.deepEqual([...f.data], [5, 5, 0, 2, 4, 0]);
    });
}

test('through selection preserves add/remove/intersect and protected state bits', async () => {
    const f = fixture();
    await f.select('mask');
    f.setHits([0, 0, 255, 255, 255, 0]);
    await f.select('rect', 'add');
    assert.deepEqual([...f.data], [1, 1, 1, 2, 4, 0]);
    f.setHits([255, 0, 0, 255, 255, 0]);
    await f.select('mask', 'remove');
    assert.deepEqual([...f.data], [0, 1, 1, 2, 4, 0]);
    f.setHits([0, 255, 0, 255, 255, 0]);
    await f.select('rect', 'intersect');
    assert.deepEqual([...f.data], [0, 1, 0, 2, 4, 0]);
});

test('visible-only mode remains opt-in and point picking is unchanged', async () => {
    const f = fixture('visible');
    await f.select('rect');
    assert.deepEqual([...f.data], [1, 0, 0, 2, 4, 0]);
    const through = fixture();
    await through.events.invoke('select.point', 'set', { x: .5, y: .5 });
    await through.queue.enqueue(() => {});
    assert.deepEqual([...through.data], [1, 0, 0, 2, 4, 0]);
    assert.ok(through.calls.includes('visible-pick'));
});
