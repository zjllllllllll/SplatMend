import { Color, Quat, Vec3, createGraphicsDevice } from 'playcanvas';
import { WorkerQueue } from '@playcanvas/splat-transform';
import { Scene } from '../vendor/supersplat/src/scene';
import { Splat } from '../vendor/supersplat/src/splat';
import { Events } from '../vendor/supersplat/src/events';
import { CommandQueue } from '../vendor/supersplat/src/command-queue';
import { EditHistory } from '../vendor/supersplat/src/edit-history';
import { getSceneConfig } from '../vendor/supersplat/src/scene-config';
import { registerEditorEvents } from '../vendor/supersplat/src/editor';
import { registerSelectionEvents } from '../vendor/supersplat/src/selection';
import { registerCameraPosesEvents } from '../vendor/supersplat/src/camera-poses';
import { registerTimelineEvents } from '../vendor/supersplat/src/timeline';
import { ToolManager } from '../vendor/supersplat/src/tools/tool-manager';
import { LassoSelection } from '../vendor/supersplat/src/tools/lasso-selection';
import { RectSelection } from '../vendor/supersplat/src/tools/rect-selection';
import { MappedReadFileSystem } from '../vendor/supersplat/src/io';
import { State } from '../vendor/supersplat/src/splat-state';
import { IndexRanges } from '../vendor/supersplat/src/index-ranges';
import { capture, cameraMetadata } from './capture';
import { requestOptions } from './model-options.mjs';
import { restoredDeletionOp } from './restore-deletion.mjs';
import { rememberedJob } from './job-navigation.mjs';
import { isEditing, keyboardNavigation } from './keyboard-navigation.mjs';
import { phase, jobProgress, renderProgress } from './job-progress.mjs';

type SceneInfo = { id: string, name: string, count: number, file: string };
type ModelProfile = { id: string, label: string, sizes: [number, number][], input_transport: string, response_format: string };
type JobView = { scene: SceneInfo, model: string, prompt: string,
    viewer_camera: ReturnType<Scene['camera']['docSerialize']> | null,
    camera: ReturnType<typeof cameraMetadata> };
type JobState = { status: string, message: string, stage: number, image_ready?: boolean,
    log_ready?: boolean, can_retry_download?: boolean, can_retry_pipeline?: boolean,
    directory: string, result?: string, result_scene?: SceneInfo };

const el = <T extends HTMLElement = HTMLElement>(id: string) => document.getElementById(id) as T;
const button = (id: string) => el<HTMLButtonElement>(id);
const status = (message: string, error = false) => {
    el('status').textContent = message;
    el('status').classList.toggle('error', error);
};
let token = '';
let busy = false;
let apiReady = false;
let modelProfiles: ModelProfile[] = [];
let keysReady: Record<string, boolean> = {};
let active: Splat = null;
let source: SceneInfo = null;
let sourceBlob: Blob = null;
let scene: Scene;
let history: EditHistory;
let tools: ToolManager;
let lastJobId: string = null;
let navigation: ReturnType<typeof keyboardNavigation>;
let currentProgress: ReturnType<typeof phase> = null;

function showProgress(next: ReturnType<typeof phase>, reveal = false) {
    currentProgress = next;
    renderProgress(document, next);
    if (reveal) el('job-progress').closest('aside').scrollTop = 0;
}

function failProgress(message: string) {
    if (currentProgress) showProgress(phase(currentProgress.index, message, 'failed'));
}

function forgetJob() {
    lastJobId = null;
    sessionStorage.removeItem('repair-studio-job');
    window.history.replaceState(null, '', '/');
    el('retry-download').hidden = true;
    el('retry-pipeline').hidden = true;
    showProgress(null);
    el('job-links').hidden = true;
}

async function api(url: string, method = 'GET', body?: BodyInit, type?: string) {
    const response = await fetch(url, { method, body, headers: {
        ...(method === 'GET' ? {} : { 'X-Studio-Token': token }),
        ...(type ? { 'Content-Type': type } : {})
    } });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || `Local request failed (${response.status}).`);
    return result;
}

function controls() {
    const editable = Boolean(active) && !busy;
    const options = modelOptions();
    for (const id of ['orbit', 'lasso', 'rect', 'clear', 'frame']) button(id).disabled = !editable;
    button('delete').disabled = !editable || active.numSelected === 0;
    button('undo').disabled = !editable || !history?.canUndo();
    button('repair').disabled = !editable || !apiReady || !options.ready;
    button('open').disabled = busy;
    button('open-empty').disabled = busy;
    button('sample').disabled = busy;
    button('retry-download').disabled = busy;
    button('retry-pipeline').disabled = busy;
    el<HTMLSelectElement>('model').disabled = busy || !apiReady;
    el<HTMLSelectElement>('resolution').disabled = busy;
    el<HTMLTextAreaElement>('prompt').disabled = busy;
    el('lock-shield').hidden = !busy || !source;
    if (active) el('counts').textContent = `${active.numSplats.toLocaleString()} 个高斯 · 选中 ${active.numSelected.toLocaleString()}`;
}

function modelOptions() {
    const selected = modelProfiles.find(item => item.id === el<HTMLSelectElement>('model').value);
    const [width, height] = el<HTMLSelectElement>('resolution').value.split(',').map(Number);
    const options = requestOptions(selected, width, height, Boolean(keysReady[selected?.id]));
    el('model-options').textContent = options.message;
    el('model-options').classList.toggle('error', !options.ready);
    return options;
}

function setBusy(value: boolean) {
    busy = value;
    if (value) { navigation?.clear(); tools?.activate(null); }
    controls();
}

async function loadAsset(blob: Blob, deleted?: Uint8Array, preserveView = false) {
    await scene.commandQueue.enqueue((): void => {});
    const cameraState = preserveView ? scene.camera.docSerialize() : null;
    await history.clear();
    scene.clear();
    active = null;
    const filesystem = new MappedReadFileSystem();
    filesystem.addFile('scene.ply', blob);
    // Keep source row ordering so deleted.bin indexes the server's original PLY.
    const loaded = await scene.assetLoader.load('scene.ply', filesystem, false, true);
    if (!loaded) throw new Error('Unable to open this Gaussian scene.');
    await scene.add(loaded);
    active = loaded;
    // Resolve scene bounds before setting distance; this initializes sceneRadius.
    void scene.bound;
    if (deleted) {
        if (deleted.length !== active.state.data.length) throw new Error('Saved selection no longer matches the scene.');
        if (deleted.some(value => value !== 0 && value !== 1)) throw new Error('Saved deletion mask is invalid.');
        if (deleted.some(value => value === 1)) {
            const ranges = IndexRanges.fromPredicate(deleted.length, i => deleted[i] === 1);
            await history.add(restoredDeletionOp(active, ranges, State.deleted));
        }
    }
    if (cameraState) scene.camera.docDeserialize(cameraState);
    else scene.events.fire('camera.focus');
    el('empty').hidden = true;
    scene.forceRender = true;
    controls();
}

async function openFile(file: File) {
    if (!file || busy) return;
    setBusy(true);
    status('正在读取高斯场景…');
    try {
        const next = await api(`/api/scenes?name=${encodeURIComponent(file.name)}`, 'POST', file, 'application/octet-stream');
        const response = await fetch(next.file);
        if (!response.ok) throw new Error('Unable to retrieve the imported scene.');
        const blob = await response.blob();
        await loadAsset(blob);
        source = next;
        sourceBlob = blob;
        forgetJob();
        el('scene-name').textContent = next.name;
        status('场景已就绪。调整视角，圈选并删除需要移除的区域。');
        el('job-links').hidden = true;
    } catch (error) { status(error.message, true); }
    finally { setBusy(false); }
}

async function openSample() {
    if (busy) return;
    setBusy(true);
    status('正在打开仓库示例，并恢复其已记录的相机视角…');
    try {
        const next = await api('/api/sample', 'POST');
        const response = await fetch(next.file);
        if (!response.ok) throw new Error('Unable to retrieve the sample scene.');
        const blob = await response.blob();
        await loadAsset(blob);
        source = next;
        sourceBlob = blob;
        forgetJob();
        const c = next.camera.camera;
        const position = new Vec3(...c.position_world as [number, number, number]);
        const rotation = new Quat(...c.rotation_world_xyzw as [number, number, number, number]);
        const forward = rotation.transformVector(new Vec3(0, 0, -1));
        scene.camera.fov = c.fov_degrees;
        scene.camera.setPose(position, position.clone().add(forward), 0);
        scene.forceRender = true;
        el('scene-name').textContent = next.name;
        status('示例已就绪，视角与仓库相机记录一致。示例已包含空洞，可直接开始补洞。');
    } catch (error) { status(error.message, true); }
    finally { setBusy(false); }
}

function resizeFrame() {
    const [width, height] = el<HTMLSelectElement>('resolution').value.split(',').map(Number);
    const stage = el('canvas-container').parentElement;
    const usableWidth = stage.clientWidth;
    const usableHeight = stage.clientHeight;
    const displayWidth = Math.min(usableWidth, usableHeight * width / height);
    const container = el('canvas-container');
    container.style.width = `${displayWidth}px`;
    container.style.height = `${displayWidth * height / width}px`;
}

async function startRepair() {
    if (busy || !active || !source) return;
    setBusy(true);
    const savedCamera = scene.camera.docSerialize();
    showProgress(phase(0, '检查配置，然后导出锁定视角的 RGB、深度和相机参数。'), true);
    try {
        // Recheck key status before GPU capture or any job creation.
        const config = await api('/api/config');
        token = config.token;
        if (!config.preflight.key_ready) throw new Error('API 密钥缺失，请配置 ark-key.txt 或 grs-key.txt。');
        modelProfiles = config.models;
        keysReady = config.preflight.keys_ready;
        const options = modelOptions();
        if (!options.ready) throw new Error(options.message);
        status('已锁定视角，正在导出带洞图片、深度和相机参数…');
        const [width, height] = el<HTMLSelectElement>('resolution').value.split(',').map(Number);
        const exported = await capture(scene, active, width, height);
        showProgress(phase(1, '正在保存带洞图片、深度、删除记录与相机参数…'));
        const request = { scene_id: source.id, model: el<HTMLSelectElement>('model').value,
            prompt: el<HTMLTextAreaElement>('prompt').value, camera: exported.camera, viewer_camera: savedCamera };
        const job = await api('/api/jobs', 'POST', JSON.stringify(request), 'application/json');
        const jobId = job.id;
        // Serial uploads are intentional: bounded memory and unambiguous completion.
        await api(`/api/jobs/${jobId}/input/point_cloud.png`, 'POST', exported.image);
        await api(`/api/jobs/${jobId}/input/point_cloud.depth.npy`, 'POST', exported.depth);
        await api(`/api/jobs/${jobId}/input/deleted.bin`, 'POST', exported.deleted);
        await followJob(jobId, 'start');
    } catch (error) { failProgress(error.message); status(error.message, true); }
    finally { setBusy(false); }
}

function restoreCamera(view: JobView) {
    // docSerialize().distance is normalized by sceneRadius. Appending a patch
    // can change that radius: restore the ACTUAL locked world-space pose instead.
    const c = view.camera.camera;
    const position = new Vec3(...c.position_world as [number, number, number]);
    const rotation = new Quat(...c.rotation_world_xyzw as [number, number, number, number]);
    scene.camera.fov = c.fov_degrees;
    if (view.viewer_camera) scene.camera.tonemapping = view.viewer_camera.tonemapping;
    scene.camera.setPose(position, position.clone().add(rotation.transformVector(new Vec3(0, 0, -1))), 0);
    scene.forceRender = true;
}

async function followJob(jobId: string, action?: 'start' | 'retry-download' | 'retry-pipeline') {
    setBusy(true);
    showProgress(phase(action === 'retry-pipeline' ? 3 : action === 'retry-download' ? 2 : 1,
        '正在读取同一个任务的状态，不会重复生图…', 'waiting'), true);
    lastJobId = jobId;
    sessionStorage.setItem('repair-studio-job', jobId);
    window.history.replaceState(null, '', `/?job=${encodeURIComponent(jobId)}`);
    let view: JobView;
    let released = false;
    try {
        const config = await api('/api/config');
        token = config.token;
        view = await api(`/api/jobs/${jobId}/view`);
        el('scene-name').textContent = view.scene.name;
        // Retired historical models remain in task records, not in the selector.
        // An unsupported value selects nothing and blocks a new job until chosen.
        el<HTMLSelectElement>('model').value = view.model;
        el<HTMLTextAreaElement>('prompt').value = view.prompt;
        el<HTMLSelectElement>('resolution').value = `${view.camera.image.width},${view.camera.image.height}`;
        resizeFrame();
        el('retry-download').hidden = true;
        el('retry-pipeline').hidden = true;
        el<HTMLAnchorElement>('source-link').href = `/api/jobs/${jobId}/source.png`;
        el<HTMLAnchorElement>('task-link').href = `/?job=${encodeURIComponent(jobId)}`;
        el<HTMLAnchorElement>('repaired-link').href = `/api/jobs/${jobId}/repaired.png`;
        el<HTMLAnchorElement>('log-link').href = `/api/jobs/${jobId}/pipeline.log`;
        el('repaired-link').hidden = true;
        el('log-link').hidden = true;
        el('result-link').hidden = true;
        el('job-links').hidden = false;
        el<HTMLImageElement>('locked-view').src = `/api/jobs/${jobId}/source.png`;
        el('locked-view').hidden = false;
        el('empty').hidden = true;
        await history.clear();
        scene.clear(); // Release GPU resources for the unchanged LingBot / SHARP pipeline.
        active = null;
        released = true;
        controls();
        el('lock-shield').hidden = false;
        if (action) {
            try { await api(`/api/jobs/${jobId}/${action}`, 'POST'); }
            catch {
                // An interrupted response does not prove submission failed. Read
                // the same task; never automatically submit or generate twice.
                status('正在确认同一个任务的提交状态，不会重复生图…');
            }
        }
        let state: JobState;
        for (;;) {
            try { state = await api(`/api/jobs/${jobId}`); }
            catch {
                status('本地连接暂时中断，正在重新读取同一个任务；不会重复调用图片 API。', true);
                showProgress(phase(currentProgress.index, '连接暂时中断，正在重新确认同一个任务的进度。', 'waiting'));
                await new Promise(resolve => setTimeout(resolve, 3000));
                continue;
            }
            status(state.message, state.status === 'failed');
            showProgress(jobProgress(state, currentProgress));
            el('repaired-link').hidden = !state.image_ready;
            el('log-link').hidden = !state.log_ready;
            if (state.status === 'uploading') throw new Error('任务尚未启动；未调用图片 API。输入已保留，可以重新开始。');
            if (state.status === 'succeeded' || state.status === 'failed') break;
            await new Promise(resolve => setTimeout(resolve, 1500));
        }
        el('retry-download').hidden = !state.can_retry_download || !modelProfiles.some(item => item.id === view.model);
        el('retry-pipeline').hidden = !state.can_retry_pipeline;
        if (state.status === 'failed') throw new Error(`${state.message} 任务目录：${state.directory}`);
        el<HTMLAnchorElement>('result-link').href = state.result;
        el('result-link').hidden = false;
        const response = await fetch(state.result);
        if (!response.ok) throw new Error('Repair succeeded, but the result could not be downloaded. Use the result link to retry.');
        const resultBlob = await response.blob();
        el<HTMLAnchorElement>('result-link').href = state.result;
        el('result-link').hidden = false;
        // Register the accepted result as a new input for any subsequent edit.
        const resultSource = state.result_scene || await api('/api/scenes?name=repaired_scene.ply', 'POST', resultBlob, 'application/octet-stream');
        await loadAsset(resultBlob, undefined, true);
        restoreCamera(view);
        source = resultSource;
        sourceBlob = resultBlob;
        released = false;
        el('scene-name').textContent = 'repaired_scene.ply';
        showProgress(phase(9, '六步原链路验收通过，结果已加载，可以下载 PLY。', 'complete'));
        status('补洞完成，六步原链路验收通过。结果已加载，也可以下载 PLY。');
    } catch (error) {
        failProgress(error.message);
        status(error.message, true);
        if (released && view) {
            try {
                if (source?.id !== view.scene.id || !sourceBlob) {
                    const response = await fetch(view.scene.file);
                    if (!response.ok) throw new Error('Original scene download failed.');
                    sourceBlob = await response.blob();
                }
                source = view.scene;
                const response = await fetch(`/api/jobs/${jobId}/deleted.bin`);
                if (!response.ok) throw new Error('Saved deletion mask download failed.');
                const removed = new Uint8Array(await response.arrayBuffer());
                await loadAsset(sourceBlob, removed, true);
                restoreCamera(view);
                el('scene-name').textContent = source.name;
            } catch { status(`${error.message} 原场景文件仍保存在本地，请重新打开。`, true); }
        }
    } finally {
        el('locked-view').hidden = true;
        setBusy(false);
    }
}

async function main() {
    const config = await api('/api/config');
    token = config.token;
    const model = el<HTMLSelectElement>('model');
    modelProfiles = config.models;
    keysReady = config.preflight.keys_ready;
    model.replaceChildren(...config.models.map((item: { id: string, label: string }) => new Option(item.label, item.id)));
    const savedModel = localStorage.getItem('repair-studio-model');
    if (config.models.some((item: { id: string }) => item.id === savedModel)) model.value = savedModel;
    else if (savedModel) localStorage.removeItem('repair-studio-model');
    model.addEventListener('change', () => {
        localStorage.setItem('repair-studio-model', model.value);
        controls();
    });
    el<HTMLTextAreaElement>('prompt').value = config.prompt;
    apiReady = config.preflight.key_ready && config.preflight.missing_files.length === 0;
    el('preflight').textContent = `${config.preflight.key_ready ? 'API 密钥已配置' : '未找到 API 密钥'} · 剩余磁盘 ${config.preflight.disk_free_gb} GB`;
    if (config.preflight.missing_files.length) el('preflight').textContent += ' · 模型或管线文件缺失';
    WorkerQueue.maxWorkers = 0;
    const events = new Events();
    const queue = new CommandQueue();
    history = new EditHistory(events, queue);
    events.function('queue', (fn: () => Promise<void>) => queue.enqueue(fn));
    registerTimelineEvents(events);
    registerCameraPosesEvents(events);
    events.function('polygonSelection.removeLastPoint', () => false);
    events.function('showPopup', async (popup: { message: string }) => { status(popup.message, true); return { action: 'cancel' }; });
    const canvas = el<HTMLCanvasElement>('scene-canvas');
    const device = await createGraphicsDevice(canvas, { deviceTypes: ['webgl2'], antialias: false,
        depth: false, stencil: false, xrCompatible: false, powerPreference: 'high-performance' });
    scene = new Scene(events, getSceneConfig([{ show: { grid: false, bound: false }, camera: { fov: 75 } }]), canvas, device, queue);
    events.function('bgClr', () => new Color(0, 0, 0, 1));
    events.function('selectedClr', () => new Color(0.47, 0.85, 0.69, 0.9));
    events.function('unselectedClr', () => new Color(0, 0, 0, 0));
    events.function('lockedClr', () => new Color(0, 0, 0, 0));
    registerEditorEvents(events, history, scene, { areaSelection: 'through' });
    registerSelectionEvents(events, scene);
    events.fire('camera.setMode', 'rings');
    events.fire('view.setOutlineSelection', true);
    const maskCanvas = document.createElement('canvas');
    maskCanvas.id = 'mask-canvas';
    const maskContext = maskCanvas.getContext('2d');
    maskContext.globalCompositeOperation = 'copy';
    el('tools').appendChild(maskCanvas);
    tools = new ToolManager(events);
    tools.register('lassoSelection', new LassoSelection(events, el('tools'), { canvas: maskCanvas, context: maskContext }));
    tools.register('rectSelection', new RectSelection(events, el('tools')));
    navigation = keyboardNavigation({ win: window, doc: document,
        fire: (name: string, down: boolean) => events.fire(name, down),
        canMove: () => Boolean(active) && !busy && !tools.active });
    events.on('tool.activated', (name: string) => {
        navigation.clear();
        events.fire('camera.setControlMode', 'orbit');
        button('lasso').classList.toggle('active', name === 'lassoSelection');
        button('rect').classList.toggle('active', name === 'rectSelection');
        button('orbit').classList.toggle('active', !name);
    });
    events.on('splat.stateChanged', controls);
    events.on('edit.canUndo', controls);
    for (const name of ['open', 'open-empty']) button(name).onclick = () => el<HTMLInputElement>('file').click();
    el<HTMLInputElement>('file').onchange = async event => {
        const input = event.target as HTMLInputElement;
        await openFile(input.files[0]); input.value = '';
    };
    button('orbit').onclick = () => {
        navigation.clear();
        tools.activate(null);
        events.fire('camera.setControlMode', 'orbit');
    };
    button('lasso').onclick = () => tools.activate('lassoSelection');
    button('rect').onclick = () => tools.activate('rectSelection');
    button('delete').onclick = async () => { events.fire('select.delete'); await queue.enqueue((): void => {}); controls(); };
    button('undo').onclick = async () => { await history.undo(); controls(); };
    button('clear').onclick = () => events.fire('select.none');
    button('frame').onclick = () => events.fire('camera.focus');
    button('repair').onclick = startRepair;
    button('retry-download').onclick = () => {
        if (!busy && lastJobId) void followJob(lastJobId, 'retry-download');
    };
    button('retry-pipeline').onclick = () => {
        if (!busy && lastJobId) void followJob(lastJobId, 'retry-pipeline');
    };
    button('sample').onclick = openSample;
    el<HTMLSelectElement>('resolution').onchange = () => { resizeFrame(); controls(); };
    new ResizeObserver(resizeFrame).observe(el('canvas-container').parentElement);
    document.addEventListener('keydown', event => {
        if (busy || !active || event.isComposing || isEditing(event.target) || event.altKey || event.metaKey) return;
        const actions: Record<string, string> = { v: 'orbit', l: 'lasso', f: 'frame', Delete: 'delete', Escape: 'clear' };
        const id = event.ctrlKey ? (event.key.toLowerCase() === 'z' ? 'undo' : null) : actions[event.key.length === 1 ? event.key.toLowerCase() : event.key];
        if (id && !button(id).disabled) { event.preventDefault(); button(id).click(); }
    });
    // Keep camera input and wheel events from changing the view while exporting / running.
    for (const type of ['pointerdown', 'pointermove', 'pointerup', 'wheel', 'dblclick']) {
        el('canvas-container').addEventListener(type, event => {
            if (busy) { event.stopImmediatePropagation(); event.preventDefault(); }
        }, { capture: true, passive: false });
    }
    scene.start();
    controls();
    status(apiReady ? '打开 PLY 文件开始。' : '请检查模型文件和 API 密钥，然后刷新页面。');
    const restoreId = rememberedJob(config.active_job, window.location.search, sessionStorage.getItem('repair-studio-job'));
    if (restoreId) await followJob(restoreId);
}

main().catch(error => { status(`启动失败：${error.message}`, true); });
