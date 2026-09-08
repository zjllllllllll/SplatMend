// Registered export derived from the user's SuperSplat unified export hook.
import { Scene } from '../vendor/supersplat/src/scene';
import { Splat } from '../vendor/supersplat/src/splat';
import { encodePng } from '../vendor/supersplat/src/png-writer';
import { State } from '../vendor/supersplat/src/splat-state';
import { matrixRows, npy, halfToFloat } from './export-math.mjs';
const multiply = (a: number[][], b: number[][]) => a.map((row) => b[0].map((_, col) => row.reduce((n, v, k) => n + v * b[k][col], 0)));
const glToCv = [[1, 0, 0, 0], [0, -1, 0, 0], [0, 0, -1, 0], [0, 0, 0, 1]];

export function cameraMetadata(scene: Scene, splat: Splat, width: number, height: number) {
    const camera = scene.camera;
    const projection = matrixRows(camera.camera.projectionMatrix);
    const view = matrixRows(camera.camera.viewMatrix);
    const inverse = matrixRows(camera.worldTransform);
    const cvView = multiply(glToCv, view);
    const model = matrixRows(splat.worldTransform);
    const position = camera.position;
    const rotation = camera.mainCamera.getRotation();
    return {
        schema_version: 2, generator: 'Gaussian Repair Studio / SuperSplat unified export',
        image: { width, height, format: 'png', projection: 'standard', transparent_background: true, show_debug: false },
        camera: { projection_type: 'perspective', fov_degrees: camera.fov,
            fov_axis: camera.camera.horizontalFov ? 'horizontal' : 'vertical', near: camera.near, far: camera.far,
            position_world: [position.x, position.y, position.z],
            rotation_world_xyzw: [rotation.x, rotation.y, rotation.z, rotation.w] },
        intrinsic_K_opencv: [[Math.abs(projection[0][0]) * width / 2, 0, width / 2],
            [0, Math.abs(projection[1][1]) * height / 2, height / 2], [0, 0, 1]],
        matrices: { projection_opengl: projection, world_to_camera_opengl: view,
            camera_to_world_opengl: inverse, world_to_camera_opencv: cvView,
            camera_to_world_opencv: multiply(inverse, glToCv) },
        splats: [{ name: splat.name, filename: 'point_cloud.ply', splat_count: splat.numSplats,
            splat_to_world_opengl: model, model_view_opencv_for_original_splat: multiply(cvView, model),
            has_per_splat_palette_transforms: false }],
        depth: { file: 'point_cloud.depth.npy', shape: [height, width, 2],
            channels: ['absolute_camera_z', 'alpha'], representation: 'Linear camera-space forward Z in scene units' }
    };
}

function renderedFrame(scene: Scene): Promise<void> {
    return new Promise((resolve, reject) => {
        const handle = scene.events.on('postrender', () => { clearTimeout(timer); handle.off(); resolve(); });
        const timer = setTimeout(() => { handle.off(); reject(new Error('The locked view did not render within 20 seconds.')); }, 20000);
        scene.forceRender = true;
    });
}

export async function capture(scene: Scene, splat: Splat, width: number, height: number) {
    await scene.commandQueue.enqueue((): void => {});
    if (scene.camera.ortho) throw new Error('Repair requires a perspective view.');
    const transform = splat.splatData.getProp('transform') as Uint16Array;
    if (transform?.some(value => value !== 0)) throw new Error('Per-Gaussian transforms are not supported by this pipeline.');
    const deleted = Uint8Array.from(splat.state.data, value => (value & State.deleted) ? 1 : 0);
    if (splat.state.data.some(value => (value & State.locked) !== 0)) throw new Error('Unlock the scene before capture.');
    const camera = scene.camera;
    const previousOverride = camera.poseOverride;
    const pose = { position: camera.position.clone(), rotation: camera.mainCamera.getRotation().clone(),
        fov: camera.fov, near: camera.near, far: camera.far };
    const overlays = camera.renderOverlays;
    const gizmos = scene.gizmoLayer.enabled;
    const enabled = scene.elements.filter(e => e.type === 'splat').map(e => ({ splat: e as Splat, enabled: (e as Splat).entity.enabled }));
    try {
        camera.setPoseOverride(pose);
        camera.startOffscreenMode(width, height);
        camera.renderOverlays = false;
        scene.gizmoLayer.enabled = false;
        await renderedFrame(scene);
        const metadata = cameraMetadata(scene, splat, width, height);
        const rgba = new Uint8Array(width * height * 4);
        scene.dataProcessor.copyRt(camera.mainTarget, camera.workTarget);
        await camera.workTarget.colorBuffer.read(0, 0, width, height, { renderTarget: camera.workTarget, data: rgba });
        if (scene.graphicsDevice.isWebGL2) {
            const row = new Uint8Array(width * 4);
            for (let y = 0; y < Math.floor(height / 2); y++) {
                const a = y * width * 4, b = (height - y - 1) * width * 4;
                row.set(rgba.subarray(a, a + width * 4));
                rgba.copyWithin(a, b, b + width * 4); rgba.set(row, b);
            }
        }
        camera.picker.prepareDepth(splat);
        const rt = camera.colorTarget;
        const pixels = await rt.colorBuffer.read(0, 0, width, height, { renderTarget: rt });
        const depth = new Float32Array(width * height * 2);
        for (let y = 0; y < height; y++) {
            const sy = scene.graphicsDevice.isWebGL2 ? height - y - 1 : y;
            for (let x = 0; x < width; x++) {
                const i = (sy * width + x) * 4, o = (y * width + x) * 2;
                const alpha = 1 - halfToFloat(pixels[i + 3]);
                const z = pose.near + halfToFloat(pixels[i]) / alpha * (pose.far - pose.near);
                const valid = alpha >= 1e-6 && Number.isFinite(z) && z > 0;
                depth[o] = valid ? z : NaN;
                depth[o + 1] = valid ? Math.max(0, Math.min(1, alpha)) : 0;
            }
        }
        return { camera: metadata, image: new Blob([await encodePng(rgba, width, height)], { type: 'image/png' }),
            depth: npy(depth, [height, width, 2]), deleted: new Blob([deleted]) };
    } finally {
        for (const entry of enabled) entry.splat.entity.enabled = entry.enabled;
        camera.setPoseOverride(previousOverride);
        camera.endOffscreenMode();
        camera.renderOverlays = overlays;
        scene.gizmoLayer.enabled = gizmos;
        scene.forceRender = true;
    }
}
