import test from 'node:test';
import assert from 'node:assert/strict';
import { matrixRows, npy, halfToFloat } from '../src/export-math.mjs';

test('column-major PlayCanvas matrices are presented as mathematical rows', () => {
    assert.deepEqual(matrixRows({ data: Array.from({ length: 16 }, (_, i) => i) }),
        [[0, 4, 8, 12], [1, 5, 9, 13], [2, 6, 10, 14], [3, 7, 11, 15]]);
});
test('half-float including subnormals and invalid depth', () => {
    assert.equal(halfToFloat(0), 0);
    assert.equal(halfToFloat(1), 2 ** -24);
    assert.equal(halfToFloat(0x3c00), 1);
    assert.equal(halfToFloat(0xbc00), -1);
    assert.equal(halfToFloat(0x7c00), Infinity);
    assert.ok(Number.isNaN(halfToFloat(0x7e00)));
});
test('NPY header, shape, alignment and float data are exact', async () => {
    const values = new Float32Array([1.25, .5, NaN, 0]);
    const raw = await npy(values, [1, 2, 2]).arrayBuffer();
    const view = new DataView(raw);
    const headerSize = view.getUint16(8, true);
    assert.equal((headerSize + 10) % 16, 0);
    const header = new TextDecoder().decode(new Uint8Array(raw, 10, headerSize));
    assert.match(header, /'descr': '<f4'/);
    assert.match(header, /'shape': \(1, 2, 2\)/);
    assert.equal(view.getFloat32(10 + headerSize, true), 1.25);
    assert.equal(view.getFloat32(14 + headerSize, true), .5);
    assert.ok(Number.isNaN(view.getFloat32(18 + headerSize, true)));
    assert.equal(view.getFloat32(22 + headerSize, true), 0);
});
test('NPY handles subarrays and refuses mismatched shapes', async () => {
    const values = new Float32Array([99, 1, 2, 99]).subarray(1, 3);
    const raw = await npy(values, [2]).arrayBuffer();
    const view = new DataView(raw);
    const offset = view.getUint16(8, true) + 10;
    assert.equal(raw.byteLength - offset, 8);
    assert.equal(view.getFloat32(offset, true), 1);
    assert.throws(() => npy(values, [5]), /shape/);
});
