// Pure, browser-independent parts of the user's registered export format.
export const matrixRows = (matrix) => {
    const a = Array.from(matrix.data);
    return [0, 1, 2, 3].map(r => [a[r], a[r + 4], a[r + 8], a[r + 12]]);
};

export function npy(values, shape) {
    if (!(values instanceof Float32Array) || shape.reduce((a, b) => a * b, 1) !== values.length) {
        throw new Error('NPY shape does not match the Float32 payload.');
    }
    if (new Uint8Array(new Uint16Array([1]).buffer)[0] !== 1) throw new Error('Little-endian browser required.');
    const shapeText = shape.join(', ') + (shape.length === 1 ? ',' : '');
    const dict = `{'descr': '<f4', 'fortran_order': False, 'shape': (${shapeText}), }`;
    const padding = (16 - ((10 + dict.length + 1) % 16)) % 16;
    const header = new TextEncoder().encode(dict + ' '.repeat(padding) + '\n');
    const prefix = new Uint8Array(10);
    prefix.set([0x93, 0x4e, 0x55, 0x4d, 0x50, 0x59, 1, 0]);
    new DataView(prefix.buffer).setUint16(8, header.length, true);
    return new Blob([prefix, header, values]);
}

export function halfToFloat(h) {
    const sign = (h & 0x8000) ? -1 : 1;
    const exponent = (h >> 10) & 31;
    const mantissa = h & 1023;
    if (!exponent) return sign * 2 ** -14 * (mantissa / 1024);
    if (exponent === 31) return mantissa ? NaN : sign * Infinity;
    return sign * 2 ** (exponent - 15) * (1 + mantissa / 1024);
}
