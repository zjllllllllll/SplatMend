import assert from 'node:assert/strict';
import test from 'node:test';
import { restoredDeletionOp } from '../src/restore-deletion.mjs';
import { rememberedJob } from '../src/job-navigation.mjs';

test('restored deletion can be undone and redone without touching unrelated bits', async () => {
    const data = new Uint8Array([0, 1, 2, 0]);
    const ranges = [0, 3];
    const updates = [];
    const splat = { state: {
        setBits: (indices, bits) => indices.forEach(i => { data[i] |= bits; }),
        clearBits: (indices, bits) => indices.forEach(i => { data[i] &= ~bits; })
    }, updateState: async bits => { updates.push(bits); } };
    const op = restoredDeletionOp(splat, ranges, 4);
    await op.do();
    assert.deepEqual([...data], [4, 1, 2, 4]);
    await op.undo();
    assert.deepEqual([...data], [0, 1, 2, 0]);
    await op.do();
    assert.deepEqual([...data], [4, 1, 2, 4]);
    assert.deepEqual(updates, [4, 4, 4]);
    op.destroy();
    assert.equal(op.splat, null);
});

test('active job wins, explicit task link wins over stale session, invalid link fails', () => {
    const a = 'a'.repeat(32), b = 'b'.repeat(32), c = 'c'.repeat(32);
    assert.equal(rememberedJob(a, `?job=${b}`, c), a);
    assert.equal(rememberedJob(null, `?job=${b}`, c), b);
    assert.equal(rememberedJob(null, '', c), c);
    assert.equal(rememberedJob(null, '', null), null);
    assert.throws(() => rememberedJob(null, '?job=../../api_key.txt', null), /invalid/);
});
