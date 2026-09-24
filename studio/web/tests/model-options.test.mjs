import assert from 'node:assert/strict';
import test from 'node:test';
import { requestOptions } from '../src/model-options.mjs';

const gpt = { id: 'gpt-image-2', input_transport: 'data_url', sizes: [[1672, 941], [1443, 1090], [1024, 1024]] };
const seedream = { id: 'doubao-seedream-5-0-260128', input_transport: 'data_url', sizes: [[2848, 1600], [2304, 1728], [2048, 2048]] };

test('changing the model changes output size without changing original dimensions', () => {
    assert.equal(requestOptions(gpt, 2560, 1440, true).size, '1672x941');
    assert.equal(requestOptions(seedream, 2560, 1440, true).size, '2848x1600');
    assert.equal(requestOptions(gpt, 2048, 1536, true).size, '1443x1090');
    assert.equal(requestOptions(seedream, 2048, 1536, true).size, '2304x1728');
    assert.ok(requestOptions(seedream, 2048, 1536, true).ready);
});

test('missing selected provider key blocks its model', () => {
    assert.equal(requestOptions(gpt, 2048, 2048, false).ready, false);
    assert.equal(requestOptions(seedream, 2048, 2048, true).ready, true);
});

test('unsupported ratio and historical model cannot silently become another', () => {
    assert.equal(requestOptions(gpt, 1600, 1000, true).ready, false);
    assert.equal(requestOptions(undefined, 2048, 2048, true).ready, false);
    assert.equal(requestOptions(gpt, 0, 0, true).ready, false);
});
