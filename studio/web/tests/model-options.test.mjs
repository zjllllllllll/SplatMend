import assert from 'node:assert/strict';
import test from 'node:test';
import { requestOptions } from '../src/model-options.mjs';

const gpt = { input_transport: 'oss_url', sizes: [[2048, 1152], [2048, 1536], [2048, 2048]] };
const seedream = { input_transport: 'data_url', sizes: [[2848, 1600], [2304, 1728], [2048, 2048]] };

test('changing the model changes output size without changing original dimensions', () => {
    assert.equal(requestOptions(gpt, 2560, 1440, true).size, '2048x1152');
    assert.equal(requestOptions(seedream, 2560, 1440, true).size, '2848x1600');
    assert.equal(requestOptions(gpt, 2048, 1536, true).size, '2048x1536');
    assert.equal(requestOptions(seedream, 2048, 1536, true).size, '2304x1728');
    assert.ok(requestOptions(seedream, 2048, 1536, true).ready);
});

test('missing OSS blocks only models which require it', () => {
    assert.equal(requestOptions(gpt, 2048, 2048, false).ready, false);
    assert.equal(requestOptions({ ...gpt, input_transport: 'data_url' }, 2048, 2048, false).ready, true);
});

test('unsupported ratio and historical model cannot silently become another', () => {
    assert.equal(requestOptions(gpt, 1600, 1000, true).ready, false);
    assert.equal(requestOptions(undefined, 2048, 2048, true).ready, false);
    assert.equal(requestOptions(gpt, 0, 0, true).ready, false);
});
