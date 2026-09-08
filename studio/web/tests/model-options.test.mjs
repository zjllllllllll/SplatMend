import assert from 'node:assert/strict';
import test from 'node:test';
import { requestOptions } from '../src/model-options.mjs';

const gpt = { input_transport: 'oss_url', sizes: [[2048, 1152], [2048, 1536], [2048, 2048]] };
const gemini = { input_transport: 'oss_url', sizes: [[2752, 1536], [2400, 1792], [2048, 2048]] };

test('changing the model changes output size without changing original dimensions', () => {
    assert.equal(requestOptions(gpt, 2560, 1440, true).size, '2048x1152');
    assert.equal(requestOptions(gemini, 2560, 1440, true).size, '2752x1536');
    assert.equal(requestOptions(gpt, 2048, 1536, true).size, '2048x1536');
    assert.equal(requestOptions(gemini, 2048, 1536, true).size, '2400x1792');
    assert.ok(requestOptions(gemini, 2048, 1536, true).ready);
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
