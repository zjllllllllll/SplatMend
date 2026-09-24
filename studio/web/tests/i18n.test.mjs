import assert from 'node:assert/strict';
import test from 'node:test';
import { readFileSync } from 'node:fs';
import { getLanguage, setLanguage, t, translateDom } from '../src/i18n.mjs';
import { phase, progressDisplay, renderProgress } from '../src/job-progress.mjs';
import { requestOptions } from '../src/model-options.mjs';

test('fresh UI defaults to English and switching localizes static and dynamic labels', () => {
    setLanguage('en');
    assert.equal(getLanguage(), 'en');
    assert.equal(t('Open PLY file'), 'Open PLY file');
    assert.match(progressDisplay(phase(0, 'Exporting…')).headline, /Lock view and export/);
    setLanguage('zh');
    assert.equal(t('Open PLY file'), '打开 PLY 文件');
    assert.match(progressDisplay(phase(0, 'Exporting…')).headline, /锁定视角并导出/);
    assert.match(requestOptions({ id: 'gpt-image-2', sizes: [[1024, 1024]] }, 1024, 1024, true).message, /原图/);
    const node = { dataset: { i18n: 'Scene repair' }, textContent: '' };
    const doc = { documentElement: { lang: 'en' }, querySelectorAll: selector => selector === '[data-i18n]' ? [node] : [] };
    translateDom(doc);
    assert.equal(doc.documentElement.lang, 'zh-CN');
    assert.equal(node.textContent, '场景补洞');
    setLanguage('en');
    translateDom(doc);
    assert.equal(doc.documentElement.lang, 'en');
    assert.equal(node.textContent, 'Scene repair');
});

test('progress detail and counts update when language switches without changing job state', () => {
    class Element {
        dataset = {}; children = [];
        setAttribute() {}
        replaceChildren(...items) { this.children = items; }
        append(...items) { this.children.push(...items); }
    }
    const ids = ['job-progress', 'progress-heading', 'progress-count', 'progress-detail', 'progress', 'progress-steps'];
    const elements = new Map(ids.map(id => [id, new Element()]));
    const doc = { getElementById: id => elements.get(id), createElement: () => new Element() };
    const state = phase(2, 'Repairing the view with the selected image model.');
    setLanguage('en');
    renderProgress(doc, state);
    assert.match(elements.get('progress-detail').textContent, /Repairing/);
    setLanguage('zh');
    renderProgress(doc, state);
    assert.match(elements.get('progress-detail').textContent, /修复视图/);
    assert.match(elements.get('progress-count').textContent, /已完成/);
    setLanguage('en');
});

test('every static user-facing translation key has Chinese text', () => {
    const html = readFileSync(new URL('../public/index.html', import.meta.url), 'utf8');
    const keys = [...html.matchAll(/data-i18n(?:-aria|-alt|-content)?="([^"]+)"/g)].map(match => match[1]);
    setLanguage('zh');
    assert.ok(keys.length > 30);
    assert.deepEqual(keys.filter(key => t(key) === key), []);
    setLanguage('en');
});