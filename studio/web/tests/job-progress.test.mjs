import assert from 'node:assert/strict';
import test from 'node:test';
import { jobProgress, phase, progressDisplay, progressSteps, renderProgress } from '../src/job-progress.mjs';

test('ten steps include export, input, image API, all six original stages, and loading', () => {
    assert.equal(progressSteps.length, 10);
    assert.equal(progressDisplay(phase(0, 'export')).completed, 0);
    assert.equal(jobProgress({status:'uploading', stage:0}).index, 1);
    assert.equal(jobProgress({status:'validating', stage:0}).index, 1);
    assert.equal(jobProgress({status:'image_api', stage:0}).index, 2);
    assert.equal(jobProgress({status:'pipeline', stage:0, image_ready:true}).index, 3);
    for (let stage=1;stage<=6;stage++) {
        const current = jobProgress({status:'pipeline', stage, image_ready:true});
        assert.equal(current.index, stage+2);
        assert.equal(progressDisplay(current).completed, stage+2);
        assert.equal(progressDisplay(current).rows[current.index].state, 'running');
    }
});

test('backend success is 9/10, only loaded output is 10/10; loading can fail', () => {
    const loading = jobProgress({status:'succeeded', stage:6});
    assert.equal(progressDisplay(loading).completed, 9);
    assert.equal(progressDisplay(phase(loading.index, 'download failed', 'failed')).completed, 9);
    const done = progressDisplay(phase(9, 'loaded', 'complete'));
    assert.equal(done.completed, 10);
    assert.ok(done.rows.every(row=>row.state==='done'));
});

test('failures distinguish input, API, geometry and final validation', () => {
    for (const [fields, index] of [
        [{stage:0, failed_phase:'validating'}, 1],
        [{stage:0, failed_phase:'image_api'}, 2],
        [{stage:0, can_retry_download:true}, 2],
        [{stage:0, image_ready:true}, 3],
        [{stage:3, image_ready:true}, 5],
        [{stage:6, image_ready:true}, 8]
    ]) {
        const display = progressDisplay(jobProgress({status:'failed', message:'error', ...fields}));
        assert.equal(display.completed, index);
        assert.equal(display.rows[index].state, 'failed');
        assert.ok(display.rows.slice(index+1).every(row=>row.state==='pending'));
    }
});

test('retry and recovery never fake completion or automatically request generation', () => {
    assert.equal(jobProgress({status:'validating',stage:0,image_ready:true}).index,3);
    const recovering=jobProgress({status:'recovering',stage:3,image_ready:true});
    assert.equal(recovering.mode,'waiting');
    assert.equal(recovering.index,5);
    const unknown=jobProgress({status:'unknown'},phase(2,'API'));
    assert.equal(unknown.index,2);
    assert.equal(unknown.mode,'waiting');
});

test('renderer exposes accessible step states, bounded meter and error text without HTML injection', () => {
    class Element {
        dataset={}; attributes={}; children=[];
        setAttribute(key,value){this.attributes[key]=value;}
        replaceChildren(...items){this.children=items;}
        append(...items){this.children.push(...items);}
    }
    const elements = new Map(['job-progress','progress-heading','progress-count','progress-detail','progress','progress-steps'].map(id=>[id,new Element()]));
    const doc = {getElementById:id=>elements.get(id),createElement:()=>new Element()};
    renderProgress(doc,phase(5,'<img onerror=bad()>','failed'));
    assert.equal(elements.get('progress').value,5);
    assert.equal(elements.get('progress').max,10);
    assert.equal(elements.get('progress-detail').textContent,'<img onerror=bad()>');
    assert.equal(elements.get('progress-steps').children.length,10);
    assert.equal(elements.get('progress-steps').children[5].attributes['aria-current'],'step');
    renderProgress(doc,null);
    assert.equal(elements.get('job-progress').hidden,true);
});
