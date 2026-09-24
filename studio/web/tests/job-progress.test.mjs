import assert from 'node:assert/strict';
import test from 'node:test';
import { jobProgress, phase, progressDisplay, progressSteps, renderProgress } from '../src/job-progress.mjs';

test('eleven steps include color correction after the image API', () => {
    assert.equal(progressSteps.length, 11);
    assert.equal(progressDisplay(phase(0, 'export')).completed, 0);
    assert.equal(jobProgress({status:'uploading', stage:0}).index, 1);
    assert.equal(jobProgress({status:'validating', stage:0}).index, 1);
    assert.equal(jobProgress({status:'image_api', stage:0}).index, 2);
    assert.equal(jobProgress({status:'color_correction', stage:0, image_ready:true}).index, 3);
    assert.equal(jobProgress({status:'pipeline', stage:0, image_ready:true}).index, 4);
    for (let stage=1;stage<=6;stage++) {
        const current = jobProgress({status:'pipeline', stage, image_ready:true});
        assert.equal(current.index, stage+3);
        assert.equal(progressDisplay(current).completed, stage+3);
        assert.equal(progressDisplay(current).rows[current.index].state, 'running');
    }
});

test('backend success is 10/11, only loaded output is 11/11; loading can fail', () => {
    const loading = jobProgress({status:'succeeded', stage:6});
    assert.equal(progressDisplay(loading).completed, 10);
    assert.equal(progressDisplay(phase(loading.index, 'download failed', 'failed')).completed, 10);
    const done = progressDisplay(phase(10, 'loaded', 'complete'));
    assert.equal(done.completed, 11);
    assert.ok(done.rows.every(row=>row.state==='done'));
});

test('failures distinguish input, API, geometry and final validation', () => {
    for (const [fields, index] of [
        [{stage:0, failed_phase:'validating'}, 1],
        [{stage:0, failed_phase:'image_api'}, 2],
        [{stage:0, can_retry_download:true}, 2],
        [{stage:0, failed_phase:'color_correction', image_ready:true}, 3],
        [{stage:0, image_ready:true}, 4],
        [{stage:3, image_ready:true}, 6],
        [{stage:6, image_ready:true}, 9]
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
    assert.equal(recovering.index,6);
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
    assert.equal(elements.get('progress').max,11);
    assert.equal(elements.get('progress-detail').textContent,'<img onerror=bad()>');
    assert.equal(elements.get('progress-steps').children.length,11);
    assert.equal(elements.get('progress-steps').children[5].attributes['aria-current'],'step');
    renderProgress(doc,null);
    assert.equal(elements.get('job-progress').hidden,true);
});
