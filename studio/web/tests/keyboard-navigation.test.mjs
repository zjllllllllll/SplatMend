import assert from 'node:assert/strict';
import test from 'node:test';
import { keyboardNavigation, isEditing } from '../src/keyboard-navigation.mjs';

function harness() {
    const doc = new EventTarget();
    const win = new EventTarget();
    const calls = [];
    let movable = true;
    const bridge = keyboardNavigation({ win, doc, fire: (...args) => calls.push(args), canMove: () => movable });
    function send(type, props = {}, target = doc) {
        const event = new Event(type, { cancelable: true });
        for (const [key, value] of Object.entries(props)) Object.defineProperty(event, key, { value });
        target.dispatchEvent(event);
        return event;
    }
    return { doc, win, calls, bridge, send, lock() { movable = false; bridge.clear(); } };
}

test('WASD and Q/E use existing camera events; repeat does not relatch', () => {
    const h = harness();
    for (const [code, direction] of Object.entries({KeyW:'forward',KeyS:'backward',KeyA:'left',KeyD:'right',KeyQ:'down',KeyE:'up'})) {
        assert.ok(h.send('keydown', {code}).defaultPrevented);
        h.send('keydown', {code, repeat:true});
        h.send('keyup', {code});
        assert.deepEqual(h.calls.filter(([name]) => name === `camera.fly.${direction}`),
            [[`camera.fly.${direction}`, true], [`camera.fly.${direction}`, false]]);
    }
    h.bridge.destroy();
});

test('typing, IME, browser modifiers and selection/busy guards block movement', () => {
    for (const props of [{isComposing:true}, {ctrlKey:true}, {metaKey:true}, {altKey:true},
        {target:{closest:()=>({})}}, {target:{isContentEditable:true}}]) {
        const h = harness();
        h.send('keydown', {code:'KeyW', ...props});
        assert.ok(!h.calls.some(([name, value]) => name === 'camera.fly.forward' && value));
    }
    const h = harness();
    h.send('keydown', {code:'KeyW'});
    h.lock();
    h.send('keydown', {code:'KeyD'});
    assert.deepEqual(h.calls.filter(([name]) => name.startsWith('camera.fly')),
        [['camera.fly.forward', true], ['camera.fly.forward', false]]);
    assert.equal(isEditing({isContentEditable:true}), true);
});

test('focus, blur, hidden page, explicit clear and key-up always release held movement', () => {
    for (const stop of [h=>h.send('blur', {}, h.win), h=>h.send('focusin', {target:{closest:()=>({})}}),
        h=>{h.doc.hidden=true;h.send('visibilitychange');}, h=>h.bridge.clear(),
        h=>h.send('keyup', {code:'KeyW', target:{closest:()=>({})}})]) {
        const h = harness();
        h.send('keydown', {code:'KeyW', shiftKey:true});
        stop(h);
        assert.deepEqual(h.calls.filter(([name]) => name === 'camera.fly.forward'),
            [['camera.fly.forward', true], ['camera.fly.forward', false]]);
    }
});

test('Shift speed modifier releases; destroyed bridge no longer handles keys', () => {
    const h = harness();
    h.send('keydown', {code:'KeyW', shiftKey:true});
    h.send('keyup', {key:'Shift'});
    assert.deepEqual(h.calls.filter(([name]) => name === 'camera.modifier.fast'),
        [['camera.modifier.fast', true], ['camera.modifier.fast', false]]);
    h.bridge.destroy();
    const count = h.calls.length;
    h.send('keydown', {code:'KeyD'});
    assert.equal(h.calls.length, count);
});
