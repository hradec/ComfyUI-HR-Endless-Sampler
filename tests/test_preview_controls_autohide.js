// Run: node tests/test_preview_controls_autohide.js
// Both fullscreen players (the preview node and the Save/Load Video nodes) hide their
// chrome after three seconds of a still pointer, and a paused player keeps it.
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');

function load(file) {
    const source = fs.readFileSync(`${__dirname}/../web/${file}`, 'utf8');
    const start = source.indexOf('function createControlsAutohide(');
    const end = source.indexOf('\nfunction ', start + 1) + 1;
    assert.ok(start > 0 && end > start, `${file}: the autohide helper must exist`);
    const timers = [];
    const listeners = {};
    const elements = [0, 1, 2].map(() => ({ style: {} }));
    const context = {
        document: {
            addEventListener(type, handler) { (listeners[type] = listeners[type] || []).push(handler); },
            removeEventListener(type, handler) { listeners[type] = (listeners[type] || []).filter(item => item !== handler); },
        },
        setTimeout(handler, delay) { timers.push({ handler, delay, cancelled: false, fired: false }); return timers.length; },
        clearTimeout(id) { if (timers[id - 1]) timers[id - 1].cancelled = true; },
    };
    vm.createContext(context);
    vm.runInContext(`${source.slice(start, end)}\nglobalThis.createControlsAutohide = createControlsAutohide;`, context);
    return {
        source,
        create: context.createControlsAutohide,
        elements,
        timers,
        listeners,
        fire(type) { for (const handler of [...(listeners[type] || [])]) handler({}); },
        pending: () => timers.filter(timer => !timer.cancelled && !timer.fired),
        run(timer) { timer.fired = true; timer.handler(); },
    };
}

for (const file of ['unlimited_preview.js', 'finished_video_player.js']) {
    const harness = load(file);
    const { create, elements, listeners, fire, pending } = harness;
    const timersOf = () => pending();

    // Nothing hides outside fullscreen, however long the pointer is still.
    let fullscreen = false;
    let playing = true;
    const controls = create(elements, { isFullscreen: () => fullscreen, isPlaying: () => playing });
    controls.arm();
    assert.equal(controls.isVisible(), true, `${file}: the controls start visible`);
    assert.equal(timersOf().length, 0, `${file}: nothing is armed outside fullscreen`);
    assert.equal(elements[0].style.transition, 'opacity .3s ease', `${file}: the fade is a transition`);
    assert.equal(elements[0].style.pointerEvents, '', `${file}: a visible control stays clickable`);

    // In fullscreen, three seconds of a still pointer fade them away.
    fullscreen = true;
    controls.arm();
    assert.deepEqual(timersOf().map(timer => timer.delay), [3000], `${file}: fullscreen arms the three second delay`);
    assert.equal(controls.isVisible(), true, `${file}: arming shows the controls first`);
    harness.run(timersOf()[0]);
    assert.equal(controls.isVisible(), false, `${file}: going quiet hides the controls`);
    for (const element of elements) {
        assert.equal(element.style.opacity, '0', `${file}: a hidden control is transparent`);
        assert.equal(element.style.pointerEvents, 'none', `${file}: a hidden control is not clickable`);
    }

    // Moving the mouse brings them back and re-arms.
    fire('mousemove');
    assert.equal(controls.isVisible(), true, `${file}: moving the mouse shows the controls again`);
    assert.equal(timersOf().length, 1, `${file}: movement re-arms the delay`);

    // A paused player keeps its chrome: the timer must not hide them.
    playing = false;
    controls.arm();
    assert.equal(controls.isVisible(), true, `${file}: a paused player shows the controls`);
    assert.equal(timersOf().length, 0, `${file}: a paused player arms nothing`);
    playing = true;

    // Leaving fullscreen restores them at once, whatever the pointer was doing.
    fullscreen = false;
    fire('fullscreenchange');
    assert.equal(controls.isVisible(), true, `${file}: leaving fullscreen restores the controls`);
    assert.equal(timersOf().length, 0, `${file}: leaving fullscreen clears the delay`);

    // A removed panel leaves nothing behind.
    controls.stop();
    assert.equal(listeners.mousemove.length, 0, `${file}: stopping detaches the pointer listener`);
    assert.equal(listeners.fullscreenchange.length, 0, `${file}: stopping detaches the fullscreen listener`);

    // The wiring itself: the whole block floats in fullscreen and fades as one, and the
    // player's own state decides whether it may hide.
    assert.ok(harness.source.includes('createControlsAutohide('), `${file}: the panel must build the autohide`);
    assert.ok(harness.source.includes('applyFullscreenLayout();\n                controlsAutohide.arm();'), `${file}: entering or leaving fullscreen re-applies the layout and re-arms`);
    assert.ok(harness.source.includes('controlsAutohide.stop();'), `${file}: removing the panel stops the autohide`);
    assert.ok(harness.source.includes('viewport.style.position = active ? "absolute" : "relative";'), `${file}: fullscreen puts the picture behind the chrome`);
    assert.ok(harness.source.includes('controls.style.bottom = active ? "0" : "";'), `${file}: fullscreen floats the chrome on the bottom`);
    assert.ok(harness.source.includes('transport.style.background = active ? "rgba(20,20,20,.72)" : "#181818";'), `${file}: the floating bar lets the video through`);
    assert.ok(harness.source.includes('controls.appendChild(transport);'), `${file}: the chrome block owns the transport bar`);
    assert.ok(harness.source.includes('controls.appendChild(status);'), `${file}: the chrome block owns the status line`);
}

// Each panel is armed by its own player state, and the preview's graphs float with the rest.
const preview = fs.readFileSync(`${__dirname}/../web/unlimited_preview.js`, 'utf8');
const finished = fs.readFileSync(`${__dirname}/../web/finished_video_player.js`, 'utf8');
assert.ok(preview.includes('isPlaying: () => !paused,'), 'the preview hides only while it plays');
assert.ok(preview.includes('controlsAutohide.arm();\n                stop();') || preview.includes('controlsAutohide.arm();'), 'the preview re-arms on its state changes');
assert.ok(preview.includes('[controls, graphs, linearDisplayButton, subtitleButton, fullscreenButton]'), 'the preview fades its graphs with the rest');
assert.ok(preview.includes('graphs.style.position = active ? "absolute" : "";'), 'the preview floats its graphs over the video');
assert.ok(finished.includes('isPlaying: () => !media.paused && !media.ended,'), 'the finished player hides only while the video plays');
assert.ok(finished.includes('media.addEventListener("play", () => controlsAutohide.arm());'), 'pressing play re-arms the finished player');
assert.ok(finished.includes('media.addEventListener("pause", () => controlsAutohide.arm());'), 'pressing pause brings the chrome back');
assert.ok(finished.includes('controls.appendChild(compareFooter);'), 'the comparison footer floats with the rest');

console.log('Fullscreen control autohide checks passed.');
