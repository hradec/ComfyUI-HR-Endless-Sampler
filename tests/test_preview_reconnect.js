// Run: node tests/test_preview_reconnect.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(`${__dirname}/../web/unlimited_preview.js`, 'utf8');
const start = source.indexOf('async function restoreServerState(');
const end = source.indexOf('function inspectFrameGroup(', start);

async function check(force, liveReset, expected) {
    const accepted = [];
    const snapshot = { execution: 1, reset: { execution: 1, action: 'reset' }, chunks: [{ execution: 1, action: 'chunk' }] };
    const context = {
        execution: 7, restoringCache: false, statusPhase: {}, statusProgressFill: { style: {} },
        fpsWidget: null, setPlaybackFps() {}, renderStatus() {}, redrawGraphs() {},
        hoverStep: null, timer: null, framePending: false, restorePlayback() {},
        setTimeout, console,
        node: { id: 1, _hrEndlessSamplerPreview(data) {
            if (data.execution === context.execution) accepted.push(data.action);
        } },
        resetExecution(data) { context.execution = data.execution; accepted.push('reset'); },
        api: { async fetchApi() {
            if (liveReset) context.execution = 8;
            return { ok: true, async json() { return snapshot; } };
        } },
    };
    vm.createContext(context);
    vm.runInContext(source.slice(start, end), context);
    await context.restoreServerState(0, force);
    assert.deepEqual(accepted, expected);
    assert.equal(context.execution, liveReset ? 8 : force ? 1 : 7);
}

(async () => {
    await check(true, false, ['reset', 'chunk']); // Reconnect adopts restarted server.
    await check(false, false, []); // Automatic restore still rejects stale snapshots.
    await check(true, true, []); // A newer live reset wins the fetch race.
    console.log('Preview reconnect checks passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
