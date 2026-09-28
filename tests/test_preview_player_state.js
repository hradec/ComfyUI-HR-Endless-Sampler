// Run: node tests/test_preview_player_state.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const source = fs.readFileSync(`${__dirname}/../web/unlimited_preview.js`, 'utf8');
assert.match(source, /const PLAYER_STATE_STORAGE_KEY = "hr_endless_sampler_preview_player"/, 'the button state needs its own browser storage key');
assert.match(source, /\.\.\.readStoredPlayerState\(\),/, 'a reload must prefer the stored button state over the workflow copy');
assert.match(source, /writeStoredPlayerState\(state\);/, 'every button-state write must reach browser storage');
assert.match(source, /node\.properties\.hr_endless_sampler_preview_player = state;/, 'the workflow copy must keep its exact shape');
assert.match(source, /rememberPlayerState\(\{ frame, muted: audioMuted, inverseGammaDisplay, subtitlesEnabled, paused \}\)/, 'a restored frame must be stored the same way as a fresh one');
assert.match(source, /paused = savedPlayerState\.paused === undefined \? true : Boolean\(savedPlayerState\.paused\);/, 'a restore must honor the stored play state and keep the old paused default');
assert.match(source, /function setPaused\(value\) \{[\s\S]*?persistPlayerState\(\);/, 'pressing play or pause must store the state at once');
// A refused cache-button change must say why instead of reverting silently.
assert.ok(source.includes('refusal = error instanceof Error ? error.message : String(error);'), 'a refused cache change must keep the reason');
assert.ok(source.includes('warnToast("Teacher audio cache policy was not changed", refusal);'), 'the reason must reach the warning the owner sees');
// The sampler composes the resume sentence, because it can keep chunks on both
// sides of the one it regenerates; the widget must show it verbatim.
assert.ok(source.includes('String(audio.run_reason || "kept the settled teacher audio and resumed the teacher")'), 'a resumed run must show the reason the sampler reported');

// The storage helpers are file scope, so they can be executed for real.
const start = source.indexOf('const PLAYER_STATE_STORAGE_KEY');
const end = source.indexOf('api.addEventListener(', start);
assert.ok(start > 0 && end > start, 'the storage helpers must stay at file scope');

function load(storage) {
    const warnings = [];
    const context = { window: { localStorage: storage }, console: { warn: message => warnings.push(String(message)) } };
    vm.createContext(context);
    vm.runInContext(`${source.slice(start, end)}\nglobalThis.read = readStoredPlayerState; globalThis.write = writeStoredPlayerState;`, context);
    return { context, warnings };
}

function memoryStorage() {
    const entries = new Map();
    return {
        entries,
        getItem: key => (entries.has(key) ? entries.get(key) : null),
        setItem: (key, value) => entries.set(key, String(value)),
    };
}

const state = { frame: 137, muted: true, inverseGammaDisplay: true, subtitlesEnabled: false, paused: false };
const working = memoryStorage();
const workingContext = load(working);
workingContext.context.write(state);
assert.deepEqual({ ...workingContext.context.read() }, state, 'the stored button state must round trip unchanged');
assert.deepEqual(workingContext.warnings, [], 'a working storage must not warn');
assert.equal(working.entries.get('hr_endless_sampler_preview_player'), JSON.stringify(state));

// A browser that blocks or corrupts storage must fall back to the defaults
// instead of throwing inside node creation.
const corrupt = load({ getItem: () => '{not json', setItem: () => {} });
assert.deepEqual({ ...corrupt.context.read() }, {});
assert.equal(corrupt.warnings.length, 1);
const blocked = load({ getItem: () => { throw new Error('blocked'); }, setItem: () => { throw new Error('blocked'); } });
assert.deepEqual({ ...blocked.context.read() }, {});
blocked.context.write(state);
assert.equal(blocked.warnings.length, 2);
// A press must reach storage even while a restored frame is still pending, which is
// always the case for an audio-only preview: this runs the real function body.
const persistStart = source.indexOf('function persistPlayerState()');
const persistEnd = source.indexOf('\n            }', persistStart) + '\n            }'.length;
assert.ok(persistStart > 0 && persistEnd > persistStart, 'persistPlayerState must stay at node scope');
const persistSource = source.slice(persistStart, persistEnd);

function runPersist({ displayedFrame, pendingFrameValue, stored }) {
    const context = {
        displayedFrameNumber: () => displayedFrame,
        pendingFrame: pendingFrameValue,
        audioMuted: true,
        inverseGammaDisplay: true,
        subtitlesEnabled: false,
        paused: true,
        rememberPlayerState: state => stored.push(state),
    };
    vm.createContext(context);
    vm.runInContext(`${persistSource}\npersistPlayerState();`, context);
    return stored;
}

// A pending, unreachable frame must not swallow the press.
let stored = runPersist({ displayedFrame: null, pendingFrameValue: 4321, stored: [] });
assert.equal(stored.length, 1, 'a press must store the state even with a pending frame');
assert.deepEqual({ ...stored[0] }, { frame: 4321, muted: true, inverseGammaDisplay: true, subtitlesEnabled: false, paused: true });
// Without a pending frame the displayed one is what the state carries.
stored = runPersist({ displayedFrame: 137, pendingFrameValue: null, stored: [] });
assert.equal(stored[0].frame, 137, 'the displayed frame is stored once nothing is pending');
// A pending frame that has arrived is replaced by the real one.
stored = runPersist({ displayedFrame: 4321, pendingFrameValue: 4321, stored: [] });
assert.equal(stored[0].frame, 4321, 'an arrived frame keeps its number');
// A restore keeps the stored play state; only a live reset starts playing again.
assert.ok(
    source.includes('paused = restoringCache && savedPlayerState.paused !== undefined ? Boolean(savedPlayerState.paused) : false;'),
    'a refreshed panel must not start playing over a stored pause',
);
assert.ok(
    source.includes('                    renderStatus();\n                    // The restored play state is only visible once the button is'),
    'the restore must repaint the play button',
);
// An audio-only preview has no frame numbers to match, so the stored position must land
// on the chunk whose frame range holds it; a group that has frames still wins with the
// exact frame, which is why the fallback skips those groups.
assert.ok(source.includes('if (group?.frameNumbers?.length) continue;'), 'only frame-less groups take the range fallback');
assert.ok(source.includes('const start = Number(group?.output_start ?? group?.start ?? chunkRanges[chunkIndex]?.start);'), 'the fallback range must come from the published chunk');
assert.ok(source.includes('playing = chunkIndex;'), 'the timeline cursor must land on that chunk');
// The yellow node message is gone: cache problems are raised as ComfyUI toasts, at the
// top of the UI, and an identical warning is not repeated by the status poll.
assert.ok(!source.includes("#e0c060"), 'no yellow node message may remain');
assert.ok(source.includes('warnToast("Teacher audio cache is incompatible with this render"'), 'an incompatible cache must warn');
assert.ok(source.includes('warnToast("Teacher audio cache policy was not changed"'), 'a refused cache change must warn');
const toastStart = source.indexOf('let lastWarnToast = "";');
const toastEnd = source.indexOf('\n            }', source.indexOf("function warnToast(", toastStart)) + '\n            }'.length;
assert.ok(toastStart > 0 && toastEnd > toastStart, 'the toast helper must stay at node scope');
const toastSource = source.slice(toastStart, toastEnd);

function runToast(api, calls) {
    const context = { app: api, String, console };
    vm.createContext(context);
    vm.runInContext(`${toastSource}\nglobalThis.warnToast = warnToast;`, context);
    const first = context.warnToast("summary", "detail");
    const second = context.warnToast("summary", "detail");
    calls.push({ first, second, context });
    return context.warnToast("other", "detail");
}

const toastCalls = [];
const sent = [];
const toastApi = { extensionManager: { toast: { add: payload => sent.push(payload) } } };
assert.equal(runToast(toastApi, toastCalls), true);
assert.deepEqual({ ...sent[0] }, { severity: "warn", summary: "summary", detail: "detail", life: 8000 });
assert.equal(sent.length, 2, 'a repeated and an identical warning must not be raised twice');
assert.equal(runToast({}, toastCalls), false, 'a frontend without the toast API must not throw');
console.log('Preview player state checks passed.');
