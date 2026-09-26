// Run with node tests/test_preview_audio.js. Exercise the real scheduling functions.
const fs = require("fs");
const assert = require("assert");
const source = fs.readFileSync("web/unlimited_preview.js", "utf8");
const begin = source.indexOf("            function stopContinuousAudio()");
const end = source.indexOf("            function stop()", begin);
const run = new Function("assert", `
    const chunks = [{ audioSource: "a", sourceFps: 24 }, { audioSource: "b", sourceFps: 24 }];
    const decodedAudio = new Map([["a", { duration: 1 }], ["b", { duration: 2 }]]);
    let continuousAudio = [];
    const continuousAudioContext = {
        currentTime: 5, resume: () => Promise.resolve(),
        createBufferSource: () => ({ playbackRate: {}, connect() {}, start(time, offset, duration) { this.time = time; this.offset = offset; this.duration = duration; }, stop() { this.stopped = true; } })
    };
    const continuousAudioGain = { gain: {} };
    const audioMuted = false, sourceFps = 24;
    const audioPlayer = { pause() {} };
    const validFps = value => value;
    const currentPlaybackFps = () => 24;
    const audioOffset = (group, frame) => (group.audioStartSeconds || 0) + frame / 24;
    ${source.slice(begin, end)}
    assert(scheduleContinuousAudio(chunks[0], 0, true));
    assert.strictEqual(continuousAudio.length, 2);
    assert.strictEqual(continuousAudio[1].start, continuousAudio[0].end);
    const first = continuousAudio[0];
    continuousAudioContext.currentTime = first.end;
    assert(scheduleContinuousAudio(chunks[1], 0, false));
    assert.strictEqual(continuousAudio[0], first);
    assert.strictEqual(continuousAudioTime(chunks[1]), 0);
    scheduleContinuousAudio(chunks[1], 24, true);
    assert(first.source.stopped);
    assert.strictEqual(continuousAudio.length, 1);
    assert.strictEqual(continuousAudio[0].offset, 1);
    stopContinuousAudio();
    assert.strictEqual(continuousAudio.length, 0);
    chunks.splice(0, chunks.length,
        { audioSource: "full", sourceFps: 24, audioStartSeconds: 0, audioEndSeconds: 2 },
        { audioSource: "full", sourceFps: 24, audioStartSeconds: 2, audioEndSeconds: 4 });
    decodedAudio.set("full", { duration: 4 });
    assert(scheduleContinuousAudio(chunks[0], 0, true));
    assert.deepStrictEqual(continuousAudio.map(entry => entry.offset), [0, 2]);
    assert.deepStrictEqual(continuousAudio.map(entry => entry.source.duration), [2, 2]);
`);
run(assert);
const offsetStart = source.indexOf("            function audioOffset(group, frameIndex)");
const offsetEnd = source.indexOf("            function syncAudio", offsetStart);
const audioOffset = new Function("validFps", "sourceFps", `${source.slice(offsetStart, offsetEnd)} return audioOffset;`)(value => value, 24);
assert.strictEqual(audioOffset({ sourceFps: 24, audioStartSeconds: 3, frameNumbers: [72, 73, 84] }, 2), 3.5);
const playBegin = source.indexOf("            function playFrameGroup(");
const playEnd = source.indexOf("            function show(", playBegin);
const waitForNext = new Function("assert", `
    const group = { frames: ["black"], audioSource: "audio", sourceFps: 24 };
    const chunks = [group], chunkCount = 3;
    const audioPlayer = { paused: true, readyState: 0 };
    let playbackSerial = 0, paused = false, hoverStep = null, complete = false;
    let playing = -1, playingFrame = -1, framePending = false, waitingAfterChunk = null, timer = null, audioGroup = null;
    let jumped = false, timerCallback = null;
    const renderTransport = () => {}, syncAudio = () => {}, preloadAudio = () => {};
    const frameDuration = () => 1, nextAvailable = () => 0, continuousAudioTime = () => null;
    const available = index => Boolean(chunks[index]?.frames?.length);
    const displaySource = (_source, _valid, done) => done();
    const setTimeout = callback => { timerCallback = callback; return 1; };
    const show = () => { jumped = true; };
    const stop = () => { playbackSerial++; timer = null; framePending = false; waitingAfterChunk = null; };
    ${source.slice(playBegin, playEnd)}
    playFrameGroup(0, group, 0, 0);
    timerCallback();
    assert.strictEqual(waitingAfterChunk, 0);
    assert.strictEqual(jumped, false);
`);
waitForNext(assert);
const timelineClock = new Function("assert", `
    const group = { frames: ["a", "b", "c"], frameNumbers: [120, 124, 128], outputStart: 120, durations: [167, 167, 167], audioSource: "full", audioStartSeconds: 5, sourceFps: 24 };
    const chunks = [group], chunkCount = 1;
    const audioPlayer = { paused: true, readyState: 0 };
    let playbackSerial = 0, paused = false, hoverStep = null, complete = false;
    let playing = -1, playingFrame = -1, framePending = false, waitingAfterChunk = null, timer = null, audioGroup = null;
    let timerCallback = null, delay = null;
    const renderTransport = () => {}, syncAudio = () => {}, preloadAudio = () => {};
    const frameDuration = () => 42, nextAvailable = () => -1, continuousAudioTime = () => 5;
    const validFps = value => value, sourceFps = 24, currentPlaybackFps = () => 24;
    const available = index => Boolean(chunks[index]?.frames?.length);
    const displaySource = (_source, _valid, done) => done();
    const setTimeout = (callback, ms) => { timerCallback = callback; delay = ms; return 1; };
    const show = () => {}, stop = () => {};
    ${source.slice(playBegin, playEnd)}
    playFrameGroup(0, group, 0, 0);
    assert(delay > 160, "the full-audio offset and sparse preview frames must retain their timing");
    timerCallback();
    assert.strictEqual(playingFrame, 1);
`);
timelineClock(assert);
const carryBegin = source.indexOf('                let restoredAudioSource = typeof data.audio');
const carryEnd = source.indexOf('                if (typeof data.gemma_detailed_description', carryBegin);
const carryAudio = new Function('data', 'chunks', 'index', 'finalized', 'pendingAudioSources', 'currentStep', `
    const encodedFrames = ['image'];
    const frameDurations = [42];
    ${source.slice(carryBegin, carryEnd)}
    return group;
`);
const audioGroups = [{ audioSource: 'full-audio', audioStartSeconds: 5, audioEndSeconds: 10, audioOnlyPlaceholder: true }];
const frameData = { frames: ['frame'], output_start: 120, output_end: 239, fps: 24 };
const latentGroup = carryAudio(frameData, audioGroups, 0, false, new Map(), 1);
audioGroups[0] = latentGroup;
const decodedGroup = carryAudio(frameData, audioGroups, 0, true, new Map(), 1);
assert.strictEqual(decodedGroup.audioSource, 'full-audio');
assert.strictEqual(decodedGroup.audioStartSeconds, 5);
const liveAudio = new Function('assert', 'carryAudio', `
    const chunks = [{ frames: ['black'], audioSource: 'full', audioStartSeconds: 0, audioEndSeconds: 5, sourceFps: 24 }];
    const decodedAudio = new Map([['full', { duration: 5 }]]);
    let continuousAudio = [];
    const continuousAudioContext = {
        currentTime: 10, resume: () => Promise.resolve(),
        createBufferSource: () => ({ playbackRate: {}, connect() {}, start() {}, stop() { this.stopped = true; } })
    };
    const continuousAudioGain = { gain: {} }, audioMuted = false, sourceFps = 24;
    const audioPlayer = { pause() {} }, validFps = value => value;
    const currentPlaybackFps = () => 24, audioOffset = (_group, frame) => frame / 24;
    ${source.slice(begin, end)}
    assert(scheduleContinuousAudio(chunks[0], 0, true));
    const scheduled = continuousAudio[0];
    const playingGroup = chunks[0];
    chunks[0] = carryAudio({ output_start: 0, output_end: 38, fps: 24 }, chunks, 0, false, new Map(), 1);
    assert.strictEqual(chunks[0], playingGroup, 'live frames must retain the scheduled chunk identity');
    assert(scheduleContinuousAudio(playingGroup, 1, false));
    assert.strictEqual(continuousAudio[0], scheduled);
    assert(!scheduled.source.stopped, 'a new inference preview must not cancel chunk 1 audio');
`);
liveAudio(assert, carryAudio);
console.log("Preview audio scheduling checks passed.");
