// Run: node tests/test_preview_eta.js
const assert = require('node:assert/strict');
const fs = require('node:fs');

const source = fs.readFileSync(`${__dirname}/../web/unlimited_preview.js`, 'utf8');
const start = source.indexOf('function projectedRenderTiming(');
const end = source.indexOf('\n\napi.addEventListener', start);
assert.ok(start >= 0 && end > start, 'projectedRenderTiming must remain independently testable');
const projectedRenderTiming = new Function(`${source.slice(start, end)} return projectedRenderTiming;`)();

assert.deepEqual(projectedRenderTiming(100, 1, 4), { etaSeconds: 300, totalSeconds: 400 });
// Four completed/current TaoMate phases out of ten use elapsed/4 for each
// remaining phase, regardless of unequal phase frame counts.
assert.deepEqual(projectedRenderTiming(240, 4, 10), { etaSeconds: 360, totalSeconds: 600 });
assert.deepEqual(projectedRenderTiming(100, 2.5, 5), { etaSeconds: 100, totalSeconds: 200 });
assert.deepEqual(projectedRenderTiming(NaN, 0, 4, 12), { etaSeconds: 12, totalSeconds: NaN });
// Retrying consumes wall time but must not inflate the cost of remaining work.
assert.deepEqual(projectedRenderTiming(300, 1, 3, NaN, 100), { etaSeconds: 200.00000000000003, totalSeconds: 500 });
const statusBegin = source.indexOf('                const audioFirstMode =');
const statusEnd = source.indexOf('                const eta =', statusBegin);
const estimate = new Function('projectedRenderTiming', 'state', `
    const {chunkRanges, elapsedSeconds, audioFirstSeconds, activeChunk, activeSubchunk, audioProgress=0, complete=false} = state;
    const chunkCount = chunkRanges.length, chunks = [], cachedChunkIndices = new Set();
    const audioCompletedChunks = new Set(), totalSteps = 3, currentStep = 0, averageStepMs = NaN;
    ${source.slice(statusBegin, statusEnd)}
    return { ...timingEstimate, audioPrepass, observedUnits };
`);
const ranges = [{taomate_audio_first: true, taomate_phase_count: 2}, {taomate_audio_first: true, taomate_phase_count: 2}];
const prepass = estimate(projectedRenderTiming, {chunkRanges: ranges, elapsedSeconds: 100, audioFirstSeconds: null, activeChunk: 1, activeSubchunk: null, audioProgress: 0.5});
assert.equal(prepass.audioPrepass, true);
assert.equal(prepass.etaSeconds, 100);
const retrying = estimate(projectedRenderTiming, {chunkRanges: [{...ranges[0], audio_retry_start_ms: 100000}, ranges[1]], elapsedSeconds: 150, audioFirstSeconds: null, activeChunk: 0, activeSubchunk: null});
assert.equal(retrying.observedUnits, 1);
assert.equal(retrying.etaSeconds, 100);
const video = estimate(projectedRenderTiming, {chunkRanges: ranges, elapsedSeconds: 720, audioFirstSeconds: 600, activeChunk: 0, activeSubchunk: 1});
assert.equal(video.etaSeconds, 360);
assert.equal(video.totalSeconds, 1080);
const alternating = estimate(projectedRenderTiming, {chunkRanges: [{taomate_phase_count: 2, audio_retry_start_ms: 100000, audio_retry_end_ms: 200000}, {taomate_phase_count: 2}], elapsedSeconds: 220, audioFirstSeconds: null, activeChunk: 0, activeSubchunk: 1});
assert.equal(alternating.audioPrepass, false);
assert.equal(alternating.etaSeconds, 360);
assert.equal(alternating.totalSeconds, 580);
console.log('Preview ETA projection checks passed.');
// Audio uses its own completed step count and measured inference rate.
const statusStart = source.indexOf('                const audioStep =', source.indexOf('function renderStatus()'));
const statusStop = source.indexOf('                const audioProgress =', statusStart);
const audioStatus = new Function('phase', 'audioStepMs', `
    const averageStepMs = 99999, completedElapsed = 0, startedAt = null;
    const chunkRanges = [], activeChunk = 0, chunkCount = 1, activeSubchunk = null;
    const hoverStep = null, currentStep = 3, totalSteps = 10, complete = false, paused = false;
    const formatEta = () => '';
    ${source.slice(statusStart, statusStop)}
    return { displayStep, displayTotalSteps, secondsPerStep, phaseLine };
`);
assert.deepEqual(audioStatus('TaoMate: audio teacher 1/4 · step 1/3', 2500), {
    displayStep: 1, displayTotalSteps: 3, secondsPerStep: '2.50s/step', phaseLine: 'TaoMate: audio teacher 1/4'
});
