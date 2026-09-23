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
        createBufferSource: () => ({ playbackRate: {}, connect() {}, start(time, offset) { this.time = time; this.offset = offset; }, stop() { this.stopped = true; } })
    };
    const continuousAudioGain = { gain: {} };
    const audioMuted = false, sourceFps = 24;
    const audioPlayer = { pause() {} };
    const validFps = value => value;
    const currentPlaybackFps = () => 24;
    const audioOffset = (group, frame) => frame / 24;
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
`);
run(assert);
console.log("Preview audio scheduling checks passed.");
