// Run: node tests/test_preview_eta.js
const assert = require('node:assert/strict');
const fs = require('node:fs');

const source = fs.readFileSync(`${__dirname}/../web/unlimited_preview.js`, 'utf8');
const start = source.indexOf('function projectedRenderTiming(');
const end = source.indexOf('\n\napi.addEventListener', start);
assert.ok(start >= 0 && end > start, 'projectedRenderTiming must remain independently testable');
const projectedRenderTiming = new Function(`${source.slice(start, end)} return projectedRenderTiming;`)();

assert.deepEqual(projectedRenderTiming(100, 1, 4), { etaSeconds: 300, totalSeconds: 400 });
assert.deepEqual(projectedRenderTiming(100, 2.5, 5), { etaSeconds: 100, totalSeconds: 200 });
assert.deepEqual(projectedRenderTiming(NaN, 0, 4, 12), { etaSeconds: 12, totalSeconds: NaN });
console.log('Preview ETA projection checks passed.');
