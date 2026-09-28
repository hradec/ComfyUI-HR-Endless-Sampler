// Run: node tests/test_taomate_timeline.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const source = fs.readFileSync(`${__dirname}/../web/unlimited_preview.js`, 'utf8');
const start = source.indexOf('const stops = [];', source.indexOf('function renderTransport()'));
const end = source.indexOf('timelineTrack.style.background', start);
// The sub-chunk helpers are file scope, so the loop under test needs them in its own scope.
const helperStart = source.indexOf('function mixHex(');
const helperEnd = source.indexOf('function canvasContext(', helperStart);
assert.ok(helperStart > 0 && helperEnd > helperStart, 'the timeline sub-chunk helpers must exist');
const helpers = new Function(`${source.slice(helperStart, helperEnd)}\nreturn { mixHex, subchunkEdges, subchunkTimelineStops };`)();
const stops = new Function('chunkRanges', 'spans', 'total', 'chunkColors', 'available', 'chunks', 'audioCompletedChunks', 'audioCompletedOutlines', 'document', `let offset = 0; ${source.slice(helperStart, helperEnd)} ${source.slice(start, end)} Object.defineProperty(stops, 'outlines', {value: audioCompletedOutlines.items}); return stops;`);
const draw = (completed, previewEnd, audioDone=[], audioOnlyPlaceholder=false, work=undefined) => {
    const audioCompletedOutlines = {items: [], replaceChildren() { this.items = []; }, appendChild(item) { this.items.push(item); }};
    return stops([{start: 0, taomate_completed_frames: completed, taomate_phase_work: work}], [124], 124, ['#ff0000'], () => true, [{outputEnd: previewEnd, audioOnlyPlaceholder}], new Set(audioDone), audioCompletedOutlines, {createElement: () => ({style: {}, title: ''})});
};
// The gradient stops are rounded to a thousandth of a percent, so a long timeline keeps a
// readable background string.
const percent = value => Math.round(value * 1000) / 1000;
assert.deepEqual(draw(0), ['color-mix(in srgb, #ff0000 50%, black) 0%', 'color-mix(in srgb, #ff0000 50%, black) 100%']);
// The rendered part holds its color right up to the boundary, and the pending part starts
// at it: one stop a hundredth of a percent earlier keeps that step where it belongs.
const boundaryOf = completed => percent(completed / 124 * 100);
const renderEdge = completed => percent(boundaryOf(completed) - 0.01);
assert.equal(draw(39).length, 4);
assert.ok(draw(39).some(stop => stop === `#ff0000e8 ${renderEdge(39)}%`), 'the rendered part keeps its color up to its boundary');
assert.ok(draw(39).some(stop => stop === `color-mix(in srgb, #ff0000 50%, black) ${boundaryOf(39)}%`), 'the pending part starts at the boundary');
assert.deepEqual(draw(124), ['#ff0000e8 0%', '#ff0000e8 100%']);
assert.deepEqual(draw(undefined), ['#ff0000e8 0%', '#ff0000e8 100%']);
assert.ok(draw(0, 38).some(stop => stop === `#ff0000e8 ${renderEdge(39)}%`), 'a live preview moves the boundary to the frames it holds');
assert.ok(draw(39, 72).some(stop => stop === `#ff0000e8 ${renderEdge(73)}%`));
assert.ok(draw(73, 38).some(stop => stop === `#ff0000e8 ${renderEdge(73)}%`), 'a preview never moves the boundary backwards');
assert.equal(draw(0, undefined, [0]).outlines[0].title, 'Audio teacher complete · chunks 1–1');
assert.match(draw(0, undefined, [0]).outlines[0].style.cssText, /#42c8ff/);
assert.deepEqual(draw(0, 123, [0], true).slice(0, 2), ['color-mix(in srgb, #ff0000 50%, black) 0%', 'color-mix(in srgb, #ff0000 50%, black) 100%']);
assert.deepEqual(draw(undefined, 123, [0], true).slice(0, 2), ['#ff000035 0%', '#ff000035 100%']);

// Each chunk alternates sub-chunk by sub-chunk between its own color and a brighter one:
// the odd sub-chunks keep the color, the even ones are a step brighter. Every band is flat
// and every boundary is a sharp step, which the doubled stop position carries.
const bright = helpers.mixHex('#ff0000', '#ffffff', 0.3);
assert.equal(bright, '#ff4d4d');
assert.equal(helpers.mixHex('#000000', '#ffffff', 0.5), '#808080');
assert.equal(helpers.mixHex('#112233', '#ffffff', 0), '#112233');
assert.deepEqual(helpers.subchunkEdges([50, 50]), [0, 0.5, 1]);
assert.deepEqual(helpers.subchunkEdges([10, 20, 30, 40]), [0, 0.1, 0.3, 0.6, 1]);
assert.equal(helpers.subchunkEdges([7]), null, 'a chunk with one sub-chunk keeps its flat bar');
assert.equal(helpers.subchunkEdges(undefined), null, 'a chunk without sub-chunk frames keeps its flat bar');
assert.equal(helpers.subchunkEdges([0, 0]), null, 'sub-chunks without frames cannot be placed');
// Two equal sub-chunks: the color, then the brighter one, and the boundary carries both
// colors at its own position so the step is sharp instead of a ramp.
assert.deepEqual(draw(undefined, undefined, [], false, [50, 50]), [
    '#ff0000e8 0%', '#ff0000e8 50%', `${bright}e8 50%`, `${bright}e8 100%`,
]);
// Unequal sub-chunks put their boundary where their frames are: 31 of 124 frames of the
// timeline is the quarter mark, and the whole even sub-chunk is the brighter band.
assert.deepEqual(draw(undefined, undefined, [], false, [31, 93]), [
    '#ff0000e8 0%', '#ff0000e8 25%', `${bright}e8 25%`, `${bright}e8 100%`,
]);
// Four sub-chunks alternate twice, so the chunk starts and ends in its own color.
const fourBands = draw(undefined, undefined, [], false, [25, 25, 25, 25]);
assert.deepEqual(fourBands.map(stop => stop.split(' ')[1]), ['0%', '25%', '25%', '50%', '50%', '75%', '75%', '100%']);
assert.deepEqual(fourBands.map(stop => stop.split(' ')[0]), ['#ff0000e8', '#ff0000e8', `${bright}e8`, `${bright}e8`, '#ff0000e8', '#ff0000e8', `${bright}e8`, `${bright}e8`]);
// The bands keep their alternation over the dimmed part of a filling chunk, and the render
// boundary is its own sharp step inside the band it falls in.
assert.deepEqual(draw(31, undefined, [], false, [50, 50]), [
    '#ff0000e8 0%', '#ff0000e8 24.99%',
    'color-mix(in srgb, #ff0000 50%, black) 25%', 'color-mix(in srgb, #ff0000 50%, black) 50%',
    'color-mix(in srgb, #ff4d4d 50%, black) 50%', 'color-mix(in srgb, #ff4d4d 50%, black) 100%',
]);
// A sub-chunk that just finished ends exactly on its band edge, which used to leave that
// band ramping from the rendered color to the dimmed one. It must stay flat and step.
assert.deepEqual(draw(62, undefined, [], false, [62, 62]), [
    '#ff0000e8 0%', '#ff0000e8 49.99%',
    'color-mix(in srgb, #ff0000 50%, black) 50%',
    'color-mix(in srgb, #ff4d4d 50%, black) 50%', 'color-mix(in srgb, #ff4d4d 50%, black) 100%',
]);
console.log('TaoMate timeline sub-chunk shading checks passed.');
console.log('TaoMate timeline completion checks passed.');

const cursorStart = source.indexOf('const before = spans.slice(0, playing)', end);
const cursorEnd = source.indexOf('timelinePlayhead.style.left', cursorStart);
const cursor = new Function('spans', 'total', 'playing', 'playingFrame', 'chunks', 'chunkRanges', `${source.slice(cursorStart, cursorEnd)} return position;`);
const group = {frames: Array(12), frameNumbers: [0, 1, 5, 9, 13, 17, 18, 22, 26, 30, 34, 35], outputStart: 0, outputEnd: 38};
assert.equal(cursor([124, 119], 243, 0, 11, [group], [{start: 0}]), 35 / 243 * 100);
const second = {...group, frameNumbers: group.frameNumbers.map(n => n + 124), outputStart: 124, outputEnd: 162};
assert.equal(cursor([124, 119], 243, 1, 11, [group, second], [{start: 0}, {start: 124}]), 159 / 243 * 100);
assert.equal(cursor([124], 124, 0, 11, [{...group, frameNumbers: []}], [{start: 0}]), 38 / 124 * 100);
console.log('TaoMate partial-preview cursor checks passed.');
const labelStart = source.indexOf('const chunk = chunkCount ?');
const labelEnd = source.indexOf(';', labelStart) + 1;
assert.ok(labelStart >= 0 && labelEnd > labelStart);
const label = new Function('chunkCount', 'activeChunk', 'activeSubchunk', 'activePhaseCount', source.slice(labelStart, labelEnd) + ' return chunk;');
assert.equal(label(13, 0, 1, 5), 'C1.1/13.5');
assert.equal(label(13, 0, 2, 5), 'C1.2/13.5');
assert.equal(label(13, 1, 4, 3), 'C2.4/13.3');
assert.equal(label(13, 0, null, 5), 'C1/13');
assert.equal(label(0, 0, null, 1), 'C—/—');
// The delete item is offered for every TaoMate chunk while the cache button is on, and
// only disabled when that chunk has no stored audio and attention rows.
assert.match(source, /const offersTeacherAudio = replayCacheEnabled && !cacheButton\.disabled && taomateMode/, 'the item must be offered for every TaoMate chunk');
assert.match(source, /if \(!canDeleteTeacherAudio\) \{/, 'a chunk without stored audio must disable the item');
assert.ok(source.includes('the audio and its attention rows are cached together, and this chunk has neither'), 'the disabled item must say why');
assert.ok(source.includes('keeps every other chunk and generates this one again'), 'the enabled item must promise only that chunk');
console.log('TaoMate sub-chunk status labels passed.');

// A drag owns the timeline cursor: a chunk still being rendered has fewer frames
// than its span, so the pointer's own position decides where the cursor sits.
const scrubStart = source.indexOf('function scrubPlayheadPosition(');
const scrubEnd = source.indexOf('function seekTimeline(', scrubStart);
assert.ok(scrubStart >= 0 && scrubEnd > scrubStart, 'the scrub position helper must exist');
const scrub = new Function('spans', 'total', 'scrub', `${source.slice(scrubStart, scrubEnd)} return scrubPlayheadPosition(spans, total, scrub);`);
assert.equal(scrub([124, 119], 243, null), null, 'no drag means the displayed frame decides');
assert.equal(scrub([124, 119], 243, {chunk: 0, fraction: 0.9}), 0.9 * 124 / 243 * 100);
assert.equal(scrub([124, 119], 243, {chunk: 1, fraction: 1}), 243 / 243 * 100);
assert.equal(scrub([124, 119], 243, {chunk: 1, fraction: 0}), 124 / 243 * 100);
assert.equal(scrub([124, 119], 243, {chunk: 0, fraction: 2}), 124 / 243 * 100, 'a fraction past the span is clamped');
assert.equal(scrub([124, 119], 243, {chunk: 7, fraction: 0.5}), (124 + 0.5 * 119) / 243 * 100, 'the last span owns an out-of-range chunk');
assert.equal(scrub([124, 119], 243, {chunk: 1, fraction: 0.5}), (124 + 0.5 * 119) / 243 * 100);
assert.equal(scrub([0, 240], 240, {chunk: 0, fraction: 1}), 0, 'a chunk with no span offers only its start');
// The pointer's chunk decides the cursor even when that chunk has no frames yet,
// and the displayed picture is not dragged to the nearest rendered chunk.
assert.ok(source.includes('timelineScrub = { chunk: targetChunk, fraction };'), 'the drag records its own position');
assert.ok(source.includes('const scrubPosition = scrubPlayheadPosition(spans, total, timelineScrub);'), 'the playhead reads the drag position');
assert.ok(!source.includes('targetChunk = candidates.reduce'), 'a drag no longer jumps to the nearest rendered chunk');
assert.ok(source.includes('timelineScrub = null;'), 'showing a frame or playing clears the drag position');
console.log('Timeline scrub cursor checks passed.');

// A drag shows the image that covers the pointed position. One preview image stands for
// the whole H3 temporal token it was decoded from, so a position inside that span shows
// that image, and only a position the render has not reached stays black.
const frameStart = source.indexOf('function scrubFrameIndex(');
const frameEnd = source.indexOf('function seekTimeline(', frameStart);
assert.ok(frameStart >= 0 && frameEnd > frameStart, 'the scrub frame resolver must exist');
const frameAt = new Function('chunks', 'chunkRanges', 'chunkSpan', `${source.slice(frameStart, frameEnd)} return scrubFrameIndex;`);
const span80 = () => 80;
// Four images decoded from tokens covering frames 0-3, 4-7, 8-11 and 12-15.
const tokens = {frames: Array(4), frameNumbers: [0, 4, 8, 12], outputEnd: 15};
const tokensAt = (fraction, group = tokens, index = 0) => frameAt([group], [{start: 0}], span80)(index, fraction);
assert.equal(tokensAt(0), 0, 'the first position shows the image covering it');
assert.equal(tokensAt(3 / 80), 0, 'a frame the image covers shows that image, never black');
assert.equal(tokensAt(4 / 80), 1, 'the next token image takes over at its own frame');
assert.equal(tokensAt(7 / 80), 1, 'the last frame of a token span shows its image');
assert.equal(tokensAt(12 / 80), 3, 'the last decoded image answers its own span');
assert.equal(tokensAt(15 / 80), 3, 'the rendered end still shows the last image');
assert.equal(tokensAt(16 / 80), -1, 'the first position past the rendered end has no image');
assert.equal(tokensAt(0.5), -1, 'a position no sub-chunk has reached has no image');
assert.equal(tokensAt(1), -1, 'the end of a half-filled chunk has no image');
// A group that starts later in the chunk answers from its own numbers, and a position
// before its first image stays black.
const later = {frames: Array(2), frameNumbers: [124, 128], outputEnd: 131};
assert.equal(frameAt([null, later], [{start: 0}, {start: 124}], span80)(1, 0), 0, 'a later chunk shows its first image at its own first frame');
assert.equal(frameAt([null, later], [{start: 0}, {start: 124}], span80)(1, 0.05), 1, 'a later chunk is read by its own numbers');
assert.equal(frameAt([null, later], [{start: 0}, {start: 124}], span80)(1, 0.9), -1, 'a position past the rendered end stays black');
assert.equal(frameAt([null, later], [{start: 0}, {start: 120}], span80)(1, 0.0375), -1, 'a position before the first decoded image stays black');
// A group without frame numbers is its own frame list, and a chunk with nothing
// rendered has no image at any position.
const unnumbered = {frames: Array(20), frameNumbers: []};
assert.equal(frameAt([unnumbered], [{start: 0}], span80)(0, 0.1), 8);
assert.equal(frameAt([unnumbered], [{start: 0}], span80)(0, 0.5), -1);
assert.equal(frameAt([], [{start: 0}], span80)(0, 0.5), -1);
assert.ok(source.includes('if (Number.isFinite(renderedEnd) && position > renderedEnd) return -1;'), 'the rendered end is the phase that finished');
assert.ok(!source.includes('Math.round(fraction * Math.max(0, chunks[targetChunk].frames.length - 1))'), 'the decoded frames are no longer stretched over the whole span');
assert.ok(source.includes('else showEmptyPosition(targetChunk);'), 'a position with nothing rendered shows black');
assert.ok(source.includes('playingFrame = -1;'), 'a black position reports no frame');
assert.ok(source.includes('if (playingFrame < 0 && available(playing))'), 'the black position survives a later redraw');
assert.ok(source.includes('if (paused) showEmptyPosition(playing);'), 'playing after a black position starts from that chunk');
console.log('Timeline scrub frame checks passed.');
