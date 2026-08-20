const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;


function findNode(rootGraph, qualifiedId) {
    const parts = String(qualifiedId).split(":");
    let graph = rootGraph;
    for (let index = 0; index < parts.length - 1; index++) {
        const node = graph?.getNodeById?.(Number(parts[index]));
        if (!node?.subgraph) return null;
        graph = node.subgraph;
    }
    return graph?.getNodeById?.(Number(parts[parts.length - 1])) || null;
}


function canvasContext(canvas) {
    const ratio = window.devicePixelRatio || 1;
    const width = canvas.clientWidth || 1;
    const height = canvas.clientHeight || 1;
    if (canvas.width !== Math.round(width * ratio) || canvas.height !== Math.round(height * ratio)) {
        canvas.width = Math.round(width * ratio);
        canvas.height = Math.round(height * ratio);
    }
    const context = canvas.getContext("2d");
    context.setTransform(ratio, 0, 0, ratio, 0, 0);
    return { context, width, height };
}


function drawGrid(context, width, height) {
    context.strokeStyle = "#292929";
    context.lineWidth = 1;
    for (let line = 1; line < 4; line++) {
        const y = Math.round(line * height / 4) + 0.5;
        context.beginPath();
        context.moveTo(4, y);
        context.lineTo(width - 4, y);
        context.stroke();
    }
}


function drawSeries(context, values, total, width, height, color, dashed=false, fill=false) {
    const points = [];
    for (let index = 0; index < values.length; index++) {
        if (Number.isFinite(values[index])) points.push([index, values[index]]);
    }
    if (!points.length) return;
    let minimum = Math.min(...points.map(point => point[1]));
    const maximum = Math.max(...points.map(point => point[1]));
    if (minimum > 0) minimum = 0;
    const range = Math.max(maximum - minimum, 1e-6);
    const x = index => 4 + index / Math.max(1, total - 1) * (width - 8);
    const y = value => 3 + (1 - (value - minimum) / range) * (height - 6);

    context.beginPath();
    for (let index = 0; index < points.length; index++) {
        const [step, value] = points[index];
        if (index === 0) context.moveTo(x(step), y(value));
        else context.lineTo(x(step), y(value));
    }
    if (fill) {
        const last = points[points.length - 1][0];
        context.lineTo(x(last), height - 3);
        context.lineTo(x(points[0][0]), height - 3);
        context.closePath();
        context.fillStyle = "rgba(230, 126, 34, 0.14)";
        context.fill();
        context.beginPath();
        for (let index = 0; index < points.length; index++) {
            const [step, value] = points[index];
            if (index === 0) context.moveTo(x(step), y(value));
            else context.lineTo(x(step), y(value));
        }
    }
    context.strokeStyle = color;
    context.lineWidth = 1.3;
    context.setLineDash(dashed ? [3, 3] : []);
    context.stroke();
    context.setLineDash([]);
}


function drawMarker(context, step, total, width, height) {
    if (!Number.isFinite(step) || total < 2) return;
    const x = 4 + step / (total - 1) * (width - 8) + 0.5;
    context.strokeStyle = "rgba(220,220,220,0.65)";
    context.lineWidth = 1;
    context.beginPath();
    context.moveTo(x, 3);
    context.lineTo(x, height - 3);
    context.stroke();
}


function drawSigmaDelta(canvas, sigmas, deltas, steps, markerStep=null) {
    const { context, width, height } = canvasContext(canvas);
    const total = Math.max(steps + 1, sigmas.length, deltas.length + 1);
    context.clearRect(0, 0, width, height);
    drawGrid(context, width, height);
    drawSeries(context, sigmas, total, width, height, "rgba(210,210,210,0.6)", true);
    drawSeries(context, [null, ...deltas], total, width, height, "#e67e22", false, true);
    drawMarker(context, markerStep, total, width, height);
}


function drawStepTimes(canvas, values, steps, markerStep=null) {
    const { context, width, height } = canvasContext(canvas);
    const aligned = [null, ...values];
    const total = Math.max(steps + 1, aligned.length);
    context.clearRect(0, 0, width, height);
    drawGrid(context, width, height);
    drawSeries(context, aligned, total, width, height, "#e67e22", false, true);
    drawMarker(context, markerStep, total, width, height);
}


function formatEta(seconds) {
    if (!Number.isFinite(seconds)) return "—";
    if (seconds < 60) return `${seconds.toFixed(1)}s`;
    const minutes = Math.floor(seconds / 60);
    return `${minutes}m ${Math.round(seconds - minutes * 60)}s`;
}


api.addEventListener("minimax_h3_unlimited_preview", event => {
    const data = event.detail;
    const node = data?.node_id == null ? null : findNode(app.graph, data.node_id);
    node?._minimaxUnlimitedPreview?.(data);
});


app.registerExtension({
    name: "MiniMaxH3.UnlimitedPreview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "MiniMaxH3UnlimitedPreview") return;

        const previousCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            previousCreated?.apply(this, arguments);
            const node = this;
            const root = document.createElement("div");
            root.style.cssText = "display:flex;flex-direction:column;width:100%;height:100%;min-height:410px;background:#111;border-radius:6px;overflow:hidden;color:#ddd;font:12px sans-serif;outline:none;";
            root.tabIndex = 0;

            const viewport = document.createElement("div");
            viewport.style.cssText = "position:relative;display:flex;width:100%;flex:1 1 auto;min-height:220px;background:#090909;overflow:hidden;";
            root.appendChild(viewport);

            const image = document.createElement("img");
            image.style.cssText = "display:block;width:100%;height:100%;object-fit:contain;background:#090909;";
            image.draggable = false;
            viewport.appendChild(image);

            const frameLabel = document.createElement("div");
            frameLabel.style.cssText = "position:absolute;right:8px;bottom:6px;color:#ffe600;font:bold 13px/1.1 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.2px;text-shadow:-1px -1px 0 #000,1px -1px 0 #000,-1px 1px 0 #000,1px 1px 0 #000,0 2px 2px #000;pointer-events:none;user-select:none;display:none;";
            viewport.appendChild(frameLabel);

            const transport = document.createElement("div");
            transport.style.cssText = "display:flex;align-items:center;gap:7px;box-sizing:border-box;height:43px;padding:4px 8px;background:#181818;border-top:1px solid #242424;";
            root.appendChild(transport);

            const playButton = document.createElement("button");
            playButton.type = "button";
            playButton.style.cssText = "display:flex;align-items:center;justify-content:center;width:22px;height:19px;padding:0;border:1px solid #555;border-radius:3px;background:#252525;color:#f4f4f4;font:12px/1 sans-serif;cursor:pointer;";
            playButton.textContent = "▶";
            playButton.title = "Play/pause (Space). Use Left/Right arrows for one preview frame.";
            transport.appendChild(playButton);

            const timelineShell = document.createElement("div");
            timelineShell.style.cssText = "position:relative;flex:1;height:33px;cursor:pointer;touch-action:none;";
            timelineShell.title = "Click or drag to seek; colors identify chunks";
            transport.appendChild(timelineShell);

            const timelineTrack = document.createElement("div");
            timelineTrack.style.cssText = "position:absolute;left:0;right:0;top:3px;height:5px;border-radius:3px;background:#333;box-shadow:0 0 0 1px #080808,0 1px 2px #000;overflow:hidden;";
            timelineShell.appendChild(timelineTrack);

            const timelinePlayhead = document.createElement("div");
            timelinePlayhead.style.cssText = "position:absolute;top:0;height:11px;width:2px;margin-left:-1px;border-radius:1px;background:#fff;box-shadow:0 0 2px #000,0 0 4px rgba(255,255,255,.75);pointer-events:none;display:none;";
            timelineShell.appendChild(timelinePlayhead);

            const shotBrackets = document.createElement("div");
            shotBrackets.style.cssText = "position:absolute;left:0;right:0;top:12px;height:19px;pointer-events:none;overflow:hidden;";
            timelineShell.appendChild(shotBrackets);

            const transportFrame = document.createElement("div");
            transportFrame.style.cssText = "min-width:62px;text-align:right;color:#aaa;font:10px/1 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap;";
            transport.appendChild(transportFrame);

            const status = document.createElement("div");
            status.style.cssText = "box-sizing:border-box;padding:7px 9px 3px;background:#1b1b1b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
            status.textContent = "Waiting for SamplerCustomAdvanced-Unlimited…";
            root.appendChild(status);

            const graphs = document.createElement("div");
            graphs.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:6px;background:#151515;";
            root.appendChild(graphs);

            function graphCell(label) {
                const cell = document.createElement("div");
                cell.style.cssText = "min-width:0;background:#101010;border-radius:3px;padding:4px;";
                const header = document.createElement("div");
                header.style.cssText = "height:16px;color:#aaa;display:flex;justify-content:space-between;gap:4px;";
                const title = document.createElement("span");
                title.textContent = label;
                const value = document.createElement("span");
                header.append(title, value);
                const canvas = document.createElement("canvas");
                canvas.style.cssText = "display:block;width:100%;height:64px;";
                cell.append(header, canvas);
                graphs.appendChild(cell);
                return { canvas, value };
            }

            const sigmaGraph = graphCell("σ / Δ");
            const timeGraph = graphCell("step time");
            sigmaGraph.canvas.style.cursor = "crosshair";
            timeGraph.canvas.style.cursor = "crosshair";
            sigmaGraph.canvas.title = "Hover to inspect the preview at a sampling step";
            timeGraph.canvas.title = "Hover to inspect the preview at a sampling step";

            let execution = null;
            let chunkCount = 0;
            let activeChunk = 0;
            let chunks = [];
            let chunkRanges = [];
            let shotRanges = [];
            let timelineTotalFrames = 0;
            let shotBracketKey = null;
            let playing = -1;
            let playingFrame = 0;
            let timer = null;
            let playbackSerial = 0;
            let framePending = false;
            let paused = false;
            let timelineDragging = false;
            let sigmas = [];
            let deltas = [];
            let stepTimes = [];
            let stepPreviews = [];
            let hoverStep = null;
            let currentStep = 0;
            let totalSteps = 0;
            let averageStepMs = null;
            let previewWidth = null;
            let previewHeight = null;
            // `sourceFps` is the rate used by the backend to make the stored
            // frame durations. `playbackFps` follows the live node widget and
            // may differ while an inference is already in progress.
            let sourceFps = null;
            let playbackFps = null;
            let fpsWidget = null;
            let startedAt = null;
            let completedElapsed = null;
            let elapsedTimer = null;
            let complete = false;

            function stop() {
                if (timer != null) clearTimeout(timer);
                timer = null;
                framePending = false;
                playbackSerial++;
            }

            function validFps(value) {
                const fps = Number(value);
                return Number.isFinite(fps) && fps > 0 ? fps : null;
            }

            function setSourceFps(value) {
                const fps = validFps(value);
                if (fps == null) return;
                sourceFps = fps;
                if (playbackFps == null) playbackFps = fps;
            }

            function currentPlaybackFps() {
                // Read the widget itself first. During workflow restoration
                // ComfyUI assigns serialized widget values without reliably
                // invoking callbacks, so a value cached during onNodeCreated
                // can otherwise remain stuck at the 24 FPS default.
                return validFps(fpsWidget?.value) || validFps(playbackFps) || validFps(sourceFps) || 24;
            }

            function frameDuration(group, frameIndex) {
                const source = validFps(group?.sourceFps) || validFps(sourceFps) || currentPlaybackFps();
                const stored = Number(group?.durations?.[frameIndex]);
                const sourceDuration = Number.isFinite(stored) && stored > 0 ? stored : 1000 / source;
                // Stored durations represent the source playback rate. Scaling
                // preserves H3's irregular latent-frame spans at any live FPS.
                return Math.max(1, Math.round(sourceDuration * source / currentPlaybackFps()));
            }

            function setPlaybackFps(value, restart=true) {
                const fps = validFps(value);
                if (fps == null || fps === playbackFps) return;
                playbackFps = fps;
                renderStatus();
                if (!restart) return;

                // Cancel the current timeout and re-schedule the currently
                // visible frame immediately at the new rate. This makes a live
                // widget change effective without waiting for a new backend
                // preview event or the next frame.
                if (hoverStep != null) {
                    const group = stepPreviews[hoverStep] || (available(playing) ? chunks[playing] : null);
                    if (group) inspectFrameGroup(group, hoverStep);
                } else if (!paused && available(playing)) {
                    show(playing, playingFrame);
                }
            }

            function available(index) {
                return Array.isArray(chunks[index]?.frames) && chunks[index].frames.length > 0;
            }

            const chunkColors = [
                "#f4b942", "#45b7d1", "#ef6f91", "#72c472", "#a98bea",
                "#f08a4b", "#55c7b3", "#d6cf57", "#6f9ee8", "#d676d4",
            ];

            function chunkSpan(index, estimate=1) {
                const group = chunks[index];
                const planned = chunkRanges[index];
                const plannedStart = Number(planned?.start);
                const plannedEnd = Number(planned?.end);
                if (Number.isFinite(plannedStart) && Number.isFinite(plannedEnd) && plannedEnd >= plannedStart) {
                    return plannedEnd - plannedStart + 1;
                }
                const start = Number(group?.outputStart);
                const end = Number(group?.outputEnd);
                if (Number.isFinite(start) && Number.isFinite(end) && end >= start) return end - start + 1;
                if (group?.frames?.length) return group.frames.length;
                return estimate;
            }

            function timelineLayout() {
                const known = chunks
                    .map((_, index) => available(index) ? chunkSpan(index, 1) : 0)
                    .filter(value => value > 0);
                const estimate = known.length ? Math.max(...known) : 1;
                const spans = Array.from({ length: chunkCount }, (_, index) => chunkSpan(index, estimate));
                const total = Math.max(1, spans.reduce((sum, value) => sum + value, 0));
                return { spans, total };
            }

            function renderShotBrackets() {
                const key = JSON.stringify([timelineTotalFrames, shotRanges]);
                if (key === shotBracketKey) return;
                shotBracketKey = key;
                shotBrackets.replaceChildren();
                const total = Math.max(0, timelineTotalFrames);
                if (!total || !shotRanges.length) return;
                for (const shot of shotRanges) {
                    const start = Math.max(0, Math.min(total, Number(shot.start) || 0));
                    const end = Math.max(start, Math.min(total - 1, Number(shot.end) || 0));
                    const sourceEnd = Number.isFinite(Number(shot.source_end)) ? Number(shot.source_end) : end;
                    const shotNumber = Number(shot.shot) || 1;
                    const left = start / total * 100;
                    const width = (end - start + 1) / total * 100;
                    const color = chunkColors[(shotNumber - 1) % chunkColors.length];
                    const bracket = document.createElement("div");
                    bracket.style.cssText = `position:absolute;left:${left}%;width:${width}%;top:0;height:8px;box-sizing:border-box;border-left:1px solid ${color};border-right:1px solid ${color};border-bottom:1px solid ${color};opacity:.9;pointer-events:auto;`;
                    bracket.title = `Shot ${shotNumber}: frames ${start}–${sourceEnd}`;
                    const label = document.createElement("div");
                    label.style.cssText = `position:absolute;left:1px;right:1px;top:8px;height:10px;overflow:hidden;text-align:center;white-space:nowrap;text-overflow:clip;color:${color};font:8px/10px ui-monospace,SFMono-Regular,Consolas,monospace;text-shadow:0 1px 1px #000;`;
                    label.textContent = `S${shotNumber} ${start}–${sourceEnd}`;
                    bracket.appendChild(label);
                    shotBrackets.appendChild(bracket);
                }
            }

            function flatFrames() {
                const entries = [];
                for (let chunkIndex = 0; chunkIndex < chunkCount; chunkIndex++) {
                    const group = chunks[chunkIndex];
                    if (!group?.frames?.length) continue;
                    for (let frameIndex = 0; frameIndex < group.frames.length; frameIndex++) {
                        entries.push({ chunkIndex, frameIndex });
                    }
                }
                return entries;
            }

            function currentFlatIndex(entries=flatFrames()) {
                return entries.findIndex(entry => entry.chunkIndex === playing && entry.frameIndex === playingFrame);
            }

            function displayedFrameNumber() {
                const group = chunks[playing];
                const exact = Number(group?.frameNumbers?.[playingFrame]);
                if (Number.isFinite(exact)) return Math.round(exact);
                const entries = flatFrames();
                const index = currentFlatIndex(entries);
                return index >= 0 ? index : null;
            }

            function formatFrameLabel(frameNumber) {
                if (frameNumber == null) return "";
                const frame = Math.round(frameNumber);
                const chunk = chunkRanges.find(range => {
                    const start = Number(range?.start);
                    const end = Number(range?.end);
                    return Number.isFinite(start) && Number.isFinite(end) && frame >= start && frame <= end;
                });
                const shot = shotRanges.find(range => {
                    const start = Number(range?.start);
                    const end = Number(range?.end);
                    return Number.isFinite(start) && Number.isFinite(end) && frame >= start && frame <= end;
                });
                const chunkNumber = Number(chunk?.chunk);
                const shotNumber = Number(shot?.shot);
                return Number.isFinite(chunkNumber) && Number.isFinite(shotNumber)
                    ? `S${Math.round(shotNumber)}/C${Math.round(chunkNumber)}/${frame}`
                    : `${frame}`;
            }

            function renderTransport() {
                const entries = flatFrames();
                const showPlay = paused || !entries.length;
                playButton.textContent = showPlay ? "▶" : "❚❚";
                playButton.setAttribute("aria-label", showPlay ? "Play preview" : "Pause preview");
                const flatIndex = currentFlatIndex(entries);
                transportFrame.textContent = flatIndex >= 0 ? `${flatIndex + 1}/${entries.length}` : `—/${entries.length || "—"}`;
                const frameNumber = displayedFrameNumber();
                frameLabel.textContent = formatFrameLabel(frameNumber);
                frameLabel.style.display = frameNumber == null ? "none" : "block";

                const { spans, total } = timelineLayout();
                let offset = 0;
                const stops = [];
                for (let index = 0; index < spans.length; index++) {
                    const start = offset / total * 100;
                    offset += spans[index];
                    const end = offset / total * 100;
                    const color = chunkColors[index % chunkColors.length];
                    const alpha = available(index) ? "e8" : "35";
                    stops.push(`${color}${alpha} ${start}%`, `${color}${alpha} ${end}%`);
                }
                timelineTrack.style.background = stops.length
                    ? `linear-gradient(to right, ${stops.join(",")})`
                    : "#333";
                renderShotBrackets();

                if (!available(playing)) {
                    timelinePlayhead.style.display = "none";
                    return;
                }
                const before = spans.slice(0, playing).reduce((sum, value) => sum + value, 0);
                const fraction = chunks[playing].frames.length < 2
                    ? 0
                    : playingFrame / (chunks[playing].frames.length - 1);
                const position = (before + fraction * spans[playing]) / total * 100;
                timelinePlayhead.style.left = `${Math.max(0, Math.min(100, position))}%`;
                timelinePlayhead.style.display = "block";
            }

            function displaySource(source, valid, displayed) {
                if (!source) {
                    displayed?.(false);
                    return;
                }
                const replacement = new Image();
                replacement.onload = () => {
                    if (valid && !valid()) return;
                    image.src = replacement.src;
                    displayed?.(true);
                };
                replacement.onerror = () => {
                    if (!valid || valid()) displayed?.(false);
                };
                replacement.src = source;
            }

            function nextAvailable(after) {
                for (let index = after + 1; index < chunkCount; index++) {
                    if (available(index)) return index;
                }
                for (let index = 0; index <= after && index < chunkCount; index++) {
                    if (available(index)) return index;
                }
                return -1;
            }

            function playFrameGroup(index, group, frameIndex, serial) {
                if (serial !== playbackSerial || paused || hoverStep != null || !group?.frames?.length) return;
                const boundedFrame = Math.max(0, Math.min(frameIndex, group.frames.length - 1));
                playing = index;
                playingFrame = boundedFrame;
                renderTransport();
                const duration = frameDuration(group, boundedFrame);
                framePending = true;
                displaySource(
                    group.frames[boundedFrame],
                    () => serial === playbackSerial && !paused && hoverStep == null,
                    () => {
                        framePending = false;
                        timer = setTimeout(() => {
                            timer = null;
                            if (serial !== playbackSerial || paused || hoverStep != null) return;
                            if (boundedFrame + 1 < group.frames.length) {
                                playFrameGroup(index, group, boundedFrame + 1, serial);
                                return;
                            }
                            const next = nextAvailable(index);
                            if (next >= 0) show(next);
                        }, duration);
                    },
                );
            }

            function show(index, frameIndex=0) {
                if (!available(index) || hoverStep != null) return;
                stop();
                playing = index;
                playingFrame = Math.max(0, Math.min(frameIndex, chunks[index].frames.length - 1));
                renderTransport();
                const group = chunks[index];
                const serial = playbackSerial;
                if (paused) {
                    framePending = true;
                    displaySource(
                        group.frames[playingFrame],
                        () => serial === playbackSerial && paused && hoverStep == null,
                        () => { framePending = false; },
                    );
                } else {
                    playFrameGroup(index, group, playingFrame, serial);
                }
            }

            function restorePlayback() {
                if (available(playing)) {
                    show(playing, playingFrame);
                    return;
                }
                const first = chunks.findIndex((_, index) => available(index));
                if (first >= 0) show(first, 0);
            }

            function leaveGraphInspection() {
                if (hoverStep == null) return;
                hoverStep = null;
                redrawGraphs();
            }

            function setPaused(value) {
                paused = Boolean(value);
                leaveGraphInspection();
                stop();
                renderTransport();
                restorePlayback();
                renderStatus();
            }

            function stepPreviewFrame(direction) {
                const entries = flatFrames();
                if (!entries.length) return;
                leaveGraphInspection();
                paused = true;
                let position = currentFlatIndex(entries);
                if (position < 0) position = direction < 0 ? entries.length : -1;
                position = Math.max(0, Math.min(entries.length - 1, position + direction));
                const target = entries[position];
                show(target.chunkIndex, target.frameIndex);
                renderStatus();
            }

            function seekTimeline(clientX) {
                const rect = timelineShell.getBoundingClientRect();
                if (rect.width <= 0 || chunkCount < 1) return;
                const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
                const { spans, total } = timelineLayout();
                let cursor = ratio * total;
                let targetChunk = spans.length - 1;
                let chunkStart = 0;
                for (let index = 0, offset = 0; index < spans.length; index++) {
                    if (cursor <= offset + spans[index] || index === spans.length - 1) {
                        targetChunk = index;
                        chunkStart = offset;
                        break;
                    }
                    offset += spans[index];
                }
                if (!available(targetChunk)) {
                    const candidates = chunks
                        .map((_, index) => available(index) ? index : -1)
                        .filter(index => index >= 0);
                    if (!candidates.length) return;
                    targetChunk = candidates.reduce((best, index) =>
                        Math.abs(index - targetChunk) < Math.abs(best - targetChunk) ? index : best
                    );
                    chunkStart = spans.slice(0, targetChunk).reduce((sum, value) => sum + value, 0);
                    cursor = chunkStart + (targetChunk < playing ? spans[targetChunk] : 0);
                }
                const group = chunks[targetChunk];
                const fraction = Math.max(0, Math.min(1, (cursor - chunkStart) / Math.max(1, spans[targetChunk])));
                const targetFrame = Math.round(fraction * Math.max(0, group.frames.length - 1));
                paused = true;
                leaveGraphInspection();
                show(targetChunk, targetFrame);
                renderStatus();
            }

            playButton.addEventListener("click", event => {
                event.preventDefault();
                event.stopPropagation();
                setPaused(!paused);
                root.focus({ preventScroll: true });
            });
            viewport.addEventListener("click", () => root.focus({ preventScroll: true }));
            timelineShell.addEventListener("pointerdown", event => {
                event.preventDefault();
                event.stopPropagation();
                timelineDragging = true;
                timelineShell.setPointerCapture?.(event.pointerId);
                seekTimeline(event.clientX);
                root.focus({ preventScroll: true });
            });
            timelineShell.addEventListener("pointermove", event => {
                if (!timelineDragging) return;
                event.preventDefault();
                seekTimeline(event.clientX);
            });
            const finishTimelineDrag = event => {
                if (!timelineDragging) return;
                timelineDragging = false;
                timelineShell.releasePointerCapture?.(event.pointerId);
            };
            timelineShell.addEventListener("pointerup", finishTimelineDrag);
            timelineShell.addEventListener("pointercancel", finishTimelineDrag);
            transport.addEventListener("mousedown", event => event.stopPropagation());
            root.addEventListener("keydown", event => {
                if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
                    event.preventDefault();
                    event.stopPropagation();
                    stepPreviewFrame(event.key === "ArrowLeft" ? -1 : 1);
                } else if (event.key === " " || event.code === "Space") {
                    event.preventDefault();
                    event.stopPropagation();
                    setPaused(!paused);
                }
            });

            function renderStatus() {
                const resolution = previewWidth && previewHeight ? `${previewWidth}×${previewHeight}` : "resolution —";
                const fps = Number.isFinite(currentPlaybackFps()) ? `${Number(currentPlaybackFps().toFixed(3))} fps` : "fps —";
                const secondsPerStep = Number.isFinite(averageStepMs) ? `${(averageStepMs / 1000).toFixed(2)}s/step` : "—s/step";
                const remainingSteps = Math.max(0, totalSteps - currentStep) + Math.max(0, chunkCount - activeChunk - 1) * totalSteps;
                const eta = Number.isFinite(averageStepMs) ? formatEta(remainingSteps * averageStepMs / 1000) : "—";
                const elapsedSeconds = completedElapsed ?? (startedAt == null ? NaN : (performance.now() - startedAt) / 1000);
                const elapsed = formatEta(elapsedSeconds);
                const chunk = chunkCount ? `Chunk ${activeChunk + 1}/${chunkCount}` : "Chunk —/—";
                const displayStep = hoverStep ?? currentStep;
                const inspecting = hoverStep == null ? "" : "inspect ";
                status.textContent = `${complete ? "Complete · " : ""}${paused ? "Paused · " : ""}${chunk} · ${resolution} · ${fps} · ${inspecting}step ${displayStep}/${totalSteps || "—"} · ${secondsPerStep} · Elapsed ${elapsed} · ETA ${eta}`;
                sigmaGraph.value.textContent = Number.isFinite(deltas[displayStep - 1]) ? deltas[displayStep - 1].toFixed(3) : "—";
                timeGraph.value.textContent = Number.isFinite(stepTimes[displayStep - 1]) ? `${(stepTimes[displayStep - 1] / 1000).toFixed(2)}s` : "—";
            }

            function redrawGraphs() {
                drawSigmaDelta(sigmaGraph.canvas, sigmas, deltas, totalSteps, hoverStep);
                drawStepTimes(timeGraph.canvas, stepTimes, totalSteps, hoverStep);
            }

            function resetExecution(data) {
                execution = data.execution;
                chunkCount = data.chunk_count || 0;
                activeChunk = data.chunk ?? 0;
                chunks = new Array(chunkCount);
                chunkRanges = Array.isArray(data.chunk_ranges) ? data.chunk_ranges.slice() : [];
                shotRanges = Array.isArray(data.shot_ranges) ? data.shot_ranges.slice() : [];
                timelineTotalFrames = Number.isFinite(Number(data.total_frames)) ? Number(data.total_frames) : 0;
                shotBracketKey = null;
                playing = -1;
                playingFrame = 0;
                paused = false;
                timelineDragging = false;
                sigmas = Array.isArray(data.sigmas) ? data.sigmas : [];
                deltas = [];
                stepTimes = [];
                stepPreviews = [];
                hoverStep = null;
                currentStep = 0;
                totalSteps = data.steps || Math.max(0, sigmas.length - 1);
                averageStepMs = null;
                previewWidth = null;
                previewHeight = null;
                setSourceFps(data.fps);
                const elapsedMs = Number.isFinite(data.elapsed_ms) ? data.elapsed_ms : 0;
                startedAt = performance.now() - elapsedMs;
                completedElapsed = null;
                complete = false;
                if (elapsedTimer != null) clearInterval(elapsedTimer);
                elapsedTimer = setInterval(renderStatus, 1000);
                stop();
                image.removeAttribute("src");
                frameLabel.style.display = "none";
                renderStatus();
                renderTransport();
                redrawGraphs();
            }

            node._minimaxUnlimitedPreview = data => {
                if (data.action === "reset") {
                    if (execution !== null && Number(data.execution) < Number(execution)) return;
                    resetExecution(data);
                    return;
                }
                if (data.execution !== execution) {
                    if (execution !== null) return;
                    resetExecution(data);
                }
                if (data.action === "sample_start") {
                    activeChunk = data.chunk ?? activeChunk;
                    sigmas = Array.isArray(data.sigmas) ? data.sigmas : [];
                    deltas = [];
                    stepTimes = [];
                    stepPreviews = [];
                    hoverStep = null;
                    currentStep = 0;
                    totalSteps = data.steps || Math.max(0, sigmas.length - 1);
                    averageStepMs = null;
                    setSourceFps(data.fps);
                    renderStatus();
                    renderTransport();
                    redrawGraphs();
                    return;
                }
                if (data.action === "complete") {
                    completedElapsed = Number.isFinite(data.elapsed_ms)
                        ? data.elapsed_ms / 1000
                        : startedAt == null ? null : (performance.now() - startedAt) / 1000;
                    if (elapsedTimer != null) clearInterval(elapsedTimer);
                    elapsedTimer = null;
                    complete = true;
                    renderStatus();
                    renderTransport();
                    if (hoverStep == null && timer == null && !framePending) restorePlayback();
                    return;
                }
                if (data.action === "progress") {
                    activeChunk = data.chunk ?? activeChunk;
                    currentStep = data.step || currentStep;
                    totalSteps = data.steps || totalSteps;
                    if (Array.isArray(data.sigmas)) sigmas = data.sigmas;
                    deltas[currentStep - 1] = data.delta;
                    stepTimes[currentStep - 1] = data.step_ms;
                    averageStepMs = data.avg_step_ms;
                    setSourceFps(data.fps);
                    renderStatus();
                    redrawGraphs();
                    return;
                }
                if (data.action !== "chunk" || (!Array.isArray(data.frames) && !data.image)) return;

                const index = data.chunk;
                activeChunk = index;
                currentStep = data.step || currentStep;
                totalSteps = data.steps || totalSteps;
                if (Array.isArray(data.sigmas)) sigmas = data.sigmas;
                const encodedFrames = Array.isArray(data.frames) ? data.frames : [data.image];
                const frameDurations = Array.isArray(data.frame_durations_ms)
                    ? data.frame_durations_ms
                    : [data.duration_ms || 1000];
                const group = {
                    frames: encodedFrames.map(frame => `data:image/webp;base64,${frame}`),
                    durations: frameDurations,
                    frameNumbers: Array.isArray(data.frame_numbers) ? data.frame_numbers : [],
                    outputStart: data.output_start,
                    outputEnd: data.output_end,
                    sourceFps: data.fps,
                    step: currentStep,
                };
                chunks[index] = group;
                stepPreviews[currentStep] = group;
                previewWidth = data.width;
                previewHeight = data.height;
                setSourceFps(data.fps);
                renderStatus();
                renderTransport();
                redrawGraphs();
                if (hoverStep == null && timer == null && !framePending) restorePlayback();
            };

            async function restoreServerState(attempt=0) {
                if (node.id == null || Number(node.id) < 0) {
                    if (attempt < 20) setTimeout(() => restoreServerState(attempt + 1), 100);
                    return;
                }
                try {
                    // Workflow widget restoration normally completes before
                    // the asynchronous history request. Synchronize it here
                    // so restored playback starts at the visible FPS value.
                    setPlaybackFps(fpsWidget?.value, false);
                    const response = await api.fetchApi(
                        `/minimax_h3_unlimited_preview/state?node_id=${encodeURIComponent(node.id)}`,
                        { cache: "no-store" },
                    );
                    if (!response.ok) return;
                    const snapshot = await response.json();
                    if (!snapshot?.reset) return;
                    if (execution !== null && Number(snapshot.execution) < Number(execution)) return;
                    node._minimaxUnlimitedPreview(snapshot.reset);
                    if (snapshot.sample_start) node._minimaxUnlimitedPreview(snapshot.sample_start);
                    if (snapshot.progress) node._minimaxUnlimitedPreview(snapshot.progress);
                    for (const chunk of snapshot.chunks || []) node._minimaxUnlimitedPreview(chunk);
                    if (Array.isArray(snapshot.deltas)) deltas = snapshot.deltas.slice();
                    if (Array.isArray(snapshot.step_times)) stepTimes = snapshot.step_times.slice();
                    renderStatus();
                    redrawGraphs();
                    if (snapshot.complete) node._minimaxUnlimitedPreview(snapshot.complete);
                    if (hoverStep == null && timer == null && !framePending) restorePlayback();
                } catch (error) {
                    console.warn("MiniMax H3 preview history restore failed", error);
                }
            }

            function inspectFrameGroup(group, inspectedStep) {
                if (!group?.frames?.length || hoverStep !== inspectedStep) return;
                stop();
                const serial = playbackSerial;
                const play = frameIndex => {
                    if (serial !== playbackSerial || hoverStep !== inspectedStep) return;
                    const boundedFrame = frameIndex % group.frames.length;
                    const inspectedNumber = Number(group.frameNumbers?.[boundedFrame]);
                    if (Number.isFinite(inspectedNumber)) {
                        frameLabel.textContent = formatFrameLabel(inspectedNumber);
                        frameLabel.style.display = "block";
                    }
                    const duration = frameDuration(group, boundedFrame);
                    framePending = true;
                    displaySource(
                        group.frames[boundedFrame],
                        () => serial === playbackSerial && hoverStep === inspectedStep,
                        () => {
                            framePending = false;
                            timer = setTimeout(() => {
                                timer = null;
                                play(boundedFrame + 1);
                            }, duration);
                        },
                    );
                };
                play(0);
            }

            function inspectGraph(event) {
                const rect = event.currentTarget.getBoundingClientRect();
                const count = Math.max(totalSteps + 1, sigmas.length, stepTimes.length + 1, stepPreviews.length);
                if (count < 2) return;
                const position = Math.max(0, Math.min(1, (event.clientX - rect.left - 4) / Math.max(1, rect.width - 8)));
                const step = Math.round(position * (count - 1));
                if (step === hoverStep) return;
                hoverStep = step;
                stop();
                if (stepPreviews[step]) inspectFrameGroup(stepPreviews[step], step);
                else if (available(playing)) inspectFrameGroup(chunks[playing], step);
                renderStatus();
                redrawGraphs();
            }

            function stopInspecting() {
                if (hoverStep == null) return;
                hoverStep = null;
                renderStatus();
                redrawGraphs();
                restorePlayback();
            }

            for (const canvas of [sigmaGraph.canvas, timeGraph.canvas]) {
                canvas.addEventListener("mousemove", inspectGraph);
                canvas.addEventListener("mouseleave", stopInspecting);
                canvas.addEventListener("mousedown", event => event.stopPropagation());
            }

            const resizeObserver = new ResizeObserver(redrawGraphs);
            resizeObserver.observe(graphs);
            node.addDOMWidget("preview", "minimax_h3_unlimited_preview", root, { serialize: false });
            node.setSize([Math.max(node.size?.[0] || 460, 460), Math.max(node.size?.[1] || 540, 540)]);

            fpsWidget = node.widgets?.find(widget => widget.name === "fps");
            const previousFpsCallback = fpsWidget?.callback;
            if (fpsWidget) {
                fpsWidget.callback = function () {
                    const result = previousFpsCallback?.apply(this, arguments);
                    setPlaybackFps(arguments[0] ?? fpsWidget.value);
                    return result;
                };
                setPlaybackFps(fpsWidget.value, false);
            }
            setTimeout(() => restoreServerState(), 0);

            const previousRemoved = node.onRemoved;
            node.onRemoved = function () {
                stop();
                if (elapsedTimer != null) clearInterval(elapsedTimer);
                resizeObserver.disconnect();
                node._minimaxUnlimitedPreview = null;
                previousRemoved?.apply(this, arguments);
            };
        };
    },
});
