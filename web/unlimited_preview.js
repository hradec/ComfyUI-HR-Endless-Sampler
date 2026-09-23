const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;


function chunkPromptDescription(chunk) {
    // The actual H3 prompt is authoritative, including runs without a director.
    const prompt = String(chunk.h3_prompt || "");
    const field = prompt.match(/(?:^|\n)[ \t]*(?:detailed_description|integrated_multimodal_description)[ \t]*:[ \t]*([\s\S]*?)(?=\n[ \t]*(?:overall_soundscape|non_diegetic_music|retention_analysis|summary)[ \t]*:|$)/i);
    return field ? field[1].trim() : String(chunk.gemma_detailed_description || "").trim();
}


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


function formatChunkClock(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds < 0) return null;
    const rounded = Math.round(seconds);
    const minutes = Math.floor(rounded / 60);
    const remainder = rounded % 60;
    return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}


function chunkTimingLines(range) {
    const lines = [];
    const totalSeconds = Number(range.chunk_total_seconds);
    const samplerSeconds = Math.max(0, Number(range.h3_render_seconds) || 0);
    const gemmaSeconds = Math.max(0, Number(range.gemma_seconds) || 0);
    if (Number.isFinite(totalSeconds) && totalSeconds >= 0) {
        const miscSeconds = Math.max(0, totalSeconds - samplerSeconds - gemmaSeconds);
        lines.push(
            `Chunk processing: ${formatChunkClock(totalSeconds)} `
            + `( sampler:${formatChunkClock(samplerSeconds)} + gemma4:${formatChunkClock(gemmaSeconds)} `
            + `+ misc:${formatChunkClock(miscSeconds)} )`
        );
    }
    const preproduction = formatChunkClock(range.gemma_preproduction_seconds);
    if (preproduction && Number(range.gemma_preproduction_seconds) > 0) {
        lines.push(`Gemma4 preproduction included above: ${preproduction}`);
    }
    return lines;
}


function shotsOverlappingChunk(chunk, shotRanges) {
    const start = Number(chunk?.start);
    const end = Number(chunk?.end);
    if (!Number.isFinite(start) || !Number.isFinite(end)) return [];
    return shotRanges
        .filter(shot => Number(shot.end) >= start && Number(shot.start) <= end)
        .sort((left, right) => Number(left.start) - Number(right.start));
}


function coloredShotPromptSegments(description, chunk, shotRanges, colors) {
    const text = String(description || "");
    if (!text) return [];
    // Full-prompt headers must not shift the description's shot-color mapping.
    const body = chunkPromptDescription({ h3_prompt: text });
    if (body && body !== text) {
        const start = text.indexOf(body);
        return [{ text: text.slice(0, start), color: "#b8b8d0" },
            ...coloredShotPromptSegments(body, chunk, shotRanges, colors),
            { text: text.slice(start + body.length), color: "#b8b8d0" }];
    }
    const overlapping = shotsOverlappingChunk(chunk, shotRanges);
    const colorFor = shot => shot
        ? colors[(Math.max(1, Number(shot.shot) || 1) - 1) % colors.length]
        : null;
    const markers = [...text.matchAll(/\[Shot\s+(\d+)\]/gi)];
    if (!markers.length) {
        return [{ text, color: colorFor(overlapping[0]) || colors[0] }];
    }

    const segments = [];
    const hasPrefix = markers[0].index > 0;
    if (hasPrefix) {
        segments.push({
            text: text.slice(0, markers[0].index),
            color: colorFor(overlapping[0]) || colors[0],
        });
    }
    // Gemma's markers are physical-chunk-local shot numbers, while the
    // timeline colors represent source/global shots. Map them by chronological
    // overlap, not by equal numbers. If continuation prose precedes the first
    // marker, it belongs to the already-active first source shot and that
    // marker begins the next overlapping shot.
    const sequentialOffset = hasPrefix ? 1 : 0;
    for (let index = 0; index < markers.length; index++) {
        const marker = markers[index];
        const next = markers[index + 1];
        const shot = overlapping[index + sequentialOffset];
        const fallbackNumber = Number(marker[1]) || index + 1;
        segments.push({
            text: text.slice(marker.index, next ? next.index : text.length),
            color: colorFor(shot) || colors[(fallbackNumber - 1) % colors.length],
        });
    }
    return segments;
}


function createColoredChunkTooltip() {
    const tooltip = document.createElement("div");
    tooltip.style.cssText = "position:fixed;z-index:100001;display:none;box-sizing:border-box;max-width:min(760px,calc(100vw - 24px));max-height:min(70vh,640px);overflow:hidden;padding:9px 11px;border:1px solid #555;border-radius:5px;background:rgba(18,18,18,.97);color:#ddd;box-shadow:0 7px 24px rgba(0,0,0,.75);font:11px/1.38 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;pointer-events:none;user-select:none;";
    document.body.appendChild(tooltip);

    function line(text, style="") {
        const element = document.createElement("div");
        element.textContent = text;
        if (style) element.style.cssText = style;
        tooltip.appendChild(element);
        return element;
    }

    function position(event) {
        const margin = 12;
        const gap = 14;
        const rect = tooltip.getBoundingClientRect();
        let left = event.clientX + gap;
        let top = event.clientY + gap;
        if (left + rect.width > window.innerWidth - margin) left = event.clientX - rect.width - gap;
        if (top + rect.height > window.innerHeight - margin) top = event.clientY - rect.height - gap;
        tooltip.style.left = `${Math.max(margin, left)}px`;
        tooltip.style.top = `${Math.max(margin, top)}px`;
    }

    return {
        show(event, { help, chunk, timing, description, retentionAnalysis, fullPrompt, showFullPrompt, shotRanges, colors, waitingText }) {
            tooltip.replaceChildren();
            line(help, "color:#999;margin-bottom:6px;");
            const chunkNumber = Number(chunk.chunk) || 1;
            const chunkColor = colors[(Math.max(1, chunkNumber) - 1) % colors.length] || "#fff";
            line(`Chunk ${chunkNumber}`, `color:${chunkColor};font-weight:700;`);
            if (timing.length) {
                for (const item of timing) line(item, "color:#bbb;");
            } else {
                line("Timing: waiting for this chunk to finish.", "color:#888;");
            }
            if (!showFullPrompt && retentionAnalysis) {
                line("Per-chunk retention_analysis:", "color:#bbb;margin-top:7px;margin-bottom:2px;");
                line(retentionAnalysis, "color:#d8c7a0;");
            }
            line(showFullPrompt ? "Full prompt sent to H3:" : "chunk detailed_description:", "color:#bbb;margin-top:7px;margin-bottom:2px;");
            const text = showFullPrompt ? fullPrompt : description;
            if (text) {
                const prompt = document.createElement("div");
                for (const segment of coloredShotPromptSegments(text, chunk, shotRanges, colors)) {
                    // H3 dialogue tags remain literal text; only their presentation changes.
                    for (const [index, part] of segment.text.split(/(<d>[\s\S]*?<\/d>)/gi).entries()) {
                        const span = document.createElement("span");
                        span.textContent = part;
                        span.style.color = segment.color;
                        if (index % 2) {
                            span.style.color = `color-mix(in srgb, ${segment.color} 65%, white)`;
                            span.style.fontWeight = "700";
                        }
                        prompt.appendChild(span);
                    }
                }
                tooltip.appendChild(prompt);
            } else {
                line(waitingText, "color:#888;");
            }
            tooltip.style.display = "block";
            position(event);
        },
        move: position,
        hide() {
            tooltip.style.display = "none";
        },
        remove() {
            tooltip.remove();
        },
    };
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


function projectedRenderTiming(elapsedSeconds, workDone, workTotal, fallbackEtaSeconds=NaN) {
    // Estimate remaining and total wall time from completed planned work units.
    const fraction = workTotal > 0 ? Math.max(0, Math.min(1, workDone / workTotal)) : NaN;
    const etaSeconds = Number.isFinite(elapsedSeconds) && fraction > 0
        ? elapsedSeconds * (1 - fraction) / fraction
        : fallbackEtaSeconds;
    return {
        etaSeconds,
        totalSeconds: Number.isFinite(elapsedSeconds) && Number.isFinite(etaSeconds)
            ? elapsedSeconds + etaSeconds
            : NaN,
    };
}


api.addEventListener("hr_endless_sampler_preview", event => {
    const data = event.detail;
    const node = data?.node_id == null ? null : findNode(app.graph, data.node_id);
    node?._hrEndlessSamplerPreview?.(data);
});

api.addEventListener("hr_endless_sampler_cache_restore", event => {
    const data = event.detail;
    const node = data?.node_id == null ? null : findNode(app.graph, data.node_id);
    node?._hrEndlessCacheRestoreProgress?.(data);
});




app.registerExtension({
    name: "HREndlessSampler.Preview",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== "HREndlessSamplerPreview") return;

        const previousCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            previousCreated?.apply(this, arguments);
            const node = this;
            // LiteGraph serializes node properties with the workflow, including this local player state.
            const savedPlayerState = node.properties?.hr_endless_sampler_preview_player || {};
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

            // Browser compositing applies this transfer to whichever preview
            // image is on screen, decoded or latent; it never requests a VAE
            // decode or new frame data.
            const inverseGammaFilterId = `hr-endless-inverse-gamma-${node.id}`;
            const filterSvg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
            filterSvg.setAttribute("width", "0");
            filterSvg.setAttribute("height", "0");
            filterSvg.style.cssText = "position:absolute;pointer-events:none;";
            const filter = document.createElementNS("http://www.w3.org/2000/svg", "filter");
            filter.setAttribute("id", inverseGammaFilterId);
            // SVG filters default to linearRGB. This preview deliberately applies
            // pow() to the displayed RGB samples themselves, matching the sampler
            // display experiment rather than doing a color-managed conversion.
            filter.setAttribute("color-interpolation-filters", "sRGB");
            const transfer = document.createElementNS("http://www.w3.org/2000/svg", "feComponentTransfer");
            for (const channel of ["R", "G", "B"]) {
                const curve = document.createElementNS("http://www.w3.org/2000/svg", `feFunc${channel}`);
                curve.setAttribute("type", "gamma");
                curve.setAttribute("amplitude", "1");
                curve.setAttribute("exponent", "0.45");
                curve.setAttribute("offset", "0");
                transfer.appendChild(curve);
            }
            filter.appendChild(transfer);
            filterSvg.appendChild(filter);
            viewport.appendChild(filterSvg);

            // Finalized chunks carry their decoded H3 soundtrack as an
            // in-memory WAV. Keep the media element invisible: the custom
            // image timeline remains the authoritative visual transport.
            let audioPlayer = document.createElement("audio");
            audioPlayer.preload = "auto";
            audioPlayer.style.display = "none";
            audioPlayer.preservesPitch = true;
            viewport.appendChild(audioPlayer);
            let audioStandbyPlayer = document.createElement("audio");
            audioStandbyPlayer.preload = "auto";
            audioStandbyPlayer.style.display = "none";
            audioStandbyPlayer.preservesPitch = true;
            viewport.appendChild(audioStandbyPlayer);

            const frameLabel = document.createElement("div");
            frameLabel.style.cssText = "position:absolute;right:8px;bottom:6px;color:#ffe600;font:bold 13px/1.1 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.2px;text-shadow:-1px -1px 0 #000,1px -1px 0 #000,-1px 1px 0 #000,1px 1px 0 #000,0 2px 2px #000;pointer-events:none;user-select:none;display:none;";
            viewport.appendChild(frameLabel);

            let previewAnnotation = "";
            const annotationLabel = document.createElement("div");
            annotationLabel.style.cssText = "position:absolute;left:12px;right:12px;top:10px;z-index:3;text-align:center;color:#ffe600;font:bold 16px/1.2 sans-serif;-webkit-text-stroke:1px #000;text-shadow:1px 1px 0 #000,-1px -1px 0 #000,1px -1px 0 #000,-1px 1px 0 #000,0 2px 3px #000;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;pointer-events:none;user-select:none;display:none;";
            viewport.appendChild(annotationLabel);

            function renderAnnotation() {
                annotationLabel.textContent = previewAnnotation;
                annotationLabel.title = previewAnnotation;
                annotationLabel.style.display = previewAnnotation ? "block" : "none";
            }

            async function saveAnnotation(value) {
                previewAnnotation = String(value || "").replace(/[\r\n]+/g, " ").trim().slice(0, 256);
                renderAnnotation();
                const response = await api.fetchApi("/hr_endless_sampler_preview/annotation", {
                    method: "POST", headers: { "Content-Type": "application/json" },
                    body: JSON.stringify({ node_id: node.id, execution, annotation: previewAnnotation }),
                });
                const body = await response.text();
                let result = {};
                if (body) {
                    try { result = JSON.parse(body); }
                    catch (error) {
                        if (response.status === 404 || response.status === 405) {
                            throw new Error("The annotation endpoint is not loaded. Restart ComfyUI, then refresh the browser.");
                        }
                        throw new Error(`Annotation save failed (HTTP ${response.status}): ${body.slice(0, 240)}`);
                    }
                }
                if (!response.ok) throw new Error(result.error || `HTTP ${response.status}`);
                if (typeof result.annotation !== "string") throw new Error("The annotation endpoint returned an invalid response.");
                previewAnnotation = result.annotation;
                renderAnnotation();
            }

            function openAnnotationWindow() {
                const backdrop = document.createElement("div");
                backdrop.style.cssText = "position:fixed;inset:0;z-index:10002;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.55);";
                const dialog = document.createElement("div");
                dialog.style.cssText = "width:min(360px,calc(100vw - 32px));padding:12px;background:#242424;border:1px solid #666;border-radius:5px;box-shadow:0 5px 24px #000;color:#eee;font:12px sans-serif;";
                const heading = document.createElement("div");
                heading.textContent = "Preview annotation";
                heading.style.cssText = "margin-bottom:8px;font-weight:bold;";
                const input = document.createElement("input");
                input.type = "text";
                input.maxLength = 256;
                input.value = previewAnnotation;
                input.placeholder = "Type one line to show over the video";
                input.style.cssText = "box-sizing:border-box;width:100%;height:30px;padding:5px 7px;background:#111;border:1px solid #666;border-radius:3px;color:#fff;font:12px sans-serif;";
                const actions = document.createElement("div");
                actions.style.cssText = "display:flex;justify-content:flex-end;gap:6px;margin-top:10px;";
                const cancel = document.createElement("button");
                cancel.textContent = "Cancel";
                const save = document.createElement("button");
                save.textContent = "Save";
                for (const button of [cancel, save]) button.style.cssText = "padding:4px 10px;border:1px solid #666;border-radius:3px;background:#333;color:#eee;cursor:pointer;";
                const close = () => backdrop.remove();
                cancel.addEventListener("click", close);
                backdrop.addEventListener("pointerdown", event => { if (event.target === backdrop) close(); });
                input.addEventListener("keydown", event => {
                    if (event.key === "Escape") close();
                    if (event.key === "Enter") save.click();
                });
                save.addEventListener("click", async () => {
                    save.disabled = true;
                    try { await saveAnnotation(input.value); close(); }
                    catch (error) { window.alert(error.message); save.disabled = false; }
                });
                actions.append(cancel, save);
                dialog.append(heading, input, actions);
                backdrop.appendChild(dialog);
                document.body.appendChild(backdrop);
                input.focus();
                input.select();
            }

            const cacheReuseLabel = document.createElement("div");
            cacheReuseLabel.style.cssText = "position:absolute;left:8px;bottom:6px;color:#69b76f;font:bold 11px/1.1 ui-monospace,SFMono-Regular,Consolas,monospace;letter-spacing:.1px;text-shadow:-1px -1px 0 #000,1px -1px 0 #000,-1px 1px 0 #000,1px 1px 0 #000,0 2px 2px #000;pointer-events:none;user-select:none;display:none;white-space:nowrap;";
            viewport.appendChild(cacheReuseLabel);

            const linearDisplayButton = document.createElement("button");
            linearDisplayButton.type = "button";
            linearDisplayButton.textContent = "L";
            linearDisplayButton.title = "Every preview image: RGB sRGB^0.45, browser-only";
            linearDisplayButton.style.cssText = "position:absolute;right:8px;top:8px;width:20px;height:20px;padding:0;border:1px solid #666;border-radius:3px;background:rgba(28,28,28,.9);color:#aaa;font:bold 12px/1 ui-monospace,SFMono-Regular,Consolas,monospace;cursor:pointer;z-index:2;";
            viewport.appendChild(linearDisplayButton);

            const transport = document.createElement("div");
            transport.style.cssText = "display:flex;align-items:center;gap:7px;box-sizing:border-box;height:43px;padding:4px 8px;background:#181818;border-top:1px solid #242424;";
            root.appendChild(transport);

            const playButton = document.createElement("button");
            playButton.type = "button";
            playButton.style.cssText = "display:flex;align-items:center;justify-content:center;width:22px;height:19px;padding:0;border:1px solid #555;border-radius:3px;background:#252525;color:#f4f4f4;font:12px/1 sans-serif;cursor:pointer;";
            playButton.textContent = "▶";
            playButton.title = "Play/pause (Space). Use Left/Right arrows for one preview frame.";
            transport.appendChild(playButton);

            const timelineHelp = "Click or drag to seek; colors identify chunks. Hold Shift to show the full prompt sent to H3.";
            const timelineShell = document.createElement("div");
            timelineShell.style.cssText = "position:relative;flex:1;height:33px;cursor:pointer;touch-action:none;";
            transport.appendChild(timelineShell);
            const chunkTooltip = createColoredChunkTooltip();

            const timelineTrack = document.createElement("div");
            timelineTrack.style.cssText = "position:absolute;left:0;right:0;top:3px;height:5px;border-radius:3px;background:#333;box-shadow:0 0 0 1px #080808,0 1px 2px #000;overflow:hidden;";
            timelineShell.appendChild(timelineTrack);

            const cachedChunkUnderlines = document.createElement("div");
            cachedChunkUnderlines.style.cssText = "position:absolute;left:0;right:0;top:10px;height:2px;z-index:2;pointer-events:none;";
            timelineShell.appendChild(cachedChunkUnderlines);

            const overlapReplacementLines = document.createElement("div");
            overlapReplacementLines.style.cssText = "position:absolute;left:0;right:0;top:6px;height:2px;z-index:2;pointer-events:none;";
            timelineShell.appendChild(overlapReplacementLines);

            const timelinePlayhead = document.createElement("div");
            timelinePlayhead.style.cssText = "position:absolute;z-index:3;top:0;height:11px;width:2px;margin-left:-1px;border-radius:1px;background:#fff;box-shadow:0 0 2px #000,0 0 4px rgba(255,255,255,.75);pointer-events:none;display:none;";
            timelineShell.appendChild(timelinePlayhead);

            const shotBrackets = document.createElement("div");
            shotBrackets.style.cssText = "position:absolute;left:0;right:0;top:13px;height:19px;pointer-events:none;overflow:hidden;";
            timelineShell.appendChild(shotBrackets);

            const muteButton = document.createElement("button");
            muteButton.type = "button";
            muteButton.style.cssText = "display:flex;flex:0 0 auto;align-items:center;justify-content:center;width:23px;height:19px;padding:0;border:1px solid #555;border-radius:3px;background:#252525;color:#f4f4f4;font:13px/1 sans-serif;cursor:pointer;";
            muteButton.title = "Mute/unmute finalized preview audio (M)";
            transport.appendChild(muteButton);

            const transportFrame = document.createElement("div");
            transportFrame.style.cssText = "min-width:62px;text-align:right;color:#aaa;font:10px/1 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap;";
            const transportRight = document.createElement("div");
            transportRight.style.cssText = "display:flex;min-width:62px;height:33px;flex-direction:column;align-items:flex-end;justify-content:center;gap:3px;";
            transportRight.appendChild(transportFrame);
            transport.appendChild(transportRight);

            const cacheButton = document.createElement("button");
            cacheButton.type = "button";
            cacheButton.textContent = "cache";
            cacheButton.style.cssText = "box-sizing:border-box;width:38px;height:14px;padding:0;border:1px solid #666;border-radius:3px;background:#292929;color:#ddd;font:9px/12px ui-monospace,SFMono-Regular,Consolas,monospace;cursor:pointer;";
            const transportButtons = document.createElement("div");
            transportButtons.style.cssText = "display:flex;gap:3px;height:14px;flex-shrink:0;align-items:center;";
            transportRight.appendChild(transportButtons);
            transportButtons.appendChild(cacheButton);

            const savePreviewButton = document.createElement("button");
            savePreviewButton.type = "button";
            savePreviewButton.textContent = "save";
            savePreviewButton.title = "Save preview: after stopping the render, join completed chunk movies beside the Save Video output, with a saved_preview suffix.";
            savePreviewButton.style.cssText = cacheButton.style.cssText;
            transportButtons.appendChild(savePreviewButton);
            savePreviewButton.addEventListener("click", async event => {
                event.stopPropagation();
                savePreviewButton.disabled = true;
                savePreviewButton.textContent = "…";
                try {
                    const response = await api.fetchApi("/hr_endless_sampler_video/save_preview", { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ node_id: node.id, execution }) });
                    if (!(response.headers.get("content-type") || "").includes("application/json")) {
                        const detail = await response.text();
                        if (response.status === 404 || response.status === 405) throw new Error("Save preview is unavailable on this running server. Restart ComfyUI to load the new endpoint, then refresh the browser.");
                        throw new Error(`Save preview failed (HTTP ${response.status}): ${detail.slice(0, 300)}`);
                    }
                    const result = await response.json();
                    if (!response.ok) throw new Error(result.error || "Could not save preview");
                    savePreviewButton.title = `Save preview. Last saved: ${result.filename}`;
                    window.alert(`Preview saved:\n${result.filename}`);
                } catch (error) {
                    window.alert(error.message);
                } finally {
                    savePreviewButton.disabled = false;
                    savePreviewButton.textContent = "save";
                }
            });
            let taomateMode = false;
            let lastCacheStatus = null;

            const cacheHelp = "Enable or disable replay-cache reuse for the next HR Endless Sampler run. Disabled ignores the existing cache, but the sampler still saves a fresh cache and overwrites the previous one.";
            let replayCacheEnabled = true;
            let reusingCachedChunks = false;
            let reusedChunkNumbers = [];
            // This is cache availability between runs, not merely the chunks
            // restored by the current sampler execution.
            let cachedChunkCount = 0;
            let cachedChunkIndices = new Set();
            function renderCacheReuseLabel() {
                const labels = reusedChunkNumbers.map(number => `#${number}`);
                const chunks = labels.length < 2
                    ? labels.join("")
                    : labels.length === 2
                        ? `${labels[0]} and ${labels[1]}`
                        : `${labels.slice(0, -1).join(", ")} and ${labels.at(-1)}`;
                cacheReuseLabel.textContent = `reusing cached chunks ${chunks}`;
                cacheReuseLabel.style.display = !taomateMode && reusingCachedChunks && labels.length ? "block" : "none";
            }
            function applyReplayCacheStatus(cacheStatus) {
                lastCacheStatus = cacheStatus;
                replayCacheEnabled = cacheStatus?.enabled !== false;
                if (cacheStatus?.has_cache) {
                    const count = Math.max(0, Number(cacheStatus.completed_chunks) || 0);
                    const cachedChunks = Array.isArray(cacheStatus.cached_chunks)
                        ? cacheStatus.cached_chunks
                        : Array.from({ length: count }, (_, index) => index + 1);
                    cachedChunkIndices = new Set(cachedChunks
                        .map(value => Math.round(Number(value)) - 1)
                        .filter(index => index >= 0 && (!chunkCount || index < chunkCount)));
                    cachedChunkCount = cachedChunkIndices.size;
                } else if (cacheStatus) {
                    cachedChunkCount = 0;
                    cachedChunkIndices = new Set();
                }
                if (taomateMode) {
                    cachedChunkCount = 0;
                    cachedChunkIndices.clear();
                }
                cacheButton.disabled = taomateMode || Boolean(cacheStatus?.active);
                cacheButton.style.background = replayCacheEnabled ? "#28662d" : "#202020";
                cacheButton.style.borderColor = replayCacheEnabled ? "#69b76f" : "#444";
                cacheButton.style.color = replayCacheEnabled ? "#e5ffe6" : "#888";
                cacheButton.style.cursor = cacheButton.disabled ? "not-allowed" : "pointer";
                renderCacheReuseLabel();
                if (taomateMode) {
                    cacheButton.title = "Replay cache is unavailable in TaoMate mode.";
                    cacheButton.style.background = "#202020";
                    cacheButton.style.color = "#888";
                    cacheButton.style.borderColor = "#444";
                } else if (cacheStatus?.active) {
                    cacheButton.title = `${cacheHelp}\nThe current render already chose its cache policy.`;
                } else {
                    const count = Array.isArray(cacheStatus?.cached_chunks)
                        ? cacheStatus.cached_chunks.length
                        : Math.max(0, Number(cacheStatus?.completed_chunks) || 0);
                    cacheButton.title = replayCacheEnabled
                        ? `${cacheHelp}\nEnabled. ${count} cached completed chunks are available.`
                        : `${cacheHelp}\nDisabled. Existing cache is ignored and will be replaced by this run.`;
                }
            }
            applyReplayCacheStatus(null);

            async function refreshReplayCacheStatus() {
                try {
                    const response = await api.fetchApi(
                        "/hr_endless_sampler_preview/replay_cache",
                        { cache: "no-store" },
                    );
                    if (!response.ok) throw new Error(`HTTP ${response.status}`);
                    applyReplayCacheStatus(await response.json());
                    // The cache endpoint is independent of preview events.
                    // Redraw its availability marks as soon as it returns.
                    renderTransport();
                } catch (error) {
                    applyReplayCacheStatus(null);
                    console.warn("HR Endless Sampler cache status request failed", error);
                }
            }

            cacheButton.addEventListener("click", async event => {
                event.preventDefault();
                event.stopPropagation();
                if (cacheButton.disabled) return;
                cacheButton.disabled = true;
                cacheButton.textContent = "…";
                try {
                    const response = await api.fetchApi(
                        `/hr_endless_sampler_preview/replay_cache_enabled?enabled=${replayCacheEnabled ? "0" : "1"}`,
                        { method: "POST", cache: "no-store" },
                    );
                    const payload = await response.json();
                    applyReplayCacheStatus(payload);
                    if (!response.ok) throw new Error(payload?.error || `HTTP ${response.status}`);
                    renderTransport();
                } catch (error) {
                    console.warn("HR Endless Sampler could not update its replay cache policy", error);
                    await refreshReplayCacheStatus();
                } finally {
                    cacheButton.textContent = "cache";
                    renderCacheReuseLabel();
                }
            });

            let previewContextMenu = null;
            function closePreviewContextMenu() {
                previewContextMenu?.remove();
                previewContextMenu = null;
            }

            function showPreviewContextMenu(chunkIndex, event) {
                closePreviewContextMenu();
                const menu = document.createElement("div");
                menu.style.cssText = "position:fixed;z-index:10000;min-width:174px;padding:3px;background:#202020;border:1px solid #5b5b5b;border-radius:4px;box-shadow:0 3px 12px rgba(0,0,0,.65);";
                menu.style.left = `${Math.max(4, Math.min(event.clientX, window.innerWidth - 182))}px`;
                menu.style.top = `${Math.max(4, Math.min(event.clientY, window.innerHeight - 32))}px`;
                const reconnectButton = document.createElement("button");
                reconnectButton.type = "button";
                reconnectButton.textContent = "Reconnect";
                reconnectButton.title = "Restore this preview from the current sampler render.";
                reconnectButton.style.cssText = "display:block;width:100%;padding:4px 7px;border:0;border-radius:2px;background:transparent;color:#f0f0f0;text-align:left;font:11px/1.2 ui-monospace,SFMono-Regular,Consolas,monospace;cursor:pointer;";
                reconnectButton.addEventListener("mouseenter", () => { reconnectButton.style.background = "#3e5f85"; });
                reconnectButton.addEventListener("mouseleave", () => { reconnectButton.style.background = "transparent"; });
                reconnectButton.addEventListener("click", async clickEvent => {
                    clickEvent.preventDefault();
                    clickEvent.stopPropagation();
                    reconnectButton.disabled = true;
                    reconnectButton.textContent = "Reconnecting…";
                    await restoreServerState(0, true);
                    await refreshReplayCacheStatus();
                    closePreviewContextMenu();
                });
                menu.appendChild(reconnectButton);
                const annotationButton = document.createElement("button");
                annotationButton.type = "button";
                annotationButton.textContent = "Annotation…";
                annotationButton.title = "Set the one-line annotation shown over the image and stored in the video timeline JSON.";
                annotationButton.style.cssText = reconnectButton.style.cssText;
                annotationButton.addEventListener("mouseenter", () => { annotationButton.style.background = "#3e5f85"; });
                annotationButton.addEventListener("mouseleave", () => { annotationButton.style.background = "transparent"; });
                annotationButton.addEventListener("click", event => { event.preventDefault(); event.stopPropagation(); closePreviewContextMenu(); openAnnotationWindow(); });
                menu.appendChild(annotationButton);
                const canDeleteChunk = replayCacheEnabled && !cacheButton.disabled && Number.isInteger(chunkIndex)
                    && chunkIndex >= 0 && cachedChunkIndices.has(chunkIndex);
                if (!canDeleteChunk) {
                    document.body.appendChild(menu);
                    previewContextMenu = menu;
                    return;
                }
                const chunkNumber = chunkIndex + 1;
                const divider = document.createElement("div");
                divider.style.cssText = "height:1px;margin:3px 2px;background:#505050;";
                menu.appendChild(divider);
                const removeButton = document.createElement("button");
                removeButton.type = "button";
                removeButton.textContent = "Delete cached chunk";
                removeButton.title = `Delete cached Chunk ${chunkNumber}. The next sampler run rerenders it and reuses intact later cached chunks.`;
                removeButton.style.cssText = "display:block;width:100%;padding:4px 7px;border:0;border-radius:2px;background:transparent;color:#f0f0f0;text-align:left;font:11px/1.2 ui-monospace,SFMono-Regular,Consolas,monospace;cursor:pointer;";
                removeButton.addEventListener("mouseenter", () => { removeButton.style.background = "#8d302d"; });
                removeButton.addEventListener("mouseleave", () => { removeButton.style.background = "transparent"; });
                removeButton.addEventListener("click", async clickEvent => {
                    clickEvent.preventDefault();
                    clickEvent.stopPropagation();
                    chunkTooltip.hide();
                    tooltipChunkIndex = null;
                    tooltipSignature = null;
                    removeButton.disabled = true;
                    removeButton.textContent = "Deleting…";
                    try {
                        const response = await api.fetchApi(
                            `/hr_endless_sampler_preview/replay_cache_chunk?chunk=${chunkNumber}`,
                            { method: "POST", cache: "no-store" },
                        );
                        const payload = await response.json();
                        applyReplayCacheStatus(payload);
                        renderTransport();
                        if (!response.ok) throw new Error(payload?.error || `HTTP ${response.status}`);
                    } catch (error) {
                        console.warn("HR Endless Sampler could not delete the cached chunk", error);
                        await refreshReplayCacheStatus();
                    } finally {
                        closePreviewContextMenu();
                    }
                });
                menu.appendChild(removeButton);
                document.body.appendChild(menu);
                previewContextMenu = menu;
            }

            const dismissPreviewContextMenu = event => {
                if (previewContextMenu && !previewContextMenu.contains(event.target)) closePreviewContextMenu();
            };
            document.addEventListener("pointerdown", dismissPreviewContextMenu, true);

            const status = document.createElement("div");
            status.style.cssText = "box-sizing:border-box;display:flex;flex-direction:column;gap:1px;min-height:36px;padding:3px 9px;background:#1b1b1b;overflow:hidden;font:10px/11px ui-monospace,SFMono-Regular,Consolas,monospace;";
            const statusPhase = document.createElement("div");
            statusPhase.style.cssText = "min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#ddd;";
            statusPhase.textContent = "Waiting for HR Endless Sampler…";
            const statusMetrics = document.createElement("div");
            statusMetrics.style.cssText = "min-width:0;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;color:#aaa;";
            const statusProgress = document.createElement("div");
            statusProgress.style.cssText = "position:relative;height:3px;margin-top:1px;border-radius:2px;background:#303030;overflow:hidden;box-shadow:inset 0 0 0 1px rgba(0,0,0,.35);";
            const statusProgressFill = document.createElement("div");
            statusProgressFill.style.cssText = "position:absolute;left:0;top:0;height:100%;width:0;border-radius:2px;background:#65b9ff;transform:translateX(0);transition:width .15s linear;";
            statusProgress.appendChild(statusProgressFill);
            status.append(statusPhase, statusMetrics, statusProgress);
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
            let activeSubchunk = null;
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
            let phase = "Preparing sampler";
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
            let statusProgressAnimation = null;
            let restoringCache = false;
            node._hrEndlessCacheRestoreProgress = data => {
                if (!restoringCache) return;
                statusPhase.textContent = data.message;
                statusProgressFill.style.width = `${Math.min(100, 100 * (Number(data.completed) || 0) / Math.max(1, Number(data.total) || 1))}%`;
            };
            let tooltipChunkIndex = null;
            let tooltipSignature = null;
            let tooltipPointer = null;
            let shiftPromptVisible = false;
            let audioGroup = null;
            let standbyAudioGroup = null;
            let audioMuted = Boolean(savedPlayerState.muted);
            let inverseGammaDisplay = Boolean(savedPlayerState.inverseGammaDisplay);
            let pendingFrame = typeof savedPlayerState.frame === "number" && Number.isFinite(savedPlayerState.frame)
                ? savedPlayerState.frame : null;
            let pendingAudioSources = new Map();
            let continuousAudioContext = null;
            let continuousAudioGain = null;
            let continuousAudio = [];
            const decodedAudio = new Map();

            function stopContinuousAudio() {
                for (const entry of continuousAudio) entry.source.stop();
                continuousAudio = [];
            }

            function prepareContinuousAudio(group) {
                if (!group?.audioSource || decodedAudio.has(group.audioSource)) return;
                const AudioContext = window.AudioContext || window.webkitAudioContext;
                if (!AudioContext) return;
                if (!continuousAudioContext) {
                    continuousAudioContext = new AudioContext();
                    continuousAudioGain = continuousAudioContext.createGain();
                    continuousAudioGain.connect(continuousAudioContext.destination);
                }
                const source = group.audioSource;
                decodedAudio.set(source, null);
                fetch(source).then(response => response.arrayBuffer())
                    .then(bytes => continuousAudioContext.decodeAudioData(bytes))
                    .then(buffer => { if (decodedAudio.has(source)) decodedAudio.set(source, buffer); })
                    .catch(error => console.warn("Preview audio decode failed", error));
            }

            function continuousAudioTime(group) {
                const entry = continuousAudio.find(item => item.group === group && item.url === group.audioSource);
                return entry ? Math.max(0, entry.offset + (continuousAudioContext.currentTime - entry.start) * entry.rate) : null;
            }

            function scheduleContinuousAudio(group, frameIndex, forceSeek) {
                prepareContinuousAudio(group);
                if (!decodedAudio.get(group.audioSource)) return false;
                if (forceSeek) stopContinuousAudio();
                if (continuousAudio.length && !continuousAudio.some(entry => entry.group === group && entry.url === group.audioSource)) stopContinuousAudio();
                continuousAudioContext.resume().catch(() => {});
                continuousAudioGain.gain.value = audioMuted ? 0 : 1;
                let index = chunks.indexOf(group);
                let start = continuousAudio.length ? continuousAudio[continuousAudio.length - 1].end : continuousAudioContext.currentTime + 0.02;
                // Schedule ready, adjacent chunks ahead of time on the audio hardware clock.
                for (; index >= 0 && index < chunks.length; index++) {
                    const next = chunks[index];
                    if (!next?.audioSource) break;
                    prepareContinuousAudio(next);
                    if (continuousAudio.some(entry => entry.group === next && entry.url === next.audioSource)) continue;
                    const buffer = decodedAudio.get(next.audioSource);
                    if (!buffer) break;
                    const rate = Math.max(0.0625, Math.min(16, currentPlaybackFps() / (validFps(next.sourceFps) || validFps(sourceFps) || 24)));
                    const offset = next === group ? audioOffset(group, frameIndex) : 0;
                    if (offset >= buffer.duration) break;
                    const source = continuousAudioContext.createBufferSource();
                    source.buffer = buffer;
                    source.playbackRate.value = rate;
                    source.connect(continuousAudioGain);
                    start = Math.max(start, continuousAudioContext.currentTime);
                    const end = start + (buffer.duration - offset) / rate;
                    source.start(start, offset);
                    continuousAudio.push({ group: next, url: next.audioSource, source, start, end, offset, rate });
                    start = end;
                }
                audioPlayer.pause();
                return continuousAudioTime(group) != null;
            }

            function stop() {
                if (timer != null) clearTimeout(timer);
                timer = null;
                framePending = false;
                playbackSerial++;
                audioPlayer.pause();
                stopContinuousAudio();
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

            function audioOffset(group, frameIndex) {
                const source = validFps(group?.sourceFps) || validFps(sourceFps) || 24;
                const first = Number(group?.frameNumbers?.[0]);
                const current = Number(group?.frameNumbers?.[frameIndex]);
                if (Number.isFinite(first) && Number.isFinite(current)) {
                    return Math.max(0, (current - first) / source);
                }
                return Math.max(0, Number(frameIndex) || 0) / source;
            }

            function syncAudio(group, frameIndex, shouldPlay, serial, forceSeek=false) {
                const source = group?.audioSource;
                if (source && shouldPlay && scheduleContinuousAudio(group, frameIndex, forceSeek)) return;
                stopContinuousAudio();
                if (!source) {
                    audioPlayer.pause();
                    audioGroup = null;
                    return;
                }
                let sourceChanged = false;
                if (audioGroup !== group) {
                    audioPlayer.pause();
                    if (standbyAudioGroup === group && audioStandbyPlayer.readyState >= 1) {
                        const previousPlayer = audioPlayer;
                        audioPlayer = audioStandbyPlayer;
                        audioStandbyPlayer = previousPlayer;
                        audioStandbyPlayer.pause();
                        audioStandbyPlayer.removeAttribute("src");
                        audioStandbyPlayer.load();
                    } else {
                        audioPlayer.src = source;
                        audioPlayer.load();
                    }
                    audioGroup = group;
                    standbyAudioGroup = null;
                    sourceChanged = true;
                }
                const sourceRate = validFps(group.sourceFps) || validFps(sourceFps) || 24;
                const desiredRate = Math.max(0.0625, Math.min(16, currentPlaybackFps() / sourceRate));
                if (Math.abs(audioPlayer.playbackRate - desiredRate) > 1e-6) {
                    audioPlayer.playbackRate = desiredRate;
                }
                const target = audioOffset(group, frameIndex);
                const positionAndPlay = () => {
                    if (serial !== playbackSerial || audioGroup !== group) return;
                    try {
                        // Reposition only when the user explicitly seeks,
                        // playback starts/restarts, or a different finalized
                        // chunk is installed. Seeking on every image frame was
                        // producing audible clicks whenever WebP decoding made
                        // the visual timer drift.
                        if (forceSeek || sourceChanged || !Number.isFinite(audioPlayer.currentTime)) {
                            audioPlayer.currentTime = target;
                        }
                    } catch (_error) {
                        return;
                    }
                    if (shouldPlay) {
                        if (audioPlayer.paused) {
                            audioPlayer.play().catch(() => {
                                // Browsers can reject autoplay until the user
                                // presses this preview's play button. Video keeps
                                // running and the next explicit play gesture retries.
                            });
                        }
                    } else {
                        audioPlayer.pause();
                    }
                };
                if (audioPlayer.readyState >= 1) positionAndPlay();
                else audioPlayer.addEventListener("loadedmetadata", positionAndPlay, { once: true });
            }

            function preloadAudio(group) {
                prepareContinuousAudio(group);
                if (!group?.audioSource || group === audioGroup || group === standbyAudioGroup) return;
                audioStandbyPlayer.pause();
                audioStandbyPlayer.src = group.audioSource;
                audioStandbyPlayer.load();
                standbyAudioGroup = group;
            }

            function persistPlayerState() {
                const frame = displayedFrameNumber();
                // Retain a restored target until its chunk arrives from the server.
                if (Number.isFinite(pendingFrame) && frame !== pendingFrame) return;
                node.properties = node.properties || {};
                node.properties.hr_endless_sampler_preview_player = { frame, muted: audioMuted, inverseGammaDisplay };
            }

            function restorePersistedFrame() {
                if (!Number.isFinite(pendingFrame)) return false;
                for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
                    const group = chunks[chunkIndex];
                    const frameIndex = group?.frameNumbers?.findIndex(value => Number(value) === pendingFrame);
                    if (frameIndex >= 0) {
                        paused = true;
                        const frame = pendingFrame;
                        pendingFrame = null;
                        show(chunkIndex, frameIndex);
                        node.properties.hr_endless_sampler_preview_player = { frame, muted: audioMuted, inverseGammaDisplay };
                        return true;
                    }
                }
                return false;
            }

            function renderMuteButton() {
                if (continuousAudioGain) continuousAudioGain.gain.value = audioMuted ? 0 : 1;
                audioPlayer.muted = audioMuted;
                audioStandbyPlayer.muted = audioMuted;
                muteButton.textContent = audioMuted ? "🔇" : "🔊";
                muteButton.setAttribute("aria-label", audioMuted ? "Unmute preview audio" : "Mute preview audio");
                muteButton.style.color = audioMuted ? "#999" : "#f4f4f4";
                muteButton.style.background = audioMuted ? "#1d1d1d" : "#252525";
            }

            function setAudioMuted(value) {
                audioMuted = Boolean(value);
                renderMuteButton();
                persistPlayerState();
            }

            function renderInverseGammaDisplay() {
                image.style.filter = inverseGammaDisplay ? `url(#${inverseGammaFilterId})` : "none";
                linearDisplayButton.style.color = inverseGammaDisplay ? "#82d7ff" : "#aaa";
                linearDisplayButton.style.background = inverseGammaDisplay ? "rgba(20,68,88,.95)" : "rgba(28,28,28,.9)";
                linearDisplayButton.style.cursor = "pointer";
                linearDisplayButton.setAttribute("aria-pressed", String(inverseGammaDisplay));
            }

            linearDisplayButton.addEventListener("click", event => {
                event.preventDefault();
                event.stopPropagation();
                inverseGammaDisplay = !inverseGammaDisplay;
                renderInverseGammaDisplay();
                persistPlayerState();
                root.focus({ preventScroll: true });
            });
            renderInverseGammaDisplay();

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

            function renderCachedChunkUnderlines(spans, total) {
                cachedChunkUnderlines.replaceChildren();
                if (taomateMode || !replayCacheEnabled || cachedChunkCount <= 0 || !total) return;
                let offset = 0;
                for (let index = 0; index < spans.length; index++) {
                    const start = offset / total * 100;
                    offset += spans[index];
                    const end = offset / total * 100;
                    if (!cachedChunkIndices.has(index)) continue;
                    const underline = document.createElement("div");
                    underline.style.cssText = `position:absolute;left:${start}%;width:${Math.max(0, end - start)}%;height:2px;border-radius:1px;background:#69b76f;box-shadow:0 0 2px rgba(105,183,111,.65);`;
                    cachedChunkUnderlines.appendChild(underline);
                }
            }

            function renderOverlapReplacementLines(spans, total) {
                overlapReplacementLines.replaceChildren();
                if (!total) return;
                let offset = 0;
                for (let index = 0; index < spans.length - 1; index++) {
                    const overlapFrames = Math.max(0, Number(chunkRanges[index]?.replacement_overlap_frames) || 0);
                    const span = spans[index];
                    if (overlapFrames > 0 && span > 0) {
                        const start = (offset + Math.max(0, span - overlapFrames)) / total * 100;
                        const end = (offset + span) / total * 100;
                        const line = document.createElement("div");
                        const color = chunkColors[(index + 1) % chunkColors.length];
                        const alpha = available(index + 1) ? "" : "35";
                        line.style.cssText = `position:absolute;left:${start}%;width:calc(${Math.max(0, end - start)}% + 1px);height:2px;border-radius:1px;background:${color}${alpha};box-shadow:0 0 2px ${color}${alpha};`;
                        overlapReplacementLines.appendChild(line);
                    }
                    offset += span;
                }
            }

            function chunkIndexAtTimelinePointer(event) {
                const rect = timelineShell.getBoundingClientRect();
                if (!rect.width) return null;
                const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
                const { spans, total } = timelineLayout();
                let offset = 0;
                for (let index = 0; index < spans.length; index++) {
                    offset += spans[index];
                    if (fraction * total <= offset || index === spans.length - 1) return index;
                }
                return null;
            }

            function setChunkTooltip(index, event) {
                if (previewContextMenu) {
                    tooltipChunkIndex = null;
                    tooltipSignature = null;
                    chunkTooltip.hide();
                    return;
                }
                if (!Number.isInteger(index) || !chunkRanges[index]) {
                    tooltipChunkIndex = null;
                    tooltipSignature = null;
                    chunkTooltip.hide();
                    return;
                }
                tooltipPointer = { clientX: event.clientX, clientY: event.clientY };
                const range = chunkRanges[index];
                const description = chunkPromptDescription(range);
                const retentionAnalysis = String(range.gemma_retention_analysis || "").trim();
                const fullPrompt = String(range.h3_prompt || "").trim();
                const showFullPrompt = Boolean(event.shiftKey || shiftPromptVisible);
                const timing = chunkTimingLines(range);
                const signature = JSON.stringify([description, retentionAnalysis, fullPrompt, showFullPrompt, timing, shotRanges]);
                if (tooltipChunkIndex === index && tooltipSignature === signature) {
                    chunkTooltip.move(event);
                    return;
                }
                tooltipChunkIndex = index;
                tooltipSignature = signature;
                chunkTooltip.show(event, {
                    help: timelineHelp,
                    chunk: range,
                    timing,
                    description,
                    retentionAnalysis,
                    fullPrompt,
                    showFullPrompt,
                    shotRanges,
                    colors: chunkColors,
                    waitingText: "Waiting for this chunk's prompt.",
                });
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
                if (!Number.isFinite(chunkNumber)) return `${frame}`;
                // A cache restore can lack source-shot brackets, but the
                // frame label must still retain its chunk identity.
                const shotNumber = Number(shot?.shot) || 1;
                return `S${Math.round(shotNumber)}/C${Math.round(chunkNumber)}/${frame}`;
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
                    const completedFrames = chunkRanges[index]?.taomate_completed_frames;
                    if (Number.isFinite(completedFrames)) {
                        // A live latent preview makes its current sub-chunk available before completion.
                        const previewFrames = available(index)
                            ? Math.max(0, (Number(chunks[index].outputEnd) + 1) - Number(chunkRanges[index]?.start)) || 0
                            : 0;
                        const fraction = Math.max(0, Math.min(1, Math.max(completedFrames, previewFrames) / spans[index]));
                        const boundary = start + (end - start) * fraction;
                        const pendingColor = `color-mix(in srgb, ${color} 50%, black)`;
                        if (fraction > 0) stops.push(`${color}e8 ${start}%`, `${color}e8 ${boundary}%`);
                        if (fraction < 1) stops.push(`${pendingColor} ${boundary}%`, `${pendingColor} ${end}%`);
                        continue;
                    }
                    const alpha = available(index) ? "e8" : "35";
                    stops.push(`${color}${alpha} ${start}%`, `${color}${alpha} ${end}%`);
                }
                timelineTrack.style.background = stops.length
                    ? `linear-gradient(to right, ${stops.join(",")})`
                    : "#333";
                renderCachedChunkUnderlines(spans, total);
                renderOverlapReplacementLines(spans, total);
                renderShotBrackets();

                if (!available(playing)) {
                    timelinePlayhead.style.display = "none";
                    return;
                }
                const before = spans.slice(0, playing).reduce((sum, value) => sum + value, 0);
                const group = chunks[playing];
                const exactFrame = Number(group.frameNumbers?.[playingFrame]);
                const rangeStart = Number(chunkRanges[playing]?.start ?? group.outputStart ?? before);
                const fraction = group.frames.length < 2
                    ? 0
                    : playingFrame / (group.frames.length - 1);
                // Use source frames, not the number of preview images stretched over the planned group.
                const previewSpan = Math.max(0, Number(group.outputEnd) - Number(group.outputStart)) || 0;
                const localFrame = Number.isFinite(exactFrame) ? exactFrame - rangeStart : fraction * previewSpan;
                const position = (before + Math.max(0, Math.min(spans[playing], localFrame))) / total * 100;
                timelinePlayhead.style.left = `${Math.max(0, Math.min(100, position))}%`;
                timelinePlayhead.style.display = "block";
            }

            function displaySource(source, valid, displayed) {
                renderInverseGammaDisplay();
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

            function playFrameGroup(index, group, frameIndex, serial, seekAudio=false) {
                if (serial !== playbackSerial || paused || hoverStep != null || !group?.frames?.length) return;
                const boundedFrame = Math.max(0, Math.min(frameIndex, group.frames.length - 1));
                playing = index;
                playingFrame = boundedFrame;
                renderTransport();
                let duration = frameDuration(group, boundedFrame);
                syncAudio(group, boundedFrame, true, serial, seekAudio);
                const next = nextAvailable(index);
                if (next >= 0 && next !== index) preloadAudio(chunks[next]);
                const continuousTime = continuousAudioTime(group);
                if (continuousTime != null || (group.audioSource && audioGroup === group && !audioPlayer.paused && audioPlayer.readyState >= 1)) {
                    const sourceRate = validFps(group.sourceFps) || validFps(sourceFps) || 24;
                    const nextMediaTime = (boundedFrame + 1) / sourceRate;
                    const wallDelay = (nextMediaTime - (continuousTime ?? audioPlayer.currentTime)) * 1000 / Math.max(0.0625, currentPlaybackFps() / sourceRate);
                    if (Number.isFinite(wallDelay)) duration = Math.max(1, wallDelay);
                }
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
                                let nextFrame = boundedFrame + 1;
                                // Finalized audio is the stable clock. If image
                                // decoding was late, skip the stale visual
                                // frame instead of seeking audio backward and
                                // creating a click.
                                const audioTime = continuousAudioTime(group);
                                if (audioTime != null || (group.audioSource && audioGroup === group && !audioPlayer.paused)) {
                                    const sourceRate = validFps(group.sourceFps) || validFps(sourceFps) || 24;
                                    nextFrame = Math.max(
                                        nextFrame,
                                        Math.floor((audioTime ?? audioPlayer.currentTime) * sourceRate + 1e-4),
                                    );
                                }
                                if (nextFrame < group.frames.length) {
                                    playFrameGroup(index, group, nextFrame, serial, false);
                                    return;
                                }
                            }
                            if (boundedFrame + 1 >= group.frames.length || group.audioSource) {
                                const next = nextAvailable(index);
                                if (next === index + 1) playFrameGroup(next, chunks[next], 0, serial, false);
                                else if (next >= 0) show(next);
                                return;
                            }
                        }, duration);
                    },
                );
            }

            function show(index, frameIndex=0) {
                if (!available(index) || hoverStep != null) return;
                stop();
                playing = index;
                playingFrame = Math.max(0, Math.min(frameIndex, chunks[index].frames.length - 1));
                persistPlayerState();
                renderTransport();
                const group = chunks[index];
                const serial = playbackSerial;
                if (paused) {
                    syncAudio(group, playingFrame, false, serial, true);
                    framePending = true;
                    displaySource(
                        group.frames[playingFrame],
                        () => serial === playbackSerial && paused && hoverStep == null,
                        () => { framePending = false; },
                    );
                } else {
                    playFrameGroup(index, group, playingFrame, serial, true);
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
            muteButton.addEventListener("click", event => {
                event.preventDefault();
                event.stopPropagation();
                setAudioMuted(!audioMuted);
                root.focus({ preventScroll: true });
            });
            viewport.addEventListener("click", () => {
                root.focus({ preventScroll: true });
                // A click is also an explicit browser media gesture. If the
                // visual preview was already autoplaying, use it to enable the
                // synchronized soundtrack without resetting the frame.
                if (!paused && available(playing)) {
                    syncAudio(chunks[playing], playingFrame, true, playbackSerial, false);
                }
            });
            timelineShell.addEventListener("pointerdown", event => {
                if (event.button !== 0) return;
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
            root.addEventListener("contextmenu", event => {
                event.preventDefault();
                event.stopPropagation();
                chunkTooltip.hide();
                tooltipChunkIndex = null;
                tooltipSignature = null;
                tooltipPointer = null;
                closePreviewContextMenu();
                const chunkIndex = timelineShell.contains(event.target) ? chunkIndexAtTimelinePointer(event) : null;
                showPreviewContextMenu(chunkIndex, event);
            });
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
                } else if (event.key === "m" || event.key === "M") {
                    event.preventDefault();
                    event.stopPropagation();
                    setAudioMuted(!audioMuted);
                }
            });
            renderMuteButton();

            function renderStatus() {
                if (restoringCache) return;
                const resolution = previewWidth && previewHeight ? `${previewWidth}×${previewHeight}` : "resolution —";
                const fps = Number.isFinite(currentPlaybackFps()) ? `${Number(currentPlaybackFps().toFixed(3))} fps` : "fps —";
                const secondsPerStep = Number.isFinite(averageStepMs) ? `${(averageStepMs / 1000).toFixed(2)}s/step` : "—s/step";
                const elapsedSeconds = completedElapsed ?? (startedAt == null ? NaN : (performance.now() - startedAt) / 1000);
                const elapsed = formatEta(elapsedSeconds);
                const activePhaseCount = Math.max(1, Number(chunkRanges[activeChunk]?.taomate_phase_count) || 1);
                const chunk = chunkCount ? activeSubchunk
                    ? `C${activeChunk + 1}.${activeSubchunk}/${chunkCount}.${activePhaseCount}`
                    : `C${activeChunk + 1}/${chunkCount}`
                    : "C—/—";
                const displayStep = hoverStep ?? currentStep;
                const inspecting = hoverStep == null ? "" : "Inspect · ";
                const statePrefix = `${complete ? "Complete · " : ""}${paused ? "Paused · " : ""}`;
                const phaseLine = `${statePrefix}${phase || "Preparing sampler"}`;
                const h3Active = !complete && /h3\s+(?:sampling|inference)/i.test(phase || "");
                // ETA is deliberately unit based. A TaoMate phase is one unit;
                // otherwise a sampler chunk is one unit. The current unit counts
                // in the elapsed average even before it has completed.
                const phaseCountFor = range => Math.max(
                    1,
                    Number(range?.taomate_phase_count) || 0,
                    Array.isArray(range?.taomate_phase_work) ? range.taomate_phase_work.length : 0,
                );
                const hasSubchunks = chunkRanges.some(range => phaseCountFor(range) > 1);
                let workTotal = 0;
                let completedUnits = 0;
                for (let index = 0; index < chunkCount; index++) {
                    const range = chunkRanges[index] || {};
                    const phaseCount = phaseCountFor(range);
                    const chunkUnits = hasSubchunks ? phaseCount : 1;
                    workTotal += chunkUnits;
                    if (index === activeChunk) {
                        const reportedPhases = Number(range.taomate_completed_phases) || 0;
                        const activePhases = activeSubchunk == null ? 0 : Number(activeSubchunk) - 1;
                        const completedPhases = Math.max(0, Math.min(phaseCount, reportedPhases));
                        completedUnits += hasSubchunks ? Math.min(phaseCount, Math.max(completedPhases, activePhases)) : 0;
                    } else if (index < activeChunk || cachedChunkIndices.has(index)) {
                        // A cached range may include the active index after a
                        // reconnect; that live range must remain estimable.
                        completedUnits += chunkUnits;
                    }
                }
                const observedUnits = complete ? workTotal : Math.min(workTotal, completedUnits + 1);
                const fallbackRemainingSteps = Math.max(0, totalSteps - currentStep) + Math.max(0, chunkCount - activeChunk - 1) * totalSteps;
                const fallbackEtaSeconds = Number.isFinite(averageStepMs) ? fallbackRemainingSteps * averageStepMs / 1000 : NaN;
                const timingEstimate = projectedRenderTiming(elapsedSeconds, observedUnits, workTotal, fallbackEtaSeconds);
                const eta = formatEta(timingEstimate.etaSeconds);
                const projected = formatEta(timingEstimate.totalSeconds);
                const metricsLine = `${chunk} · ${resolution} · ${fps} · ${inspecting}S ${displayStep}/${totalSteps || "—"} · ${secondsPerStep} · E ${elapsed} · ETA ${eta} · Est. total ${projected}`;
                statusPhase.textContent = phaseLine;
                statusMetrics.textContent = metricsLine;
                const gemmaActive = !complete && /gemma\s*4/i.test(phase || "");
                let progressHelp = "";
                if (gemmaActive) {
                    statusProgressFill.style.width = "32%";
                    statusProgressFill.style.background = "#bd78ff";
                    statusProgressFill.style.transition = "none";
                    if (statusProgressAnimation == null) {
                        statusProgressAnimation = statusProgressFill.animate(
                            [
                                { transform: "translateX(-115%)" },
                                { transform: "translateX(315%)" },
                            ],
                            { duration: 950, iterations: Infinity, easing: "ease-in-out" },
                        );
                    }
                    progressHelp = "Gemma generation is active; its final token count is not known in advance.";
                } else {
                    statusProgressAnimation?.cancel();
                    statusProgressAnimation = null;
                    statusProgressFill.style.transform = "translateX(0)";
                    statusProgressFill.style.transition = "width .15s linear";
                    statusProgressFill.style.background = complete ? "#63d38a" : "#65b9ff";
                    const samplingProgress = totalSteps > 0 ? Math.max(0, Math.min(1, currentStep / totalSteps)) : 0;
                    statusProgressFill.style.width = `${complete ? 100 : h3Active ? samplingProgress * 100 : 0}%`;
                    progressHelp = h3Active
                        ? `H3 sampling step ${currentStep}/${totalSteps || "—"}.`
                        : complete ? "Render complete." : "Waiting for H3 or Gemma progress.";
                }
                status.title = `${phaseLine}\n${metricsLine}\n${progressHelp}`;
                sigmaGraph.value.textContent = Number.isFinite(deltas[displayStep - 1]) ? deltas[displayStep - 1].toFixed(3) : "—";
                timeGraph.value.textContent = Number.isFinite(stepTimes[displayStep - 1]) ? `${(stepTimes[displayStep - 1] / 1000).toFixed(2)}s` : "—";
            }

            function redrawGraphs() {
                drawSigmaDelta(sigmaGraph.canvas, sigmas, deltas, totalSteps, hoverStep);
                drawStepTimes(timeGraph.canvas, stepTimes, totalSteps, hoverStep);
            }

            function resetExecution(data) {
                stopContinuousAudio();
                decodedAudio.clear();
                execution = data.execution;
                chunkCount = data.chunk_count || 0;
                cachedChunkCount = Math.max(0, Math.min(Number(data.cached_chunk_count) || 0, chunkCount));
                cachedChunkIndices = new Set(Array.from({ length: cachedChunkCount }, (_, index) => index));
                activeChunk = data.chunk ?? 0;
                activeSubchunk = null;
                chunks = new Array(chunkCount);
                chunkRanges = Array.isArray(data.chunk_ranges) ? data.chunk_ranges.map(range => ({ ...range })) : [];
                previewAnnotation = String(data.annotation || "").replace(/[\r\n]+/g, " ").slice(0, 256);
                if (node.properties) delete node.properties.hr_endless_sampler_annotation;
                renderAnnotation();
                taomateMode = chunkRanges.some(range => Number(range.taomate_phase_count) > 0);
                applyReplayCacheStatus(lastCacheStatus);
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
                phase = typeof data.phase === "string" ? data.phase : "Preparing sampler";
                setSourceFps(data.fps);
                reusingCachedChunks = Boolean(data.reusing_cached_chunks);
                const suppliedReusedChunks = Array.isArray(data.reused_chunk_numbers)
                    ? data.reused_chunk_numbers
                    : Array.from({ length: cachedChunkCount }, (_, index) => index + 1);
                reusedChunkNumbers = Array.from(new Set(suppliedReusedChunks
                    .map(value => Math.round(Number(value)))
                    .filter(number => number >= 1 && number <= chunkCount))).sort((left, right) => left - right);
                cachedChunkIndices = new Set(reusedChunkNumbers.map(number => number - 1));
                cachedChunkCount = cachedChunkIndices.size;
                const elapsedMs = Number.isFinite(data.elapsed_ms) ? data.elapsed_ms : 0;
                // A reset cached by the server retains its wall-clock start, so
                // elapsed time continues correctly after a browser refresh.
                const startedAtMs = Number(data.started_at_ms);
                const elapsedFromStart = Number.isFinite(startedAtMs) && startedAtMs > 0
                    ? Math.max(0, Date.now() - startedAtMs)
                    : 0;
                startedAt = performance.now() - Math.max(elapsedMs, elapsedFromStart);
                completedElapsed = null;
                complete = false;
                audioPlayer.pause();
                audioPlayer.removeAttribute("src");
                audioPlayer.load();
                audioGroup = null;
                audioStandbyPlayer.removeAttribute("src");
                audioStandbyPlayer.load();
                standbyAudioGroup = null;
                pendingAudioSources.clear();
                chunkTooltip.hide();
                tooltipChunkIndex = null;
                tooltipSignature = null;
                if (elapsedTimer != null) clearInterval(elapsedTimer);
                elapsedTimer = setInterval(renderStatus, 1000);
                stop();
                image.removeAttribute("src");
                frameLabel.style.display = "none";
                renderCacheReuseLabel();
                renderStatus();
                renderTransport();
                redrawGraphs();
            }

            function adoptElapsed(data) {
                // Old running renders have no wall-clock reset field, but their
                // cached phase/progress payload still carries server elapsed time.
                const elapsedMs = Number(data?.elapsed_ms);
                if (!Number.isFinite(elapsedMs) || elapsedMs < 0) return;
                const localElapsedMs = startedAt == null ? 0 : performance.now() - startedAt;
                if (elapsedMs > localElapsedMs) startedAt = performance.now() - elapsedMs;
            }

            node._hrEndlessSamplerPreview = data => {
                if (data.action === "reset") {
                    if (execution !== null && Number(data.execution) < Number(execution)) return;
                    resetExecution(data);
                    return;
                }
                if (data.execution !== execution) {
                    // A newly mounted widget can receive this lightweight
                    // metadata event before its asynchronous state restore has
                    // supplied the required reset timeline. Ignore it until
                    // that reset arrives instead of replacing the playlist
                    // with an incomplete event.
                    if (execution !== null || data.action === "chunk_metadata") return;
                    resetExecution(data);
                }
                adoptElapsed(data);
                if (data.action === "chunk_metadata") {
                    const index = Number(data.chunk);
                    const range = chunkRanges[index];
                    if (range && typeof data.gemma_detailed_description === "string") {
                        range.gemma_detailed_description = data.gemma_detailed_description;
                    }
                    if (range && typeof data.gemma_retention_analysis === "string") {
                        range.gemma_retention_analysis = data.gemma_retention_analysis;
                    }
                    if (range && typeof data.h3_prompt === "string") {
                        range.h3_prompt = data.h3_prompt;
                    }
                    if (range) {
                        if (Number.isFinite(data.taomate_completed_frames) && data.taomate_completed_frames >= 0) {
                            range.taomate_completed_frames = data.taomate_completed_frames;
                            renderTransport();
                        }
                        if (Number.isFinite(data.taomate_completed_phases) && data.taomate_completed_phases >= 0) {
                            range.taomate_completed_phases = data.taomate_completed_phases;
                        }
                        for (const key of ["h3_render_seconds", "gemma_seconds", "gemma_preproduction_seconds", "chunk_total_seconds"]) {
                            const value = Number(data[key]);
                            if (Number.isFinite(value) && value >= 0) range[key] = value;
                        }
                    }
                    return;
                }
                if (data.action === "phase") {
                    if (typeof data.phase === "string") phase = data.phase;
                    if (data.chunk != null) {
                        if (data.chunk !== activeChunk) activeSubchunk = null;
                        activeChunk = data.chunk;
                    }
                    renderStatus();
                    renderTransport();
                    return;
                }
                if (data.action === "sample_start") {
                    activeChunk = data.chunk ?? activeChunk;
                    activeSubchunk = data.subchunk ?? null;
                    phase = "H3 sampling";
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
                    phase = "Complete";
                    renderStatus();
                    renderTransport();
                    if (hoverStep == null && timer == null && !framePending) restorePlayback();
                    return;
                }
                if (data.action === "progress") {
                    activeChunk = data.chunk ?? activeChunk;
                    activeSubchunk = data.subchunk ?? null;
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
                if (data.action === "chunk_audio_update") {
                    const index = Number(data.chunk);
                    const group = chunks[index];
                    if (typeof data.audio !== "string" || !data.audio) return;
                    const updatedSource = `data:${data.audio_mime || "audio/wav"};base64,${data.audio}`;
                    pendingAudioSources.set(index, updatedSource);
                    // Websocket delivery can race a large chunk_final payload.
                    // Keep the update pending until authoritative frames exist.
                    if (!group?.finalized) return;
                    const wasActiveAudio = audioGroup === group;
                    const shouldResume = wasActiveAudio && !paused && hoverStep == null;
                    if (wasActiveAudio) {
                        audioPlayer.pause();
                        audioPlayer.removeAttribute("src");
                        audioPlayer.load();
                        audioGroup = null;
                    }
                    group.audioSource = updatedSource;
                    stopContinuousAudio();
                    prepareContinuousAudio(group);
                    pendingAudioSources.delete(index);
                    if (shouldResume && playing === index && timer == null && !framePending) {
                        show(index, playingFrame);
                    }
                    return;
                }
                const finalized = data.action === "chunk_final";
                if ((!finalized && data.action !== "chunk") || (!Array.isArray(data.frames) && !data.image)) return;

                const index = data.chunk;
                // Never let a delayed live latent-preview event replace the
                // authoritative full-VAE frame/audio group. Browser refresh
                // restores server state while websocket events are still in
                // flight, so their arrival order is not guaranteed.
                if (!finalized && chunks[index]?.finalized) return;
                const replacingPlayingChunk = finalized && index === playing;
                const displayedBeforeReplacement = replacingPlayingChunk ? displayedFrameNumber() : null;
                activeChunk = index;
                activeSubchunk = finalized ? null : (data.subchunk ?? null);
                currentStep = data.step || currentStep;
                totalSteps = data.steps || totalSteps;
                if (Array.isArray(data.sigmas)) sigmas = data.sigmas;
                const encodedFrames = Array.isArray(data.frames) ? data.frames : [data.image];
                const frameDurations = Array.isArray(data.frame_durations_ms)
                    ? data.frame_durations_ms
                    : [data.duration_ms || 1000];
                let restoredAudioSource = typeof data.audio === "string" && data.audio
                    ? `data:${data.audio_mime || "audio/wav"};base64,${data.audio}`
                    : null;
                if (finalized && pendingAudioSources.has(index)) {
                    restoredAudioSource = pendingAudioSources.get(index);
                    pendingAudioSources.delete(index);
                }
                const group = {
                    frames: encodedFrames.map(frame => `data:image/webp;base64,${frame}`),
                    durations: frameDurations,
                    frameNumbers: Array.isArray(data.frame_numbers) ? data.frame_numbers : [],
                    outputStart: data.output_start,
                    outputEnd: data.output_end,
                    sourceFps: data.fps,
                    step: currentStep,
                    finalized,
                    audioSource: restoredAudioSource,
                };
                if (typeof data.gemma_detailed_description === "string" && chunkRanges[index]) {
                    chunkRanges[index].gemma_detailed_description = data.gemma_detailed_description;
                }
                if (typeof data.gemma_retention_analysis === "string" && chunkRanges[index]) {
                    chunkRanges[index].gemma_retention_analysis = data.gemma_retention_analysis;
                }
                if (typeof data.h3_prompt === "string" && chunkRanges[index]) {
                    chunkRanges[index].h3_prompt = data.h3_prompt;
                }
                if (chunkRanges[index]) {
                    for (const key of ["h3_render_seconds", "gemma_seconds", "gemma_preproduction_seconds", "chunk_total_seconds"]) {
                        const value = Number(data[key]);
                        if (Number.isFinite(value) && value >= 0) chunkRanges[index][key] = value;
                    }
                }
                chunks[index] = group;
                prepareContinuousAudio(group);
                if (!finalized) stepPreviews[currentStep] = group;
                previewWidth = data.width;
                previewHeight = data.height;
                setSourceFps(data.fps);
                renderStatus();
                renderTransport();
                redrawGraphs();
                if (restorePersistedFrame()) return;
                if (replacingPlayingChunk && hoverStep == null) {
                    let replacementFrame = 0;
                    if (Number.isFinite(displayedBeforeReplacement) && group.frameNumbers.length) {
                        replacementFrame = group.frameNumbers.reduce((best, value, candidate) =>
                            Math.abs(Number(value) - displayedBeforeReplacement)
                                < Math.abs(Number(group.frameNumbers[best]) - displayedBeforeReplacement)
                                ? candidate : best,
                        0);
                    }
                    show(index, replacementFrame);
                } else if (hoverStep == null && timer == null && !framePending) {
                    restorePlayback();
                }
            };

            async function restoreServerState(attempt=0, force=false) {
                if (node.id == null || Number(node.id) < 0) {
                    if (attempt < 20) setTimeout(() => restoreServerState(attempt + 1, force), 100);
                    return;
                }
                const executionAtRequest = execution;
                try {
                    restoringCache = true;
                    statusPhase.textContent = force ? "Reconnecting preview…" : "Retrieving cached previous run…";
                    statusProgressFill.style.width = "0%";
                    // Workflow widget restoration normally completes before
                    // the asynchronous history request. Synchronize it here
                    // so restored playback starts at the visible FPS value.
                    setPlaybackFps(fpsWidget?.value, false);
                    const response = await api.fetchApi(
                        `/hr_endless_sampler_preview/state?node_id=${encodeURIComponent(node.id)}`,
                        { cache: "no-store" },
                    );
                    if (!response.ok) return;
                    let snapshot = await response.json();
                    // A normal refresh restores lightweight in-memory preview
                    // history. If ComfyUI no longer has that history but the
                    // green cache toggle says a finalized replay checkpoint
                    // exists, rebuild the player straight from its CPU media.
                    if (!snapshot?.reset) {
                        const maxResolution = node.widgets?.find(widget => widget.name === "max_resolution")?.value ?? 0;
                        const quality = node.widgets?.find(widget => widget.name === "quality")?.value ?? 75;
                        const cachedResponse = await api.fetchApi(
                            `/hr_endless_sampler_preview/cached_preview?node_id=${encodeURIComponent(node.id)}&max_resolution=${encodeURIComponent(maxResolution)}&quality=${encodeURIComponent(quality)}&fps=${encodeURIComponent(currentPlaybackFps())}`,
                            { cache: "no-store" },
                        );
                        if (cachedResponse.ok) snapshot = await cachedResponse.json();
                    }
                    if (!snapshot?.reset) return;
                    // A live reset arriving during fetch owns the newer render.
                    if (execution !== executionAtRequest && snapshot.execution !== execution) return;
                    if (!force && execution !== null && Number(snapshot.execution) < Number(execution)) return;
                    // Explicit reconnect must bypass the stale-reset guard after a server restart.
                    if (force) resetExecution(snapshot.reset);
                    else node._hrEndlessSamplerPreview(snapshot.reset);
                    if (snapshot.phase) node._hrEndlessSamplerPreview(snapshot.phase);
                    if (snapshot.sample_start) node._hrEndlessSamplerPreview(snapshot.sample_start);
                    if (snapshot.progress) node._hrEndlessSamplerPreview(snapshot.progress);
                    for (const chunk of snapshot.chunks || []) {
                        if (execution !== snapshot.execution) return;
                        node._hrEndlessSamplerPreview(chunk);
                        // Let input, drawing and playback run between restored chunks.
                        await new Promise(resolve => setTimeout(resolve, 0));
                    }
                    if (execution !== snapshot.execution) return;
                    if (Array.isArray(snapshot.deltas)) deltas = snapshot.deltas.slice();
                    if (Array.isArray(snapshot.step_times)) stepTimes = snapshot.step_times.slice();
                    renderStatus();
                    redrawGraphs();
                    if (snapshot.complete) node._hrEndlessSamplerPreview(snapshot.complete);
                    if (hoverStep == null && timer == null && !framePending) restorePlayback();
                } catch (error) {
                    console.warn("HR Endless Sampler preview history restore failed", error);
                } finally {
                    restoringCache = false;
                    renderStatus();
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

            timelineShell.addEventListener("mousemove", event => setChunkTooltip(chunkIndexAtTimelinePointer(event), event));
            timelineShell.addEventListener("mouseleave", () => {
                tooltipChunkIndex = null;
                tooltipSignature = null;
                tooltipPointer = null;
                chunkTooltip.hide();
            });
            const updateShiftPrompt = event => {
                if (event.key !== "Shift") return;
                shiftPromptVisible = event.type === "keydown";
                if (tooltipChunkIndex == null || tooltipPointer == null) return;
                setChunkTooltip(tooltipChunkIndex, {
                    clientX: tooltipPointer.clientX,
                    clientY: tooltipPointer.clientY,
                    shiftKey: shiftPromptVisible,
                });
            };
            document.addEventListener("keydown", updateShiftPrompt);
            document.addEventListener("keyup", updateShiftPrompt);

            const resizeObserver = new ResizeObserver(redrawGraphs);
            resizeObserver.observe(graphs);
            node.addDOMWidget("preview", "hr_endless_sampler_preview", root, { serialize: false });
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
            setTimeout(() => refreshReplayCacheStatus(), 0);
            const replayCacheStatusTimer = setInterval(refreshReplayCacheStatus, 3000);

            const previousRemoved = node.onRemoved;
            node.onRemoved = function () {
                stopContinuousAudio();
                decodedAudio.clear();
                continuousAudioContext?.close();
                stop();
                audioPlayer.removeAttribute("src");
                audioPlayer.load();
                audioStandbyPlayer.removeAttribute("src");
                audioStandbyPlayer.load();
                if (elapsedTimer != null) clearInterval(elapsedTimer);
                clearInterval(replayCacheStatusTimer);
                document.removeEventListener("pointerdown", dismissPreviewContextMenu, true);
                document.removeEventListener("keydown", updateShiftPrompt);
                document.removeEventListener("keyup", updateShiftPrompt);
                closePreviewContextMenu();
                resizeObserver.disconnect();
                chunkTooltip.remove();
                node._hrEndlessSamplerPreview = null;
                previousRemoved?.apply(this, arguments);
            };
        };
    },
});
