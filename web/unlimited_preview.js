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


function drawSigmaDelta(canvas, sigmas, deltas, steps) {
    const { context, width, height } = canvasContext(canvas);
    context.clearRect(0, 0, width, height);
    drawGrid(context, width, height);
    drawSeries(context, sigmas, Math.max(steps + 1, sigmas.length), width, height, "rgba(210,210,210,0.6)", true);
    drawSeries(context, [null, ...deltas], Math.max(steps + 1, deltas.length + 1), width, height, "#e67e22", false, true);
}


function drawStepTimes(canvas, values, steps) {
    const { context, width, height } = canvasContext(canvas);
    context.clearRect(0, 0, width, height);
    drawGrid(context, width, height);
    drawSeries(context, values, Math.max(steps, values.length), width, height, "#e67e22", false, true);
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
            root.style.cssText = "display:flex;flex-direction:column;width:100%;height:100%;min-height:410px;background:#111;border-radius:6px;overflow:hidden;color:#ddd;font:12px sans-serif;";

            const image = document.createElement("img");
            image.style.cssText = "display:block;width:100%;flex:1 1 auto;min-height:220px;object-fit:contain;background:#090909;";
            image.draggable = false;
            root.appendChild(image);

            const status = document.createElement("div");
            status.style.cssText = "box-sizing:border-box;padding:7px 9px 3px;background:#1b1b1b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
            status.textContent = "Waiting for SamplerCustomAdvanced-Unlimited…";
            root.appendChild(status);

            const decoderStatus = document.createElement("div");
            decoderStatus.style.cssText = "box-sizing:border-box;padding:0 9px 7px;color:#999;background:#1b1b1b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
            decoderStatus.textContent = "Previewer: waiting";
            root.appendChild(decoderStatus);

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

            let execution = null;
            let chunkCount = 0;
            let chunks = [];
            let durations = [];
            let playing = 0;
            let timer = null;
            let imageUpdate = 0;
            let sigmas = [];
            let deltas = [];
            let stepTimes = [];
            let currentStep = 0;
            let totalSteps = 0;
            let averageStepMs = null;
            let previewWidth = null;
            let previewHeight = null;
            let previewFps = null;
            let previewer = null;
            let complete = false;
            let windowCount = 0;
            let activeWindow = 0;

            function stop() {
                if (timer != null) clearTimeout(timer);
                timer = null;
            }

            function available(index) {
                return typeof chunks[index] === "string" && chunks[index].length > 0;
            }

            function show(index) {
                if (!available(index)) return;
                playing = index;
                const update = ++imageUpdate;
                image.removeAttribute("src");
                requestAnimationFrame(() => {
                    if (update === imageUpdate) image.src = chunks[index];
                });
                stop();
                if (chunkCount <= 1) return;
                timer = setTimeout(() => {
                    let next = index + 1;
                    while (next < chunkCount && !available(next)) next++;
                    if (next >= chunkCount) next = 0;
                    while (next < chunkCount && !available(next)) next++;
                    if (next < chunkCount) show(next);
                }, Math.max(100, durations[index] || 1000));
            }

            function renderStatus() {
                const resolution = previewWidth && previewHeight ? `${previewWidth}×${previewHeight}` : "resolution —";
                const fps = Number.isFinite(previewFps) ? `${Number(previewFps.toFixed(3))} fps` : "fps —";
                const secondsPerStep = Number.isFinite(averageStepMs) ? `${(averageStepMs / 1000).toFixed(2)}s/step` : "—s/step";
                const eta = Number.isFinite(averageStepMs) ? formatEta((totalSteps - currentStep) * averageStepMs / 1000) : "—";
                const chunk = windowCount ? `Chunk ${activeWindow + 1}/${windowCount}` : "Chunk —/—";
                status.textContent = `${complete ? "Complete · " : ""}${chunk} · ${resolution} · ${fps} · step ${currentStep}/${totalSteps || "—"} · ${secondsPerStep} · ETA ${eta}`;
                decoderStatus.textContent = `Previewer: ${previewer || "waiting"}`;
                sigmaGraph.value.textContent = Number.isFinite(deltas[currentStep - 1]) ? deltas[currentStep - 1].toFixed(3) : "—";
                timeGraph.value.textContent = Number.isFinite(stepTimes[currentStep - 1]) ? `${(stepTimes[currentStep - 1] / 1000).toFixed(2)}s` : "—";
            }

            function redrawGraphs() {
                drawSigmaDelta(sigmaGraph.canvas, sigmas, deltas, totalSteps);
                drawStepTimes(timeGraph.canvas, stepTimes, totalSteps);
            }

            node._minimaxUnlimitedPreview = data => {
                if (data.action === "reset") {
                    execution = data.execution;
                    chunkCount = data.chunk_count || 0;
                    chunks = new Array(chunkCount);
                    durations = new Array(chunkCount);
                    playing = 0;
                    sigmas = [];
                    deltas = [];
                    stepTimes = [];
                    currentStep = 0;
                    totalSteps = 0;
                    averageStepMs = null;
                    previewWidth = null;
                    previewHeight = null;
                    previewFps = null;
                    previewer = null;
                    complete = false;
                    windowCount = data.window_count || 0;
                    activeWindow = 0;
                    stop();
                    imageUpdate++;
                    image.removeAttribute("src");
                    renderStatus();
                    redrawGraphs();
                    return;
                }
                if (data.execution !== execution) return;
                if (data.action === "sample_start") {
                    sigmas = Array.isArray(data.sigmas) ? data.sigmas : [];
                    totalSteps = data.steps || Math.max(0, sigmas.length - 1);
                    previewFps = data.fps;
                    previewer = data.previewer;
                    renderStatus();
                    redrawGraphs();
                    return;
                }
                if (data.action === "window") {
                    windowCount = data.window_count || windowCount;
                    activeWindow = data.window ?? activeWindow;
                    renderStatus();
                    return;
                }
                if (data.action === "complete") {
                    complete = true;
                    renderStatus();
                    if (timer == null && available(0)) show(0);
                    return;
                }
                if (data.action === "progress") {
                    currentStep = data.step || currentStep;
                    totalSteps = data.steps || totalSteps;
                    deltas[currentStep - 1] = data.delta;
                    stepTimes[currentStep - 1] = data.step_ms;
                    averageStepMs = data.avg_step_ms;
                    previewFps = data.fps ?? previewFps;
                    previewer = data.previewer || previewer;
                    renderStatus();
                    redrawGraphs();
                    return;
                }
                if (data.action !== "chunk" || !data.image) return;

                const index = data.chunk;
                chunks[index] = `data:image/webp;base64,${data.image}`;
                durations[index] = data.duration_ms;
                previewWidth = data.width;
                previewHeight = data.height;
                previewFps = data.fps ?? previewFps;
                previewer = data.previewer || previewer;
                renderStatus();
                if (timer == null || index === playing) show(index === playing ? index : 0);
            };

            const resizeObserver = new ResizeObserver(redrawGraphs);
            resizeObserver.observe(graphs);
            node.addDOMWidget("preview", "minimax_h3_unlimited_preview", root, { serialize: false });
            node.setSize([Math.max(node.size?.[0] || 460, 460), Math.max(node.size?.[1] || 540, 540)]);

            const previousRemoved = node.onRemoved;
            node.onRemoved = function () {
                stop();
                resizeObserver.disconnect();
                imageUpdate++;
                node._minimaxUnlimitedPreview = null;
                previousRemoved?.apply(this, arguments);
            };
        };
    },
});
