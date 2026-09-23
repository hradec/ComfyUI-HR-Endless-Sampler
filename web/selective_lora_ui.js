// Artist-facing view, and the only editor, of what the selected H3 LoRA contains.
//
// The whole panel is one DOM widget, so it renders and behaves the same in the
// classic canvas UI and under Nodes 2.0, and every pixel it draws is its own:
//   * the coloured bar behind each group row is that group's share of the
//     adapter. The dim band is the share the adapter stores, and the brighter
//     part is how much of that share the current strength keeps.
//   * the strength field at the right of each row is the single control for that
//     group. The node's own sliders are hidden, so each group has exactly one
//     editor and the panel can never disagree with the graph. The values stay in
//     the widgets, so they keep serializing exactly as before.
//   * the radar compares the adapter as trained against the current settings, so
//     changing a strength never hides the reference.

const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const NODE_NAME = "HREndlessSamplerSelectiveLora";
const ANALYSIS_ENDPOINT = "/hr_endless_sampler_lora/analysis";
const POLL_MS = 1200;
const POLL_LIMIT = 400;
// 1.0 applies a group exactly as the adapter was trained; the widget default.
const DEFAULT_STRENGTH = 1;

// Widget names are fixed by the node definition; only their displayed labels change.
const GROUPS = [
    { id: "attention", label: "Composition & Reference", short: "Composition", color: "#4aa3ff" },
    { id: "feed_forward", label: "Texture & Light", short: "Texture", color: "#f0a03c" },
    { id: "token_refiner", label: "Prompt Reading", short: "Prompt", color: "#b06cff" },
    { id: "modulation", label: "Conditioning Response", short: "Conditioning", color: "#3fbf8f" },
    { id: "other", label: "Everything Else", short: "Other", color: "#8a8f98" },
];

// Row bars are dimmer than the radar's because the row's text sits on top of them.
const BAR_ALPHA_KEPT = 0.45;
const BAR_ALPHA_DROPPED = 0.18;
const BAR_ALPHA_TRACK = 0.05;

const REFERENCE_STROKE = "rgba(226,226,226,0.85)";
const REFERENCE_FILL = "rgba(206,206,206,0.13)";
const LIVE_STROKE = "#ffe600";
const LIVE_FILL = "rgba(255,230,0,0.17)";

const SHARE_COLOR = "#e6e6e6";
const SHARE_MUTED = "#7a7f87";
const SHARE_WARNING = "#e07a5f";


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


function clampUnit(value) {
    const number = Number(value);
    return Number.isFinite(number) ? Math.max(0, Math.min(1, number)) : 0;
}


function formatPercent(value) {
    if (!Number.isFinite(value)) return "--";
    const percent = value * 100;
    return (percent >= 10 ? percent.toFixed(0) : percent.toFixed(1)) + "%";
}


// Strengths are shown as percentages of the adapter as trained, which is what the
// artist is editing: 100% keeps the group exactly as the adapter stored it.
function formatStrength(value) {
    const number = Number(value);
    if (!Number.isFinite(number)) return "";
    const percent = number * 100;
    const rounded = Math.abs(percent - Math.round(percent)) < 0.001
        ? Math.round(percent)
        : Math.round(percent * 10) / 10;
    return `${rounded}%`;
}


// Accepts "50", "50%", "-12.5%" and any other field text; null means "not a number".
function parseStrength(text) {
    const cleaned = String(text).replace(/[^0-9eE+\-.]/g, "");
    if (!cleaned) return null;
    const percent = Number(cleaned);
    return Number.isFinite(percent) ? percent / 100 : null;
}


// The node declares one step, but the frontend rewrites it at runtime into the
// pair it actually uses: `step2` is the real arrow/drag granularity and `step` is
// its tenfold coarse twin, kept for the legacy canvas widget. `round` is the
// granularity a value is stored at. Older widgets only carry `step`.
function strengthSteps(widget) {
    const options = widget?.options || {};
    const coarse = Number(options.step);
    const fine = Number(options.step2);
    if (Number.isFinite(fine) && fine > 0) {
        return { fine, coarse: Number.isFinite(coarse) && coarse > fine ? coarse : fine * 10 };
    }
    if (Number.isFinite(coarse) && coarse > 0) {
        return coarse > 10 ? { fine: coarse / 10, coarse } : { fine: coarse, coarse: coarse * 10 };
    }
    return { fine: 0.05, coarse: 0.5 };
}


function strengthBounds(widget) {
    const options = widget?.options || {};
    return {
        min: Number.isFinite(options.min) ? options.min : -10,
        max: Number.isFinite(options.max) ? options.max : 10,
    };
}


// Store strengths at the same granularity the rest of ComfyUI uses, so typed
// values do not drift into float noise.
function roundStrength(widget, value) {
    const options = widget?.options || {};
    const round = Number(options.round);
    const decimals = Number.isFinite(round) && round > 0
        ? Math.max(0, Math.round(-Math.log10(round)))
        : Number.isFinite(options.precision) ? Number(options.precision) : null;
    return decimals === null ? value : Number(value.toFixed(decimals));
}


function tint(hex, alpha) {
    const value = Number.parseInt(String(hex).slice(1), 16);
    if (!Number.isFinite(value)) return "transparent";
    return `rgba(${(value >> 16) & 255}, ${(value >> 8) & 255}, ${value & 255}, ${alpha})`;
}


// The group's share as a band, with the part the current strength keeps painted
// brighter than the part it drops. Gradient stops have to stay in ascending
// order, so a strength pushed past 100% simply extends the bright band.
function barGradient(bar) {
    const reference = clampUnit(bar.reference);
    const live = clampUnit(bar.live);
    const stops = [`${tint(bar.color, BAR_ALPHA_KEPT)} 0 ${live * 100}%`];
    if (reference > live) {
        stops.push(`${tint(bar.color, BAR_ALPHA_DROPPED)} ${live * 100}% ${reference * 100}%`);
    }
    stops.push(`${tint("#ffffff", BAR_ALPHA_TRACK)} ${Math.max(live, reference) * 100}% 100%`);
    return `linear-gradient(to right, ${stops.join(", ")})`;
}


function drawRadar(canvas, reference, live) {
    const { context, width, height } = canvasContext(canvas);
    context.clearRect(0, 0, width, height);

    const centreX = width / 2;
    const centreY = height / 2 + 2;
    const radius = Math.max(16, Math.min(width, height) / 2 - 26);
    const count = GROUPS.length;
    // Start at the top and go clockwise, so the first group reads naturally.
    const angleAt = index => (-Math.PI / 2) + (index * 2 * Math.PI / count);
    // Both rings share one scale, so pushing a strength past 100% rescales the
    // chart instead of being silently clipped.
    const peak = Math.max(...reference, ...live, 1e-9);

    context.save();
    for (const fraction of [0.25, 0.5, 0.75, 1]) {
        context.beginPath();
        for (let index = 0; index < count; index++) {
            const spoke = angleAt(index);
            const x = centreX + Math.cos(spoke) * radius * fraction;
            const y = centreY + Math.sin(spoke) * radius * fraction;
            if (index) context.lineTo(x, y);
            else context.moveTo(x, y);
        }
        context.closePath();
        context.strokeStyle = fraction === 1 ? "rgba(255,255,255,0.28)" : "rgba(255,255,255,0.10)";
        context.lineWidth = 1;
        context.stroke();
    }

    context.font = "9px sans-serif";
    for (let index = 0; index < count; index++) {
        const spoke = angleAt(index);
        const cos = Math.cos(spoke);
        const sin = Math.sin(spoke);
        context.beginPath();
        context.moveTo(centreX, centreY);
        context.lineTo(centreX + cos * radius, centreY + sin * radius);
        context.strokeStyle = "rgba(255,255,255,0.10)";
        context.stroke();

        // Keep every axis label inside the canvas, and colour it like its bar.
        context.fillStyle = GROUPS[index].color;
        context.textAlign = cos > 0.3 ? "left" : cos < -0.3 ? "right" : "center";
        context.textBaseline = sin > 0.3 ? "top" : sin < -0.3 ? "bottom" : "middle";
        context.fillText(GROUPS[index].short, centreX + cos * (radius + 4), centreY + sin * (radius + 4));
    }

    const polygon = (values, fill, stroke, dashed) => {
        context.beginPath();
        for (let index = 0; index < count; index++) {
            const spoke = angleAt(index);
            const scaled = radius * Math.max(0, values[index]) / peak;
            const x = centreX + Math.cos(spoke) * scaled;
            const y = centreY + Math.sin(spoke) * scaled;
            if (index) context.lineTo(x, y);
            else context.moveTo(x, y);
        }
        context.closePath();
        context.fillStyle = fill;
        context.fill();
        context.setLineDash(dashed ? [4, 3] : []);
        context.strokeStyle = stroke;
        context.lineWidth = dashed ? 1.2 : 1.6;
        context.stroke();
        context.setLineDash([]);
    };

    // The adapter as trained stays visible at all times.
    polygon(reference, REFERENCE_FILL, REFERENCE_STROKE, true);
    polygon(live, LIVE_FILL, LIVE_STROKE, false);
    context.restore();
}


app.registerExtension({
    name: "HREndlessSampler.SelectiveLora",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData?.name !== NODE_NAME) return;

        const previousCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            previousCreated?.apply(this, arguments);
            const node = this;

            let report = null;
            let deepState = "idle";
            let statusText = "Choose a LoRA to see what it contains.";
            let requestToken = 0;
            let pollTimer = null;
            let pollCount = 0;
            let disposed = false;

            // The panel is the control surface for these values, so the node's own
            // rows would be a second, equally authoritative slider. Hide them, but
            // keep the widgets themselves: their values still serialize, so saved
            // workflows and the loader are untouched.
            const widgets = new Map();
            for (const widget of node.widgets || []) {
                const group = GROUPS.find(candidate => candidate.id === widget.name);
                if (!group) continue;
                widget.label = group.label;
                widget.hidden = true;
                widgets.set(group.id, widget);
            }

            const root = document.createElement("div");
            // The frontend sizes a DOM widget from these, so the panel keeps room
            // for the chart and every row even when the node is short.
            root.style.cssText = "display:flex;flex-direction:column;width:100%;box-sizing:border-box;padding:4px 6px 2px;background:#1b1b1b;border-radius:5px;color:#ccc;font:11px sans-serif;overflow:hidden;--comfy-widget-min-height:296px;--comfy-widget-height:296px;";

            const header = document.createElement("div");
            header.style.cssText = "display:flex;justify-content:space-between;gap:6px;align-items:baseline;white-space:nowrap;overflow:hidden;";
            const title = document.createElement("span");
            title.style.cssText = "color:#eee;font-weight:600;overflow:hidden;text-overflow:ellipsis;";
            const metric = document.createElement("span");
            metric.style.cssText = "color:#8a8f98;font-size:10px;flex:0 0 auto;";
            header.append(title, metric);

            const chart = document.createElement("div");
            chart.style.cssText = "position:relative;width:100%;height:132px;flex:0 0 auto;margin-top:2px;";
            const chartCanvas = document.createElement("canvas");
            chartCanvas.style.cssText = "display:block;width:100%;height:100%;";
            chart.append(chartCanvas);

            const legend = document.createElement("div");
            legend.style.cssText = "display:flex;flex-direction:column;gap:1px;margin-top:3px;";

            const status = document.createElement("div");
            status.style.cssText = "margin-top:3px;padding-top:3px;border-top:1px solid #333;color:#9aa0a6;font-size:10px;line-height:1.35;max-height:52px;overflow:hidden;";

            root.append(header, chart, legend, status);

            // One row per group: the bar is its background, the share is what the
            // adapter holds, and the field is the strength the artist edits.
            const rows = new Map();
            for (const group of GROUPS) {
                const widget = widgets.get(group.id);
                const row = document.createElement("div");
                row.style.cssText = "display:flex;align-items:center;gap:5px;padding:1px 3px;border-radius:3px;box-shadow:inset 0 0 0 1px rgba(255,255,255,.07);white-space:nowrap;overflow:hidden;";
                row.title = widget?.options?.tooltip || widget?.tooltip || group.label;
                row.dataset.hrGroupRow = group.id;

                const swatch = document.createElement("span");
                swatch.style.cssText = `flex:0 0 auto;width:8px;height:8px;border-radius:2px;background:${group.color};`;
                const name = document.createElement("span");
                name.style.cssText = "flex:1 1 auto;overflow:hidden;text-overflow:ellipsis;";
                name.textContent = group.label;
                const share = document.createElement("span");
                share.style.cssText = `flex:0 0 auto;min-width:48px;text-align:right;font-size:10px;font-variant-numeric:tabular-nums;color:${SHARE_COLOR};`;
                share.textContent = "--";

                const field = document.createElement("input");
                field.type = "text";
                field.inputMode = "decimal";
                field.autocomplete = "off";
                field.spellcheck = false;
                field.style.cssText = "flex:0 0 auto;width:52px;box-sizing:border-box;padding:0 3px;border:1px solid rgba(255,255,255,.16);border-radius:3px;background:rgba(0,0,0,.34);color:#f2f2f2;font:11px/16px sans-serif;text-align:right;font-variant-numeric:tabular-nums;";
                field.title = "Group strength. Type a percentage, drag up and down, or use the arrow keys. 100% applies this part of the adapter exactly as trained, 0 disables it. Double click to reset.";
                field.dataset.hrGroupField = group.id;

                row.append(swatch, name, share, field);
                legend.append(row);
                rows.set(group.id, { group, row, share, field, widget, gradient: "" });
            }

            const showField = entry => {
                const widget = widgets.get(entry.group.id);
                const text = formatStrength(widget?.value);
                if (entry.field.value !== text) entry.field.value = text;
            };

            const applyStrength = (group, strength) => {
                const widget = widgets.get(group.id);
                if (!widget) return;
                const { min, max } = strengthBounds(widget);
                const wanted = Number.isFinite(strength)
                    ? roundStrength(widget, Math.max(min, Math.min(max, strength)))
                    : DEFAULT_STRENGTH;
                if (widget.value !== wanted) {
                    widget.value = wanted;
                    // The node's own callback is what the canvas uses when a widget
                    // is edited, so extensions watching this group still see it.
                    try {
                        widget.callback?.(wanted);
                    } catch (error) {
                        // A failing callback must not leave the panel unusable.
                    }
                    app.graph?.setDirtyCanvas?.(true, true);
                }
                render();
                showField(rows.get(group.id));
            };

            const commitField = entry => {
                const parsed = parseStrength(entry.field.value);
                if (parsed === null) showField(entry);
                else applyStrength(entry.group, parsed);
            };

            const nudgeField = (entry, delta) => {
                const widget = widgets.get(entry.group.id);
                const current = Number(widget?.value);
                applyStrength(entry.group, (Number.isFinite(current) ? current : DEFAULT_STRENGTH) + delta);
            };

            for (const entry of rows.values()) {
                const field = entry.field;
                // Keep the canvas from panning, zooming or dragging the node while
                // the field is being used, and keep its own keys off the shortcut
                // handlers that listen on the document.
                const swallow = event => event.stopPropagation();
                field.addEventListener("keydown", event => {
                    event.stopPropagation();
                    const steps = strengthSteps(widgets.get(entry.group.id));
                    if (event.key === "Enter") {
                        event.preventDefault();
                        commitField(entry);
                        field.blur();
                    } else if (event.key === "Escape") {
                        event.preventDefault();
                        showField(entry);
                        field.blur();
                    } else if (event.key === "ArrowUp" || event.key === "ArrowDown") {
                        event.preventDefault();
                        const size = event.shiftKey ? steps.coarse : steps.fine;
                        nudgeField(entry, (event.key === "ArrowUp" ? 1 : -1) * size);
                    } else if (event.key === "PageUp" || event.key === "PageDown") {
                        event.preventDefault();
                        nudgeField(entry, (event.key === "PageUp" ? 1 : -1) * steps.coarse);
                    }
                });
                field.addEventListener("keyup", swallow);
                field.addEventListener("change", () => commitField(entry));
                // A browser fires `change` on blur too, but never rely on it: the
                // field must never keep text that disagrees with the node.
                field.addEventListener("blur", () => showField(entry));
                field.addEventListener("focus", () => field.select());
                field.addEventListener("dblclick", () => applyStrength(entry.group, DEFAULT_STRENGTH));
                // No wheel handler: ComfyUI intercepts wheel events at document
                // capture over every DOM widget and forwards them to the canvas, so
                // one here could never run. Scrolling zooms the graph, as elsewhere.

                // Dragging up and down scrubs the strength, four pixels a step.
                // Vertically, because a horizontal drag would fight text selection
                // in the field. A press that never moves just selects the text.
                let drag = null;
                field.addEventListener("pointerdown", event => {
                    event.stopPropagation();
                    if (event.button !== 0) return;
                    const widget = widgets.get(entry.group.id);
                    const current = Number(widget?.value);
                    drag = {
                        startY: event.clientY,
                        startValue: Number.isFinite(current) ? current : DEFAULT_STRENGTH,
                        moved: false,
                    };
                    field.setPointerCapture?.(event.pointerId);
                });
                field.addEventListener("pointermove", event => {
                    if (!drag) return;
                    event.stopPropagation();
                    const widget = widgets.get(entry.group.id);
                    const steps = Math.round((drag.startY - event.clientY) / 4);
                    if (!steps) return;
                    drag.moved = true;
                    applyStrength(entry.group, drag.startValue + steps * strengthSteps(widget).fine);
                });
                field.addEventListener("pointerup", event => {
                    if (!drag) return;
                    event.stopPropagation();
                    field.releasePointerCapture?.(event.pointerId);
                    if (!drag.moved) field.select();
                    drag = null;
                });
                field.addEventListener("pointercancel", () => { drag = null; });
            }

            // Group shares, re-read from the current widget values on every redraw.
            const readShares = () => {
                const measured = deepState === "ready" && !!report;
                const reference = {};
                for (const group of GROUPS) {
                    const entry = report?.groups?.find(candidate => candidate.id === group.id);
                    const share = measured ? entry?.applied_share : entry?.parameter_share;
                    reference[group.id] = Number.isFinite(share) ? share : 0;
                }
                const live = {};
                for (const group of GROUPS) {
                    const weight = Number(widgets.get(group.id)?.value);
                    live[group.id] = reference[group.id] * Math.abs(Number.isFinite(weight) ? weight : 1);
                }
                return { reference, live, measured };
            };

            const render = () => {
                const { reference, live, measured } = readShares();
                const described = new Map((report?.groups || []).map(entry => [entry.id, entry]));

                for (const group of GROUPS) {
                    const entry = rows.get(group.id);
                    const info = described.get(group.id);
                    const stored = !info || !!info.modules;
                    const dead = measured && stored && !!info.dead;

                    const gradient = barGradient({
                        reference: reference[group.id],
                        live: live[group.id],
                        color: group.color,
                    });
                    if (entry.gradient !== gradient) {
                        entry.gradient = gradient;
                        entry.row.style.backgroundImage = gradient;
                    }

                    let text = "--";
                    let color = SHARE_COLOR;
                    if (report) {
                        if (!stored) {
                            text = "not stored";
                            color = SHARE_MUTED;
                        } else if (dead) {
                            // The adapter stores these weights but applies nothing.
                            text = "no effect";
                            color = SHARE_WARNING;
                        } else if (measured) {
                            text = `${formatPercent(reference[group.id])} \u2192 ${formatPercent(live[group.id])}`;
                        } else {
                            text = formatPercent(reference[group.id]);
                        }
                    }
                    entry.share.textContent = text;
                    entry.share.style.color = color;
                    entry.field.style.opacity = stored ? "1" : "0.55";
                    // Never fight the artist for the field they are typing in.
                    if (document.activeElement !== entry.field) showField(entry);
                }

                // drawRadar scales both rings by its own peak, so pushing a strength
                // past 100% rescales the chart rather than clipping.
                drawRadar(chartCanvas,
                    GROUPS.map(group => reference[group.id]),
                    GROUPS.map(group => live[group.id]));

                title.textContent = report ? (report.provenance?.trained_name || report.name) : "No LoRA selected";
                title.title = report?.name || "";
                metric.textContent = !report ? "" : measured ? "share of applied change" : "share of stored weights";
                status.textContent = [statusText, ...(report?.notes || []).slice(0, 2)].join(" \u00b7 ");
                status.title = status.textContent;

                app.graph?.setDirtyCanvas(true, true);
            };

            const schedulePoll = (name) => {
                clearTimeout(pollTimer);
                if (disposed || pollCount >= POLL_LIMIT) return;
                pollCount += 1;
                pollTimer = setTimeout(() => load(name, true), POLL_MS);
            };

            const load = async (name, withDeep) => {
                const token = ++requestToken;
                if (!name) {
                    report = null;
                    deepState = "idle";
                    statusText = "Choose a LoRA to see what it contains.";
                    render();
                    return;
                }
                try {
                    const query = `name=${encodeURIComponent(name)}&deep=${withDeep ? 1 : 0}`;
                    const response = await api.fetchApi(`${ANALYSIS_ENDPOINT}?${query}`, { cache: "no-store" });
                    const body = await response.json().catch(() => ({}));
                    if (disposed || token !== requestToken) return;
                    if (!response.ok || body.ok === false) {
                        report = null;
                        deepState = "failed";
                        statusText = response.status === 404 || response.status === 405
                            ? "The LoRA analysis endpoint is not loaded. Restart ComfyUI, then refresh the browser."
                            : `Analysis unavailable: ${body.error || `HTTP ${response.status}`}`;
                        render();
                        return;
                    }
                    report = body;
                    deepState = body.deep?.state || "skipped";
                    if (deepState === "computing") {
                        statusText = "Measuring how much each group actually changes the model...";
                        render();
                        schedulePoll(name);
                        return;
                    }
                    if (deepState === "failed") {
                        statusText = `Could not measure group influence: ${body.deep?.error || "unknown error"}. Showing stored weight sizes instead.`;
                    } else {
                        statusText = (body.notes || [])[0] || "Loaded.";
                    }
                    render();
                } catch (error) {
                    if (disposed || token !== requestToken) return;
                    report = null;
                    deepState = "failed";
                    statusText = `Analysis request failed: ${error?.message || error}`;
                    render();
                }
            };

            const refresh = () => {
                pollCount = 0;
                clearTimeout(pollTimer);
                load(node.widgets?.find(widget => widget.name === "lora_name")?.value, true);
            };

            // A strength change never needs a refetch: the shape is already known.
            // Nodes 2.0 routes widget edits through this callback and never calls
            // onWidgetChanged, so the LoRA picker has to be caught here too.
            const wrapWidget = (widget) => {
                if (!widget || widget._hrLoraWrapped) return;
                widget._hrLoraWrapped = true;
                const previous = widget.callback;
                widget.callback = function () {
                    const result = previous?.apply(this, arguments);
                    if (widget.name === "lora_name") {
                        clearTimeout(node._hrLoraRefresh);
                        node._hrLoraRefresh = setTimeout(refresh, 0);
                    } else {
                        clearTimeout(node._hrLoraRedraw);
                        node._hrLoraRedraw = setTimeout(render, 0);
                    }
                    return result;
                };
            };
            for (const widget of node.widgets || []) {
                wrapWidget(widget);
            }

            const previousConfigure = node.onConfigure;
            node.onConfigure = function () {
                previousConfigure?.apply(this, arguments);
                setTimeout(refresh, 0);
            };

            const previousWidgetChanged = node.onWidgetChanged;
            node.onWidgetChanged = function (name, value, oldValue, widget) {
                previousWidgetChanged?.apply(this, arguments);
                if (name === "lora_name") {
                    refresh();
                    return;
                }
                if (GROUPS.some(group => group.id === name)) {
                    clearTimeout(node._hrLoraRedraw);
                    node._hrLoraRedraw = setTimeout(render, 0);
                }
            };

            // `widget.serialize` keeps the panel out of `widgets_values` (so the
            // stored group strengths keep their positions), and `options.serialize`
            // keeps it out of the API prompt. The frontend treats the two flags
            // separately, and neither is implied by the other.
            const panel = node.addDOMWidget("lora_groups", "hr_endless_sampler_lora", root, { serialize: false });
            if (panel) {
                panel.serialize = false;
                panel.serializeValue = () => undefined;
            }
            node.setSize([
                Math.max(node.size?.[0] || 380, 380),
                Math.max(node.size?.[1] || 340, 340),
            ]);

            const resizeObserver = new ResizeObserver(render);
            resizeObserver.observe(chart);
            render();
            setTimeout(refresh, 0);

            const previousRemoved = node.onRemoved;
            node.onRemoved = function () {
                disposed = true;
                clearTimeout(pollTimer);
                clearTimeout(node._hrLoraRedraw);
                clearTimeout(node._hrLoraRefresh);
                resizeObserver.disconnect();
                previousRemoved?.apply(this, arguments);
            };
        };
    },
});
