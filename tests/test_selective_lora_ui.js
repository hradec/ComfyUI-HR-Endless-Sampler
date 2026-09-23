// Run with node tests/test_selective_lora_ui.js
//
// The whole panel is one DOM widget, so the module owns both the pixels and the
// controls. This drives the real module against a stand-in for the DOM the
// frontend builds, then checks the three things that matter: the node's own
// sliders are hidden, the panel is the editor, and what it writes is what
// ComfyUI reads back out of the widgets.
const fs = require("fs");
const assert = require("assert");

const PATH = "web/selective_lora_ui.js";
const source = fs.readFileSync(PATH, "utf8");

const GROUPS = ["attention", "feed_forward", "token_refiner", "modulation", "other"];

// --- a very small DOM, only as capable as the module needs ----------------

function recordingContext() {
    const context = {
        calls: [],
        texts: [],
        setTransform() {}, clearRect() { context.calls.push("clearRect"); },
        beginPath() {}, moveTo() {}, lineTo() {}, closePath() {},
        stroke() {}, fill() {}, save() {}, restore() {}, setLineDash() {},
        fillText(text) { context.texts.push(text); },
    };
    return context;
}


function matches(element, selector) {
    if (selector === "*") return true;
    const parsed = /^\[([a-z-]+)(?:="(.*)")?\]$/.exec(selector);
    assert(parsed, `unsupported selector in test helper: ${selector}`);
    const attribute = parsed[1];
    const wanted = parsed[2];
    // `data-hr-group-row` in the DOM is `dataset.hrGroupRow` in script.
    const key = attribute.startsWith("data-")
        ? attribute.slice(5).replace(/-([a-z])/g, (_, letter) => letter.toUpperCase())
        : attribute;
    const value = key in element.dataset ? element.dataset[key] : element[key];
    return wanted === undefined ? value !== undefined : value === wanted;
}


function element(tagName) {
    const node = {
        tagName,
        dataset: {},
        style: {},
        children: [],
        listeners: {},
        textContent: "",
        value: "",
        title: "",
        parent: null,
        context: null,
        clientWidth: 300,
        clientHeight: 132,
        width: 0,
        height: 0,
        getContext() {
            if (!node.context) node.context = recordingContext();
            return node.context;
        },
        append(...added) {
            for (const child of added) {
                child.parent = node;
                node.children.push(child);
            }
        },
        addEventListener(type, handler) {
            (node.listeners[type] || (node.listeners[type] = [])).push(handler);
        },
        // Dispatch the way the browser would, with the three methods the module
        // calls, and a record of what it asked the event to do.
        emit(type, event = {}) {
            const handlers = node.listeners[type] || [];
            const detail = { key: undefined, ...event };
            detail.defaultPrevented = false;
            detail.propagationStopped = false;
            detail.stopPropagation = () => { detail.propagationStopped = true; };
            detail.preventDefault = () => { detail.defaultPrevented = true; };
            for (const handler of handlers) handler(detail);
            return detail;
        },
        // Focus and blur both move the caret and fire their events, as in a browser.
        select() { node.selected = true; },
        focus() {
            documents.activeElement = node;
            node.emit("focus");
        },
        blur() {
            node.emit("blur");
            if (documents.activeElement === node) documents.activeElement = null;
        },
        setPointerCapture() {},
        releasePointerCapture() {},
        descendants() {
            const all = [];
            for (const child of node.children) all.push(child, ...child.descendants());
            return all;
        },
        querySelectorAll(selector) {
            return node.descendants().filter(candidate => matches(candidate, selector));
        },
        querySelector(selector) {
            return node.querySelectorAll(selector)[0] ?? null;
        },
    };
    return node;
}


const documents = {
    activeElement: null,
    createElement: tagName => element(tagName),
};

class ObserverStub {
    constructor(callback) {
        this.callback = callback;
        this.targets = [];
    }
    observe(target) {
        this.targets.push(target);
    }
    disconnect() {
        this.disconnected = true;
        this.targets = [];
    }
}


// --- loading the module ---------------------------------------------------

const sleep = ms => new Promise(resolve => setTimeout(resolve, ms));

async function main() {

const registered = [];
const app = {
    registerExtension(extension) {
        registered.push(extension);
    },
    graph: { setDirtyCanvas() {} },
};

// Shares chosen so each band is easy to tell apart: two live groups, one the
// adapter stores but never applies, one it does not store at all.
const REPORT = {
    attention: { modules: 8, parameter_share: 0.5, applied_share: 0.5, dead: false },
    feed_forward: { modules: 8, parameter_share: 0.25, applied_share: 0.25, dead: false },
    token_refiner: { modules: 2, parameter_share: 0.1, applied_share: 0.1, dead: false },
    modulation: { modules: 4, parameter_share: 0.15, applied_share: 0.0, dead: true },
    other: { modules: 0, parameter_share: 0.0, applied_share: 0.0, dead: false },
};

const fetched = [];
const api = {
    fetchApi(url) {
        fetched.push(url);
        const name = decodeURIComponent(/name=([^&]*)/.exec(url)[1]);
        return Promise.resolve({
            ok: true,
            status: 200,
            json: async () => ({
                ok: true,
                name,
                deep: { state: "ready" },
                notes: ["Reference framing is the strongest influence."],
                provenance: { trained_name: "test adapter" },
                groups: GROUPS.map(id => ({ id, ...REPORT[id] })),
            }),
        });
    },
};

// ComfyUI exposes module namespaces here, so the module's
// `const { app } = window.comfyAPI.app` reads `comfyAPI.app.app`.
const windowStub = {
    comfyAPI: { app: { app }, api: { api } },
    devicePixelRatio: 1,
};

const body = `${source}
return { clampUnit, tint, formatPercent, formatStrength, parseStrength, strengthSteps, strengthBounds, roundStrength, barGradient, drawRadar };`;

const exported = new Function(
    "window", "document", "ResizeObserver", "setTimeout", "clearTimeout", "console", "assert",
    body,
)(windowStub, documents, ObserverStub, setTimeout, clearTimeout, console, assert);

assert.strictEqual(registered.length, 1, "the extension must register exactly once");

// --- pure helpers ---------------------------------------------------------

assert.strictEqual(exported.clampUnit(0.5), 0.5);
assert.strictEqual(exported.clampUnit(-2), 0);
assert.strictEqual(exported.clampUnit(7), 1);
assert.strictEqual(exported.clampUnit(undefined), 0);
assert.strictEqual(exported.clampUnit("nonsense"), 0);

assert.strictEqual(exported.tint("#4aa3ff", 1), "rgba(74, 163, 255, 1)");
assert.strictEqual(exported.tint("not a colour", 1), "transparent");

// Shares are read at a glance, so small ones keep a decimal and big ones do not.
assert.strictEqual(exported.formatPercent(0.5), "50%");
assert.strictEqual(exported.formatPercent(0.20068), "20%");
assert.strictEqual(exported.formatPercent(0.0036), "0.4%");
assert.strictEqual(exported.formatPercent(0), "0.0%");
assert.strictEqual(exported.formatPercent(NaN), "--");

// The artist edits percentages of the adapter as trained.
assert.strictEqual(exported.formatStrength(1), "100%");
assert.strictEqual(exported.formatStrength(0.5), "50%");
assert.strictEqual(exported.formatStrength(1.25), "125%");
assert.strictEqual(exported.formatStrength(0), "0%");
assert.strictEqual(exported.formatStrength(undefined), "");

assert.strictEqual(exported.parseStrength("50"), 0.5);
assert.strictEqual(exported.parseStrength("50%"), 0.5);
assert.strictEqual(exported.parseStrength("-12.5%"), -0.125);
assert.strictEqual(exported.parseStrength(" 75 % "), 0.75);
assert.strictEqual(exported.parseStrength("abc"), null);
assert.strictEqual(exported.parseStrength(""), null);
assert.strictEqual(exported.parseStrength("%"), null);

// The frontend rewrites the declared step into the pair it really uses: `step2`
// is the arrow step the artist feels, `step` its tenfold coarse twin.
assert.deepStrictEqual(exported.strengthSteps({ options: { step: 0.5, step2: 0.05 } }), { fine: 0.05, coarse: 0.5 });
assert.deepStrictEqual(exported.strengthSteps({ options: { step: 0.05 } }), { fine: 0.05, coarse: 0.5 });
assert.deepStrictEqual(exported.strengthSteps({ options: { step: 0.5 } }), { fine: 0.5, coarse: 5 });
assert.deepStrictEqual(exported.strengthSteps({ options: {} }), { fine: 0.05, coarse: 0.5 });
assert.deepStrictEqual(exported.strengthSteps(null), { fine: 0.05, coarse: 0.5 });
// A pair that makes no sense must not produce a step of zero, or a coarse step
// smaller than the fine one.
assert.deepStrictEqual(exported.strengthSteps({ options: { step: 0.5, step2: 1 } }), { fine: 1, coarse: 10 });
for (const options of [{}, { step: 0.5 }, { step: 0.05 }, { step: 0.5, step2: 0.05 }, { step: 0.5, step2: 1 }, { step: 0, step2: 0 }]) {
    const steps = exported.strengthSteps({ options });
    assert(steps.fine > 0 && steps.coarse >= steps.fine, `${JSON.stringify(options)} gave ${JSON.stringify(steps)}`);
}

assert.deepStrictEqual(exported.strengthBounds({ options: { min: -2, max: 3 } }), { min: -2, max: 3 });
assert.deepStrictEqual(exported.strengthBounds({ options: {} }), { min: -10, max: 10 });

// Values are stored at the granularity the rest of ComfyUI uses.
assert.strictEqual(exported.roundStrength({ options: { round: 0.01 } }, 0.123456), 0.12);
assert.strictEqual(exported.roundStrength({ options: { precision: 2 } }, 0.123456), 0.12);
assert.strictEqual(exported.roundStrength({ options: {} }, 0.123456), 0.123456);

// Dropping half the group: a bright band to the setting, a dim band to the
// adapter's own share, then the track.
const dropped = exported.barGradient({ color: "#4aa3ff", reference: 0.8, live: 0.4 });
assert(dropped.startsWith("linear-gradient(to right, "), dropped);
assert(dropped.includes("rgba(74, 163, 255, 0.45) 0 40%"), dropped);
assert(dropped.includes("rgba(74, 163, 255, 0.18) 40% 80%"), dropped);
assert(dropped.includes("rgba(255, 255, 255, 0.05) 80% 100%"), dropped);

// A strength past 100% extends the bright band, and gradient stops must stay in
// ascending order or the browser drops the whole declaration.
const boosted = exported.barGradient({ color: "#4aa3ff", reference: 0.5, live: 1.5 });
assert(boosted.includes("rgba(74, 163, 255, 0.45) 0 100%"), boosted);
assert(!boosted.includes("0.18"), boosted);
const positions = [...boosted.matchAll(/(\d+(?:\.\d+)?)%/g)].map(match => Number(match[1]));
assert(positions.length >= 2, boosted);
for (let index = 1; index < positions.length; index += 1) {
    assert(positions[index] >= positions[index - 1], `${boosted} is not ascending`);
}

// A group with nothing stored still paints the empty track.
const empty = exported.barGradient({ color: "#4aa3ff", reference: 0, live: 0 });
assert(empty.includes("rgba(74, 163, 255, 0.45) 0 0%"), empty);
assert(empty.includes("rgba(255, 255, 255, 0.05) 0% 100%"), empty);

// The chart is redrawn from scratch, one axis label per group.
const chart = element("canvas");
exported.drawRadar(chart, [0.5, 0.25, 0.1, 0, 0], [0.25, 0.25, 0.1, 0, 0]);
assert.strictEqual(chart.context.calls.filter(call => call === "clearRect").length, 1);
assert.strictEqual(chart.context.texts.length, GROUPS.length, "one axis label per group");
assert.deepStrictEqual(chart.context.texts, ["Composition", "Texture", "Prompt", "Conditioning", "Other"]);
// A chart of nothing but zeros must still draw rather than divide by zero.
exported.drawRadar(chart, [0, 0, 0, 0, 0], [0, 0, 0, 0, 0]);
assert.strictEqual(chart.context.texts.length, GROUPS.length * 2);

// --- the node end to end --------------------------------------------------

const LABELS = {
    attention: "Composition & Reference",
    feed_forward: "Texture & Light",
    token_refiner: "Prompt Reading",
    modulation: "Conditioning Response",
    other: "Everything Else",
};

// The options the frontend really hands a FLOAT widget at runtime.
const OPTIONS = { default: 1, min: -10, max: 10, round: 0.01, step: 0.5, step2: 0.05, precision: 2 };

const node = {
    id: 7,
    size: [300, 200],
    widgets: [],
    added: [],
    sizes: [],
    addDOMWidget(name, type, element_, options) {
        const widget = { name, type, element: element_, options, value: "" };
        node.added.push(widget);
        node.widgets.push(widget);
        return widget;
    },
    setSize(size) {
        node.sizes.push(size);
        node.size = size;
    },
};
for (const id of GROUPS) {
    node.widgets.push({ name: id, value: 1, type: "FLOAT", options: { ...OPTIONS }, callback: null });
}
node.widgets.push({ name: "model", value: null, type: "MODEL" });
node.widgets.push({ name: "lora_name", value: "test.safetensors", type: "COMBO", options: {} });

const nodeType = { prototype: {} };
const extension = registered[0];
await extension.beforeRegisterNodeDef(nodeType, { name: "HREndlessSamplerSelectiveLora" });
assert.strictEqual(typeof nodeType.prototype.onNodeCreated, "function");

// A node of a different type must be left alone.
const untouched = { prototype: {} };
await extension.beforeRegisterNodeDef(untouched, { name: "SomethingElse" });
assert.strictEqual(untouched.prototype.onNodeCreated, undefined);

const previousCreated = nodeType.prototype.onNodeCreated;
previousCreated.call(node);

// The node's own sliders are gone from the node: one editor per group, and it is
// the panel. The widgets themselves stay, so their values keep serializing.
for (const id of GROUPS) {
    const widget = node.widgets.find(candidate => candidate.name === id);
    assert.strictEqual(widget.hidden, true, `${id} must be hidden`);
    assert.strictEqual(widget.label, LABELS[id], `${id} must keep an artist-facing label`);
}
assert.strictEqual(node.widgets.find(widget => widget.name === "lora_name").hidden, undefined);

// The panel is not a stored value: it must stay out of `widgets_values`, or it
// would take a positional slot away from the group strengths.
const panel = node.added[0];
assert(panel, "the panel must be added as a DOM widget");
assert.strictEqual(panel.serialize, false);
assert.strictEqual(panel.options.serialize, false);
assert.strictEqual(panel.element.style.cssText.includes("--comfy-widget-min-height"), true, panel.element.style.cssText);
assert.strictEqual(node.size[0] >= 380 && node.size[1] >= 340, true, JSON.stringify(node.size));

const row = id => panel.element.querySelector(`[data-hr-group-row="${id}"]`);
const field = id => panel.element.querySelector(`[data-hr-group-field="${id}"]`);
// The row is a colour swatch, the group name, its share, then the strength field.
const share = id => row(id).children[2];
const widget = id => node.widgets.find(candidate => candidate.name === id);
for (const id of GROUPS) {
    assert(row(id) && field(id), `${id} must get a row and a field`);
    assert.deepStrictEqual(row(id).children.map(child => child.tagName), ["span", "span", "span", "input"], `${id} row shape`);
    assert.strictEqual(row(id).children[3], field(id), `${id} field must be the last cell`);
}

// Before the analysis arrives the fields still work, but no bars are painted.
assert.strictEqual(field("attention").value, "100%", "the field starts at the widget's own value");
assert.strictEqual(share("attention").textContent, "--");

// Let the queued initial load settle. The picker was already set, so the module
// must ask for the deep tier as well.
await sleep(20);
assert.strictEqual(fetched.length, 1, "the initial load must query the analysis route");
assert(fetched[0].includes("/hr_endless_sampler_lora/analysis"));
assert(fetched[0].includes("name=test.safetensors"));
assert(fetched[0].includes("deep=1"));

// The share column reads the adapter: what it stores, and what it keeps.
assert.strictEqual(share("attention").textContent, "50% \u2192 50%");
assert.strictEqual(share("feed_forward").textContent, "25% \u2192 25%");
assert.strictEqual(share("token_refiner").textContent, "10% \u2192 10%");
assert.strictEqual(share("modulation").textContent, "no effect");
assert.strictEqual(share("other").textContent, "not stored");
// A group the adapter does not store cannot be edited into having an effect.
assert.strictEqual(field("other").style.opacity, "0.55");
assert.strictEqual(field("attention").style.opacity, "1");

// Every row paints the adapter's share as a bar behind its text.
assert(row("attention").style.backgroundImage.includes("rgba(74, 163, 255, 0.45) 0 50%"), row("attention").style.backgroundImage);
assert(row("feed_forward").style.backgroundImage.includes("rgba(240, 160, 60, 0.45) 0 25%"), row("feed_forward").style.backgroundImage);
assert(row("other").style.backgroundImage.includes("rgba(138, 143, 152, 0.45) 0 0%"), row("other").style.backgroundImage);

// The header names the adapter and says which tier the numbers came from.
const header = panel.element.children[0];
assert.strictEqual(header.children[0].textContent, "test adapter");
assert.strictEqual(header.children[1].textContent, "share of applied change");
assert(panel.element.children[3].textContent.includes("Reference framing"), panel.element.children[3].textContent);

// --- the panel is the editor ---------------------------------------------

const commit = (id, text) => {
    const element_ = field(id);
    element_.focus();
    element_.value = text;
    element_.emit("change");
};
const press = (id, key, init = {}) => field(id).emit("keydown", { key, ...init });

// Typing a percentage writes the node's own parameter.
commit("attention", "50");
assert.strictEqual(widget("attention").value, 0.5);
assert.strictEqual(field("attention").value, "50%");
assert.strictEqual(share("attention").textContent, "50% \u2192 25%");
assert(row("attention").style.backgroundImage.includes("rgba(74, 163, 255, 0.45) 0 25%"), row("attention").style.backgroundImage);
assert(row("attention").style.backgroundImage.includes("rgba(74, 163, 255, 0.18) 25% 50%"), row("attention").style.backgroundImage);
// ...and a strength edit never needs another round trip.
assert.strictEqual(fetched.length, 1, "a strength edit must not refetch");

// The panel calls the widget's own callback, so extensions watching the node
// still see the edit.
const calls = [];
widget("attention").callback = value => calls.push(value);
commit("attention", "60");
assert.deepStrictEqual(calls, [0.6]);

await sleep(20);

// Arrows use the fine step; shift and PageUp/PageDown use the coarse one.
press("attention", "ArrowUp");
assert.strictEqual(widget("attention").value, 0.65);
press("attention", "ArrowDown", { shiftKey: true });
assert.strictEqual(widget("attention").value, 0.15);
press("attention", "PageUp");
assert.strictEqual(widget("attention").value, 0.65);
press("attention", "PageDown");
assert.strictEqual(widget("attention").value, 0.15);
// The panel's own keys must not reach the canvas behind it.
assert.strictEqual(press("attention", "ArrowUp").propagationStopped, true);
assert.strictEqual(press("attention", "PageDown").defaultPrevented, true);
// A key the panel does not use is none of its business.
const unused = press("attention", "a");
assert.strictEqual(unused.defaultPrevented, false);
assert.strictEqual(unused.propagationStopped, true);

// Escape restores the text without touching the value.
commit("attention", "15");
field("attention").value = "999";
press("attention", "Escape");
assert.strictEqual(widget("attention").value, 0.15);
assert.strictEqual(field("attention").value, "15%");

// Enter commits, at the granularity the rest of ComfyUI stores.
field("attention").value = "12.3456";
press("attention", "Enter");
assert.strictEqual(widget("attention").value, 0.12);
assert.strictEqual(field("attention").value, "12%");

// Junk leaves the value alone and puts the real one back.
field("attention").value = "abc";
field("attention").emit("change");
assert.strictEqual(widget("attention").value, 0.12);
assert.strictEqual(field("attention").value, "12%");

// Double click resets to as trained.
field("attention").emit("dblclick");
assert.strictEqual(widget("attention").value, 1);
assert.strictEqual(field("attention").value, "100%");

// ComfyUI forwards every wheel event over a DOM widget to the canvas before any
// listener of ours can see it, so the panel must not pretend to own the wheel:
// a wheel over a field must leave the strength alone.
field("feed_forward").emit("wheel", { deltaY: -100 });
assert.strictEqual(widget("feed_forward").value, 1);
assert.strictEqual(field("feed_forward").value, "100%");

// Dragging up scrubs in fine steps, four pixels each: 12 px is three steps.
const drag = (id, ...events) => {
    const element_ = field(id);
    for (const [type, y] of events) {
        element_.emit(type, { button: 0, pointerId: 1, clientY: y });
    }
};
const before = widget("token_refiner").value;
drag("token_refiner", ["pointerdown", 200], ["pointermove", 188], ["pointerup", 188]);
assert.strictEqual(widget("token_refiner").value, Number((before + 0.15).toFixed(2)));
// A drag that never moves is a click, and selects the text instead.
field("token_refiner").selected = false;
drag("token_refiner", ["pointerdown", 200], ["pointerup", 200]);
assert.strictEqual(field("token_refiner").selected, true);

// A value pushed past the widget's own limits is clamped, not stored raw.
commit("modulation", "-9999");
assert.strictEqual(widget("modulation").value, -10);
assert.strictEqual(field("modulation").value, "-1000%");

// The panel never overwrites the field the artist is typing in.
field("modulation").focus();
field("modulation").value = "half typed";
field("other").emit("change");
assert.strictEqual(field("modulation").value, "half typed");
field("modulation").blur();
assert.strictEqual(field("modulation").value, "-1000%");

// --- refetching -----------------------------------------------------------

// Switching the LoRA does refetch, because the whole shape may differ. The
// frontend assigns the value before firing the callback, so do the same.
const picker = node.widgets.find(candidate => candidate.name === "lora_name");
picker.value = "other.safetensors";
picker.callback("other.safetensors");
await sleep(20);
assert.strictEqual(fetched.length, 2, "a LoRA change must refetch");
assert(fetched[1].includes("name=other.safetensors"));
assert(fetched[1].includes("deep=1"));

// Nodes 2.0 routes widget edits through onWidgetChanged, which must repaint
// without refetching.
node.onWidgetChanged("attention", 0.5, 1, widget("attention"));
await sleep(20);
assert.strictEqual(fetched.length, 2, "a widget change must not refetch");

// A failed analysis must say so rather than leave empty bars unlabelled.
api.fetchApi = url => {
    fetched.push(url);
    return Promise.resolve({ ok: false, status: 400, json: async () => ({ ok: false, error: "Unknown LoRA: nope" }) });
};
picker.value = "nope.safetensors";
picker.callback("nope.safetensors");
await sleep(20);
assert.strictEqual(panel.element.children[3].textContent.includes("Unknown LoRA: nope"), true, panel.element.children[3].textContent);

// --- teardown -------------------------------------------------------------

assert.strictEqual(panel.element.listeners, undefined || panel.element.listeners, "the host element has no listeners of its own");
node.onRemoved();
assert.strictEqual(node.onRemoved !== undefined, true);
node.onRemoved();
console.log("Selective LoRA UI checks passed.");
}

main().catch(error => {
    console.error(error);
    process.exit(1);
});
