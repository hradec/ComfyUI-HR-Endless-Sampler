// Run with: node tests/test_preview_prompt.js (no dependencies).
function checkPreviewPrompts(sources) {
    const assert = (condition, message) => { if (!condition) throw new Error(message); };
    for (const [name, source] of Object.entries(sources)) {
        const finished = name.includes("finished");
        const tooltipName = finished ? "createFinishedChunkTooltip" : "createColoredChunkTooltip";
        const end = source.indexOf(finished ? "function formatBrowserBytes(" : "function canvasContext(");
        const element = () => ({
            style: {}, children: [], textContent: "",
            appendChild(child) { this.children.push(child); },
            replaceChildren() { this.children = []; },
            getBoundingClientRect() { return { width: 100, height: 100 }; },
        });
        const document = { body: element(), createElement: element };
        const window = { comfyAPI: { app: {}, api: {} }, innerWidth: 1000, innerHeight: 1000 };
        const api = new Function("window", "document", source.slice(0, end) + `\nreturn { chunkPromptDescription, tooltip: ${tooltipName}() };`)(window, document);
        const description = '[Shot 1] She says <d>[English] Hello <Subject 1>!</d>\n[Shot 2] She leaves.';
        const fullPrompt = `summary: Continue.\ndetailed_description: ${description}\noverall_soundscape: Wind.`;
        assert(api.chunkPromptDescription({ h3_prompt: fullPrompt, gemma_detailed_description: "stale" }) === description, name + ": actual prompt must win");
        assert(api.chunkPromptDescription({ gemma_detailed_description: "legacy" }) === "legacy", name + ": legacy fallback");
        assert(api.chunkPromptDescription({ h3_prompt: "integrated_multimodal_description: Plain chunk." }) === "Plain chunk.", name + ": alternate field");
        const chunk = { chunk: 1, start: 0, end: 19 };
        const shotRanges = [{ shot: 1, start: 0, end: 9 }, { shot: 2, start: 10, end: 19 }];
        let normalColors;
        for (const showFullPrompt of [false, true]) {
            api.tooltip.show({ clientX: 0, clientY: 0 }, { help: "", chunk, timing: [], description, fullPrompt, showFullPrompt, shotRanges, colors: ["#f4b942", "#45b7d1"] });
            const tooltip = document.body.children[0];
            const spans = tooltip.children.flatMap(child => child.children);
            assert(spans.map(span => span.textContent).join("") === (showFullPrompt ? fullPrompt : description), name + ": preserve literal prompt text");
            const dialogue = spans.find(span => span.textContent.startsWith("<d>"));
            const shotColor = spans.find(span => span.textContent.includes("[Shot 1]")).style.color;
            assert(dialogue?.style.fontWeight === "700" && dialogue.style.color === `color-mix(in srgb, ${shotColor} 65%, white)`, name + ": bold dialogue brightens its own shot color in both modes");
            const colors = spans.filter(span => /\[Shot [12]\]/.test(span.textContent)).map(span => span.style.color);
            if (!showFullPrompt) {
                normalColors = colors;
                assert(tooltip.children.some(child => child.textContent === "chunk detailed_description:"), name + ": neutral heading");
            } else {
                assert(JSON.stringify(colors) === JSON.stringify(normalColors), name + ": Shift must preserve shot colors");
            }
        }
    }
}

if (typeof require !== "undefined" && require.main === module) {
    const fs = require("node:fs");
    const path = require("node:path");
    const sources = {};
    for (const name of ["unlimited_preview.js", "finished_video_player.js"]) sources[name] = fs.readFileSync(path.join(__dirname, "../web", name), "utf8");
    checkPreviewPrompts(sources);
    console.log("Preview prompt checks passed.");
}
