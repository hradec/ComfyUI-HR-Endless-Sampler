// Run with: node tests/test_preview_prompt.js (no dependencies).
function checkPreviewPrompts(sources) {
    const assert = (condition, message) => { if (!condition) throw new Error(message); };
    for (const [name, source] of Object.entries(sources)) {
        const finished = name.includes("finished");
        const tooltipName = finished ? "createFinishedChunkTooltip" : "createColoredChunkTooltip";
        const end = source.indexOf(finished ? "function formatBrowserBytes(" : "function canvasContext(");
        const element = () => ({
            style: {}, children: [], textContent: "", scrollTop: 0, scrollHeight: 0, clientHeight: 0, listeners: {},
            appendChild(child) { this.children.push(child); },
            replaceChildren() { this.children = []; },
            getBoundingClientRect() { return { width: 100, height: 100 }; },
            remove() { this.removed = true; },
            addEventListener(type, handler) { (this.listeners[type] = this.listeners[type] || []).push(handler); },
            dispatchEvent(type, event = {}) { for (const handler of this.listeners[type] || []) handler(event); },
        });
        const document = { body: element(), createElement: element };
        // The tooltip takes the wheel from the window's capture phase, ahead of the
        // frontend's own document.body handler.
        const windowListeners = {};
        const window = {
            comfyAPI: { app: {}, api: {} }, innerWidth: 1000, innerHeight: 1000,
            addEventListener(type, handler) { (windowListeners[type] = windowListeners[type] || []).push(handler); },
            removeEventListener(type, handler) { windowListeners[type] = (windowListeners[type] || []).filter(item => item !== handler); },
        };
        const dispatchWindowEvent = (type, event) => {
            let calls = 0;
            for (const handler of [...(windowListeners[type] || [])]) {
                handler(event);
                calls++;
            }
            return calls;
        };
        const extra = finished ? "" : ", audioTeacherAttempts, audioTeacherAttemptTranscripts, promptSections, subchunkAt, subchunkPrompt, mergeFrameGroup";
        const api = new Function("window", "document", source.slice(0, end) + `\nreturn { chunkPromptDescription${extra}, tooltip: ${tooltipName}() };`)(window, document);
        const description = '[Shot 1] She says <d>[English] Hello <Subject 1>!</d>\n[Shot 2] She leaves.';
        const fullPrompt = `summary: Continue.\ndetailed_description: ${description}\noverall_soundscape: Wind.`;
        assert(api.chunkPromptDescription({ h3_prompt: fullPrompt, gemma_detailed_description: "stale" }) === description, name + ": actual prompt must win");
        assert(api.chunkPromptDescription({ gemma_detailed_description: "legacy" }) === "legacy", name + ": legacy fallback");
        assert(api.chunkPromptDescription({ h3_prompt: "integrated_multimodal_description: Plain chunk." }) === "Plain chunk.", name + ": alternate field");
        const chunk = { chunk: 1, start: 0, end: 19 };
        const shotRanges = [{ shot: 1, start: 0, end: 9 }, { shot: 2, start: 10, end: 19 }];
        // The preview frames the description section, header and all; the finished player shows
        // the description body on its own.
        const plainText = finished ? description : `detailed_description: ${description}\n`;
        let normalColors;
        for (const showFullPrompt of [false, true]) {
            api.tooltip.show({ clientX: 0, clientY: 0 }, { help: "", chunk, timing: [], description, fullPrompt, showFullPrompt, shotRanges, colors: ["#f4b942", "#45b7d1"] });
            const tooltip = document.body.children[0];
            const spans = collectPromptSpans(tooltip, finished);
            assert(spans.map(span => span.textContent).join("") === (showFullPrompt ? fullPrompt : plainText), name + ": preserve literal prompt text");
            const dialogue = spans.find(span => span.textContent.startsWith("<d>"));
            const shotColor = spans.find(span => span.textContent.includes("[Shot 1]")).style.color;
            assert(dialogue?.style.fontWeight === "700" && dialogue.style.color === `color-mix(in srgb, ${shotColor} 65%, white)`, name + ": bold dialogue brightens its own shot color in both modes");
            const colors = spans.filter(span => /\[Shot [12]\]/.test(span.textContent)).map(span => span.style.color);
            if (!showFullPrompt) {
                normalColors = colors;
                // The preview's plain view is the Shift view with one section, so it carries
                // the same framing: the description on its own section color, named by its own
                // header. The finished player's tooltip is untouched.
                if (!finished) {
                    const framed = contentBlocks(tooltip);
                    assert(framed.length === 1 && boxAccent(framed[0], 2) === "#45b7d1", name + ": the plain view frames the description section");
                    assert(leavesOf([tooltip]).some(leaf => leaf.textContent === "detailed_description:"), name + ": the section names itself");
                    assert(!leavesOf([tooltip]).some(leaf => leaf.textContent === "chunk detailed_description:"), name + ": no separate grey heading is left");
                }
            } else {
                assert(JSON.stringify(colors) === JSON.stringify(normalColors), name + ": Shift must preserve shot colors");
            }
            if (!finished) checkPromptBlocks(assert, api, tooltip, chunk, description, fullPrompt, showFullPrompt);
        if (!finished) {
            // The retention analysis rides with the views that show more than the plain one:
            // Shift shows everything, the plain view keeps to the prompt and the transcript.
            api.tooltip.show({ clientX: 0, clientY: 0 }, { help: "", chunk, timing: [], description, fullPrompt, retentionAnalysis: "Keeps the framing.", showFullPrompt, shotRanges, colors: ["#f4b942", "#45b7d1"] });
            const retention = leavesOf([document.body.children[0]]).some(leaf => leaf.textContent === "Keeps the framing.");
            assert(retention === showFullPrompt, name + ": the retention analysis belongs to the Shift view, not the plain one");
            // A description with no field of its own (a legacy chunk) still shows its body.
            api.tooltip.show({ clientX: 0, clientY: 0 }, { help: "", chunk, timing: [], description: "legacy body", fullPrompt: "", showFullPrompt: false, shotRanges, colors: ["#f4b942", "#45b7d1"] });
            assert(contentBlocks(document.body.children[0]).some(box => leavesOf([box]).map(leaf => leaf.textContent).join("") === "legacy body"), name + ": a description without that field still shows its body");
        }
        }
        if (!finished) {
            checkAttemptView(assert, api, document, dispatchWindowEvent, chunk, description, shotRanges, source);
        }
    }
}


function elementTree(root) {
    // Every element the tooltip holds, however deeply the chunk card nests its content.
    const nodes = [];
    const walk = node => {
        for (const child of node.children || []) {
            nodes.push(child);
            walk(child);
        }
    };
    walk(root);
    return nodes;
}


function leavesOf(nodes) {
    // The leaves of the given elements: the tooltip's text lines and the spans its
    // blocks are built from.
    const leaves = [];
    const walk = node => {
        if (!node.children?.length) {
            leaves.push(node);
            return;
        }
        for (const child of node.children) walk(child);
    };
    for (const node of nodes) walk(node);
    return leaves;
}


function promptBlocks(tooltip) {
    // Every framed block: the chunk card, its section blocks and the attempt blocks.
    return elementTree(tooltip).filter(node => String(node.style?.cssText || "").includes("background:"));
}


function contentBlocks(tooltip) {
    // The blocks that hold prompt text: a section or attempt block, never the card
    // that contains them.
    return promptBlocks(tooltip).filter(box => !(box.children || []).some(child => String(child.style?.cssText || "").includes("background:")));
}


function cardOf(tooltip, line) {
    // The chunk card that owns a header line, found by the line it starts with.
    return promptBlocks(tooltip).find(box => leavesOf([box]).some(leaf => leaf.textContent === line));
}


function collectPromptSpans(tooltip, finished=false) {
    // Every span a prompt holds: the finished player's tooltip is flat, while the
    // preview nests its prompt sections inside the chunk card.
    if (finished) return leavesOf(tooltip.children.flatMap(child => child.children));
    return leavesOf(contentBlocks(tooltip));
}


function boxAccent(box, width) {
    return String(box.style.cssText).match(new RegExp(`border-left:${width}px solid (#[0-9a-f]{6})`))[1];
}


function checkPromptBlocks(assert, api, tooltip, chunk, description, fullPrompt, showFullPrompt) {
    // The chunk card starts at Chunk N and holds everything the chunk owns, on a very
    // dark shade of the chunk color; each prompt section sits on a darker shade of its
    // own distinct color.
    const chunkColor = "#f4b942";
    const card = cardOf(tooltip, "Chunk 1");
    assert(Boolean(card), "prompt blocks: the chunk card starts at the chunk number");
    assert(
        card.style.cssText.includes(`background:color-mix(in srgb, ${chunkColor} 7%, black)`) &&
        card.style.cssText.includes(`border-left:3px solid ${chunkColor}`),
        "prompt blocks: the card is a very dark shade of the chunk color"
    );
    const inside = leavesOf([card]).map(leaf => leaf.textContent);
    assert(!inside.includes("help"), "prompt blocks: the help line stays outside the card");
    assert(inside.includes("Timing: waiting for this chunk to finish."), "prompt blocks: the timing lines are inside the card");
    if (showFullPrompt) {
        // Two prompts keep their names, one carries only its own section headers.
        const labelled = inside.includes("Chunk prompt:") || inside.includes("Audio prompt:") || inside.includes("Video prompt:");
        assert(!labelled, "prompt blocks: a single prompt in the Shift view needs no label line");
    }

    const sections = contentBlocks(tooltip);
    const expected = showFullPrompt ? ["summary:", "detailed_description:", "overall_soundscape:"] : ["detailed_description:"];
    const labels = api.promptSections(fullPrompt).map(section => section.label).filter(label => showFullPrompt || label === "detailed_description:");
    assert(JSON.stringify(labels) === JSON.stringify(expected), "prompt blocks: the sections are the prompt's own labels");
    assert(sections.length === labels.length, "prompt blocks: one block per section");
    const accents = sections.map(section => boxAccent(section, 2));
    for (const [index, section] of sections.entries()) {
        assert(
            String(section.style.cssText).includes(`background:color-mix(in srgb, ${accents[index]} 14%, black)`),
            "prompt blocks: a section sits on a darker shade of its own color"
        );
    }
    if (showFullPrompt) {
        assert(
            JSON.stringify(accents) === JSON.stringify(["#f4b942", "#45b7d1", "#72c472"]),
            "prompt blocks: each section keeps its own distinct color"
        );
        assert(new Set(accents).size === accents.length, "prompt blocks: no two sections share a color");
    } else {
        assert(JSON.stringify(accents) === JSON.stringify(["#45b7d1"]), "prompt blocks: the plain view uses the description section's own color");
    }
    // The section's own header is emphasised where it stands, never repeated.
    const headers = collectPromptSpans(tooltip).filter(span => /^[a-z_]+:$/.test(span.textContent));
    assert(
        JSON.stringify(headers.map(span => span.textContent)) === JSON.stringify(showFullPrompt ? expected : ["detailed_description:"]),
        "prompt blocks: each section header stays in place"
    );
    for (const header of headers) {
        const section = sections.find(box => leavesOf([box]).includes(header));
        assert(
            String(header.style.cssText).includes("font-weight:700") && String(header.style.cssText).includes(`color:color-mix(in srgb, ${boxAccent(section, 2)} 55%, white)`),
            "prompt blocks: a header is bold in its own section's color"
        );
    }
    const text = showFullPrompt ? fullPrompt : `detailed_description: ${description}\n`;
    assert(
        api.promptSections(text).map(section => text.slice(section.start, section.end)).join("") === text,
        "prompt blocks: the sections cover the prompt exactly"
    );
}


function checkAttemptView(assert, api, document, dispatchWindowEvent, chunk, description, shotRanges, source) {
    // Ctrl names every teacher-audio attempt: each prompt cut to its detailed_description, and
    // below it what that take heard. The take a later attempt follows caused that retry, so its
    // hearing is red; an accepted take's own hearing keeps the attempt's tint.
    const firstPrompt = `summary: Timed audio.\ndetailed_description: ${description}\noverall_soundscape: Wind.`;
    const retryPrompt = `summary: Timed audio.\ndetailed_description: [Shot 1] She speaks much faster.\noverall_soundscape: Wind.`;
    const range = {
        ...chunk,
        h3_prompt: retryPrompt,
        audio_teacher_prompt: retryPrompt,
        audio_teacher_attempts: [firstPrompt, retryPrompt],
        audio_teacher_attempt_transcripts: ["[Shot 1] She speak quickly.", "she speaks much faster"],
    };
    assert(JSON.stringify(api.audioTeacherAttempts(range)) === JSON.stringify([firstPrompt, retryPrompt]), "ctrl view: every stored attempt is used");
    assert(JSON.stringify(api.audioTeacherAttempts({ ...chunk, audio_teacher_prompt: retryPrompt })) === JSON.stringify([retryPrompt]), "ctrl view: a chunk without a retry has one attempt");
    assert(JSON.stringify(api.audioTeacherAttempts({ ...chunk, audio_teacher_attempts: ["", "  "] })) === JSON.stringify([]), "ctrl view: blank attempts are dropped");
    assert(JSON.stringify(api.audioTeacherAttemptTranscripts(range)) === JSON.stringify(["[Shot 1] She speak quickly.", "she speaks much faster"]), "ctrl view: each attempt's hearing is read in its own order");
    assert(JSON.stringify(api.audioTeacherAttemptTranscripts({ ...chunk, audio_teacher_attempt_transcripts: [] })) === JSON.stringify([]), "ctrl view: a chunk without hearings has none");
    assert(api.chunkPromptDescription({ h3_prompt: "no section here" }, false) === "", "ctrl view: the section is required, never Gemma's fallback");
    api.tooltip.show({ clientX: 0, clientY: 0 }, {
        help: "", chunk, timing: [], fullPrompt: retryPrompt, teacherAudioPrompt: retryPrompt,
        attempts: range.audio_teacher_attempts, attemptTranscripts: range.audio_teacher_attempt_transcripts,
        showAttempts: true, shotRanges, colors: ["#f4b942", "#45b7d1"],
    });
    const tooltip = document.body.children[0];
    const blocks = contentBlocks(tooltip);
    // The attempt blocks carry the tint itself; the hearing block is its darker shade.
    const attemptBlocks = blocks.filter(box => !String(box.style.cssText).includes("color-mix"));
    const hearingBlocks = blocks.filter(box => String(box.style.cssText).includes("color-mix"));
    assert(attemptBlocks.length === 2, "ctrl view: one prompt block per attempt");
    assert(
        attemptBlocks.map(block => block.children.map(span => span.textContent).join("")).join("|") === `${description}|[Shot 1] She speaks much faster.`,
        "ctrl view: each block holds only the detailed_description section"
    );
    assert(attemptBlocks[0].style.cssText !== attemptBlocks[1].style.cssText, "ctrl view: the first take and the retry carry different tints");
    const card = cardOf(tooltip, "Chunk 1");
    assert(Boolean(card), "ctrl view: the attempt blocks sit inside the chunk card");
    assert(
        attemptBlocks.every(block => leavesOf([card]).length && elementTree(card).includes(block)),
        "ctrl view: the card holds the attempts, not the other way round"
    );
    assert(!leavesOf([tooltip]).some(leaf => leaf.textContent === "chunk detailed_description:"), "ctrl view: the single description is replaced, not repeated");
    const labels = leavesOf([tooltip]).filter(leaf => /^(Teacher audio prompt|Retry [0-9]+):$/.test(leaf.textContent)).map(leaf => leaf.textContent);
    assert(JSON.stringify(labels) === JSON.stringify(["Teacher audio prompt:", "Retry 1:"]), "ctrl view: the first take is called the teacher audio prompt");
    assert(leavesOf([tooltip]).filter(leaf => leaf.textContent === "Teacher audio heard:").length === 2, "ctrl view: every attempt names its own hearing");
    // Each hearing sits under its own attempt, and the one that caused the retry is red.
    assert(hearingBlocks.length === 2, "ctrl view: one hearing per attempt");
    assert(
        hearingBlocks.map(block => block.textContent).join("|") === "[Shot 1] She speak quickly.|she speaks much faster",
        "ctrl view: the hearing below an attempt is that take's own"
    );
    assert(
        hearingBlocks[0].style.cssText.includes("border-left:3px solid #e05252") && hearingBlocks[0].style.cssText.includes("color-mix(in srgb, #e05252 26%, black)"),
        "ctrl view: the hearing that caused the retry is red"
    );
    const retryAccent = boxAccent(attemptBlocks[1], 3);
    assert(
        hearingBlocks[1].style.cssText.includes(`background:color-mix(in srgb, ${retryAccent} 32%, black)`) &&
        hearingBlocks[1].style.cssText.includes(`border-left:3px solid ${retryAccent}`),
        "ctrl view: an accepted take's hearing keeps its attempt's tint"
    );
    assert(!hearingBlocks[0].style.cssText.includes("#45b7d1"), "ctrl view: the red hearing never borrows an attempt tint");
    assert(!leavesOf([tooltip]).some(leaf => leaf.textContent === "Heard instead (caused this retry):"), "ctrl view: the old block under the retry is gone");
    // A prompt with no such section still shows its real text rather than nothing.
    api.tooltip.show({ clientX: 0, clientY: 0 }, {
        help: "", chunk, timing: [], attempts: ["summary: only a summary."], showAttempts: true, shotRanges, colors: ["#f4b942", "#45b7d1"],
    });
    const bare = contentBlocks(document.body.children[0]);
    assert(bare.length === 1 && bare[0].children.map(span => span.textContent).join("") === "summary: only a summary.", "ctrl view: a prompt without the section keeps its own text");
    // A take that heard nothing says so where its own hearing belongs, so the list never shifts.
    api.tooltip.show({ clientX: 0, clientY: 0 }, {
        help: "", chunk, timing: [], attempts: [firstPrompt, retryPrompt], attemptTranscripts: ["", "only the retry spoke"],
        showAttempts: true, shotRanges, colors: ["#f4b942", "#45b7d1"],
    });
    const quiet = document.body.children[0];
    assert(leavesOf([quiet]).some(leaf => leaf.textContent === "nothing heard in this window"), "ctrl view: a silent take says nothing was heard");
    assert(contentBlocks(quiet).some(box => box.textContent === "only the retry spoke"), "ctrl view: the retry's own hearing still shows");
    assert(source.includes("data.audio_teacher_attempt_transcripts"), "ctrl view: a live hearing reaches the range");
    // Ctrl+Shift names the sub-chunk the pointer sits in, so it needs the sub-chunk of a
    // position and the prompt that sub-chunk is sampled with.
    checkSubchunkView(assert, api, document, chunk, shotRanges, source);
    // A sub-chunk only owns its own frames, so they are merged where they belong instead
    // of replacing the sub-chunks that already rendered.
    checkFrameMerge(assert, api, source);
    // A prompt taller than the tooltip is read with the wheel or the scrollbar.
    checkTooltipScroll(assert, api, document, dispatchWindowEvent, source, chunk, shotRanges);
    // Ctrl owns the prompt area, so it wins over Shift.
    assert(source.includes("const showAttempts = Boolean(event.ctrlKey || ctrlPromptVisible);"), "ctrl view: Ctrl owns the teacher-audio attempts");
    assert(source.includes("const showFullPrompt = Boolean(!showAttempts && (event.shiftKey || shiftPromptVisible));"), "ctrl view: Shift shows the full prompt when Ctrl is not held");
    assert(source.includes("data.audio_teacher_attempts"), "ctrl view: a live retry reaches the range");
}


function checkSubchunkView(assert, api, document, chunk, shotRanges, source) {
    // A chunk with sub-chunks reads sub-chunk by sub-chunk: every view names the one under
    // the pointer, and the prompt it shows is that sub-chunk's own.
    const phased = { taomate_phase_work: [10, 20, 30, 40] };
    assert(JSON.stringify(api.subchunkAt(phased, 0)) === JSON.stringify({ number: 1, count: 4 }), "sub-chunk view: the start is the first sub-chunk");
    assert(JSON.stringify(api.subchunkAt(phased, 0.09)) === JSON.stringify({ number: 1, count: 4 }), "sub-chunk view: a phase keeps its own frames");
    assert(JSON.stringify(api.subchunkAt(phased, 0.1)) === JSON.stringify({ number: 2, count: 4 }), "sub-chunk view: the second phase starts where the first ends");
    assert(JSON.stringify(api.subchunkAt(phased, 0.29)) === JSON.stringify({ number: 2, count: 4 }), "sub-chunk view: a longer phase owns more of the span");
    assert(JSON.stringify(api.subchunkAt(phased, 0.95)) === JSON.stringify({ number: 4, count: 4 }), "sub-chunk view: the end is the last sub-chunk");
    assert(JSON.stringify(api.subchunkAt(phased, 1)) === JSON.stringify({ number: 4, count: 4 }), "sub-chunk view: the very end is still the last sub-chunk");
    assert(JSON.stringify(api.subchunkAt({ taomate_phase_count: 3 }, 0.5)) === JSON.stringify({ number: 2, count: 3 }), "sub-chunk view: phases without frame counts are split evenly");
    assert(JSON.stringify(api.subchunkAt({}, 0.5)) === JSON.stringify({ number: 1, count: 1 }), "sub-chunk view: a chunk without phases has one sub-chunk");
    const silentRange = { h3_prompt: "chunk text", subchunk_silent_numbers: [3], subchunk_silent_prompt: "no dialogue here" };
    assert(JSON.stringify(api.subchunkPrompt(silentRange, 3)) === JSON.stringify({ prompt: "no dialogue here", silent: true }), "sub-chunk view: a silent sub-chunk drops the dialogue");
    assert(JSON.stringify(api.subchunkPrompt(silentRange, 2)) === JSON.stringify({ prompt: "chunk text", silent: false }), "sub-chunk view: every other sub-chunk keeps the chunk prompt");
    assert(JSON.stringify(api.subchunkPrompt({ h3_prompt: "chunk text" }, 1)) === JSON.stringify({ prompt: "chunk text", silent: false }), "sub-chunk view: a chunk with no silent sub-chunk is one prompt");

    // The plain view of the third sub-chunk: its own detailed_description, its name and what
    // its window of the take heard.
    const silentPrompt = "summary: Continue.\ndetailed_description: [Shot 1] She stays silent.";
    const heardRange = { ...chunk, subchunk_transcripts: ["", "heard one", "heard three"] };
    api.tooltip.show({ clientX: 0, clientY: 0 }, {
        help: "", chunk: heardRange, timing: [], description: "[Shot 1] She stays silent.",
        subchunk: { number: 3, count: 4 }, subchunkBody: { prompt: silentPrompt, silent: true },
        subchunkHeard: "heard three", shotRanges, colors: ["#f4b942", "#45b7d1"],
    });
    const tooltip = document.body.children[0];
    const heading = leavesOf([tooltip]).find(leaf => /^Chunk 1\.3/.test(leaf.textContent));
    assert(Boolean(heading), "sub-chunk view: the pointing sub-chunk is named on the chunk title");
    assert(heading.textContent === "Chunk 1.3 / 4", "sub-chunk view: the title carries the sub-chunk number and count");
    assert(contentBlocks(tooltip).some(box => leavesOf([box]).map(leaf => leaf.textContent).join("") === "[Shot 1] She stays silent."), "sub-chunk view: the plain view shows that sub-chunk's description");
    const heardBlock = contentBlocks(tooltip).find(box => box.textContent === "heard three");
    assert(Boolean(heardBlock) && String(heardBlock.style.cssText).includes("border-left:3px solid #8d8d8d"), "sub-chunk view: the transcript sits in its own plain block");
    assert(leavesOf([tooltip]).some(leaf => leaf.textContent === "Teacher audio transcript (this sub-chunk's window):"), "sub-chunk view: the transcript line names the window it was cut to");
    // The Shift view keeps the same name and transcript, with the whole prompt of that sub-chunk.
    api.tooltip.show({ clientX: 0, clientY: 0 }, {
        help: "", chunk: heardRange, timing: [], fullPrompt: silentPrompt, showFullPrompt: true,
        subchunk: { number: 3, count: 4 }, subchunkBody: { prompt: silentPrompt, silent: true },
        subchunkHeard: "heard three", shotRanges, colors: ["#f4b942", "#45b7d1"],
    });
    const shifted = document.body.children[0];
    assert(leavesOf([shifted]).some(leaf => leaf.textContent === "Chunk 1.3 / 4"), "sub-chunk view: the Shift view names the same sub-chunk");
    // The Shift view frames the prompt section by section, so its blocks add up to the whole text.
    const shiftedPromptText = contentBlocks(shifted)
        .map(box => leavesOf([box]).map(leaf => leaf.textContent).join(""))
        .filter(text => text !== "heard three")
        .join("");
    assert(shiftedPromptText === silentPrompt, "sub-chunk view: the Shift view shows that sub-chunk's whole prompt");
    assert(leavesOf([shifted]).some(leaf => leaf.textContent === "heard three" || leaf.textContent === "Teacher audio transcript (this sub-chunk's window):"), "sub-chunk view: the Shift view keeps the transcript");
    // A window the take said nothing in says so, and a chunk with no transcript at all shows
    // no transcript line.
    api.tooltip.show({ clientX: 0, clientY: 0 }, {
        help: "", chunk, timing: [], description: "x", subchunk: { number: 1, count: 4 },
        subchunkBody: { prompt: "summary: Nothing.", silent: false }, subchunkHeard: "", shotRanges, colors: ["#f4b942", "#45b7d1"],
    });
    assert(leavesOf([document.body.children[0]]).some(leaf => leaf.textContent === "nothing heard in this window"), "sub-chunk view: a window the take said nothing in says so");
    api.tooltip.show({ clientX: 0, clientY: 0 }, {
        help: "", chunk, timing: [], description: "x", subchunk: { number: 1, count: 4 },
        subchunkBody: { prompt: "summary: Nothing.", silent: false }, subchunkHeard: null, shotRanges, colors: ["#f4b942", "#45b7d1"],
    });
    assert(!leavesOf([document.body.children[0]]).some(leaf => leaf.textContent === "Teacher audio transcript (this sub-chunk's window):"), "sub-chunk view: a chunk without transcripts shows no transcript line");
    // CTRL+SHIFT clears the prompt text, and the chunk card keeps its own numbers.
    api.tooltip.show({ clientX: 0, clientY: 0 }, {
        help: "", chunk, timing: ["Chunk processing: 00:12"], description: "x", showNothing: true,
        subchunk: { number: 2, count: 4 }, subchunkBody: { prompt: "summary: Nothing.", silent: false },
        subchunkHeard: "heard two", shotRanges, colors: ["#f4b942", "#45b7d1"],
    });
    const hidden = document.body.children[0];
    const hiddenText = contentBlocks(hidden).map(box => leavesOf([box]).map(leaf => leaf.textContent).join("")).join("|");
    assert(!hiddenText.includes("summary: Nothing."), "sub-chunk view: the cleared view shows no prompt blocks");
    assert(leavesOf([hidden]).some(leaf => leaf.textContent === "Prompt text hidden."), "sub-chunk view: the cleared view says the text is hidden");
    assert(leavesOf([hidden]).some(leaf => leaf.textContent === "Chunk processing: 00:12"), "sub-chunk view: the cleared view keeps the chunk's numbers");
    assert(leavesOf([hidden]).some(leaf => leaf.textContent === "heard two"), "sub-chunk view: the cleared view keeps the transcript, which is not prompt text");
    // A chunk without sub-chunks reads exactly as it always did, with no sub-chunk line.
    api.tooltip.show({ clientX: 0, clientY: 0 }, {
        help: "", chunk, timing: [], description: "whole chunk body", subchunk: { number: 1, count: 1 },
        subchunkBody: { prompt: "summary: Whole.", silent: false }, subchunkHeard: null, shotRanges, colors: ["#f4b942", "#45b7d1"],
    });
    assert(leavesOf([document.body.children[0]]).some(leaf => leaf.textContent === "Chunk 1"), "sub-chunk view: a chunk without sub-chunks keeps its plain title");
    // The number rides on the title of every view: the Ctrl attempts and the cleared one too.
    api.tooltip.show({ clientX: 0, clientY: 0 }, {
        help: "", chunk, timing: [], showAttempts: true, attempts: ["take one"], showFullPrompt: false,
        subchunk: { number: 2, count: 4 }, subchunkBody: { prompt: "summary: Nothing.", silent: false }, subchunkHeard: null, shotRanges, colors: ["#f4b942", "#45b7d1"],
    });
    assert(leavesOf([document.body.children[0]]).some(leaf => leaf.textContent === "Chunk 1.2 / 4"), "sub-chunk view: the Ctrl attempts view names the sub-chunk too");
    assert(leavesOf([document.body.children[0]]).some(leaf => leaf.textContent === "Teacher-audio prompt:"), "sub-chunk view: the Ctrl attempts view still shows the takes");
    // The wiring: no chord is left, the sub-chunk owns the prompt text, and Ctrl+Shift clears it.
    assert(!source.includes("showSubchunk"), "sub-chunk view: the Ctrl+Shift chord is gone");
    assert(source.includes("const showNothing = Boolean((event.ctrlKey || ctrlPromptVisible) && (event.shiftKey || shiftPromptVisible));"), "sub-chunk view: Ctrl+Shift clears the prompt text");
    assert(source.includes("const fullPrompt = String(subchunkBody?.prompt || range.h3_prompt || \"\").trim();"), "sub-chunk view: the shown prompt is the pointed sub-chunk's");
    assert(source.includes("chunkPromptDescription({ h3_prompt: fullPrompt, gemma_detailed_description: range.gemma_detailed_description })"), "sub-chunk view: its description keeps the Gemma fallback");
    assert(source.includes("data.subchunk_transcripts"), "sub-chunk view: a live transcript reaches the range");
    assert(source.includes("subchunkBody?.silent, subchunkHeard, timing"), "sub-chunk view: a transcript landing refreshes an open tooltip");
}


function checkFrameMerge(assert, api, source) {
    // Dragging through a chunk that is still filling must show every sub-chunk that
    // rendered where it rendered, so a sub-chunk's frames are merged by their own numbers.
    const first = { frames: ["a", "b"], durations: [1, 2], frameNumbers: [0, 1] };
    const second = api.mergeFrameGroup(first, ["c", "d"], [3, 4], [2, 3]);
    assert(JSON.stringify(second.frameNumbers) === JSON.stringify([0, 1, 2, 3]), "frame merge: the chunk keeps every sub-chunk in frame order");
    assert(JSON.stringify(second.frames) === JSON.stringify(["a", "b", "c", "d"]), "frame merge: the earlier sub-chunks keep their frames");
    assert(JSON.stringify(second.durations) === JSON.stringify([1, 2, 3, 4]), "frame merge: durations stay parallel to the frames");
    // A sub-chunk re-renders its replacement overlap, so the new version wins there.
    const overlapped = api.mergeFrameGroup({ frames: ["a", "b"], durations: [1, 2], frameNumbers: [17, 18] }, ["x", "y"], [3, 4], [18, 19]);
    assert(JSON.stringify(overlapped.frames) === JSON.stringify(["a", "x", "y"]), "frame merge: a re-rendered frame takes the new version");
    assert(JSON.stringify(overlapped.frameNumbers) === JSON.stringify([17, 18, 19]), "frame merge: the numbers stay sorted");
    // Numbers arrive in the order the server lists them, and a hole between two
    // sub-chunks stays a hole.
    const hole = api.mergeFrameGroup({ frames: ["a"], durations: [1], frameNumbers: [0] }, ["c"], [3], [5]);
    assert(JSON.stringify(hole.frameNumbers) === JSON.stringify([0, 5]), "frame merge: an unrendered position stays unrendered");
    const unordered = api.mergeFrameGroup(null, ["c", "b"], [3, 2], [5, 4]);
    assert(JSON.stringify(unordered.frameNumbers) === JSON.stringify([4, 5]), "frame merge: an unordered event is sorted into place");
    assert(JSON.stringify(unordered.frames) === JSON.stringify(["b", "c"]), "frame merge: frames follow their new order");
    // The first event of a chunk, and an event without numbers, both become the group.
    const fresh = api.mergeFrameGroup(undefined, ["a"], [1], [0]);
    assert(JSON.stringify(fresh.frames) === JSON.stringify(["a"]) && JSON.stringify(fresh.frameNumbers) === JSON.stringify([0]), "frame merge: the first sub-chunk of a chunk is the group");
    const unnumbered = api.mergeFrameGroup({ frames: ["a"], durations: [1], frameNumbers: [0] }, ["b"], [2], []);
    assert(JSON.stringify(unnumbered.frames) === JSON.stringify(["b"]) && JSON.stringify(unnumbered.frameNumbers) === JSON.stringify([]), "frame merge: a payload without numbers cannot be placed, so it is the group");
    // A placeholder is span coverage with a black picture, not frames: keeping its black
    // images would put empty steps between the frames the arrow keys walk.
    assert(source.includes("? { frames: incomingFrames, durations: frameDurations, frameNumbers: incomingNumbers }"), "frame merge: the final decoded video still replaces the group");
    assert(source.includes("mergeFrameGroup(oldGroup?.audioOnlyPlaceholder ? null : oldGroup, incomingFrames, frameDurations, incomingNumbers)"), "frame merge: the audio placeholder is replaced, not merged into");
    assert(source.includes("Math.max(Number(data.output_end) || 0, merged.frameNumbers[merged.frameNumbers.length - 1])"), "frame merge: the rendered end is the phase that finished");
    assert(source.includes("? { frames: incomingFrames, durations: frameDurations, frameNumbers: incomingNumbers }"), "frame merge: the final decoded video still replaces the group");
}


function checkTooltipScroll(assert, api, document, dispatchWindowEvent, source, chunk, shotRanges) {
    // A prompt taller than the tooltip is scrolled with the wheel or its scrollbar, so the
    // tooltip takes the mouse and must stay open while the pointer is on it.
    const tooltip = document.body.children[0];
    const style = String(tooltip.style.cssText);
    assert(style.includes("overflow-y:auto") && style.includes("overflow-x:hidden"), "tooltip scroll: the tall content scrolls inside the tooltip");
    assert(style.includes("pointer-events:auto") && style.includes("user-select:text"), "tooltip scroll: the tooltip takes the mouse so its scrollbar and text are usable");
    assert(style.includes("scrollbar-width:thin"), "tooltip scroll: the scrollbar is visible in the tooltip's own theme");
    api.tooltip.show({ clientX: 0, clientY: 0 }, { help: "", chunk, timing: [], description: "x", shotRanges, colors: ["#f4b942"] });
    assert(api.tooltip.isHovered() === false, "tooltip scroll: a shown tooltip is not hovered yet");
    tooltip.dispatchEvent("mouseenter");
    assert(api.tooltip.isHovered() === true, "tooltip scroll: the pointer on the tooltip is known");
    // Leaving for the timeline hands the hover back instead of closing the tooltip, so the
    // pointer can travel to the scrollbar and back.
    api.tooltip.setLeaveGuard(() => true);
    tooltip.dispatchEvent("mouseleave", { relatedTarget: {} });
    assert(api.tooltip.isHovered() === false && tooltip.style.display === "block", "tooltip scroll: leaving for the timeline keeps it open");
    // A rebuild of the content keeps the reader where they scrolled to, and a tooltip that
    // opens again starts at its own top.
    tooltip.scrollTop = 300;
    api.tooltip.show({ clientX: 0, clientY: 0 }, { help: "", chunk, timing: [], description: "y", shotRanges, colors: ["#f4b942"] });
    assert(tooltip.scrollTop === 300, "tooltip scroll: new content for the same hover keeps the reader's place");
    api.tooltip.setLeaveGuard(() => false);
    tooltip.dispatchEvent("mouseleave", { relatedTarget: null });
    assert(tooltip.style.display === "none" && tooltip.scrollTop === 0, "tooltip scroll: leaving for anything else closes it at the top");
    assert(source.includes("chunkTooltip.setLeaveGuard(event => Boolean(event.relatedTarget && timelineShell.contains(event.relatedTarget)))"), "tooltip scroll: the timeline is the one target that keeps it open");
    assert(source.includes("if (chunkTooltip.isHovered()) return;"), "tooltip scroll: the timeline does not close a tooltip the pointer is on");
    assert(source.includes('tooltip.addEventListener("pointerdown", event => event.stopPropagation())'), "tooltip scroll: a press in the tooltip never starts a canvas drag");
    // The wheel is taken on the window's capture phase, ahead of the frontend's own
    // document.body Ctrl+wheel handler, so Ctrl+wheel scrolls the prompt instead of
    // switching tabs or zooming the page.
    assert(source.includes('window.addEventListener("wheel", onWheel, { capture: true, passive: false })'), "tooltip scroll: the wheel is taken before the frontend's body handler");
    const wheelEvent = () => {
        const event = { deltaY: 0, deltaMode: 0, prevented: false, stopped: false };
        event.preventDefault = () => { event.prevented = true; };
        event.stopImmediatePropagation = () => { event.stopped = true; };
        return event;
    };
    const tooltipElement = document.body.children[0];
    api.tooltip.show({ clientX: 0, clientY: 0 }, { help: "", chunk, timing: [], description: "z", shotRanges, colors: ["#f4b942"] });
    // A tooltip whose text fits leaves the wheel to the graph.
    tooltipElement.scrollHeight = 600;
    tooltipElement.clientHeight = 600;
    const fits = wheelEvent();
    assert(dispatchWindowEvent("wheel", fits) === 1 && !fits.prevented, "tooltip scroll: a tooltip that fits lets the wheel through");
    // The pointer stays on the timeline, which is what keeps the tooltip showing, so the
    // wheel over a longer prompt is the tooltip's whatever the pointer is on.
    tooltipElement.scrollHeight = 900;
    tooltipElement.scrollTop = 0;
    const down = wheelEvent();
    down.deltaY = 120;
    dispatchWindowEvent("wheel", down);
    assert(down.prevented && down.stopped, "tooltip scroll: the wheel over a long prompt is owned by the tooltip");
    assert(tooltipElement.scrollTop === 120, "tooltip scroll: the wheel moves the text");
    const lines = wheelEvent();
    lines.deltaY = 3;
    lines.deltaMode = 1;
    dispatchWindowEvent("wheel", lines);
    assert(tooltipElement.scrollTop === 168, "tooltip scroll: a line-mode wheel is converted to pixels");
    const past = wheelEvent();
    past.deltaY = 9999;
    dispatchWindowEvent("wheel", past);
    assert(tooltipElement.scrollTop === 300, "tooltip scroll: scrolling stops at the end of the text");
    // A tooltip that is not showing leaves the wheel alone.
    tooltipElement.dispatchEvent("mouseleave", { relatedTarget: {} });
    const away = wheelEvent();
    away.deltaY = 120;
    dispatchWindowEvent("wheel", away);
    assert(!away.prevented && tooltipElement.scrollTop === 0, "tooltip scroll: a wheel at a closed tooltip is left alone");
    api.tooltip.remove();
    assert(dispatchWindowEvent("wheel", wheelEvent()) === 0, "tooltip scroll: a removed tooltip detaches its wheel listener");
}


if (typeof require !== "undefined" && require.main === module) {
    const fs = require("node:fs");
    const path = require("node:path");
    const sources = {};
    for (const name of ["unlimited_preview.js", "finished_video_player.js"]) sources[name] = fs.readFileSync(path.join(__dirname, "../web", name), "utf8");
    checkPreviewPrompts(sources);
    console.log("Preview prompt checks passed.");
}
