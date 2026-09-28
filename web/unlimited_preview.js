const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;


function createControlsAutohide(elements, { delayMs = 3000, isFullscreen, isPlaying, doc = document } = {}) {
    // Fullscreen playback hides the chrome the way a video player does: after the pointer
    // has been quiet for the delay the transport bar, the status line and the viewport
    // buttons fade away, and any movement brings them back at once. A paused player keeps
    // them, because there is no picture to watch behind them.
    let timer = null;

    function setVisible(visible) {
        for (const element of elements) {
            element.style.opacity = visible ? "1" : "0";
            // A faded control must not answer a click either.
            element.style.pointerEvents = visible ? "" : "none";
        }
    }

    function shouldHide() {
        return Boolean(isFullscreen() && isPlaying());
    }

    function arm() {
        if (timer != null) clearTimeout(timer);
        timer = null;
        setVisible(true);
        if (!shouldHide()) return;
        timer = setTimeout(() => {
            timer = null;
            if (shouldHide()) setVisible(false);
        }, delayMs);
    }

    const onMove = () => arm();
    const onFullscreen = () => arm();
    doc.addEventListener("mousemove", onMove);
    doc.addEventListener("fullscreenchange", onFullscreen);
    for (const element of elements) element.style.transition = "opacity .3s ease";

    return {
        arm,
        isVisible: () => elements.every(element => element.style.opacity !== "0"),
        stop() {
            if (timer != null) clearTimeout(timer);
            timer = null;
            doc.removeEventListener("mousemove", onMove);
            doc.removeEventListener("fullscreenchange", onFullscreen);
            setVisible(true);
        },
    };
}


function chunkPromptDescription(chunk, fallback=true) {
    // The actual H3 prompt is authoritative, including runs without a director.
    const prompt = String(chunk.h3_prompt || "");
    const field = prompt.match(/(?:^|\n)[ \t]*(?:detailed_description|integrated_multimodal_description)[ \t]*:[ \t]*([\s\S]*?)(?=\n[ \t]*(?:overall_soundscape|non_diegetic_music|retention_analysis|summary)[ \t]*:|$)/i);
    if (field) return field[1].trim();
    // The teacher-audio attempt view asks for that section alone, so it never
    // falls back to Gemma's description: two attempts would read identically.
    return fallback ? String(chunk.gemma_detailed_description || "").trim() : "";
}


function audioTeacherAttemptTranscripts(range) {
    // What each take heard, in the order the takes were sampled, so entry N belongs to the
    // attempt the prompt list names at N. Empty slots are kept: a take is identified by its
    // position, and the take a later attempt follows is the one that caused that retry.
    const listed = Array.isArray(range.audio_teacher_attempt_transcripts) ? range.audio_teacher_attempt_transcripts : [];
    return listed.map(text => String(text || "").trim());
}


function subchunkAt(range, fraction) {
    // Which sub-chunk of a chunk a position falls in. Every phase owns its own frames, in
    // order, so the pointer's share of the chunk's frames names it; a chunk without frame
    // counts per phase is split evenly.
    const work = Array.isArray(range?.taomate_phase_work) ? range.taomate_phase_work.map(value => Math.max(0, Number(value) || 0)) : [];
    const counts = work.length ? work : new Array(Math.max(1, Number(range?.taomate_phase_count) || 1)).fill(1);
    const total = counts.reduce((sum, value) => sum + value, 0);
    if (!total) return { number: 1, count: 1 };
    let cursor = Math.max(0, Math.min(0.999999, Number(fraction) || 0)) * total;
    for (const [index, value] of counts.entries()) {
        if (cursor < value) return { number: index + 1, count: counts.length };
        cursor -= value;
    }
    return { number: counts.length, count: counts.length };
}


function subchunkPrompt(range, number) {
    // A sub-chunk whose own audio window holds no speech is prompted without the
    // dialogue, so H3 is not told to speak where the take was silent. Every other
    // sub-chunk is prompted with the chunk's own text.
    const silent = Array.isArray(range?.subchunk_silent_numbers) ? range.subchunk_silent_numbers.map(value => Number(value)) : [];
    if (silent.includes(Number(number))) {
        return { prompt: String(range.subchunk_silent_prompt || "").trim(), silent: true };
    }
    return { prompt: String(range?.h3_prompt || "").trim(), silent: false };
}


function mergeFrameGroup(oldGroup, frames, durations, numbers) {
    // A TaoMate sub-chunk reports only its own frames, numbered where they really sit in
    // the chunk, so they are merged into the chunk's group instead of replacing it: every
    // sub-chunk that already rendered keeps its own frames and nothing is smeared across
    // the chunk. A frame a later sub-chunk re-renders (its replacement overlap) takes the
    // new version, because the incoming entries are written last.
    const incoming = new Map();
    for (const [index, frame] of (frames || []).entries()) {
        const number = Number(numbers?.[index]);
        if (Number.isFinite(number)) incoming.set(number, { frame, duration: durations?.[index] });
    }
    // Without frame numbers there is no position to merge into: the event is the group.
    if (!incoming.size) return { frames: [...(frames || [])], durations: [...(durations || [])], frameNumbers: [...(numbers || [])] };
    const merged = new Map();
    const kept = Array.isArray(oldGroup?.frames) ? oldGroup.frames : [];
    for (const [index, frame] of kept.entries()) {
        const number = Number(oldGroup?.frameNumbers?.[index]);
        if (Number.isFinite(number)) merged.set(number, { frame, duration: oldGroup?.durations?.[index] });
    }
    for (const [number, entry] of incoming) merged.set(number, entry);
    const ordered = [...merged.keys()].sort((left, right) => left - right);
    return {
        frames: ordered.map(number => merged.get(number).frame),
        durations: ordered.map(number => merged.get(number).duration),
        frameNumbers: ordered,
    };
}


function audioTeacherAttempts(range) {
    // Every prompt this chunk's teacher audio was sampled with: the take that was
    // accepted as it was, or the first take and every retry that rejected one.
    const listed = Array.isArray(range.audio_teacher_attempts) ? range.audio_teacher_attempts : [];
    const attempts = listed.map(text => String(text || "").trim()).filter(Boolean);
    if (!attempts.length) {
        const single = String(range.audio_teacher_prompt || range.h3_prompt || "").trim();
        if (single) attempts.push(single);
    }
    return attempts;
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


// One tint per teacher-audio attempt, cycled: the first take and every retry must
// stay recognizable at a glance, so one attempt index always means one color.
const ATTEMPT_TINTS = [
    { accent: "#45b7d1", background: "rgba(69,183,209,.14)" },
    { accent: "#f4b942", background: "rgba(244,185,66,.14)" },
    { accent: "#b27ce8", background: "rgba(178,124,232,.14)" },
    { accent: "#78d078", background: "rgba(120,208,120,.14)" },
];


function attemptTint(index) {
    return ATTEMPT_TINTS[index % ATTEMPT_TINTS.length];
}


// One distinct color per prompt section, so a section is recognizable wherever it
// appears: summary is always the summary color, in every view and every prompt.
const PROMPT_SECTION_COLORS = {
    subject_definitions: "#f08a4b",
    summary: "#f4b942",
    retention_analysis: "#d6cf57",
    detailed_description: "#45b7d1",
    integrated_multimodal_description: "#45b7d1",
    overall_soundscape: "#72c472",
    non_diegetic_music: "#a98bea",
    non_diegetic_audio: "#ef6f91",
};
// Prose that precedes the first field label belongs to no section and stays neutral.
const PROMPT_SECTION_NEUTRAL = "#b8b8d0";
// The section the plain tooltip shows on its own, framed exactly as the Shift view frames it.
const PROMPT_DESCRIPTION_FIELDS = ["detailed_description", "integrated_multimodal_description"];


// The field a section header names, without its colon: `detailed_description`.
function promptSectionField(label) {
    return String(label || "").replace(/\s*:\s*$/, "").toLowerCase();
}


function promptSectionColor(label) {
    return PROMPT_SECTION_COLORS[promptSectionField(label)] || PROMPT_SECTION_NEUTRAL;
}


function promptSections(text) {
    // A prompt is the sections its own field labels name. The match is anchored on a
    // line start and every section keeps its header verbatim, so the text is only
    // grouped, never rewritten on its way to the tooltip.
    const source = String(text || "");
    const headers = [...source.matchAll(/^[ \t]*(?:subject_definitions|summary|retention_analysis|detailed_description|integrated_multimodal_description|overall_soundscape|non_diegetic_music|non_diegetic_audio)[ \t]*:/gim)];
    const sections = [];
    if (!headers.length) {
        if (source) sections.push({ label: "", start: 0, headerEnd: 0, end: source.length });
        return sections;
    }
    if (headers[0].index > 0) {
        sections.push({ label: "", start: 0, headerEnd: 0, end: headers[0].index });
    }
    for (const [index, header] of headers.entries()) {
        sections.push({
            label: header[0].trim(),
            start: header.index,
            headerEnd: header.index + header[0].length,
            end: index + 1 < headers.length ? headers[index + 1].index : source.length,
        });
    }
    return sections;
}


function appendColoredRange(parent, segments, offsets, start, end) {
    // Render one character range of an already color-mapped prompt, so a section can be
    // framed without deciding again which shot owns which marker.
    for (const [index, segment] of segments.entries()) {
        const from = Math.max(start, offsets[index]);
        const to = Math.min(end, offsets[index] + segment.text.length);
        if (to <= from) continue;
        const piece = segment.text.slice(from - offsets[index], to - offsets[index]);
        // H3 dialogue tags remain literal text; only their presentation changes.
        for (const [part, text] of piece.split(/(<d>[\s\S]*?<\/d>)/gi).entries()) {
            const span = document.createElement("span");
            span.textContent = text;
            span.style.color = segment.color;
            if (part % 2) {
                span.style.color = `color-mix(in srgb, ${segment.color} 65%, white)`;
                span.style.fontWeight = "700";
            }
            parent.appendChild(span);
        }
    }
    return parent;
}


function promptSectionBlocks(text, chunk, shotRanges, colors, fields=null) {
    // One block per prompt section, each on a darker shade of that section's own color.
    // The chunk card holds the prompt, and its sections are the parts of it.
    const segments = coloredShotPromptSegments(text, chunk, shotRanges, colors);
    const offsets = [];
    let cursor = 0;
    for (const segment of segments) {
        offsets.push(cursor);
        cursor += segment.text.length;
    }
    const blocks = [];
    for (const section of promptSections(text)) {
        // A caller that wants one section of a prompt asks for it by field name.
        if (fields && !fields.includes(promptSectionField(section.label))) continue;
        const color = promptSectionColor(section.label);
        const accent = `color-mix(in srgb, ${color} 55%, white)`;
        const body = document.createElement("div");
        body.style.cssText = `margin-top:4px;padding:4px 6px;border-left:2px solid ${color};border-radius:3px;background:color-mix(in srgb, ${color} 14%, black);`;
        if (section.headerEnd > section.start) {
            // The section's own header, emphasised in place instead of repeated.
            const header = document.createElement("span");
            header.textContent = text.slice(section.start, section.headerEnd);
            header.style.cssText = `color:${accent};font-weight:700;`;
            body.appendChild(header);
        }
        appendColoredRange(body, segments, offsets, Math.max(section.start, section.headerEnd), section.end);
        blocks.push(body);
    }
    return blocks;
}


function coloredPromptBody(text, chunk, shotRanges, colors) {
    const body = document.createElement("div");
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
            body.appendChild(span);
        }
    }
    return body;
}


function createColoredChunkTooltip() {
    const tooltip = document.createElement("div");
    // The tooltip takes the mouse so a prompt taller than the panel can be scrolled with
    // the wheel or its scrollbar, and so its text can be selected.
    tooltip.style.cssText = "position:fixed;z-index:100001;display:none;box-sizing:border-box;max-width:min(760px,calc(100vw - 24px));max-height:min(70vh,640px);overflow-y:auto;overflow-x:hidden;scrollbar-width:thin;scrollbar-color:#6a6a6a transparent;padding:9px 11px;border:1px solid #555;border-radius:5px;background:rgba(18,18,18,.97);color:#ddd;box-shadow:0 7px 24px rgba(0,0,0,.75);font:11px/1.38 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:pre-wrap;pointer-events:auto;user-select:text;";
    document.body.appendChild(tooltip);
    // True while the pointer is on the tooltip: it is showing its own content, so the
    // timeline must not take it away before the pointer is back on the timeline.
    let hovered = false;
    let leaveGuard = null;

    // The panel and the canvas react to presses, and scrolling a prompt must not start a
    // node drag or a pan behind the tooltip.
    tooltip.addEventListener("pointerdown", event => event.stopPropagation());
    tooltip.addEventListener("mouseenter", () => { hovered = true; });
    tooltip.addEventListener("mouseleave", event => {
        hovered = false;
        if (leaveGuard && leaveGuard(event)) return;
        hide();
    });

    // The frontend claims Ctrl+wheel on document.body in the capture phase to switch tabs
    // and zoom, which cancels the tooltip's own scrolling and hands the rest to the
    // browser's page zoom. The tooltip therefore takes the wheel first, from the window's
    // capture phase, and scrolls itself.
    function onWheel(event) {
        // The pointer stays on the timeline, which is what keeps the tooltip on screen, so
        // what decides is that a tooltip is showing a prompt taller than itself.
        if (tooltip.style.display === "none") return;
        const limit = tooltip.scrollHeight - tooltip.clientHeight;
        if (limit <= 0) return;
        event.preventDefault();
        event.stopImmediatePropagation();
        // A wheel can report lines rather than pixels.
        const delta = event.deltaMode === 1 ? event.deltaY * 16 : event.deltaY;
        tooltip.scrollTop = Math.max(0, Math.min(limit, tooltip.scrollTop + delta));
    }
    window.addEventListener("wheel", onWheel, { capture: true, passive: false });

    function hide() {
        hovered = false;
        tooltip.style.display = "none";
        // A tooltip that opens again starts at its own top, while a rebuild of the content
        // it is already showing keeps the reader where they scrolled to.
        tooltip.scrollTop = 0;
    }

    function line(text, style="", container=tooltip) {
        const element = document.createElement("div");
        element.textContent = text;
        if (style) element.style.cssText = style;
        container.appendChild(element);
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
        element: tooltip,
        isHovered: () => hovered,
        setLeaveGuard(guard) {
            leaveGuard = guard;
        },
        show(event, { help, chunk, timing, description, retentionAnalysis, fullPrompt, teacherAudioPrompt, showFullPrompt, attempts, attemptTranscripts, showAttempts, showNothing, subchunk, subchunkBody, subchunkHeard, shotRanges, colors, waitingText }) {
            tooltip.replaceChildren();
            line(help, "color:#999;margin-bottom:6px;");
            const chunkNumber = Number(chunk.chunk) || 1;
            const chunkColor = colors[(Math.max(1, chunkNumber) - 1) % colors.length] || "#fff";
            // The chunk's own card starts at its number and holds everything the chunk
            // owns, on a very dark shade of the chunk color, so the prompt sections and
            // their own colors are what stands out.
            const card = document.createElement("div");
            card.style.cssText = `padding:6px 8px;border-left:3px solid ${chunkColor};border-radius:3px;background:color-mix(in srgb, ${chunkColor} 7%, black);`;
            tooltip.appendChild(card);
            const entry = (text, style="") => line(text, style, card);
            // A chunk with sub-chunks names the one under the pointer on its own title, so the
            // prompt below is never mistaken for the whole chunk's.
            const pointed = subchunk && subchunk.count > 1 ? subchunk : null;
            entry(pointed ? `Chunk ${chunkNumber}.${pointed.number} / ${pointed.count}` : `Chunk ${chunkNumber}`, `color:${chunkColor};font-weight:700;`);
            if (timing.length) {
                for (const item of timing) entry(item, "color:#bbb;");
            } else {
                entry("Timing: waiting for this chunk to finish.", "color:#888;");
            }
            // The chunk's own reasoning belongs to the views that show more than the plain one:
            // Shift shows everything, Ctrl shows the teacher takes, and the plain view keeps to
            // the prompt itself.
            if ((showFullPrompt || showAttempts) && retentionAnalysis) {
                entry("Per-chunk retention_analysis:", "color:#bbb;margin-top:7px;margin-bottom:2px;");
                entry(retentionAnalysis, "color:#d8c7a0;");
            }
            if (showNothing) {
                // Ctrl+Shift clears the prompt text, leaving the card's own numbers.
                entry("Prompt text hidden.", "color:#888;margin-top:7px;");
            } else if (showAttempts) {
                // Ctrl: one tinted block per prompt the teacher audio was sampled
                // with, each cut down to its detailed_description section.
                const listed = attempts || [];
                entry(
                    listed.length > 1 ? `Teacher-audio prompts: ${listed.length} attempts` : "Teacher-audio prompt:",
                    "color:#bbb;margin-top:7px;margin-bottom:2px;"
                );
                if (!listed.length) entry(waitingText, "color:#888;");
                for (const [index, attempt] of listed.entries()) {
                    const tint = attemptTint(index);
                    entry(
                        index === 0 ? "Teacher audio prompt:" : `Retry ${index}:`,
                        `color:${tint.accent};font-weight:700;margin-top:7px;margin-bottom:2px;`
                    );
                    // A prompt without that section still shows its real text
                    // rather than an empty block.
                    const text = chunkPromptDescription({ h3_prompt: attempt }, false) || String(attempt);
                    const body = coloredPromptBody(text, chunk, shotRanges, colors);
                    body.style.cssText = `padding:5px 7px;border-left:3px solid ${tint.accent};border-radius:3px;background:${tint.background};`;
                    card.appendChild(body);
                    // What this take heard, below its own prompt. The take a later attempt
                    // follows is the one that caused that retry, so its hearing is red: that is
                    // the text the retry answered.
                    const heard = String((attemptTranscripts || [])[index] || "").trim();
                    entry("Teacher audio heard:", `color:${tint.accent};margin-top:5px;margin-bottom:2px;font-weight:700;`);
                    if (!heard) {
                        entry("nothing heard in this window", "color:#888;");
                        continue;
                    }
                    const causedRetry = index + 1 < listed.length;
                    const block = document.createElement("div");
                    block.textContent = heard;
                    block.style.cssText = causedRetry
                        ? "padding:5px 7px;border-left:3px solid #e05252;border-radius:3px;color:#ffe3e3;background:color-mix(in srgb, #e05252 26%, black);"
                        : `padding:5px 7px;border-left:3px solid ${tint.accent};border-radius:3px;color:#cfcfcf;background:color-mix(in srgb, ${tint.accent} 32%, black);`;
                    card.appendChild(block);
                }
            } else if (showFullPrompt) {
                // Shift shows everything: both prompts when they differ, every section of each.
                // Two prompts need their names to be told apart; one carries its own headers.
                const prompts = teacherAudioPrompt && teacherAudioPrompt !== fullPrompt
                    ? [["Audio prompt:", teacherAudioPrompt], ["Video prompt:", fullPrompt]]
                    : [["", fullPrompt || teacherAudioPrompt]];
                for (const [label, text] of prompts) {
                    if (label) entry(label, "color:#bbb;margin-top:7px;margin-bottom:2px;");
                    if (!text) {
                        entry(waitingText, "color:#888;");
                        continue;
                    }
                    for (const section of promptSectionBlocks(text, chunk, shotRanges, colors)) card.appendChild(section);
                }
            } else {
                // The plain view is the Shift view's detailed_description section alone, framed
                // the same way: its own color, its bold header and its own background.
                const framed = fullPrompt ? promptSectionBlocks(fullPrompt, chunk, shotRanges, colors, PROMPT_DESCRIPTION_FIELDS) : [];
                // A description that carries no such field (a legacy chunk, or Gemma's own text)
                // still shows its body, unframed.
                const blocks = framed.length ? framed : promptSectionBlocks(description, chunk, shotRanges, colors);
                if (!blocks.length) entry(waitingText, "color:#888;margin-top:7px;");
                for (const section of blocks) card.appendChild(section);
            }
            // What the pointed sub-chunk's own slice of the take heard, under whichever prompt
            // this view showed. It is speech rather than prompt text, so it carries no shot
            // colors of its own, and a window the take said nothing in says so.
            if (pointed && subchunkHeard !== null) {
                entry("Teacher audio transcript (this sub-chunk's window):", "color:#bbb;margin-top:7px;margin-bottom:2px;");
                if (subchunkHeard) {
                    const heard = document.createElement("div");
                    heard.textContent = subchunkHeard;
                    heard.style.cssText = "padding:5px 7px;border-left:3px solid #8d8d8d;border-radius:3px;color:#eee;background:rgba(255,255,255,.07);";
                    card.appendChild(heard);
                } else {
                    entry("nothing heard in this window", "color:#888;");
                }
            }
            tooltip.style.display = "block";
            position(event);
        },
        move: position,
        hide,
        remove() {
            window.removeEventListener("wheel", onWheel, { capture: true });
            tooltip.remove();
        },
    };
}


function mixHex(from, to, amount) {
    // Mix two six digit colors, so a chunk can carry its sub-chunks in luminance alone.
    const parse = value => [1, 3, 5].map(at => parseInt(String(value).slice(at, at + 2), 16));
    const left = parse(from);
    const right = parse(to);
    return `#${left.map((value, index) => Math.round(value + (right[index] - value) * amount).toString(16).padStart(2, "0")).join("")}`;
}


function subchunkEdges(work) {
    // Where each sub-chunk of a chunk ends, as a fraction of the chunk. A chunk whose
    // sub-chunk frames are unknown has no edges, and keeps the flat bar it always had.
    const counts = Array.isArray(work) ? work.map(value => Math.max(0, Number(value) || 0)) : [];
    const total = counts.reduce((sum, value) => sum + value, 0);
    if (counts.length < 2 || total <= 0) return null;
    const edges = [0];
    let done = 0;
    for (const value of counts) {
        done += value;
        edges.push(done / total);
    }
    return edges;
}


function subchunkTimelineStops(stops, { color, start, end, work, alpha = "e8", renderedFraction = null }) {
    // One chunk's timeline span alternates sub-chunk by sub-chunk between the chunk's own
    // color and a step brighter one — the odd sub-chunks keep the color, the even ones are
    // brighter — so where the sub-chunks are reads at a glance, each band staying flat. The
    // part of a filling chunk that is still pending keeps the dimmed recipe it always had,
    // in the same bands, and the step at the render boundary stays a step.
    const edges = subchunkEdges(work);
    const span = end - start;
    const bands = edges || [0, 1];
    const shadeOf = band => edges && (band + 1) % 2 === 0 ? mixHex(color, "#ffffff", 0.3) : color;
    const painted = (fraction, band) => {
        const shade = shadeOf(band);
        if (renderedFraction === null || fraction < renderedFraction - 1e-9) return `${shade}${alpha}`;
        return `color-mix(in srgb, ${shade} 50%, black)`;
    };
    const cutAt = fraction => Math.round((start + span * Math.max(0, Math.min(1, fraction))) * 1000) / 1000;
    for (let band = 0; band < bands.length - 1; band++) {
        const from = bands[band];
        const to = bands[band + 1];
        if (to <= from) continue;
        if (renderedFraction !== null && renderedFraction > from && renderedFraction <= to) {
            // The band holds the frames the render has reached, which usually ends exactly on a
            // sub-chunk edge: it keeps the rendered color right up to that step and the pending
            // color starts there, so the band stays flat instead of ramping across itself.
            const edge = Math.max(from, renderedFraction - 1e-4);
            stops.push(`${painted(from, band)} ${cutAt(from)}%`, `${painted(edge, band)} ${cutAt(edge)}%`);
            stops.push(`${painted(renderedFraction, band)} ${cutAt(renderedFraction)}%`);
            if (renderedFraction < to) stops.push(`${painted(to, band)} ${cutAt(to)}%`);
            continue;
        }
        stops.push(`${painted(from, band)} ${cutAt(from)}%`, `${painted(to, band)} ${cutAt(to)}%`);
    }
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


function projectedRenderTiming(elapsedSeconds, workDone, workTotal, fallbackEtaSeconds=NaN, rateElapsedSeconds=elapsedSeconds) {
    // Estimate remaining and total wall time from completed planned work units.
    const fraction = workTotal > 0 ? Math.max(0, Math.min(1, workDone / workTotal)) : NaN;
    const etaSeconds = Number.isFinite(rateElapsedSeconds) && fraction > 0
        ? Math.max(0, rateElapsedSeconds) * (1 - fraction) / fraction
        : fallbackEtaSeconds;
    return {
        etaSeconds,
        totalSeconds: Number.isFinite(elapsedSeconds) && Number.isFinite(etaSeconds)
            ? elapsedSeconds + etaSeconds
            : NaN,
    };
}


const PLAYER_STATE_STORAGE_KEY = "hr_endless_sampler_preview_player";


function readStoredPlayerState() {
    // The player buttons must survive a browser refresh and a ComfyUI restart, and
    // the workflow's node properties only survive a workflow *save*, so the same
    // state is mirrored into browser storage as well.
    try {
        const stored = JSON.parse(window.localStorage.getItem(PLAYER_STATE_STORAGE_KEY) || "null");
        return stored && typeof stored === "object" ? stored : {};
    } catch (error) {
        console.warn("HR Endless Sampler could not read the stored player state", error);
        return {};
    }
}


function writeStoredPlayerState(state) {
    try {
        window.localStorage.setItem(PLAYER_STATE_STORAGE_KEY, JSON.stringify(state));
    } catch (error) {
        console.warn("HR Endless Sampler could not store the player state", error);
    }
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
            // LiteGraph serializes node properties with the workflow, but only when
            // that workflow is saved. The stored state is newer in every other case,
            // because each button press writes it, so it wins the merge.
            const savedPlayerState = {
                ...(node.properties?.hr_endless_sampler_preview_player || {}),
                ...readStoredPlayerState(),
            };
            const root = document.createElement("div");
            root.style.cssText = "position:relative;display:flex;flex-direction:column;width:100%;height:100%;min-height:410px;background:#111;border-radius:6px;overflow:hidden;color:#ddd;font:12px sans-serif;outline:none;";
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

            const subtitleButton = document.createElement("button");
            subtitleButton.type = "button";
            subtitleButton.textContent = "S";
            subtitleButton.title = "Turn dialogue subtitles on or off";
            subtitleButton.setAttribute("aria-label", subtitleButton.title);
            subtitleButton.style.cssText = "position:absolute;right:34px;top:8px;width:20px;height:20px;padding:0;border:1px solid #666;border-radius:3px;background:rgba(28,28,28,.9);color:#eee;font:bold 12px/1 ui-monospace,SFMono-Regular,Consolas,monospace;cursor:pointer;z-index:2;";
            viewport.appendChild(subtitleButton);
            const subtitleLabel = document.createElement("div");
            subtitleLabel.style.cssText = "position:absolute;left:7%;right:7%;bottom:27px;z-index:3;text-align:center;color:#fff;font:bold 18px/1.25 sans-serif;-webkit-text-stroke:1px #000;text-shadow:1px 1px 0 #000,-1px -1px 0 #000,1px 1px 0 #000,-1px 1px 0 #000,0 2px 4px #000;pointer-events:none;user-select:none;display:none;";
            viewport.appendChild(subtitleLabel);

            const fullscreenButton = document.createElement("button");
            fullscreenButton.type = "button";
            fullscreenButton.textContent = "⛶";
            fullscreenButton.title = "Enter fullscreen";
            fullscreenButton.setAttribute("aria-label", "Enter fullscreen");
            fullscreenButton.style.cssText = "position:absolute;right:60px;top:8px;width:20px;height:20px;padding:0;border:1px solid #666;border-radius:3px;background:rgba(28,28,28,.9);color:#ddd;font:bold 15px/18px sans-serif;cursor:pointer;z-index:2;";
            viewport.appendChild(fullscreenButton);

            function updateFullscreenButton() {
                const active = document.fullscreenElement === root;
                fullscreenButton.textContent = active ? "×" : "⛶";
                fullscreenButton.title = active ? "Exit fullscreen (Esc)" : "Enter fullscreen";
                fullscreenButton.setAttribute("aria-label", active ? "Exit fullscreen" : "Enter fullscreen");
            }

            const fullscreenChanged = () => {
                updateFullscreenButton();
                applyFullscreenLayout();
                controlsAutohide.arm();
            };
            document.addEventListener("fullscreenchange", fullscreenChanged);
            fullscreenButton.addEventListener("click", event => {
                event.preventDefault();
                event.stopPropagation();
                if (document.fullscreenElement === root) document.exitFullscreen?.();
                else if (root.requestFullscreen) root.requestFullscreen().catch(error => console.warn("HR Endless Sampler fullscreen failed", error));
                root.focus({ preventScroll: true });
            });

            // The transport bar and the status line share one block, so fullscreen can float
            // them over the whole picture as a single panel.
            const controls = document.createElement("div");
            controls.style.cssText = "display:flex;flex-direction:column;flex:0 0 auto;";
            root.appendChild(controls);

            const transport = document.createElement("div");
            transport.style.cssText = "display:flex;align-items:center;gap:7px;box-sizing:border-box;height:43px;padding:4px 8px;background:#181818;border-top:1px solid #242424;";
            controls.appendChild(transport);

            const playButton = document.createElement("button");
            playButton.type = "button";
            playButton.style.cssText = "display:flex;align-items:center;justify-content:center;width:22px;height:19px;padding:0;border:1px solid #555;border-radius:3px;background:#252525;color:#f4f4f4;font:12px/1 sans-serif;cursor:pointer;";
            playButton.textContent = "▶";
            playButton.title = "Play/pause (Space). Use Left/Right arrows for one preview frame.";
            transport.appendChild(playButton);

            const muteButton = document.createElement("button");
            muteButton.type = "button";
            muteButton.style.cssText = "display:flex;flex:0 0 auto;align-items:center;justify-content:center;width:23px;height:19px;padding:0;border:1px solid #555;border-radius:3px;background:#252525;color:#f4f4f4;font:13px/1 sans-serif;cursor:pointer;";
            muteButton.title = "Mute/unmute finalized preview audio (M)";
            transport.appendChild(muteButton);

            const timelineHelp = "Click or drag to seek; colors identify chunks. Hold Shift to show the full prompt sent to H3; hold Ctrl for every teacher-audio attempt, the first take and each retry.";
            const timelineShell = document.createElement("div");
            timelineShell.style.cssText = "position:relative;flex:1;height:33px;cursor:pointer;touch-action:none;";
            transport.appendChild(timelineShell);
            const chunkTooltip = createColoredChunkTooltip();

            const timelineTrack = document.createElement("div");
            timelineTrack.style.cssText = "position:absolute;left:0;right:0;top:3px;height:5px;border-radius:3px;background:#333;box-shadow:0 0 0 1px #080808,0 1px 2px #000;overflow:hidden;";
            timelineShell.appendChild(timelineTrack);

            const audioCompletedOutlines = document.createElement("div");
            audioCompletedOutlines.style.cssText = "position:absolute;left:0;right:0;top:1px;height:9px;z-index:2;pointer-events:none;";
            timelineShell.appendChild(audioCompletedOutlines);

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
            const teacherAudioCacheHelp = "Enable or disable teacher-audio reuse for the next TaoMate run. TaoMate records no video replay cache, so this button governs its audio-first teacher pass instead: the same prompt and seed reuse the stored teacher audio instead of sampling, decoding and transcribing it again. Disabled clears the stored audio as the render starts, and the run still records a fresh replacement.";
            let replayCacheEnabled = true;
            let reusingCachedChunks = false;
            let reusedChunkNumbers = [];
            // This is cache availability between runs, not merely the chunks
            // restored by the current sampler execution.
            let cachedChunkCount = 0;
            let cachedChunkIndices = new Set();
            // TaoMate chunks that have stored teacher audio on disk; one of them
            // can be deleted from the timeline menu to force a new take.
            let audioCacheChunkIndices = new Set();
            function renderCacheReuseLabel() {
                // TaoMate has no cached video chunks; its only reusable state is
                // the teacher audio stored by the last audio-first pass.
                if (taomateMode) {
                    const audio = lastCacheStatus?.audio_cache || null;
                    if (audio?.run_state === "incompatible") {
                        // The cache cannot be used for this render, and the run replaces
                        // it, so warn at the top of the UI instead of painting the node.
                        cacheReuseLabel.style.display = "none";
                        warnToast("Teacher audio cache is incompatible with this render", audio.run_reason || "");
                        return;
                    }
                    cacheReuseLabel.style.color = "#69b76f";
                    const settled = audio?.run_state === "reused" || audio?.run_state === "resumed";
                    const reused = settled && Number(audio.reused_chunks) > 0;
                    const reusedChunks = Math.max(0, Number(audio?.reused_chunks) || 0);
                    // The sampler composes this sentence: a resume can keep chunks on
                    // both sides of the one it generates again, so only it knows which.
                    cacheReuseLabel.textContent = audio?.run_state === "resumed"
                        ? String(audio.run_reason || "kept the settled teacher audio and resumed the teacher")
                        : `reusing cached teacher audio (${reusedChunks} chunks)`;
                    cacheReuseLabel.style.display = reused ? "block" : "none";
                    return;
                }
                const labels = reusedChunkNumbers.map(number => `#${number}`);
                const chunks = labels.length < 2
                    ? labels.join("")
                    : labels.length === 2
                        ? `${labels[0]} and ${labels[1]}`
                        : `${labels.slice(0, -1).join(", ")} and ${labels.at(-1)}`;
                cacheReuseLabel.textContent = `reusing cached chunks ${chunks}`;
                cacheReuseLabel.style.display = reusingCachedChunks && labels.length ? "block" : "none";
            }
            function teacherAudioCacheNote(audioStatus) {
                const chunks = Math.max(0, Number(audioStatus?.chunks) || 0);
                const bytes = Math.max(0, Number(audioStatus?.bytes) || 0);
                const size = bytes ? ` (${(bytes / 1048576).toFixed(1)} MiB)` : "";
                if (!chunks) return "No teacher audio is stored yet.";
                const recorded = Math.max(0, Number(audioStatus?.recorded) || 0);
                // A stopped pass stores fewer chunks than it planned: its prompt is
                // still the pass a refresh restores and the next render continues.
                const progress = recorded && recorded < chunks ? `${recorded} of ${chunks} chunks are stored so far (the render was stopped)` : `${chunks} chunks of teacher audio are stored${size}`;
                return `${progress}${audioStatus?.created ? `, recorded ${audioStatus.created}` : ""}.`;
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
                audioCacheChunkIndices = new Set(
                    (Array.isArray(cacheStatus?.audio_cache?.kv_chunks) ? cacheStatus.audio_cache.kv_chunks : [])
                        .map(value => Math.round(Number(value)))
                        .filter(index => index >= 0 && (!chunkCount || index < chunkCount)),
                );
                cacheButton.disabled = Boolean(cacheStatus?.active);
                cacheButton.style.background = replayCacheEnabled ? "#28662d" : "#202020";
                cacheButton.style.borderColor = replayCacheEnabled ? "#69b76f" : "#444";
                cacheButton.style.color = replayCacheEnabled ? "#e5ffe6" : "#888";
                cacheButton.style.cursor = cacheButton.disabled ? "not-allowed" : "pointer";
                renderCacheReuseLabel();
                if (taomateMode) {
                    cacheButton.title = replayCacheEnabled
                        ? `${teacherAudioCacheHelp}\nEnabled. ${teacherAudioCacheNote(cacheStatus?.audio_cache)}`
                        : `${teacherAudioCacheHelp}\nDisabled. The stored teacher audio is cleared when the next render starts.`;
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

            // ComfyUI's own warning toast, at the top of the UI and dismissed with its X.
            // The status poll runs every few seconds, so an identical warning is only
            // raised once until the message changes.
            let lastWarnToast = "";
            function warnToast(summary, detail="") {
                const text = `${summary || ""}\n${detail || ""}`;
                if (text === lastWarnToast) return true;
                const toast = app?.extensionManager?.toast;
                if (!toast || typeof toast.add !== "function") {
                    // Say so in the console: a missing toast API must not turn a cache
                    // problem into silence the owner cannot see.
                    console.warn("HR Endless Sampler has no toast API to show:", summary, detail);
                    return false;
                }
                lastWarnToast = text;
                toast.add({
                    severity: "warn",
                    summary: String(summary || ""),
                    detail: String(detail || ""),
                    life: 8000,
                });
                return true;
            }

            cacheButton.addEventListener("click", async event => {
                event.preventDefault();
                event.stopPropagation();
                if (cacheButton.disabled) return;
                let refusal = "";
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
                    // A refused change used to look like the button flipping back by
                    // itself. Say why instead: the owner cannot act on a silent revert.
                    refusal = error instanceof Error ? error.message : String(error);
                    console.warn("HR Endless Sampler could not update its replay cache policy", error);
                    await refreshReplayCacheStatus();
                } finally {
                    cacheButton.textContent = "cache";
                    renderCacheReuseLabel();
                    if (refusal) {
                        // A refused change is a warning, not node decoration: raise it at
                        // the top of the UI where ComfyUI shows its own warnings.
                        warnToast("Teacher audio cache policy was not changed", refusal);
                    }
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
                // TaoMate stores no video chunks, so the same gesture deletes this
                // chunk's stored teacher audio instead and the next render rebuilds it.
                // Offered for every TaoMate chunk while the cache button is on: the
                // item exists so the owner can see it and knows which chunk still has
                // stored audio, instead of wondering why the menu is empty.
                const offersTeacherAudio = replayCacheEnabled && !cacheButton.disabled && taomateMode
                    && Number.isInteger(chunkIndex) && chunkIndex >= 0;
                const canDeleteTeacherAudio = offersTeacherAudio && audioCacheChunkIndices.has(chunkIndex);
                if (!canDeleteChunk && !offersTeacherAudio) {
                    document.body.appendChild(menu);
                    previewContextMenu = menu;
                    return;
                }
                const chunkNumber = chunkIndex + 1;
                const menuItemStyle = "display:block;width:100%;padding:4px 7px;border:0;border-radius:2px;background:transparent;color:#f0f0f0;text-align:left;font:11px/1.2 ui-monospace,SFMono-Regular,Consolas,monospace;cursor:pointer;";
                const divider = document.createElement("div");
                divider.style.cssText = "height:1px;margin:3px 2px;background:#505050;";
                menu.appendChild(divider);
                if (canDeleteChunk) {
                const removeButton = document.createElement("button");
                removeButton.type = "button";
                removeButton.textContent = "Delete cached chunk";
                removeButton.title = `Delete cached Chunk ${chunkNumber}. The next sampler run rerenders it and reuses intact later cached chunks.`;
                removeButton.style.cssText = menuItemStyle;
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
                }
                if (offersTeacherAudio) {
                    const teacherAudioButton = document.createElement("button");
                    teacherAudioButton.type = "button";
                    teacherAudioButton.textContent = "Delete teacher audio cached chunk";
                    teacherAudioButton.style.cssText = menuItemStyle;
                    if (!canDeleteTeacherAudio) {
                        // The audio and its attention rows are cached together, so a
                        // chunk without rows has no stored audio to drop either.
                        teacherAudioButton.disabled = true;
                        teacherAudioButton.style.color = "#777";
                        teacherAudioButton.style.cursor = "default";
                        teacherAudioButton.title = `Chunk ${chunkNumber} has no stored teacher audio: the audio and its attention rows are cached together, and this chunk has neither. Chunks that do have a stored pair: ${[...audioCacheChunkIndices].map(index => index + 1).join(", ") || "none"}.`;
                        menu.appendChild(teacherAudioButton);
                    } else {
                    teacherAudioButton.title = `Delete the stored teacher audio for Chunk ${chunkNumber}. The next TaoMate render keeps every other chunk and generates this one again.`;
                    teacherAudioButton.addEventListener("mouseenter", () => { teacherAudioButton.style.background = "#8d302d"; });
                    teacherAudioButton.addEventListener("mouseleave", () => { teacherAudioButton.style.background = "transparent"; });
                    teacherAudioButton.addEventListener("click", async clickEvent => {
                        clickEvent.preventDefault();
                        clickEvent.stopPropagation();
                        chunkTooltip.hide();
                        tooltipChunkIndex = null;
                        tooltipSignature = null;
                        teacherAudioButton.disabled = true;
                        teacherAudioButton.textContent = "Deleting…";
                        try {
                            const response = await api.fetchApi(
                                `/hr_endless_sampler_preview/teacher_audio_chunk?chunk=${chunkIndex}`,
                                { method: "POST", cache: "no-store" },
                            );
                            const payload = await response.json();
                            applyReplayCacheStatus(payload);
                            renderTransport();
                            if (!response.ok) throw new Error(payload?.error || `HTTP ${response.status}`);
                        } catch (error) {
                            console.warn("HR Endless Sampler could not delete the cached teacher audio", error);
                            await refreshReplayCacheStatus();
                        } finally {
                            closePreviewContextMenu();
                        }
                    });
                    menu.appendChild(teacherAudioButton);
                    }
                }
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
            controls.appendChild(status);

            const graphs = document.createElement("div");
            graphs.style.cssText = "display:grid;grid-template-columns:1fr 1fr;gap:6px;padding:6px;background:#151515;";
            root.appendChild(graphs);

            // In fullscreen the picture owns the whole screen and the chrome floats on top of
            // it, translucent, so the transport, the status line and the graphs never resize
            // the video. Leaving fullscreen puts every one of them back in the column.
            function applyFullscreenLayout() {
                const active = document.fullscreenElement === root;
                const blur = active ? "blur(3px)" : "";
                viewport.style.position = active ? "absolute" : "relative";
                viewport.style.inset = active ? "0" : "";
                controls.style.position = active ? "absolute" : "";
                controls.style.left = active ? "0" : "";
                controls.style.right = active ? "0" : "";
                controls.style.bottom = active ? "0" : "";
                controls.style.zIndex = active ? "6" : "";
                graphs.style.position = active ? "absolute" : "";
                graphs.style.left = active ? "8px" : "";
                graphs.style.top = active ? "8px" : "";
                graphs.style.width = active ? "min(430px,44vw)" : "";
                graphs.style.zIndex = active ? "6" : "";
                transport.style.background = active ? "rgba(20,20,20,.72)" : "#181818";
                transport.style.backdropFilter = blur;
                status.style.background = active ? "rgba(24,24,24,.72)" : "#1b1b1b";
                status.style.backdropFilter = blur;
                graphs.style.background = active ? "rgba(18,18,18,.72)" : "#151515";
                graphs.style.backdropFilter = blur;
            }

            // Fullscreen playback hides that chrome once the pointer has been quiet for three
            // seconds; a paused preview keeps it, and any movement brings it back.
            const controlsAutohide = createControlsAutohide(
                [controls, graphs, linearDisplayButton, subtitleButton, fullscreenButton],
                {
                    isFullscreen: () => document.fullscreenElement === root,
                    isPlaying: () => !paused,
                },
            );

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
            let audioCompletedChunks = new Set();
            let shotRanges = [];
            let timelineTotalFrames = 0;
            let shotBracketKey = null;
            let playing = -1;
            let playingFrame = 0;
            let timer = null;
            let playbackSerial = 0;
            let framePending = false;
            let waitingAfterChunk = null;
            let paused = false;
            let timelineDragging = false;
            // Where a drag put the cursor: a chunk the render has only partly filled
            // cannot place the cursor, but the pointer can, so this is what the
            // playhead follows until playback or a step moves the position again.
            let timelineScrub = null;
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
            let audioFirstSeconds = null;
            let audioStepMs = null;
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
            let ctrlPromptVisible = false;
            let audioGroup = null;
            let standbyAudioGroup = null;
            let audioMuted = Boolean(savedPlayerState.muted);
            let inverseGammaDisplay = Boolean(savedPlayerState.inverseGammaDisplay);
            let subtitlesEnabled = savedPlayerState.subtitlesEnabled !== false;
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
                    const offset = next === group ? audioOffset(group, frameIndex) : audioOffset(next, 0);
                    if (offset >= buffer.duration) break;
                    const source = continuousAudioContext.createBufferSource();
                    source.buffer = buffer;
                    source.playbackRate.value = rate;
                    source.connect(continuousAudioGain);
                    start = Math.max(start, continuousAudioContext.currentTime);
                    const segmentEnd = Number(next.audioEndSeconds);
                    const playableEnd = Number.isFinite(segmentEnd) ? Math.min(buffer.duration, segmentEnd) : buffer.duration;
                    if (playableEnd <= offset) break;
                    const duration = (playableEnd - offset) / rate;
                    const end = start + duration;
                    source.start(start, offset, duration);
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
                waitingAfterChunk = null;
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
                const timelineOffset = Number(group?.audioStartSeconds) || 0;
                const first = Number(group?.frameNumbers?.[0]);
                const current = Number(group?.frameNumbers?.[frameIndex]);
                if (Number.isFinite(first) && Number.isFinite(current)) {
                    return Math.max(0, timelineOffset + (current - first) / source);
                }
                return Math.max(0, timelineOffset + (Number(frameIndex) || 0) / source);
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
                if (audioGroup !== group || audioPlayer.getAttribute("src") !== source) {
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

            function rememberPlayerState(state) {
                // Both copies get the same mapping: the workflow keeps the buttons
                // across a saved reload, browser storage across a plain one.
                node.properties = node.properties || {};
                node.properties.hr_endless_sampler_preview_player = state;
                writeStoredPlayerState(state);
            }

            function persistPlayerState() {
                const frame = displayedFrameNumber();
                // A restored frame may still be on its way, so keep asking for it while
                // storing everything else: a button press must never be dropped just
                // because no frame has arrived yet, which is the normal case for a
                // TaoMate preview that carries audio and no video at all.
                const frameStillPending = Number.isFinite(pendingFrame) && frame !== pendingFrame;
                rememberPlayerState({
                    frame: frameStillPending ? pendingFrame : frame,
                    muted: audioMuted,
                    inverseGammaDisplay,
                    subtitlesEnabled,
                    paused,
                });
            }

            function restorePersistedFrame() {
                if (!Number.isFinite(pendingFrame)) return false;
                for (let chunkIndex = 0; chunkIndex < chunks.length; chunkIndex++) {
                    const group = chunks[chunkIndex];
                    const frameIndex = group?.frameNumbers?.findIndex(value => Number(value) === pendingFrame);
                    if (frameIndex >= 0) {
                        // Resume the button's state with the frame: an explicit
                        // "playing" keeps playing, and a state written before the
                        // play button existed still restores paused, as it did then.
                        paused = savedPlayerState.paused === undefined ? true : Boolean(savedPlayerState.paused);
                        const frame = pendingFrame;
                        pendingFrame = null;
                        show(chunkIndex, frameIndex);
                        rememberPlayerState({ frame, muted: audioMuted, inverseGammaDisplay, subtitlesEnabled, paused });
                        return true;
                    }
                    // A TaoMate preview carries audio and no frames, so there is no frame
                    // number to match: land on the chunk whose frame range holds the
                    // stored position, which is what the timeline cursor shows. Only
                    // frame-less groups take this path, so a later real frame still wins.
                    if (group?.frameNumbers?.length) continue;
                    const start = Number(group?.output_start ?? group?.start ?? chunkRanges[chunkIndex]?.start);
                    const end = Number(group?.output_end ?? group?.end ?? chunkRanges[chunkIndex]?.end);
                    if (!Number.isFinite(start) || !Number.isFinite(end) || pendingFrame < start || pendingFrame > end) continue;
                    paused = savedPlayerState.paused === undefined ? true : Boolean(savedPlayerState.paused);
                    const frame = pendingFrame;
                    pendingFrame = null;
                    playing = chunkIndex;
                    playingFrame = 0;
                    show(chunkIndex, 0);
                    renderTransport();
                    renderStatus();
                    rememberPlayerState({ frame, muted: audioMuted, inverseGammaDisplay, subtitlesEnabled, paused });
                    return true;
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

            function renderSubtitles() {
                const text = subtitlesEnabled ? String(chunkRanges[playing]?.subtitle || "").trim() : "";
                subtitleLabel.textContent = text;
                subtitleLabel.style.display = text ? "block" : "none";
                subtitleButton.style.color = subtitlesEnabled ? "#ffe600" : "#888";
                subtitleButton.style.background = subtitlesEnabled ? "rgba(70,58,12,.95)" : "rgba(28,28,28,.9)";
                subtitleButton.setAttribute("aria-pressed", String(subtitlesEnabled));
            }

            subtitleButton.addEventListener("click", event => {
                event.preventDefault();
                event.stopPropagation();
                subtitlesEnabled = !subtitlesEnabled;
                renderSubtitles();
                persistPlayerState();
                root.focus({ preventScroll: true });
            });
            renderSubtitles();

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

            // The picture of a position nothing has rendered yet. The teacher-audio
            // placeholder groups and the scrub use one asset, so an empty timeline
            // position looks exactly like a chunk whose video has not started.
            const EMPTY_FRAME_SOURCE = "data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' width='2' height='2'%3E%3Crect width='2' height='2' fill='black'/%3E%3C/svg%3E";

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

            function chunkPointerPosition(event) {
                // Where on the timeline a pointer sits: the chunk under it and how far
                // into that chunk it is.
                const rect = timelineShell.getBoundingClientRect();
                if (!rect.width || chunkCount < 1) return null;
                const ratio = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
                const { spans, total } = timelineLayout();
                const cursor = ratio * total;
                let offset = 0;
                for (let index = 0; index < spans.length; index++) {
                    const next = offset + spans[index];
                    if (cursor <= next || index === spans.length - 1) {
                        return { index, fraction: Math.max(0, Math.min(1, (cursor - offset) / Math.max(1, spans[index]))) };
                    }
                    offset = next;
                }
                return null;
            }


            function chunkIndexAtTimelinePointer(event) {
                const position = chunkPointerPosition(event);
                return position === null ? null : position.index;
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
                const retentionAnalysis = String(range.gemma_retention_analysis || "").trim();
                const teacherAudioPrompt = String(range.audio_teacher_prompt || "").trim();
                const attempts = audioTeacherAttempts(range);
                const attemptTranscripts = audioTeacherAttemptTranscripts(range);
                // The sub-chunk the pointer sits in owns the prompt the tooltip reads, so a
                // chunk with sub-chunks reads differently along its own timeline: no key shows
                // that sub-chunk's detailed_description, Shift its whole prompt. Ctrl alone
                // keeps the teacher-audio attempts, and Ctrl+Shift clears the prompt text.
                const pointer = chunkPointerPosition(event);
                const subchunk = pointer ? subchunkAt(range, pointer.fraction) : null;
                const subchunkBody = subchunk ? subchunkPrompt(range, subchunk.number) : null;
                const subchunkHeard = subchunk && Array.isArray(range.subchunk_transcripts)
                    ? String(range.subchunk_transcripts[subchunk.number - 1] || "").trim()
                    : null;
                const fullPrompt = String(subchunkBody?.prompt || range.h3_prompt || "").trim();
                const description = chunkPromptDescription({ h3_prompt: fullPrompt, gemma_detailed_description: range.gemma_detailed_description });
                const showAttempts = Boolean(event.ctrlKey || ctrlPromptVisible);
                const showFullPrompt = Boolean(!showAttempts && (event.shiftKey || shiftPromptVisible));
                const showNothing = Boolean((event.ctrlKey || ctrlPromptVisible) && (event.shiftKey || shiftPromptVisible));
                const timing = chunkTimingLines(range);
                const signature = JSON.stringify([description, retentionAnalysis, fullPrompt, teacherAudioPrompt, attempts, attemptTranscripts, showAttempts, showFullPrompt, showNothing, subchunk?.number, subchunk?.count, subchunkBody?.prompt, subchunkBody?.silent, subchunkHeard, timing, shotRanges]);
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
                    teacherAudioPrompt,
                    showFullPrompt,
                    attempts,
                    attemptTranscripts,
                    showAttempts,
                    showNothing,
                    subchunk,
                    subchunkBody,
                    subchunkHeard,
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
                renderSubtitles();
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
                audioCompletedOutlines.replaceChildren();
                let audioCompletedSpan = 0;
                let audioCompletedCount = 0;
                while (audioCompletedChunks.has(audioCompletedCount) && audioCompletedCount < spans.length) {
                    audioCompletedSpan += spans[audioCompletedCount];
                    audioCompletedCount++;
                }
                if (audioCompletedCount > 0) {
                    const outline = document.createElement("div");
                    outline.title = `Audio teacher complete · chunks 1–${audioCompletedCount}`;
                    outline.style.cssText = `position:absolute;left:0;width:${audioCompletedSpan / total * 100}%;top:0;height:9px;box-sizing:border-box;border:1px solid #42c8ff;border-radius:2px;`;
                    if (audioCompletedCount < spans.length) outline.style.cssText += "border-right:0;border-top-right-radius:0;border-bottom-right-radius:0;";
                    audioCompletedOutlines.appendChild(outline);
                }
                for (let index = 0; index < spans.length; index++) {
                    const start = offset / total * 100;
                    offset += spans[index];
                    const end = offset / total * 100;
                    const color = chunkColors[index % chunkColors.length];
                    // Each sub-chunk of the chunk is drawn as a brighter step in the same
                    // color, so the timeline shows where they are inside it.
                    const work = chunkRanges[index]?.taomate_phase_work;
                    const completedFrames = chunkRanges[index]?.taomate_completed_frames;
                    if (Number.isFinite(completedFrames)) {
                        // A live latent preview makes its current sub-chunk available before completion.
                        const previewFrames = available(index) && !chunks[index].audioOnlyPlaceholder
                            ? Math.max(0, (Number(chunks[index].outputEnd) + 1) - Number(chunkRanges[index]?.start)) || 0
                            : 0;
                        const fraction = Math.max(0, Math.min(1, Math.max(completedFrames, previewFrames) / spans[index]));
                        subchunkTimelineStops(stops, { color, start, end, work, renderedFraction: fraction >= 1 ? null : fraction });
                        continue;
                    }
                    const alpha = available(index) && !chunks[index].audioOnlyPlaceholder ? "e8" : "35";
                    subchunkTimelineStops(stops, { color, start, end, work, alpha });
                }
                timelineTrack.style.background = stops.length
                    ? `linear-gradient(to right, ${stops.join(",")})`
                    : "#333";
                renderCachedChunkUnderlines(spans, total);
                renderOverlapReplacementLines(spans, total);
                renderShotBrackets();

                const scrubPosition = scrubPlayheadPosition(spans, total, timelineScrub);
                if (scrubPosition !== null) {
                    // A drag owns the cursor: it follows the pointer even where no frame
                    // exists yet, so it never lags behind the mouse or leaps to the
                    // nearest rendered chunk.
                    timelinePlayhead.style.left = `${Math.max(0, Math.min(100, scrubPosition))}%`;
                    timelinePlayhead.style.display = "block";
                    return;
                }
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
                // Playback moves the position itself, so a drag's cursor is over.
                timelineScrub = null;
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
                    const firstFrame = Number(group.outputStart);
                    const currentFrame = Number(group.frameNumbers?.[boundedFrame]);
                    const nextFrame = Number(group.frameNumbers?.[boundedFrame + 1]);
                    const nextMediaTime = Number.isFinite(firstFrame) && Number.isFinite(nextFrame)
                        ? (nextFrame - firstFrame) / sourceRate
                        : Number.isFinite(firstFrame) && Number.isFinite(currentFrame)
                            ? (currentFrame - firstFrame) / sourceRate + (Number(group.durations?.[boundedFrame]) || 1000 / sourceRate) / 1000
                            : (boundedFrame + 1) / sourceRate;
                    const localAudioTime = (continuousTime ?? audioPlayer.currentTime) - (Number(group.audioStartSeconds) || 0);
                    const wallDelay = (nextMediaTime - localAudioTime) * 1000 / Math.max(0.0625, currentPlaybackFps() / sourceRate);
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
                                    const localAudioTime = (audioTime ?? audioPlayer.currentTime) - (Number(group.audioStartSeconds) || 0);
                                    const firstFrame = Number(group.outputStart);
                                    const targetFrame = (Number.isFinite(firstFrame) ? firstFrame : 0) + localAudioTime * sourceRate;
                                    let audioFrame = 0;
                                    if (group.frameNumbers?.length === group.frames.length) {
                                        for (let candidate = 1; candidate < group.frameNumbers.length && Number(group.frameNumbers[candidate]) <= targetFrame; candidate++) audioFrame = candidate;
                                    } else {
                                        audioFrame = Math.floor(localAudioTime * sourceRate + 1e-4);
                                    }
                                    nextFrame = Math.max(
                                        nextFrame,
                                        audioFrame,
                                    );
                                }
                                if (nextFrame < group.frames.length) {
                                    playFrameGroup(index, group, nextFrame, serial, false);
                                    return;
                                }
                            }
                            if (boundedFrame + 1 >= group.frames.length || group.audioSource) {
                                if (available(index + 1)) playFrameGroup(index + 1, chunks[index + 1], 0, serial, false);
                                else if (!complete && index + 1 < chunkCount) {
                                    // Wait for the next rendered chunk instead of replaying
                                    // chunk 1 or jumping over a missing audio segment.
                                    stop();
                                    waitingAfterChunk = index;
                                    renderTransport();
                                } else if (complete || index + 1 >= chunkCount) {
                                    const next = nextAvailable(index);
                                    if (next >= 0) show(next);
                                }
                                return;
                            }
                        }, duration);
                    },
                );
            }

            function show(index, frameIndex=0) {
                if (!available(index) || hoverStep != null) return;
                stop();
                // Showing a frame is a position of its own: the drag's own cursor is
                // over, and the playhead follows what is displayed again.
                timelineScrub = null;
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

            function showEmptyPosition(index) {
                // Nothing has been rendered at this position yet, so the picture is black
                // and the transport reports no frame: the timeline shows the frames that
                // exist, never the frames that will exist. The drag's own cursor still
                // follows the pointer, because renderTransport reads it.
                stop();
                playing = index;
                playingFrame = -1;
                framePending = false;
                displaySource(EMPTY_FRAME_SOURCE, () => true, () => {});
            }

            function restorePlayback() {
                if (playingFrame < 0 && available(playing)) {
                    // The last position asked for has nothing rendered there yet. It stays
                    // black, unless playback is asked for: that starts from the chunk
                    // instead of leaving the picture black for good.
                    if (paused) showEmptyPosition(playing);
                    else show(playing, 0);
                    return;
                }
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
                // A paused player keeps its chrome: there is nothing to watch behind it.
                controlsAutohide.arm();
                leaveGraphInspection();
                stop();
                renderTransport();
                restorePlayback();
                renderStatus();
                // The play button is a state like the others: store it at once so a
                // refresh resumes the same way even before a frame is displayed.
                persistPlayerState();
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

            function scrubPlayheadPosition(spans, total, scrub) {
                // Return the playhead's percentage for a scrub position, or null. The
                // pointer's own position on the slider is the position: a chunk that is
                // still being rendered has fewer frames than its span, so placing the
                // cursor at its newest frame would leave the cursor behind the mouse.
                if (scrub === null || !spans.length) return null;
                const chunk = Math.max(0, Math.min(spans.length - 1, Number(scrub.chunk) || 0));
                const before = spans.slice(0, chunk).reduce((sum, value) => sum + value, 0);
                const fraction = Math.max(0, Math.min(1, Number(scrub.fraction) || 0));
                // A span of zero is a chunk with no frames and no known length: its
                // start is the only position it can offer.
                return (before + fraction * Math.max(0, spans[chunk])) / Math.max(1, total) * 100;
            }

            function scrubFrameIndex(chunkIndex, fraction) {
                // The pointer names a position inside the chunk. One preview image stands
                // for the whole H3 temporal token it was decoded from, so a position shows
                // the last image numbered at or before it, and only a position the render
                // has not reached has nothing to show. A chunk whose later sub-chunks are
                // still to come is black past its rendered end, never stretched.
                const group = chunks[chunkIndex];
                if (!Array.isArray(group?.frames) || !group.frames.length) return -1;
                const frames = group.frames.length;
                const span = Math.max(1, chunkSpan(chunkIndex, frames));
                const local = Math.min(Math.max(0, Math.floor(fraction * span)), span - 1);
                const numbers = Array.isArray(group.frameNumbers) && group.frameNumbers.length === frames ? group.frameNumbers : null;
                if (!numbers) return local < frames ? local : -1;
                const start = Number(chunkRanges[chunkIndex]?.start);
                const position = Number.isFinite(start) ? start + local : Number(numbers[0]) + local;
                const renderedEnd = Number(group.outputEnd);
                if (Number.isFinite(renderedEnd) && position > renderedEnd) return -1;
                // The numbers are ordered, so the covering image is the last one at or
                // before the position.
                let covering = -1;
                for (let index = 0; index < numbers.length && Number(numbers[index]) <= position; index++) covering = index;
                return covering;
            }

            function seekTimeline(clientX) {
                const rect = timelineShell.getBoundingClientRect();
                if (rect.width <= 0 || chunkCount < 1) return;
                const ratio = Math.max(0, Math.min(1, (clientX - rect.left) / rect.width));
                const { spans, total } = timelineLayout();
                const cursor = ratio * total;
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
                const fraction = Math.max(0, Math.min(1, (cursor - chunkStart) / Math.max(1, spans[targetChunk])));
                paused = true;
                leaveGraphInspection();
                // Show the frame that really lives at the pointed position. A position
                // inside a chunk whose later sub-chunks are not rendered yet, and a chunk
                // the pass has not reached at all, both have no frame and stay black.
                const frame = scrubFrameIndex(targetChunk, fraction);
                if (frame >= 0) show(targetChunk, frame);
                else showEmptyPosition(targetChunk);
                // Set after show(): showing a frame is the position, and this is the
                // position the pointer asked for, which the cursor keeps.
                timelineScrub = { chunk: targetChunk, fraction };
                renderTransport();
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
                const audioStep = /audio teacher\s+(\d+)\s*\/\s*(\d+).*?step\s+(\d+)\s*\/\s*(\d+)/i.exec(phase || "");
                const displayedStepMs = audioStep ? audioStepMs : averageStepMs;
                const secondsPerStep = Number.isFinite(displayedStepMs) ? `${(displayedStepMs / 1000).toFixed(2)}s/step` : "—s/step";
                const elapsedSeconds = completedElapsed ?? (startedAt == null ? NaN : (performance.now() - startedAt) / 1000);
                const elapsed = formatEta(elapsedSeconds);
                const activePhaseCount = Math.max(1, Number(chunkRanges[activeChunk]?.taomate_phase_count) || 1);
                const chunk = chunkCount ? activeSubchunk
                    ? `C${activeChunk + 1}.${activeSubchunk}/${chunkCount}.${activePhaseCount}`
                    : `C${activeChunk + 1}/${chunkCount}`
                    : "C—/—";
                const displayStep = audioStep ? Number(audioStep[3]) : (hoverStep ?? currentStep);
                const displayTotalSteps = audioStep ? Number(audioStep[4]) : totalSteps;
                const inspecting = hoverStep == null ? "" : "Inspect · ";
                const statePrefix = `${complete ? "Complete · " : ""}${paused ? "Paused · " : ""}`;
                const phaseLine = `${statePrefix}${(phase || "Preparing sampler").replace(/\s*·\s*step\s+\d+\s*\/\s*\d+/i, "")}`;
                const h3Active = !complete && /h3\s+(?:sampling|inference)/i.test(phase || "");
                const audioProgress = audioStep ? Math.max(0, Math.min(1, (Number(audioStep[1]) - 1 + Number(audioStep[3]) / Math.max(1, Number(audioStep[4]))) / Math.max(1, Number(audioStep[2])))) : 0;
                const audioFirstMode = chunkRanges.some(range => range.taomate_audio_first === true) || audioFirstSeconds != null;
                const audioPrepass = audioFirstMode && audioFirstSeconds == null && !complete;
                const retrySeconds = chunkRanges.reduce((sum, range) => {
                    if (!Number.isFinite(range.audio_retry_start_ms)) return sum;
                    const end = Number.isFinite(range.audio_retry_end_ms) ? range.audio_retry_end_ms / 1000 : elapsedSeconds;
                    return sum + Math.max(0, end - range.audio_retry_start_ms / 1000);
                }, 0);
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
                    if (audioFirstMode) {
                        // The audio cursor cannot mark earlier video groups complete.
                        const reported = Math.min(chunkUnits, Number(range.taomate_completed_phases) || 0);
                        const active = index === activeChunk && activeSubchunk != null ? Math.max(0, Number(activeSubchunk) - 1) : 0;
                        completedUnits += cachedChunkIndices.has(index) || chunks[index]?.finalized ? chunkUnits : Math.min(chunkUnits, Math.max(reported, active));
                    } else if (index === activeChunk) {
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
                let observedUnits = complete ? workTotal : Math.min(workTotal, completedUnits + (!audioFirstMode || activeSubchunk != null ? 1 : 0));
                const fallbackRemainingSteps = Math.max(0, totalSteps - currentStep) + Math.max(0, chunkCount - activeChunk - 1) * totalSteps;
                const fallbackEtaSeconds = Number.isFinite(averageStepMs) ? fallbackRemainingSteps * averageStepMs / 1000 : NaN;
                let rateElapsedSeconds = Math.max(0, elapsedSeconds - retrySeconds);
                if (audioPrepass) {
                    workTotal = chunkCount;
                    const attemptedAudio = chunkRanges.reduce((count, range, index) => Number.isFinite(range.audio_retry_start_ms) ? Math.max(count, index + 1) : count, 0);
                    observedUnits = Math.max(audioCompletedChunks.size, audioProgress * chunkCount, attemptedAudio);
                } else if (audioFirstMode && audioFirstSeconds != null) {
                    rateElapsedSeconds = Math.max(0, elapsedSeconds - audioFirstSeconds);
                }
                const timingEstimate = projectedRenderTiming(elapsedSeconds, observedUnits, workTotal, audioFirstMode ? NaN : fallbackEtaSeconds, rateElapsedSeconds);
                const eta = formatEta(timingEstimate.etaSeconds);
                const projected = audioPrepass ? "—" : formatEta(timingEstimate.totalSeconds);
                const metricsLine = `${chunk} · ${resolution} · ${fps} · ${inspecting}S ${displayStep}/${displayTotalSteps || "—"} · ${secondsPerStep} · E ${elapsed} · ${audioPrepass ? "ETA audio" : "ETA"} ${eta} · Est. total ${projected}`;
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
                    statusProgressFill.style.background = complete ? "#63d38a" : audioStep ? "#e5b94f" : "#65b9ff";
                    const samplingProgress = totalSteps > 0 ? Math.max(0, Math.min(1, currentStep / totalSteps)) : 0;
                    statusProgressFill.style.width = `${complete ? 100 : audioStep ? audioProgress * 100 : h3Active ? samplingProgress * 100 : 0}%`;
                    progressHelp = audioStep
                        ? `Audio teacher chunk ${audioStep[1]}/${audioStep[2]}, step ${audioStep[3]}/${audioStep[4]}.`
                        : h3Active
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
                audioCompletedChunks = new Set();
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
                // A restored previous render keeps the play button as it was; a live
                // reset is a new run and always starts playing.
                paused = restoringCache && savedPlayerState.paused !== undefined ? Boolean(savedPlayerState.paused) : false;
                timelineDragging = false;
                timelineScrub = null;
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
                audioFirstSeconds = null;
                audioStepMs = null;
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
                    if (range && typeof data.audio_teacher_prompt === "string") {
                        range.audio_teacher_prompt = data.audio_teacher_prompt;
                    }
                    if (range && Array.isArray(data.audio_teacher_attempts)) {
                        range.audio_teacher_attempts = data.audio_teacher_attempts.map(text => String(text));
                    }
                    if (range && Array.isArray(data.audio_teacher_attempt_transcripts)) {
                        range.audio_teacher_attempt_transcripts = data.audio_teacher_attempt_transcripts.map(text => String(text || ""));
                    }
                    if (range && Array.isArray(data.subchunk_silent_numbers)) {
                        range.subchunk_silent_numbers = data.subchunk_silent_numbers.map(number => Number(number));
                        range.subchunk_silent_prompt = String(data.subchunk_silent_prompt || "");
                    }
                    if (range && Array.isArray(data.subchunk_transcripts)) {
                        range.subchunk_transcripts = data.subchunk_transcripts.map(text => String(text || ""));
                    }
                    if (range && typeof data.subtitle === "string") {
                        range.subtitle = data.subtitle;
                    }
                    if (range) {
                        if (Number.isFinite(data.taomate_completed_frames) && data.taomate_completed_frames >= 0) {
                            range.taomate_completed_frames = data.taomate_completed_frames;
                            renderTransport();
                        }
                        if (Number.isFinite(data.taomate_completed_phases) && data.taomate_completed_phases >= 0) {
                            range.taomate_completed_phases = data.taomate_completed_phases;
                        }
                        for (const key of ["h3_render_seconds", "gemma_seconds", "gemma_preproduction_seconds", "chunk_total_seconds", "audio_retry_start_ms", "audio_retry_end_ms"]) {
                            const value = Number(data[key]);
                            if (Number.isFinite(value) && value >= 0) range[key] = value;
                        }
                        if (tooltipChunkIndex === index && tooltipPointer) setChunkTooltip(index, { ...tooltipPointer, shiftKey: shiftPromptVisible, ctrlKey: ctrlPromptVisible });
                    }
                    return;
                }
                if (data.action === "phase") {
                    if (typeof data.phase === "string") phase = data.phase;
                    if (Number.isFinite(data.audio_step_ms)) audioStepMs = data.audio_step_ms;
                    else if (!/audio teacher.*step/i.test(phase || "")) audioStepMs = null;
                    if (data.chunk != null) {
                        if (data.chunk !== activeChunk) activeSubchunk = null;
                        activeChunk = data.chunk;
                    }
                    renderStatus();
                    renderTransport();
                    return;
                }
                if (data.action === "audio_chunk_complete") {
                    const index = Number(data.chunk);
                    if (Number.isInteger(index) && index >= 0 && index < chunkCount) {
                        audioCompletedChunks.add(index);
                        renderTransport();
                    }
                    return;
                }
                if (data.action === "audio_first") {
                    if (typeof data.audio !== "string" || !data.audio) return;
                    audioFirstSeconds = Number.isFinite(data.elapsed_ms) ? data.elapsed_ms / 1000 : (startedAt == null ? 0 : (performance.now() - startedAt) / 1000);
                    const source = `data:${data.audio_mime || "audio/wav"};base64,${data.audio}`;
                    const fps = validFps(data.fps) || currentPlaybackFps();
                    let firstPlaceholder = -1;
                    for (let index = 0; index < chunkCount; index++) {
                        const range = chunkRanges[index] || {};
                        const start = Math.max(0, Math.round(Number(range.start) || 0));
                        const end = Math.max(start, Math.round(Number(range.end) || start));
                        const numbers = Array.from({ length: end - start + 1 }, (_, offset) => start + offset);
                        chunks[index] = {
                            frames: numbers.map(() => EMPTY_FRAME_SOURCE),
                            durations: numbers.map(() => 1000 / fps),
                            frameNumbers: numbers,
                            outputStart: start,
                            outputEnd: end,
                            sourceFps: fps,
                            finalized: false,
                            audioOnlyPlaceholder: true,
                            audioSource: source,
                            audioStartSeconds: start / fps,
                            audioEndSeconds: (end + 1) / fps,
                        };
                        if (firstPlaceholder < 0 && numbers.length) firstPlaceholder = index;
                    }
                    setSourceFps(fps);
                    renderTransport();
                    if (waitingAfterChunk != null && available(waitingAfterChunk + 1) && !paused) {
                        const next = waitingAfterChunk + 1;
                        waitingAfterChunk = null;
                        playFrameGroup(next, chunks[next], 0, playbackSerial, true);
                    } else if (firstPlaceholder >= 0 && timer == null && !framePending) show(firstPlaceholder, 0);
                    return;
                }
                if (data.action === "chunk_audio_first") {
                    if (typeof data.audio !== "string" || !data.audio) return;
                    const index = Number(data.chunk);
                    if (!Number.isInteger(index) || index < 0 || index >= chunkCount) return;
                    const oldGroup = chunks[index];
                    const source = `data:${data.audio_mime || "audio/wav"};base64,${data.audio}`;
                    if (oldGroup?.frames?.length && !oldGroup.audioOnlyPlaceholder) {
                        const fps = validFps(data.fps) || currentPlaybackFps();
                        const start = Math.max(0, Math.round(Number(data.output_start) || 0));
                        const end = Math.max(start, Math.round(Number(data.output_end) || start));
                        oldGroup.audioSource = source;
                        oldGroup.audioStartSeconds = 0;
                        oldGroup.audioEndSeconds = (end - start + 1) / fps;
                        prepareContinuousAudio(oldGroup);
                        if (waitingAfterChunk === index - 1 && !paused) {
                            waitingAfterChunk = null;
                            playFrameGroup(index, oldGroup, 0, playbackSerial, true);
                        }
                        return;
                    }
                    const fps = validFps(data.fps) || currentPlaybackFps();
                    const start = Math.max(0, Math.round(Number(data.output_start) || 0));
                    const end = Math.max(start, Math.round(Number(data.output_end) || start));
                    const numbers = Array.from({ length: end - start + 1 }, (_, offset) => start + offset);
                    chunks[index] = {
                        frames: numbers.map(() => EMPTY_FRAME_SOURCE),
                        durations: numbers.map(() => 1000 / fps),
                        frameNumbers: numbers,
                        outputStart: start,
                        outputEnd: end,
                        sourceFps: fps,
                        finalized: false,
                        audioOnlyPlaceholder: true,
                        audioSource: source,
                        audioStartSeconds: 0,
                        audioEndSeconds: (end - start + 1) / fps,
                    };
                    prepareContinuousAudio(chunks[index]);
                    renderTransport();
                    if (waitingAfterChunk === index - 1 && !paused) {
                        waitingAfterChunk = null;
                        playFrameGroup(index, chunks[index], 0, playbackSerial, true);
                    } else if (playing < 0 && timer == null && !framePending) show(index, 0);
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
                const wasAudioPlaceholder = Boolean(chunks[index]?.audioOnlyPlaceholder);
                const replacingPlayingChunk = (finalized || wasAudioPlaceholder) && index === playing;
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
                const oldGroup = chunks[index];
                if (finalized && pendingAudioSources.has(index)) {
                    restoredAudioSource = pendingAudioSources.get(index);
                    pendingAudioSources.delete(index);
                }
                // The full teacher timeline must survive both placeholder →
                // latent and latent → decoded-video replacements.
                const keptTimelineAudio = !restoredAudioSource && Boolean(oldGroup?.audioSource);
                if (keptTimelineAudio) restoredAudioSource = oldGroup.audioSource;
                // The final decoded video covers the whole chunk and is authoritative; a
                // live sub-chunk only owns its own frames and is merged in. The teacher-audio
                // placeholder is span coverage with a black picture rather than frames, so a
                // real sub-chunk replaces it instead of merging into its black frames, which
                // would put empty steps between the frames the arrow keys walk.
                const incomingFrames = encodedFrames.map(frame => `data:image/webp;base64,${frame}`);
                const incomingNumbers = Array.isArray(data.frame_numbers) ? data.frame_numbers.map(Number) : [];
                const merged = finalized
                    ? { frames: incomingFrames, durations: frameDurations, frameNumbers: incomingNumbers }
                    : mergeFrameGroup(oldGroup?.audioOnlyPlaceholder ? null : oldGroup, incomingFrames, frameDurations, incomingNumbers);
                // Audio scheduling and the running frame callback retain this
                // object. Update it in place as live video frames arrive.
                const group = Object.assign(oldGroup || {}, {
                    frames: merged.frames,
                    durations: merged.durations,
                    frameNumbers: merged.frameNumbers,
                    outputStart: merged.frameNumbers.length ? merged.frameNumbers[0] : data.output_start,
                    // The rendered end is the phase that finished, not the last image: an
                    // image stands for its whole token, and its frames are rendered too.
                    outputEnd: merged.frameNumbers.length ? Math.max(Number(data.output_end) || 0, merged.frameNumbers[merged.frameNumbers.length - 1]) : data.output_end,
                    sourceFps: data.fps,
                    step: currentStep,
                    finalized,
                    audioSource: restoredAudioSource,
                    audioStartSeconds: keptTimelineAudio ? oldGroup.audioStartSeconds : 0,
                    audioEndSeconds: keptTimelineAudio ? oldGroup.audioEndSeconds : undefined,
                    audioOnlyPlaceholder: false,
                });
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
                if (!finalized) stepPreviews[currentStep] = { ...group };
                previewWidth = data.width;
                previewHeight = data.height;
                setSourceFps(data.fps);
                renderStatus();
                renderTransport();
                redrawGraphs();
                if (restorePersistedFrame()) return;
                if (waitingAfterChunk === index - 1 && !paused) {
                    waitingAfterChunk = null;
                    playFrameGroup(index, group, 0, playbackSerial, true);
                    return;
                }
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
                    for (const chunk of snapshot.audio_chunks || []) node._hrEndlessSamplerPreview({ action: "audio_chunk_complete", execution: snapshot.execution, node_id: node.id, chunk });
                    if (snapshot.audio_first) node._hrEndlessSamplerPreview(snapshot.audio_first);
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
                    // The restored play state is only visible once the button is
                    // painted again, and a restore does not repaint it by itself.
                    renderTransport();
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
            // The tooltip takes the mouse so its scrollbar can be dragged and its text
            // selected. It therefore stays open while the pointer is on it, and closes when
            // the pointer leaves it for anything but the timeline, which owns its hover.
            chunkTooltip.setLeaveGuard(event => Boolean(event.relatedTarget && timelineShell.contains(event.relatedTarget)));
            timelineShell.addEventListener("mouseleave", () => {
                // Moving onto the tooltip hands the pointer over without closing it.
                if (chunkTooltip.isHovered()) return;
                tooltipChunkIndex = null;
                tooltipSignature = null;
                tooltipPointer = null;
                chunkTooltip.hide();
            });
            const updatePromptKeys = event => {
                if (event.key !== "Shift" && event.key !== "Control") return;
                if (event.key === "Shift") shiftPromptVisible = event.type === "keydown";
                else ctrlPromptVisible = event.type === "keydown";
                if (tooltipChunkIndex == null || tooltipPointer == null) return;
                setChunkTooltip(tooltipChunkIndex, {
                    clientX: tooltipPointer.clientX,
                    clientY: tooltipPointer.clientY,
                    shiftKey: shiftPromptVisible,
                    ctrlKey: ctrlPromptVisible,
                });
            };
            document.addEventListener("keydown", updatePromptKeys);
            document.addEventListener("keyup", updatePromptKeys);

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
                document.removeEventListener("keydown", updatePromptKeys);
                document.removeEventListener("keyup", updatePromptKeys);
                document.removeEventListener("fullscreenchange", fullscreenChanged);
                controlsAutohide.stop();
                closePreviewContextMenu();
                resizeObserver.disconnect();
                chunkTooltip.remove();
                node._hrEndlessSamplerPreview = null;
                previousRemoved?.apply(this, arguments);
            };
        };
    },
});
