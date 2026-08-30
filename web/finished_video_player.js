const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;


function findEndlessPlayerNode(rootGraph, qualifiedId) {
    const parts = String(qualifiedId).split(":");
    let graph = rootGraph;
    for (let index = 0; index < parts.length - 1; index++) {
        const node = graph?.getNodeById?.(Number(parts[index]));
        if (!node?.subgraph) return null;
        graph = node.subgraph;
    }
    return graph?.getNodeById?.(Number(parts[parts.length - 1])) || null;
}


const playerColors = [
    "#f4b942", "#45b7d1", "#ef6f91", "#72c472", "#a98bea",
    "#f08a4b", "#55c7b3", "#d6cf57", "#6f9ee8", "#d676d4",
];


function validEndlessFps(value) {
    const fps = Number(value);
    return Number.isFinite(fps) && fps > 0 ? fps : null;
}


function formatRenderDuration(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds < 0) return null;
    if (seconds >= 3600) {
        const hours = Math.floor(seconds / 3600);
        const minutes = Math.floor((seconds % 3600) / 60);
        return `${hours}h ${minutes}m ${(seconds % 60).toFixed(1)}s`;
    }
    if (seconds >= 60) {
        const minutes = Math.floor(seconds / 60);
        return `${minutes}m ${(seconds % 60).toFixed(1)}s`;
    }
    return `${seconds.toFixed(2)}s`;
}


function formatChunkClock(value) {
    const seconds = Number(value);
    if (!Number.isFinite(seconds) || seconds < 0) return null;
    const rounded = Math.round(seconds);
    const minutes = Math.floor(rounded / 60);
    const remainder = rounded % 60;
    return `${String(minutes).padStart(2, "0")}:${String(remainder).padStart(2, "0")}`;
}


function finishedChunkTimingLines(chunk) {
    const lines = [];
    const totalSeconds = Number(chunk.chunk_total_seconds);
    const samplerSeconds = Math.max(0, Number(chunk.h3_render_seconds) || 0);
    const gemmaSeconds = Math.max(0, Number(chunk.gemma_seconds) || 0);
    if (Number.isFinite(totalSeconds) && totalSeconds >= 0) {
        const miscSeconds = Math.max(0, totalSeconds - samplerSeconds - gemmaSeconds);
        lines.push(
            `Chunk processing: ${formatChunkClock(totalSeconds)} `
            + `( sampler:${formatChunkClock(samplerSeconds)} + gemma4:${formatChunkClock(gemmaSeconds)} `
            + `+ misc:${formatChunkClock(miscSeconds)} )`
        );
    }
    const preproduction = formatChunkClock(chunk.gemma_preproduction_seconds);
    if (preproduction && Number(chunk.gemma_preproduction_seconds) > 0) {
        lines.push(`Gemma4 preproduction included above: ${preproduction}`);
    }
    return lines;
}


function finishedShotsOverlappingChunk(chunk, shotRanges) {
    const start = Number(chunk?.start);
    const end = Number(chunk?.end);
    if (!Number.isFinite(start) || !Number.isFinite(end)) return [];
    return shotRanges
        .filter(shot => Number(shot.end) >= start && Number(shot.start) <= end)
        .sort((left, right) => Number(left.start) - Number(right.start));
}


function finishedColoredShotSegments(description, chunk, shotRanges) {
    const text = String(description || "");
    if (!text) return [];
    const overlapping = finishedShotsOverlappingChunk(chunk, shotRanges);
    const colorFor = shot => shot
        ? playerColors[(Math.max(1, Number(shot.shot) || 1) - 1) % playerColors.length]
        : null;
    const markers = [...text.matchAll(/\[Shot\s+(\d+)\]/gi)];
    if (!markers.length) {
        return [{ text, color: colorFor(overlapping[0]) || playerColors[0] }];
    }

    const segments = [];
    const hasPrefix = markers[0].index > 0;
    if (hasPrefix) {
        segments.push({
            text: text.slice(0, markers[0].index),
            color: colorFor(overlapping[0]) || playerColors[0],
        });
    }
    // H3 marker numbers are local to this physical chunk. Source-shot colors
    // are global, so associate the sections by chronological overlap. Plain
    // prose before the first marker is the continuing first source shot.
    const sequentialOffset = hasPrefix ? 1 : 0;
    for (let index = 0; index < markers.length; index++) {
        const marker = markers[index];
        const next = markers[index + 1];
        const shot = overlapping[index + sequentialOffset];
        const fallbackNumber = Number(marker[1]) || index + 1;
        segments.push({
            text: text.slice(marker.index, next ? next.index : text.length),
            color: colorFor(shot) || playerColors[(fallbackNumber - 1) % playerColors.length],
        });
    }
    return segments;
}


function createFinishedChunkTooltip() {
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
        show(event, { help, chunk, timing, description, shotRanges }) {
            tooltip.replaceChildren();
            line(help, "color:#999;margin-bottom:6px;");
            const chunkNumber = Number(chunk.chunk) || 1;
            const chunkColor = playerColors[(Math.max(1, chunkNumber) - 1) % playerColors.length] || "#fff";
            line(`Chunk ${chunkNumber}`, `color:${chunkColor};font-weight:700;`);
            if (timing.length) {
                for (const item of timing) line(item, "color:#bbb;");
            } else {
                line("No render timing was saved for this chunk.", "color:#888;");
            }
            line("Gemma detailed_description:", "color:#bbb;margin-top:7px;margin-bottom:2px;");
            if (description) {
                const prompt = document.createElement("div");
                for (const segment of finishedColoredShotSegments(description, chunk, shotRanges)) {
                    const span = document.createElement("span");
                    span.textContent = segment.text;
                    span.style.color = segment.color;
                    prompt.appendChild(span);
                }
                tooltip.appendChild(prompt);
            } else {
                line("No Gemma detailed_description was saved for this chunk.", "color:#888;");
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


function formatBrowserBytes(value) {
    let size = Number(value) || 0;
    const units = ["B", "KB", "MB", "GB", "TB"];
    let unit = 0;
    while (size >= 1024 && unit < units.length - 1) {
        size /= 1024;
        unit += 1;
    }
    return `${size >= 10 || unit === 0 ? size.toFixed(0) : size.toFixed(1)} ${units[unit]}`;
}


function setLoadVideoWidget(node, widget, path) {
    if (!widget || !path) return;
    widget.value = path;
    widget.callback?.(path);
    node.graph?.setDirtyCanvas?.(true, true);
    app.graph?.setDirtyCanvas?.(true, true);
}


async function uploadEndlessVideo(file, onProgress) {
    const chunkSize = 16 * 1024 * 1024;
    const totalChunks = Math.max(1, Math.ceil(file.size / chunkSize));
    const uploadId = globalThis.crypto?.randomUUID?.() || `${Date.now()}-${Math.random().toString(16).slice(2)}`;
    let complete = null;
    for (let index = 0; index < totalChunks; index++) {
        const start = index * chunkSize;
        const end = Math.min(file.size, start + chunkSize);
        const body = new FormData();
        body.append("upload_id", uploadId);
        body.append("filename", file.name);
        body.append("chunk_index", String(index));
        body.append("total_chunks", String(totalChunks));
        body.append("chunk", file.slice(start, end), `${file.name}.part`);
        const response = await api.fetchApi("/hr_endless_sampler_video/upload_chunk", { method: "POST", body });
        const payload = await response.json().catch(() => ({}));
        if (!response.ok) throw new Error(payload.error || `Upload failed (${response.status})`);
        complete = payload;
        onProgress?.((index + 1) / totalChunks);
    }
    if (!complete?.path) throw new Error("The upload completed without an output path.");
    return complete;
}


function openEndlessOutputBrowser(initialPath = "") {
    return new Promise(resolve => {
        const overlay = document.createElement("div");
        overlay.style.cssText = "position:fixed;inset:0;z-index:100000;display:flex;align-items:center;justify-content:center;background:rgba(0,0,0,.68);padding:24px;box-sizing:border-box;";
        const dialog = document.createElement("div");
        dialog.style.cssText = "display:flex;flex-direction:column;width:min(860px,92vw);height:min(650px,86vh);overflow:hidden;border:1px solid #555;border-radius:8px;background:#171717;color:#ddd;box-shadow:0 18px 60px #000;font:13px sans-serif;";
        overlay.appendChild(dialog);

        const header = document.createElement("div");
        header.style.cssText = "display:flex;align-items:center;gap:8px;padding:9px 10px;border-bottom:1px solid #333;background:#202020;";
        dialog.appendChild(header);
        const up = document.createElement("button");
        up.textContent = "↑ Up";
        up.style.cssText = "padding:5px 10px;border:1px solid #555;border-radius:4px;background:#292929;color:#eee;cursor:pointer;";
        header.appendChild(up);
        const refresh = document.createElement("button");
        refresh.textContent = "↻";
        refresh.title = "Refresh";
        refresh.style.cssText = up.style.cssText;
        header.appendChild(refresh);
        const location = document.createElement("div");
        location.style.cssText = "flex:1;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;padding:5px 8px;border:1px solid #383838;border-radius:4px;background:#111;font-family:ui-monospace,SFMono-Regular,Consolas,monospace;";
        header.appendChild(location);
        const sortControls = document.createElement("div");
        sortControls.style.cssText = "display:flex;gap:2px;white-space:nowrap;";
        header.appendChild(sortControls);
        const sortButtons = {};
        for (const [field, label] of [["name", "Name"], ["size", "Size"], ["date", "Date"]]) {
            const button = document.createElement("button");
            button.type = "button";
            button.textContent = label;
            button.title = `Sort by ${label.toLowerCase()}`;
            button.style.cssText = "min-width:42px;padding:4px 5px;border:1px solid #484848;border-radius:3px;background:#292929;color:#bbb;font:10px/1 sans-serif;cursor:pointer;";
            sortControls.appendChild(button);
            sortButtons[field] = button;
        }

        const list = document.createElement("div");
        list.style.cssText = "flex:1;overflow:auto;padding:6px;background:#111;";
        dialog.appendChild(list);
        const message = document.createElement("div");
        message.style.cssText = "padding:8px 10px;color:#aaa;border-top:1px solid #2d2d2d;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;";
        message.textContent = "Choose a video, an Endless EXR sequence, or a standalone EXR frame.";
        dialog.appendChild(message);

        const footer = document.createElement("div");
        footer.style.cssText = "display:flex;justify-content:flex-end;gap:8px;padding:9px 10px;border-top:1px solid #333;background:#202020;";
        dialog.appendChild(footer);
        const cancel = document.createElement("button");
        cancel.textContent = "Cancel";
        cancel.style.cssText = up.style.cssText;
        footer.appendChild(cancel);
        const choose = document.createElement("button");
        choose.textContent = "Load selected";
        choose.disabled = true;
        choose.style.cssText = "padding:5px 12px;border:1px solid #77651f;border-radius:4px;background:#675615;color:#fff;cursor:pointer;";
        footer.appendChild(choose);

        let currentPath = initialPath || "";
        let parentPath = "";
        let selected = null;
        let currentEntries = [];
        let sortField = "date";
        let sortDescending = true;

        function close(value = null) {
            document.removeEventListener("keydown", onKeyDown, true);
            overlay.remove();
            resolve(value);
        }

        function onKeyDown(event) {
            if (event.key === "Escape") {
                event.preventDefault();
                close();
            } else if (event.key === "Enter" && selected) {
                event.preventDefault();
                close(selected.path);
            }
        }

        function selectRow(row, entry) {
            for (const child of list.children) child.style.background = "transparent";
            row.style.background = "#3b3420";
            selected = entry.kind === "directory" ? null : entry;
            choose.disabled = !selected;
            message.textContent = entry.kind === "directory"
                ? `Folder: ${entry.path}`
                : `${entry.kind === "sequence" ? `${entry.frames}-frame EXR sequence` : entry.kind.toUpperCase()} · ${formatBrowserBytes(entry.size)} · ${entry.path}`;
        }

        function updateSortButtons() {
            for (const [field, button] of Object.entries(sortButtons)) {
                const active = field === sortField;
                const label = field[0].toUpperCase() + field.slice(1);
                button.textContent = active ? `${label} ${sortDescending ? "↓" : "↑"}` : label;
                button.style.background = active ? "#61531d" : "#292929";
                button.style.color = active ? "#fff" : "#bbb";
            }
        }

        function sortedEntries() {
            return [...currentEntries].sort((left, right) => {
                if ((left.kind === "directory") !== (right.kind === "directory")) return left.kind === "directory" ? -1 : 1;
                let comparison = 0;
                if (sortField === "name") comparison = String(left.name).localeCompare(String(right.name), undefined, { numeric: true, sensitivity: "base" });
                else if (sortField === "size") comparison = (Number(left.size) || 0) - (Number(right.size) || 0);
                else comparison = (Number(left.modified) || 0) - (Number(right.modified) || 0);
                if (!comparison) comparison = String(left.name).localeCompare(String(right.name), undefined, { numeric: true, sensitivity: "base" });
                return sortDescending ? -comparison : comparison;
            });
        }

        function renderEntries() {
            selected = null;
            choose.disabled = true;
            list.replaceChildren();
            if (!currentEntries.length) {
                const empty = document.createElement("div");
                empty.textContent = "No supported videos or EXR sequences in this folder.";
                empty.style.cssText = "padding:14px;color:#777;";
                list.appendChild(empty);
                return;
            }
            for (const entry of sortedEntries()) {
                const row = document.createElement("div");
                row.style.cssText = "display:grid;grid-template-columns:24px minmax(0,1fr) 95px 150px;align-items:center;gap:6px;padding:7px 8px;border-bottom:1px solid #252525;cursor:pointer;user-select:none;";
                const icon = entry.kind === "directory" ? "📁" : entry.kind === "sequence" ? "🎞" : entry.kind === "exr" ? "🖼" : "▶";
                const date = entry.modified ? new Date(entry.modified * 1000).toLocaleString() : "";
                row.innerHTML = `<span>${icon}</span><span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span><span style="text-align:right;color:#999"></span><span style="text-align:right;color:#777"></span>`;
                row.children[1].textContent = entry.name;
                row.children[2].textContent = entry.kind === "directory" ? "" : formatBrowserBytes(entry.size);
                row.children[3].textContent = date;
                row.addEventListener("click", () => selectRow(row, entry));
                row.addEventListener("dblclick", () => entry.kind === "directory" ? load(entry.path) : close(entry.path));
                list.appendChild(row);
            }
        }

        async function load(path) {
            selected = null;
            choose.disabled = true;
            list.textContent = "Loading output folder…";
            try {
                const response = await api.fetchApi(`/hr_endless_sampler_video/browse_output?path=${encodeURIComponent(path || "")}`, { cache: "no-store" });
                const payload = await response.json().catch(() => ({}));
                if (!response.ok) throw new Error(payload.error || `Could not list output folder (${response.status})`);
                currentPath = payload.path || "";
                parentPath = payload.parent || "";
                location.textContent = `output/${currentPath}`.replace(/\/$/, "") || "output";
                up.disabled = !currentPath;
                currentEntries = Array.isArray(payload.entries) ? payload.entries : [];
                renderEntries();
            } catch (error) {
                list.textContent = "";
                const failed = document.createElement("div");
                failed.style.cssText = "padding:14px;color:#ff9292;white-space:pre-wrap;";
                failed.textContent = String(error?.message || error);
                list.appendChild(failed);
            }
        }

        up.addEventListener("click", () => load(parentPath));
        refresh.addEventListener("click", () => load(currentPath));
        for (const [field, button] of Object.entries(sortButtons)) {
            button.addEventListener("click", () => {
                if (sortField === field) sortDescending = !sortDescending;
                else {
                    sortField = field;
                    sortDescending = field !== "name";
                }
                updateSortButtons();
                renderEntries();
            });
        }
        cancel.addEventListener("click", () => close());
        choose.addEventListener("click", () => selected && close(selected.path));
        overlay.addEventListener("mousedown", event => { if (event.target === overlay) close(); });
        document.addEventListener("keydown", onKeyDown, true);
        document.body.appendChild(overlay);
        updateSortButtons();
        load(currentPath);
    });
}


let closeActiveMatchingVideoDropdown = null;


function openMatchingVideoDropdown(button, value, stripCounter) {
    return new Promise(async resolve => {
        closeActiveMatchingVideoDropdown?.();
        const popup = document.createElement("div");
        popup.dataset.hrEndlessMatchingDropdown = "1";
        popup.style.cssText = "position:fixed;z-index:100001;width:min(540px,90vw);max-height:360px;overflow:auto;border:1px solid #555;border-radius:6px;background:#171717;color:#ddd;box-shadow:0 12px 36px #000;font:12px sans-serif;";
        const rect = button.getBoundingClientRect();
        popup.style.left = `${Math.max(6, Math.min(window.innerWidth - Math.min(540, window.innerWidth * .9) - 6, rect.left))}px`;
        popup.style.top = `${Math.max(6, Math.min(window.innerHeight - 366, rect.bottom + 4))}px`;
        popup.textContent = "Finding matching renders…";
        popup.style.padding = "8px";
        document.body.appendChild(popup);

        let settled = false;
        let outsideTimer = null;
        function close(path = null) {
            if (settled) return;
            settled = true;
            if (outsideTimer != null) clearTimeout(outsideTimer);
            document.removeEventListener("pointerdown", outside, true);
            document.removeEventListener("keydown", keyboard, true);
            popup.remove();
            if (closeActiveMatchingVideoDropdown === close) closeActiveMatchingVideoDropdown = null;
            resolve(path);
        }
        function outside(event) {
            if (!popup.contains(event.target) && event.target !== button) close();
        }
        function keyboard(event) {
            if (event.key === "Escape") {
                event.preventDefault();
                close();
            }
        }
        outsideTimer = setTimeout(() => document.addEventListener("pointerdown", outside, true), 0);
        document.addEventListener("keydown", keyboard, true);
        closeActiveMatchingVideoDropdown = close;

        try {
            const query = new URLSearchParams({ value: String(value || ""), strip_counter: stripCounter ? "1" : "0" });
            const response = await api.fetchApi(`/hr_endless_sampler_video/matching_output?${query}`, { cache: "no-store" });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(payload.error || `Could not list matching renders (${response.status})`);
            popup.style.padding = "0";
            popup.replaceChildren();
            const title = document.createElement("div");
            title.style.cssText = "position:sticky;top:0;padding:7px 9px;border-bottom:1px solid #333;background:#222;color:#aaa;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;font:11px ui-monospace,SFMono-Regular,Consolas,monospace;";
            title.textContent = `output/${payload.prefix || ""}* · newest first`;
            popup.appendChild(title);
            if (!payload.entries?.length) {
                const empty = document.createElement("div");
                empty.style.cssText = "padding:12px;color:#888;";
                empty.textContent = "No matching saved videos or EXR sequences.";
                popup.appendChild(empty);
                return;
            }
            for (const entry of payload.entries) {
                const row = document.createElement("button");
                row.type = "button";
                row.style.cssText = "display:grid;grid-template-columns:minmax(0,1fr) 82px 145px;gap:7px;align-items:center;width:100%;padding:7px 9px;border:0;border-bottom:1px solid #292929;background:transparent;color:#ddd;text-align:left;cursor:pointer;";
                const date = entry.modified ? new Date(entry.modified * 1000).toLocaleString() : "";
                row.innerHTML = '<span style="overflow:hidden;text-overflow:ellipsis;white-space:nowrap"></span><span style="text-align:right;color:#999"></span><span style="text-align:right;color:#777"></span>';
                row.children[0].textContent = entry.name;
                row.children[1].textContent = formatBrowserBytes(entry.size);
                row.children[2].textContent = date;
                row.addEventListener("mouseenter", () => { row.style.background = "#3a321c"; });
                row.addEventListener("mouseleave", () => { row.style.background = "transparent"; });
                row.addEventListener("click", () => close(entry.path));
                popup.appendChild(row);
            }
        } catch (error) {
            popup.style.color = "#ff9292";
            popup.textContent = String(error?.message || error);
        }
    });
}


api.addEventListener("hr_endless_sampler_saved_video", event => {
    const data = event.detail;
    const node = data?.node_id == null ? null : findEndlessPlayerNode(app.graph, data.node_id);
    node?._hrEndlessSamplerFinishedVideo?.(data);
});


app.registerExtension({
    name: "HREndlessSampler.FinishedVideoPlayer",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (!new Set(["HREndlessSamplerSaveVideo", "HREndlessSamplerLoadVideo"]).has(nodeData?.name)) return;

        const previousCreated = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            previousCreated?.apply(this, arguments);
            const node = this;
            const isLoadNode = nodeData?.name === "HREndlessSamplerLoadVideo";
            const isSaveNode = nodeData?.name === "HREndlessSamplerSaveVideo";
            const loadPathWidget = isLoadNode ? node.widgets?.find(widget => widget.name === "video") : null;
            const filenamePrefixWidget = isSaveNode ? node.widgets?.find(widget => widget.name === "filename_prefix") : null;
            const root = document.createElement("div");
            root.style.cssText = "display:flex;flex-direction:column;width:100%;height:100%;min-height:355px;background:#111;border-radius:6px;overflow:hidden;color:#ddd;font:12px sans-serif;outline:none;";
            root.tabIndex = 0;

            let selectedPathLabel = null;
            let browseButton = null;
            let uploadButton = null;
            let uploadInput = null;
            let matchingButton = null;
            if (isLoadNode || isSaveNode) {
                const picker = document.createElement("div");
                picker.style.cssText = "display:flex;align-items:center;gap:6px;box-sizing:border-box;padding:6px 8px;border-bottom:1px solid #2d2d2d;background:#1b1b1b;";
                root.appendChild(picker);
                const pickerButtonStyle = "padding:4px 8px;border:1px solid #555;border-radius:4px;background:#292929;color:#eee;cursor:pointer;white-space:nowrap;";
                if (isLoadNode) {
                    browseButton = document.createElement("button");
                    browseButton.type = "button";
                    browseButton.textContent = "Browse output…";
                    browseButton.title = "Browse videos, folders, and EXR sequences in ComfyUI's output directory.";
                    browseButton.style.cssText = pickerButtonStyle;
                    picker.appendChild(browseButton);
                    uploadButton = document.createElement("button");
                    uploadButton.type = "button";
                    uploadButton.textContent = "Upload video…";
                    uploadButton.title = "Upload a video from this computer into output/hr_endless_sampler_uploads/.";
                    uploadButton.style.cssText = pickerButtonStyle;
                    picker.appendChild(uploadButton);
                }
                matchingButton = document.createElement("button");
                matchingButton.type = "button";
                matchingButton.textContent = "Matching videos ▾";
                matchingButton.title = isLoadNode
                    ? "List videos sharing the loaded filename before its generated _number_ suffix."
                    : "List videos beginning with this node's filename_prefix.";
                matchingButton.style.cssText = pickerButtonStyle;
                picker.appendChild(matchingButton);
                selectedPathLabel = document.createElement("div");
                selectedPathLabel.style.cssText = "flex:1;min-width:0;overflow:hidden;text-overflow:ellipsis;white-space:nowrap;color:#aaa;font:11px/1.2 ui-monospace,SFMono-Regular,Consolas,monospace;";
                selectedPathLabel.textContent = isLoadNode
                    ? String(loadPathWidget?.value || "No file selected")
                    : String(filenamePrefixWidget?.value || "No filename prefix");
                selectedPathLabel.title = selectedPathLabel.textContent;
                picker.appendChild(selectedPathLabel);
                if (isLoadNode) {
                    uploadInput = document.createElement("input");
                    uploadInput.type = "file";
                    uploadInput.accept = "video/*,.mp4,.mkv,.webm,.mov,.avi,.m4v,.mpg,.mpeg";
                    uploadInput.style.display = "none";
                    picker.appendChild(uploadInput);
                }
            }

            const viewport = document.createElement("div");
            viewport.style.cssText = "position:relative;display:flex;flex:1 1 auto;min-height:250px;background:#090909;overflow:hidden;";
            root.appendChild(viewport);

            const media = document.createElement("video");
            media.style.cssText = "display:block;width:100%;height:100%;object-fit:contain;background:#090909;";
            media.preload = "metadata";
            media.playsInline = true;
            viewport.appendChild(media);

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
            playButton.title = "Play/pause (Space). Use Left/Right arrows for one frame.";
            transport.appendChild(playButton);

            const timelineHelp = "Click or drag to seek; colors identify sampler chunks";
            const timelineShell = document.createElement("div");
            timelineShell.style.cssText = "position:relative;flex:1;height:33px;cursor:pointer;touch-action:none;";
            transport.appendChild(timelineShell);
            const chunkTooltip = createFinishedChunkTooltip();

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
            transportFrame.style.cssText = "min-width:68px;text-align:right;color:#aaa;font:10px/1 ui-monospace,SFMono-Regular,Consolas,monospace;white-space:nowrap;";
            transport.appendChild(transportFrame);

            const status = document.createElement("div");
            status.style.cssText = "box-sizing:border-box;padding:7px 9px;background:#1b1b1b;white-space:nowrap;overflow:hidden;text-overflow:ellipsis;";
            status.textContent = "Waiting for a saved HR Endless Sampler render…";
            root.appendChild(status);

            let state = null;
            let timeline = null;
            let sourceFps = 24;
            let fpsWidget = null;
            let dragging = false;
            let animation = null;
            let bracketKey = null;
            let loadRequestSerial = 0;
            let tooltipChunk = null;
            let tooltipSignature = null;

            function totalFrames() {
                const declared = Number(timeline?.total_frames);
                if (Number.isFinite(declared) && declared > 0) return Math.round(declared);
                const fromDuration = Math.round((media.duration || 0) * sourceFps);
                return Math.max(0, fromDuration);
            }

            function chunks() {
                return Array.isArray(timeline?.chunks) ? timeline.chunks : [];
            }

            function shots() {
                return Array.isArray(timeline?.shots) ? timeline.shots : [];
            }

            function desiredFps() {
                // Load defaults to 0, which means use the embedded source rate.
                return validEndlessFps(fpsWidget?.value) || sourceFps;
            }

            function applyPlaybackRate() {
                media.playbackRate = Math.max(0.01, desiredFps() / Math.max(0.001, sourceFps));
            }

            function currentFrame() {
                const total = totalFrames();
                if (!total) return null;
                const frame = Math.floor(Math.max(0, media.currentTime || 0) * sourceFps + 1e-6);
                return Math.max(0, Math.min(total - 1, frame));
            }

            function containing(rangeList, frame) {
                if (frame == null) return null;
                return rangeList.find(item => frame >= Number(item.start) && frame <= Number(item.end)) || null;
            }

            function formatFrameLabel(frame) {
                if (frame == null) return "";
                const shot = containing(shots(), frame);
                const chunk = containing(chunks(), frame);
                const suffix = shot || chunk ? ` · ${shot ? `S${shot.shot}` : "S—"}/${chunk ? `C${chunk.chunk}` : "C—"}` : "";
                return `F${frame}${suffix}`;
            }

            function renderShotBrackets() {
                const total = totalFrames();
                const key = JSON.stringify([total, shots()]);
                if (key === bracketKey) return;
                bracketKey = key;
                shotBrackets.replaceChildren();
                if (!total) return;
                for (const shot of shots()) {
                    const start = Math.max(0, Math.min(total - 1, Number(shot.start) || 0));
                    const end = Math.max(start, Math.min(total - 1, Number(shot.end) || 0));
                    const shotNumber = Number(shot.shot) || 1;
                    const left = start / total * 100;
                    const width = (end - start + 1) / total * 100;
                    const color = playerColors[(shotNumber - 1) % playerColors.length];
                    const bracket = document.createElement("div");
                    bracket.style.cssText = `position:absolute;left:${left}%;width:${width}%;top:0;height:8px;box-sizing:border-box;border-left:1px solid ${color};border-right:1px solid ${color};border-bottom:1px solid ${color};opacity:.9;pointer-events:auto;`;
                    bracket.title = `Shot ${shotNumber}: frames ${start}–${Number(shot.source_end ?? end)}`;
                    const label = document.createElement("div");
                    label.style.cssText = `position:absolute;left:1px;right:1px;top:8px;height:10px;overflow:hidden;text-align:center;white-space:nowrap;text-overflow:clip;color:${color};font:8px/10px ui-monospace,SFMono-Regular,Consolas,monospace;text-shadow:0 1px 1px #000;`;
                    label.textContent = `S${shotNumber} ${start}–${Number(shot.source_end ?? end)}`;
                    bracket.appendChild(label);
                    shotBrackets.appendChild(bracket);
                }
            }

            function renderTransport() {
                const total = totalFrames();
                const stops = ["#333 0%"];
                for (let index = 0; index < chunks().length; index++) {
                    const chunk = chunks()[index];
                    const start = Math.max(0, Math.min(total, Number(chunk.start) || 0));
                    const end = Math.max(start, Math.min(total, Number(chunk.end) + 1 || 0));
                    const color = playerColors[index % playerColors.length];
                    stops.push(`#333 ${(start / Math.max(1, total)) * 100}%`, `${color} ${(start / Math.max(1, total)) * 100}%`, `${color} ${(end / Math.max(1, total)) * 100}%`, `#333 ${(end / Math.max(1, total)) * 100}%`);
                }
                timelineTrack.style.background = stops.length > 1 ? `linear-gradient(to right, ${stops.join(", ")})` : "#333";
                const frame = currentFrame();
                if (frame == null || !total) {
                    timelinePlayhead.style.display = "none";
                } else {
                    timelinePlayhead.style.left = `${(frame / Math.max(1, total - 1)) * 100}%`;
                    timelinePlayhead.style.display = "block";
                }
                renderShotBrackets();
            }

            function renderStatus() {
                const frame = currentFrame();
                const dimensions = media.videoWidth && media.videoHeight ? `${media.videoWidth}×${media.videoHeight}` : "resolution —";
                const rate = validEndlessFps(desiredFps()) || sourceFps;
                const mode = media.paused ? "Paused" : "Playing";
                const fullRenderTime = formatRenderDuration(timeline?.render_total_seconds);
                status.textContent = `${state?.title || "Saved render"}`
                    + (fullRenderTime ? ` · Total render: ${fullRenderTime}` : "")
                    + ` · ${mode} · ${dimensions} · ${Number(rate.toFixed(3))} fps · ${frame == null ? "Frame —" : formatFrameLabel(frame)}`;
                if (frame == null) {
                    frameLabel.style.display = "none";
                    transportFrame.textContent = "F—";
                } else {
                    const label = formatFrameLabel(frame);
                    frameLabel.textContent = label;
                    frameLabel.style.display = "block";
                    transportFrame.textContent = label;
                }
                playButton.textContent = media.paused ? "▶" : "❚❚";
                renderTransport();
            }

            function tick() {
                renderStatus();
                if (!media.paused && !media.ended) animation = requestAnimationFrame(tick);
                else animation = null;
            }

            function startTick() {
                if (animation == null) animation = requestAnimationFrame(tick);
            }

            function seekFrame(frame) {
                const total = totalFrames();
                if (!total) return;
                const target = Math.max(0, Math.min(total - 1, Math.round(frame)));
                media.currentTime = target / sourceFps;
                renderStatus();
            }

            function frameAtPointer(event) {
                const rect = timelineShell.getBoundingClientRect();
                if (!rect.width) return null;
                const fraction = Math.max(0, Math.min(1, (event.clientX - rect.left) / rect.width));
                return Math.round(fraction * Math.max(0, totalFrames() - 1));
            }

            function tooltipForPointer(event) {
                const frame = frameAtPointer(event);
                const chunk = containing(chunks(), frame);
                if (!chunk) {
                    tooltipChunk = null;
                    tooltipSignature = null;
                    chunkTooltip.hide();
                    return;
                }
                const description = String(chunk.gemma_detailed_description || "").trim();
                const timing = finishedChunkTimingLines(chunk);
                const signature = JSON.stringify([description, timing, shots()]);
                if (tooltipChunk === chunk && tooltipSignature === signature) {
                    chunkTooltip.move(event);
                    return;
                }
                tooltipChunk = chunk;
                tooltipSignature = signature;
                chunkTooltip.show(event, {
                    help: timelineHelp,
                    chunk,
                    timing,
                    description,
                    shotRanges: shots(),
                });
            }

            function setState(data) {
                if (!data?.media_url || !data?.timeline) return;
                state = data;
                timeline = data.timeline;
                sourceFps = validEndlessFps(data.source_fps) || validEndlessFps(timeline.fps) || 24;
                bracketKey = null;
                media.pause();
                media.removeAttribute("src");
                media.src = data.media_url;
                media.load();
                applyPlaybackRate();
                renderStatus();
            }

            async function previewSelectedPath(path, updateLoadWidget = false) {
                if (!path) return;
                const requestSerial = ++loadRequestSerial;
                if (updateLoadWidget && isLoadNode) setLoadVideoWidget(node, loadPathWidget, path);
                if (selectedPathLabel) {
                    selectedPathLabel.textContent = path;
                    selectedPathLabel.title = path;
                }
                state = null;
                timeline = null;
                bracketKey = null;
                media.pause();
                media.removeAttribute("src");
                media.load();
                status.textContent = `Loading output/${path}…`;
                renderTransport();
                try {
                    const response = await api.fetchApi("/hr_endless_sampler_video/load_preview", {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                            video: path,
                            fps: Number(fpsWidget?.value) || 0,
                            node_id: String(node.id),
                        }),
                    });
                    const payload = await response.json().catch(() => ({}));
                    if (!response.ok) throw new Error(payload.error || `Could not load video (${response.status})`);
                    if (requestSerial !== loadRequestSerial) return;
                    setState(payload);
                } catch (error) {
                    if (requestSerial !== loadRequestSerial) return;
                    const detail = String(error?.message || error);
                    status.textContent = `Could not load output/${path}: ${detail}`;
                    if (selectedPathLabel) selectedPathLabel.title = detail;
                }
            }

            node._hrEndlessSamplerFinishedVideo = setState;

            if (isLoadNode) {
                browseButton.addEventListener("click", async event => {
                    event.preventDefault();
                    event.stopPropagation();
                    const value = String(loadPathWidget?.value || "").replace(/\\/g, "/");
                    const parts = value.startsWith("/") || /^[A-Za-z]:\//.test(value)
                        ? []
                        : value.split("/").filter(Boolean);
                    if (parts.length) parts.pop();
                    const selected = await openEndlessOutputBrowser(parts.join("/"));
                    if (selected) await previewSelectedPath(selected, true);
                });
                uploadButton.addEventListener("click", event => {
                    event.preventDefault();
                    event.stopPropagation();
                    uploadInput.click();
                });
                uploadInput.addEventListener("change", async () => {
                    const file = uploadInput.files?.[0];
                    uploadInput.value = "";
                    if (!file) return;
                    browseButton.disabled = true;
                    uploadButton.disabled = true;
                    try {
                        selectedPathLabel.textContent = `Uploading ${file.name}: 0%`;
                        const result = await uploadEndlessVideo(file, fraction => {
                            selectedPathLabel.textContent = `Uploading ${file.name}: ${Math.round(fraction * 100)}%`;
                        });
                        await previewSelectedPath(result.path, true);
                    } catch (error) {
                        const detail = String(error?.message || error);
                        selectedPathLabel.textContent = `Upload failed: ${detail}`;
                        selectedPathLabel.title = detail;
                        status.textContent = `Upload failed: ${detail}`;
                    } finally {
                        browseButton.disabled = false;
                        uploadButton.disabled = false;
                    }
                });
            }

            matchingButton.addEventListener("click", async event => {
                event.preventDefault();
                event.stopPropagation();
                const source = isLoadNode ? loadPathWidget?.value : filenamePrefixWidget?.value;
                if (!String(source || "").trim()) {
                    status.textContent = isLoadNode
                        ? "Choose a video before listing matching renders."
                        : "Enter a filename_prefix before listing matching renders.";
                    return;
                }
                const selected = await openMatchingVideoDropdown(matchingButton, source, isLoadNode);
                if (selected) await previewSelectedPath(selected, isLoadNode);
            });

            playButton.addEventListener("click", event => {
                event.preventDefault();
                event.stopPropagation();
                if (media.paused) media.play().catch(error => console.warn("HR Endless Sampler video playback failed", error));
                else media.pause();
                root.focus({ preventScroll: true });
            });
            viewport.addEventListener("click", () => root.focus({ preventScroll: true }));
            timelineShell.addEventListener("pointerdown", event => {
                event.preventDefault();
                event.stopPropagation();
                dragging = true;
                timelineShell.setPointerCapture?.(event.pointerId);
                media.pause();
                seekFrame(frameAtPointer(event));
                root.focus({ preventScroll: true });
            });
            timelineShell.addEventListener("pointermove", event => {
                tooltipForPointer(event);
                if (dragging) seekFrame(frameAtPointer(event));
            });
            for (const eventName of ["pointerup", "pointercancel"]) {
                timelineShell.addEventListener(eventName, event => {
                    dragging = false;
                    timelineShell.releasePointerCapture?.(event.pointerId);
                });
            }
            timelineShell.addEventListener("mouseleave", () => {
                tooltipChunk = null;
                tooltipSignature = null;
                chunkTooltip.hide();
            });
            root.addEventListener("keydown", event => {
                if (event.key === "ArrowLeft" || event.key === "ArrowRight") {
                    event.preventDefault();
                    event.stopPropagation();
                    media.pause();
                    seekFrame((currentFrame() ?? 0) + (event.key === "ArrowLeft" ? -1 : 1));
                } else if (event.key === " " || event.code === "Space") {
                    event.preventDefault();
                    event.stopPropagation();
                    if (media.paused) media.play().catch(error => console.warn("HR Endless Sampler video playback failed", error));
                    else media.pause();
                }
            });
            for (const eventName of ["loadedmetadata", "loadeddata", "timeupdate", "seeking", "seeked", "pause", "play", "ended", "ratechange"]) {
                media.addEventListener(eventName, () => {
                    applyPlaybackRate();
                    renderStatus();
                    if (eventName === "play") startTick();
                });
            }

            node.addDOMWidget("player", "hr_endless_sampler_finished_video", root, { serialize: false });
            node.setSize([Math.max(node.size?.[0] || 480, 480), Math.max(node.size?.[1] || 420, 420)]);
            fpsWidget = node.widgets?.find(widget => widget.name === "fps");
            const previousFpsCallback = fpsWidget?.callback;
            if (fpsWidget) {
                fpsWidget.callback = function () {
                    const result = previousFpsCallback?.apply(this, arguments);
                    applyPlaybackRate();
                    renderStatus();
                    return result;
                };
            }
            const previousPrefixCallback = filenamePrefixWidget?.callback;
            if (filenamePrefixWidget) {
                filenamePrefixWidget.callback = function (value) {
                    const result = previousPrefixCallback?.apply(this, arguments);
                    selectedPathLabel.textContent = String(value || "No filename prefix");
                    selectedPathLabel.title = selectedPathLabel.textContent;
                    return result;
                };
            }

            async function restoreState(attempt=0) {
                if (node.id == null || Number(node.id) < 0) {
                    if (attempt < 20) setTimeout(() => restoreState(attempt + 1), 100);
                    return;
                }
                try {
                    const response = await api.fetchApi(
                        `/hr_endless_sampler_video/state?node_id=${encodeURIComponent(node.id)}`,
                        { cache: "no-store" },
                    );
                    if (response.ok) setState(await response.json());
                } catch (error) {
                    console.warn("HR Endless Sampler finished-video state restore failed", error);
                }
            }
            setTimeout(() => restoreState(), 0);

            const previousRemoved = node.onRemoved;
            node.onRemoved = function () {
                if (animation != null) cancelAnimationFrame(animation);
                media.pause();
                media.removeAttribute("src");
                chunkTooltip.remove();
                node._hrEndlessSamplerFinishedVideo = null;
                previousRemoved?.apply(this, arguments);
            };
        };
    },
});
