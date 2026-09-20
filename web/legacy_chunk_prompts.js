const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

app.registerExtension({
    name: "HR.Endless.LegacyChunkPrompts",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HREndlessLegacyChunkPrompts") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            const editor = this.widgets.find(widget => widget.name === "chunk_prompts");
            const button = this.addWidget("button", "Generate chunk prompts", null, async () => {
                if (this.legacyBakePending) return;
                try {
                    const graph = await app.graphToPrompt();
                    const ownEntry = Object.entries(graph.output).find(([id, node]) => node.class_type === "HREndlessLegacyChunkPrompts" && (id === String(this.id) || id === String(this.getNodeLocatorId?.())));
                    if (!ownEntry) throw new Error("Could not locate this provider in the executable graph.");
                    const providerId = ownEntry[0];
                    const samplers = Object.entries(graph.output).filter(([, node]) => node.class_type === "HREndlessSampler" && String(node.inputs.pre_production?.[0]) === providerId);
                    if (samplers.length !== 1) throw new Error("Connect this provider to exactly one HR Endless Sampler before generating prompts.");
                    if (editor.value.trim() && !window.confirm("Replace the edited chunk prompts with newly generated legacy prompts?")) return;
                    const [samplerId, sampler] = samplers[0];
                    // HTTP LAN pages may lack randomUUID; this ID only correlates UI replies.
                    const requestId = globalThis.crypto?.randomUUID?.() ?? `${Date.now()}-${Math.random().toString(36).slice(2)}`;
                    const inputs = {};
                    for (const name of ["prompt", "fps", "chunk_frames", "video_continuation", "video_continuation_method"]) {
                        if (!(name in sampler.inputs)) throw new Error(`Sampler input '${name}' is missing.`);
                        inputs[name] = sampler.inputs[name];
                    }
                    // Read native H3 metadata without executing model/CLIP/VAE nodes.
                    const sources = new Set(["EmptyMiniMaxH3LatentAV", "MiniMaxH3ImageToVideo", "MiniMaxH3ReferenceToVideo"]);
                    const trace = (link, field, seen = new Set()) => {
                        if (!Array.isArray(link)) return null;
                        const id = String(link[0]);
                        if (seen.has(id)) throw new Error("Cycle in the sampler input graph.");
                        seen.add(id);
                        const node = graph.output[id];
                        if (!node) return null;
                        if (sources.has(node.class_type)) return node;
                        // Unknown latent transforms may change duration; never guess.
                        if (field === "latent" && node.class_type !== "MiniMaxH3AddGuide") return null;
                        const next = field === "latent" ? (node.inputs.latent ?? node.inputs.latent_image ?? node.inputs.samples) : (node.inputs.positive ?? node.inputs.conditioning);
                        return trace(next, field, seen);
                    };
                    const latentSource = trace(sampler.inputs.latent_image, "latent");
                    const conditionSource = trace(sampler.inputs.guider, "positive");
                    if (!latentSource || !conditionSource) throw new Error("Prompt-only generation needs traceable native H3 latent and conditioning nodes. No models were queued.");
                    inputs.length = latentSource.inputs.length;
                    const kinds = [];
                    if (conditionSource.class_type === "MiniMaxH3ReferenceToVideo") {
                        // Autogrow sockets serialize as named inputs; each image socket
                        // contributes one reference, even when its source is a batch.
                        const refs = conditionSource.inputs;
                        for (const [name, value] of Object.entries(refs)) {
                            if (value == null) continue;
                            if (/^ref_image_\d+$/.test(name)) kinds.push("image");
                            if (/^ref_video_\d+$/.test(name)) kinds.push(refs[`ref_video_audio_${name.split("_").pop()}`] != null ? "video_audio" : "video");
                            if (/^ref_audio_\d+$/.test(name)) kinds.push("audio");
                        }
                    }
                    inputs.reference_kinds = JSON.stringify(kinds);
                    inputs.target_node = providerId;
                    inputs.request_id = requestId;
                    // Queue only the prompt baker and its real input ancestors.
                    const output = { [samplerId]: { class_type: "HREndlessLegacyPromptBake", inputs } };
                    const visit = node => {
                        for (const value of Object.values(node.inputs)) {
                            if (!Array.isArray(value) || value.length !== 2 || typeof value[1] !== "number") continue;
                            const id = String(value[0]);
                            if (output[id]) continue;
                            if (!graph.output[id]) throw new Error(`Missing upstream node ${id}.`);
                            output[id] = graph.output[id];
                            visit(output[id]);
                        }
                    };
                    visit(output[samplerId]);
                    this.legacyBakePending = { requestId, originalText: editor.value };
                    button.name = "Generating chunk prompts…";
                    await api.queuePrompt(0, { output, workflow: graph.workflow });
                } catch (error) {
                    this.legacyBakePending = null;
                    button.name = "Generate chunk prompts";
                    window.alert(error.message || String(error));
                }
            });
            const receive = event => {
                const pending = this.legacyBakePending;
                if (!pending || event.detail.request_id !== pending.requestId) return;
                this.legacyBakePending = null;
                button.name = "Generate chunk prompts";
                if (editor.value !== pending.originalText && !window.confirm("The text changed while generating. Replace it with the generated prompts?")) return;
                editor.value = event.detail.text;
                editor.callback?.(editor.value);
                this.setDirtyCanvas(true, true);
            };
            const failed = () => {
                this.legacyBakePending = null;
                button.name = "Generate chunk prompts";
                this.setDirtyCanvas(true, true);
            };
            api.addEventListener("hr_endless_legacy_prompts", receive);
            api.addEventListener("execution_error", failed);
            api.addEventListener("execution_interrupted", failed);
            const removed = this.onRemoved;
            this.onRemoved = function () {
                api.removeEventListener("hr_endless_legacy_prompts", receive);
                api.removeEventListener("execution_error", failed);
                api.removeEventListener("execution_interrupted", failed);
                return removed?.apply(this, arguments);
            };
            this.size = [Math.max(this.size[0], 520), Math.max(this.size[1], 400)];
        };
    },
});
