const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

function createHistorySelector(node) {
    const widget = node.widgets?.find(item => item.name === "run_id");
    if (!widget) return;
    widget.hidden = true;
    widget.computeSize = () => [0, -4];
    widget.draw = () => {};

    const root = document.createElement("div");
    root.style.cssText = "display:flex;gap:8px;align-items:center;padding:6px;color:#ddd;font:12px sans-serif";
    const label = document.createElement("span");
    label.textContent = "续写来源";
    const select = document.createElement("select");
    select.style.cssText = "min-width:260px;background:#294963;color:#fff;border:1px solid #5683a4;border-radius:5px;padding:6px";
    const refresh = document.createElement("button");
    refresh.textContent = "刷新";
    refresh.style.cssText = "background:#294963;color:#fff;border:1px solid #5683a4;border-radius:5px;padding:6px 10px";
    root.append(label, select, refresh);

    function save(value) {
        widget.value = value;
        widget.callback?.(value);
        node.setDirtyCanvas?.(true, true);
    }
    async function load() {
        const current = String(widget.value || "");
        select.replaceChildren();
        const latest = document.createElement("option");
        latest.value = "";
        latest.textContent = "当前/最近一次完整生成";
        select.append(latest);
        try {
            const response = await api.fetchApi("/hr_endless_sampler_retake/runs", { cache: "no-store" });
            const payload = await response.json();
            for (const item of payload.runs || []) {
                const option = document.createElement("option");
                option.value = item.run_id;
                option.textContent = `${item.created || item.run_id} · ${item.completed_chunks || 0} 段`;
                select.append(option);
            }
            if ([...select.options].some(option => option.value === current)) select.value = current;
            else save("");
        } catch (error) {
            latest.textContent = `历史读取失败：${error.message || error}`;
        }
    }
    select.onchange = () => save(select.value);
    refresh.onclick = load;
    node.addDOMWidget("continuation_history", "div", root, { serialize: false });
    node.setSize([Math.max(node.size[0], 380), Math.max(node.size[1], 180)]);
    load();
}

app.registerExtension({
    name: "hr-endless-sampler.continuation-history",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HREndlessContinuationCheckpoint") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            createHistorySelector(this);
        };
    },
});
