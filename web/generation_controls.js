const { app } = window.comfyAPI.app;
const { api } = window.comfyAPI.api;

const CONTROL_STYLE = `
.hr-generation-controls{display:flex;flex-wrap:wrap;gap:7px;align-items:center;padding:7px;color:#ddd;font:12px sans-serif}.hr-generation-controls button{border:1px solid #5683a4;border-radius:5px;padding:6px 10px;background:#294963;color:#fff}.hr-generation-controls button.delete{background:#733b3b;border-color:#ad6262}.hr-generation-controls button.restart{background:#6d5527;border-color:#a98945}.hr-generation-controls button:disabled{opacity:.45}.hr-generation-controls span{color:#9bb5c8}
`;

function installStyle() {
    if (document.getElementById("hr-generation-controls-style")) return;
    const style = document.createElement("style");
    style.id = "hr-generation-controls-style";
    style.textContent = CONTROL_STYLE;
    document.head.append(style);
}

function sleep(milliseconds) {
    return new Promise(resolve => setTimeout(resolve, milliseconds));
}

function createControls(node) {
    installStyle();
    const root = document.createElement("div");
    root.className = "hr-generation-controls";
    const keep = document.createElement("button");
    keep.textContent = "停止并保留进度";
    const remove = document.createElement("button");
    remove.className = "delete";
    remove.textContent = "停止并删除本次";
    const restart = document.createElement("button");
    restart.className = "restart";
    restart.textContent = "删除并从头开始";
    const status = document.createElement("span");
    status.textContent = "仅控制当前正在运行的 HR Endless Sampler";
    root.append(keep, remove, restart, status);

    function busy(value) {
        keep.disabled = value;
        remove.disabled = value;
        restart.disabled = value;
    }

    async function control(action) {
        busy(true);
        status.textContent = "正在请求停止…";
        try {
            const queue = await api.getQueue();
            const running = queue.Running || [];
            if (!running.length) throw new Error("当前没有正在生成的任务");
            const promptId = running[0]?.prompt?.[1];
            const response = await api.fetchApi("/hr_endless_sampler_retake/control", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ action }),
            });
            const payload = await response.json().catch(() => ({}));
            if (!response.ok) throw new Error(payload.error || "无法设置生成控制状态");
            await api.interrupt(promptId);
            status.textContent = action === "keep" ? "已请求停止，正在保存断点…" : "已请求停止，正在删除本次缓存…";
            if (action === "restart") {
                for (let attempt = 0; attempt < 120; attempt++) {
                    await sleep(500);
                    const current = await api.getQueue();
                    if (!(current.Running || []).some(item => item.prompt?.[1] === promptId)) {
                        status.textContent = "正在从头重新排队…";
                        await app.queuePrompt(0, 1);
                        status.textContent = "已从头重新排队";
                        return;
                    }
                }
                throw new Error("等待当前任务停止超时，请手动重新排队");
            }
        } catch (error) {
            status.textContent = `操作失败：${error.message || error}`;
        } finally {
            busy(false);
        }
    }

    keep.onclick = () => control("keep");
    remove.onclick = () => {
        if (confirm("停止生成并删除本次所有已完成分段？最近5次完整历史不会被删除。")) control("delete");
    };
    restart.onclick = () => {
        if (confirm("停止并删除本次进度，然后把当前工作流从第1段重新排队？")) control("restart");
    };
    node.addDOMWidget("generation_controls", "div", root, { serialize: false });
    node.setSize([Math.max(node.size[0], 520), Math.max(node.size[1], 220)]);
}

app.registerExtension({
    name: "hr-endless-sampler.generation-controls",
    async beforeRegisterNodeDef(nodeType, nodeData) {
        if (nodeData.name !== "HREndlessSampler") return;
        const original = nodeType.prototype.onNodeCreated;
        nodeType.prototype.onNodeCreated = function () {
            original?.apply(this, arguments);
            createControls(this);
        };
    },
});
