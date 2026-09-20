// Run with: node tests/test_legacy_chunk_prompts.js
const assert = require('node:assert/strict');
const fs = require('node:fs');
const vm = require('node:vm');
const handlers = new Map();
let extension, queued;
const graph = {
    output: {
        1: {class_type: 'HREndlessLegacyChunkPrompts', inputs: {chunk_prompts: ''}},
        2: {class_type: 'MiniMaxH3ReferenceToVideo', inputs: {length: 73, ref_image_1: ['5', 0], ref_video_1: ['5', 1], ref_video_audio_1: ['5', 2], clip: ['5', 3]}},
        5: {class_type: 'ExpensiveModelAndMedia', inputs: {}},
        6: {class_type: 'CFGGuider', inputs: {positive: ['2', 0], model: ['5', 3]}},
        3: {class_type: 'HREndlessSampler', inputs: {pre_production: ['1', 0], guider: ['6', 0], latent_image: ['2', 1], prompt: 'source prompt', fps: 24, chunk_frames: 124, video_continuation: 39, video_continuation_method: 'Masked AV overlap'}},
        4: {class_type: 'Save', inputs: {images: ['3', 4]}},
    }, workflow: {},
};
const api = {
    addEventListener: (name, fn) => handlers.set(name, fn),
    removeEventListener: name => handlers.delete(name),
    queuePrompt: async (_, data) => { queued = data; },
};
const app = {registerExtension: value => { extension = value; }, graphToPrompt: async () => graph};
// Reproduce an HTTP LAN browser without crypto.randomUUID.
vm.runInNewContext(fs.readFileSync('web/legacy_chunk_prompts.js', 'utf8'), {window: {comfyAPI: {app: {app}, api: {api}}, confirm: () => true, alert: error => { throw Error(error); }}, crypto: {}});
(async () => {
    function Node() { this.id = 1; this.size = [100, 100]; this.widgets = [{name: 'chunk_prompts', value: ''}]; }
    Node.prototype.addWidget = function (type, name, value, callback) { const widget = {name, callback}; this.widgets.push(widget); return widget; };
    Node.prototype.setDirtyCanvas = () => {};
    await extension.beforeRegisterNodeDef(Node, {name: 'HREndlessLegacyChunkPrompts'});
    const node = new Node();
    node.onNodeCreated();
    await node.widgets[1].callback();
    assert.equal(queued.output['3'].class_type, 'HREndlessLegacyPromptBake');
    assert.equal(queued.output['4'], undefined);
    assert.equal(queued.output['1'], undefined);
    assert.equal(queued.output['2'], undefined);
    assert.equal(queued.output['5'], undefined);
    assert.equal(queued.output['6'], undefined);
    assert.equal(queued.output['3'].inputs.length, 73);
    assert.deepEqual(JSON.parse(queued.output['3'].inputs.reference_kinds), ['image', 'video_audio']);
    const requestId = queued.output['3'].inputs.request_id;
    assert.ok(typeof requestId === 'string' && requestId.length > 0);
    handlers.get('hr_endless_legacy_prompts')({detail: {request_id: requestId, text: 'generated prompts'}});
    assert.equal(node.widgets[0].value, 'generated prompts');
    assert.equal(node.legacyBakePending, null);
    node.onRemoved();
    assert.equal(handlers.size, 0);
    console.log('Legacy prompt button check passed.');
})().catch(error => { console.error(error); process.exitCode = 1; });
