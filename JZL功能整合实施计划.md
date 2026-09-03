# JZL 功能整合实施计划

> 项目：ComfyUI-MiniMax-H3-Sampler-Unlimited（mickeylan fork）
> 计划日期：2026-09-03（Asia/Singapore）
> 状态：实施中
> 目标：以 HR Endless Sampler 的低显存连续分块采样为核心，整合 JZL 的剧本处理、提示词增强、素材调度、漫剧资产管理和 `ref_scale`。

---

## 架构决策记录：原生 JZL 四合一工作流

- **状态**：已接受
- **日期**：2026-09-03
- **决策**：完整迁移 JZL 的独立分段工作流，以 `[SHOT_START]...[SHOT_END]` 及其中的 `H3_PROMPT`、`SCENE_INSTRUCTION`、`VIDEO_INSTRUCTION`、`AUDIO_INSTRUCTION` 为权威接口。
- **编号所有权**：`<Picture N>`、`<Subject N>`、`<Video N>` 和 `<Audio N>` 由每个 JZL 分段及其有序 `slots` 决定，不让模型输出额外的整数映射字段。
- **状态所有权**：剧本生成和四合一解析属于规划层；素材名称到媒体槽位的确定性映射属于调度层；编码属于 Conditioning；显存管理和物理分块属于 HR Endless Sampler。
- **兼容边界**：保留 Qwen3.5/Qwen3.8 disposable worker 和现有 HR 采样能力，但不把多个 JZL 独立视频段伪装成一条全局 Storyboard JSON。
- **拒绝方案**：删除自创的 `image_subjects/shots` JSON 协议及其整数校验，不在 sampler 内解析或猜测素材名称。
- **后果**：每个 JZL 分段保持独立 `[Shot 1]` 和局部时间轴；场景、视频、音频引用按调度指令分别绑定；若需要跨段成片，由外层顺序执行与保存层组合，而不是改写段内语义。

---

## 0. 当前最高优先级：最小可用前置规划器（MVP）

当前最需要的不是一次性迁移 JZL 的全部导演台，而是先完成下面这条最短闭环：

```text
用户提供的参考图片
       +
用户故事内容
       +
用户规划的总时长/FPS
       │
       ▼
本地多模态 LLM 预制作规划
├─ 理解图片中的人物、服装、场景和道具
├─ 将故事拆成全局连续分镜
├─ 为每个镜头分配准确的开始/结束帧
├─ 决定每个镜头引用哪些图片
└─ 保留中文对白、歌词和可视文字
       │
       ▼
MiniMax H3 提示词编译
├─ subject_definitions
├─ summary
├─ retention_analysis
├─ detailed_description
├─ overall_soundscape
├─ non_diegetic_music
├─ <Picture N>/<Subject N> 引用
└─ 严格递增的 [Shot N] At MM:SS.mmm
       │
       ▼
全局 H3 prompt + 单一长 AV latent + 最小 Story Plan
       │
       ▼
现有 HR Endless Sampler
├─ 按 12GB 显存限制划分 physical chunks
├─ Qwen/Gemma 为每个 chunk 进行连续性导演
├─ Video1/Audio1 续接
└─ Replay、Preview、Timeline、Save
```

### 0.1 MVP 输入

首版只要求：

- `images`：用户提供的参考图片，按连接顺序形成 `<Picture 1>...<Picture N>`；
- `story`：用户提供的故事、剧情梗概或完整剧本；
- `duration_seconds`：计划生成的总时长；
- `fps`：默认 24；
- `director_config`：连接 `HR Qwen3.8 Director Config` 的共享配置输出；
- 可选风格、镜头密度、对白保留和自定义规则。

`HR Qwen3.8 Director Config` 的同一个输出必须同时连接 Storyboard Planner 和 HR Endless Sampler。两者共享模型、mmproj、MTP、draft tokens、reasoning effort、MoE 和 debug 设置，但每次操作仍启动独立 disposable worker，不共享常驻显存实例。

首版暂不要求：

- 参考视频调度；
- 独立音频调度；
- 完整资产库导入/导出；
- 分段视频自动保存；
- standalone llama-server；
- 一键导演台大型前端。

这些功能继续保留在后续阶段，但不阻塞 MVP。

### 0.2 MVP 输出

新增一个职责明确的节点，暂定：

```text
HR MiniMax H3 Storyboard Planner
```

输出：

1. `prompt`：可直接交给现有 H3 conditioning/HR Endless Sampler 的完整全局提示词；
2. `story_plan`：包含总帧数、图片说明、镜头帧范围和引用关系的最小 `HR_STORY_PLAN`；
3. `shot_report`：供用户检查和修改的中文分镜表；
4. `warnings`：图片未使用、故事覆盖不足、镜头时间错误、引用超限等诊断。

### 0.3 MVP 的两步 LLM 流程

不要要求模型一次同时完成图片分析、剧情拆分和最终 H3 六字段提示词。首版采用两个受验证步骤：

#### 步骤 A：多模态分镜规划

本地 VLM 接收：

- 所有参考图片；
- 图片的稳定编号；
- 故事原文；
- 总帧数、总时长和 FPS；
- 用户风格与镜头密度。

返回结构化 JSON：

```json
{
  "image_subjects": [
    {
      "picture": 1,
      "name": "女主角",
      "observable_features": "黑色短发、灰色风衣"
    }
  ],
  "shots": [
    {
      "shot": 1,
      "start_frame": 0,
      "end_frame": 72,
      "pictures": [1],
      "visual_action": "...",
      "camera": "...",
      "dialogue": "...",
      "sound": "..."
    }
  ]
}
```

程序必须验证：

- 第一镜从帧 0 开始；
- 镜头连续且无空洞、无重叠；
- 最后一镜准确结束在总帧数；
- 每个镜头至少持续合理帧数；
- `pictures` 只能引用实际输入图片；
- 中文对白保持原文；
- 分镜动作必须是可观察、可拍摄的内容。

#### 步骤 B：H3 提示词生成/编译

基于已经验证的分镜 JSON，生成 MiniMax H3 六字段提示词。时间码和引用编号尽可能由程序确定性生成，而不是让 LLM 自由计算：

- `[Shot N]` 编号由程序生成；
- `At MM:SS.mmm` 由 `start_frame / fps` 生成；
- `<Picture N>` 直接对应输入图片顺序；
- `<Subject N>` 与图片主体映射由结构化计划驱动；
- LLM 负责丰富视觉、动作、场景、镜头和声音语言；
- 程序负责最终格式、时间和引用校验。

### 0.4 与现有分块采样器的边界

MVP 不修改现有 physical chunk 算法。规划器生成的是整部目标视频的全局 prompt；现有 sampler 继续负责：

- 将完整 latent 按 `chunk_frames` 划分；
- 计算全局 Shot 与每个 chunk 的交集；
- 把全局 cut 时间改写为 chunk-local 时间；
- 让 Qwen/Gemma 根据上一块实际生成画面编写当前 chunk 描述；
- 执行 Video1/Audio1 连续参考和最终拼接。

因此，故事分镜数量不需要等于 chunk 数量，一个镜头也可以跨多个 chunks。

### 0.5 MVP 实施顺序

- [x] M1：定义和测试图片分析/分镜 JSON schema。
- [x] M2：实现总时长、FPS、总帧数和镜头边界的确定性校验。
- [x] M3：为现有 disposable local VLM worker增加通用预制作请求入口。
- [x] M4：实现图片 + 故事 + 总时长 → 分镜 JSON。
- [x] M5：实现分镜 JSON → MiniMax H3 全局提示词。
- [ ] M6：在真实 ComfyUI 中验证 prompt 能被现有 `_parse_prompt_shots` 正确解析。
- [x] M7：新增 `HR MiniMax H3 Storyboard Planner` 节点。
- [ ] M8：在 ComfyUI 中建立并保存最小工作流：Planner → H3 Conditioning → HR Endless Sampler。
- [ ] M9：无 GPU 单元测试已完成；本地 VLM capture/replay 尚待真实模型运行。
- [ ] M10：在 12GB VRAM 上完成两 chunk 端到端验证。

### 0.6 MVP 验收标准

- [ ] 输入 1–9 张图片、中文故事和总时长即可生成完整分镜。
- [ ] 所有 Shot 覆盖从 0 到总帧数的完整区间。
- [ ] 每张被引用图片的编号与 H3 conditioning 顺序一致。
- [ ] 最终 prompt 符合 MiniMax H3 六字段和 Shot 时间规则。
- [ ] 中文对白、歌词、可视文字不被翻译或改写。
- [ ] 视觉、动作和镜头描述可按当前项目契约使用英文。
- [ ] 现有 sampler 无需按故事段独立采样。
- [ ] 12GB 环境继续使用 Qwen3.6/3.8 UD-IQ2-MTP 和现有显存串行策略。
- [ ] 一个故事镜头跨越多个 physical chunks 时仍能连续生成。
- [ ] Planner 失败不会启动 H3 采样，并给出可定位的 schema/字段错误。

---

## 1. 总体原则

### 1.1 产品职责

- **JZL 能力负责“写什么、使用哪些素材”**：故事扩展、叙事分段、提示词增强、资产登记和素材调度。
- **HR Endless Sampler 负责“怎样连续生成”**：显存分块、导演观察、Video1/Audio1 续接、Replay、Timeline、预览和保存。
- JZL 的“故事段”是叙事边界，不能代替 HR 的 physical chunk。
- 整剧只执行一次 HR Endless Sampler；不得恢复为“每个故事段独立采样后拼接”。

### 1.2 两层时间结构

| 层级 | 作用 | 决定因素 |
|---|---|---|
| Story Segment / Shot | 内容、镜头和素材组织 | 剧本与导演意图 |
| Physical Chunk | 单次 H3 推理范围 | 显存、分辨率和 H3 时间网格 |

两层边界必须分别保存，并通过全局帧区间求交。

### 1.3 开发约束

- 保持旧工作流兼容；新输入尽量采用可选输入。
- 不整包复制 JZL 的大型节点或全局状态。
- 每阶段先补测试，再修改实现。
- 每个阶段单独验证、单独提交，保持可回退。
- 不覆盖或删除用户诊断文件。
- 不在工作流、日志、导出包中保存 API Key。
- 开始涉及 Gemma/MTP 的开发前，检查 llama.cpp issue `#27439` 和最新 `llama-cpp-python` release，并更新 `dependency.md`。

### 1.4 节点界面规范（强制）

所有新增节点必须保持当前 HR Endless Sampler 的界面和实现风格：

- 使用 ComfyUI V3 `comfy_api.latest.io.Schema` 和 `io.ComfyNode`；
- 节点 ID 使用 `HR...` / `HREndless...` 前缀；
- 显示名称使用简洁的 `HR ...` 英文命名，与现有四个节点一致；
- 分类优先沿用现有 `model/sampling/custom`、`image/video`，新增故事类节点统一放入一个稳定的 HR 分类；
- 参数直接使用原生 `io.String/Input/Combo/Int/Float/Boolean/Autogrow`，提供简洁 tooltip、合理默认值和 advanced 分组；
- 默认界面只显示完成主要工作所需参数，高级 LLM、MTP、MoE 和调试参数保持 advanced；
- 复用 HR 的 `director_backend`、`director_model`、`director_mmproj` 选择语言，避免第二套模型选择概念；
- 错误通过明确的 Python 异常和 ComfyUI 日志报告，不用弹窗吞错；
- 只有资产选择、媒体预览等原生 widget 无法完成的能力才添加前端 JS；
- 前端 JS 使用 HR 自己的路由命名空间，不全局 monkey-patch `app.graphToPrompt`；
- 不照搬 JZL 的 emoji 密集节点名、大型八按钮导演台、隐藏 widget 状态协议或 tkinter 文件选择器；
- 第一版使用职责单一的小节点，功能稳定后才考虑 HR 风格的一键包装节点。

界面一致性需在每个新节点验收时检查：旧工作流加载、默认 widget 值、输入顺序、节点尺寸、advanced 参数和浏览器刷新恢复。

---

## 2. 目标架构

```text
故事/剧本
   │
   ▼
HR Story Processor
├─ 故事拆解
├─ 故事扩展
├─ 风格预设
└─ 镜头规划
   │
   ▼
HR H3 Prompt Enhancer
├─ H3 六字段提示词
├─ 中文对白/歌词/可视文字保留
└─ 可视动作与镜头语言增强
   │
   ▼
HR_STORY_PLAN v1
├─ 故事段
├─ 全局镜头时间
├─ 素材清单
└─ 每段素材绑定
   │
   ▼
HR Story Asset Manager + Dispatcher
├─ 图片
├─ 参考视频及同步音轨
└─ 独立音频
   │
   ▼
HR Story Plan Compiler
├─ 全局 H3 prompt
├─ 单一长 AV latent
└─ Segment/Shot 全局帧映射
   │
   ▼
HR Endless Sampler
├─ 依据显存生成 physical chunks
├─ 选择当前 chunk 所需素材
├─ 重编号 Picture/Video/Audio 标签
├─ Gemma/Qwen chunk director
├─ Video1/Audio1 continuation
├─ Replay
└─ Timeline
   │
   ▼
Preview / Save / Load / Segment Export
```

---

## 3. LLM 运行时决策

### 3.1 第一阶段采用的方案

继续使用当前项目的 **disposable Python worker + llama-cpp-python** 作为本地 LLM 基础：

- 已支持 Qwen3.5、Qwen3.6、Qwen3.8；
- 已实现视觉 MTMD；
- 已实现 Qwen3.6/3.8 embedded MTP；
- 已实现 Qwen3.8 chat template 适配；
- 已支持 `cpu_moe` / `n_cpu_moe`；
- worker 退出是明确的 CUDA 释放边界；
- 当前 MTP 失败恢复与统计信息不会丢失。

### 3.2 llama-server 的定位

JZL 的 standalone `llama-server` 不直接替换当前 director runtime。后续作为并列、可选 backend 实现。

引入前必须补齐：

- operation-scoped 启停，禁止跨 H3 阶段驻留；
- Q8 K/V、Flash Attention、SWA、MoE offload；
- embedded MTP 和 draft token 设置；
- `response_format` 和 reasoning 参数；
- Qwen3.8 MTMD 模板等价性；
- usage、finish reason 和 MTP stats；
- 启动失败、超时和请求失败时的 `finally` 清理。

### 3.3 统一接口

新增通用协议：

```python
class LLMBackend:
    def complete(
        self,
        *,
        system_prompt: str,
        user_prompt: str,
        images=(),
        seed: int,
        max_tokens: int,
        response_format=None,
    ) -> LLMResponse:
        ...
```

返回结果至少包含：

```python
{
    "content": "...",
    "finish_reason": "...",
    "prompt_tokens": 0,
    "completion_tokens": 0,
    "backend": "qwen3.8",
    "mtp_enabled": True,
    "mtp_stats": {}
}
```

---

## 4. 核心数据格式：HR_STORY_PLAN v1

建议使用版本化 ComfyUI 自定义类型 `HR_STORY_PLAN`。

```python
{
    "version": 1,
    "fps": 24.0,
    "story_name": "故事名称",
    "style": "热血战斗",
    "segments": [
        {
            "id": "segment-001",
            "title": "竹林对决",
            "frame_start": 0,
            "frame_end": 240,
            "h3_fields": {
                "subject_definitions": "...",
                "summary": "...",
                "retention_analysis": "...",
                "detailed_description": "...",
                "overall_soundscape": "...",
                "non_diegetic_music": "..."
            },
            "shots": [
                {
                    "shot_number": 1,
                    "frame_start": 0,
                    "frame_end": 72
                }
            ],
            "assets": {
                "pictures": ["asset-character-a"],
                "videos": ["asset-motion-a"],
                "audios": ["asset-voice-a"]
            }
        }
    ],
    "assets": {
        "asset-character-a": {
            "kind": "image",
            "name": "角色A",
            "description": "孙悟空，橙色武道服",
            "path": "hr_endless_assets/image/sunwukong.png"
        }
    }
}
```

### 4.1 数据规则

- 所有运行时范围使用半开区间 `[frame_start, frame_end)`。
- Story Segment、Shot 和 Chunk 均使用全局帧坐标。
- 工作流只持久化资产元数据和相对路径，不持久化 tensor。
- JZL 的 `[SHOT_START]` 四段文本保留为导入/导出格式，不作为 sampler 内部唯一数据格式。
- schema 变更必须提升 `version` 并提供迁移函数。

---

# 5. 分阶段实施清单

## 阶段 0：准备、许可与基线

### 任务

- [x] 检查 JZL 项目 LICENSE 与当前项目 LICENSE 的兼容性（JZL 为 MIT，当前项目为 Apache-2.0；可移植，但须保留 MIT 版权和许可声明）。
- [x] 记录 JZL 代码来源、版本/commit 和需要保留的署名（来源 commit `a87b270abc1098704a0a1c023e793fd01b4ed644`，Copyright (c) 2026 wjluoxiao）。
- [ ] 保存至少三份真实 JZL 剧本输出 fixture：纯文本、图片 Ref2VA、视频+音频 Ref2VA。
- [ ] 记录当前项目完整测试基线。
- [x] 检查工作树，保留未跟踪用户诊断文件（未跟踪 `.rscode/` 未触碰）。
- [x] 重新检查 llama.cpp issue `#27439`（2026-09-03 仍为 Open，`bug-unconfirmed`，无确认修复）。
- [ ] 检查最新 `llama-cpp-python` release 及其 vendored llama.cpp commit。
- [ ] 把依赖检查结果写入 `dependency.md`。

### 验收

- [x] 代码来源和许可边界明确。
- [ ] fixture 可以稳定读取。
- [ ] 当前测试基线有明确的通过数量或环境阻断记录。

---

## 阶段 1：JZL 文本格式 characterization tests

### 新增文件

```text
story_format.py
tests/test_story_format.py
tests/fixtures/story/
```

### 迁移纯逻辑

- [x] `[SHOT_START]...[SHOT_END]` 分块解析。
- [x] `===H3_PROMPT===` 解析。
- [x] `===SCENE_INSTRUCTION===` 解析。
- [x] `===VIDEO_INSTRUCTION===` 解析。
- [x] `===AUDIO_INSTRUCTION===` 解析。
- [x] `_parse_slots` 等价行为。
- [x] `normalize_slots` 等价行为。
- [ ] H3 六字段解析和序列化。

### 明确不迁移

- ComfyUI 节点类；
- 动态端口；
- 文件弹窗；
- LLM 调用；
- 全局素材池；
- 采样和 ffmpeg。

### 测试

- [x] 单故事段。
- [x] 多故事段。
- [x] 缺失字段。
- [ ] 重复字段。
- [x] 非法调度 JSON。
- [ ] 空 slots。
- [x] 中文/英文标点差异。
- [ ] 旧文本 parse → serialize → parse 语义一致。

### 验收门槛

- [x] 当前新增的 6 个 characterization tests 全部通过。
- [x] 未修改 sampler 行为。

---

## 阶段 2：结构化 Story Plan

### 新增文件

```text
story_plan.py
tests/test_story_plan.py
```

### 任务

- [ ] 定义 `HR_STORY_PLAN` schema v1。
- [ ] 实现 schema 校验器。
- [ ] 实现旧 JZL 文本 → Story Plan。
- [ ] 实现 Story Plan → 兼容文本。
- [ ] 将段时长转换为累计全局帧区间。
- [ ] 将段内 Shot 时间转换为全局帧区间。
- [ ] 统一 FPS 舍入策略。
- [ ] 定义 H3 `17k+5` 对齐发生在哪一层。
- [ ] 定义 schema fingerprint。

### 测试

- [ ] 多段帧区间无缝衔接。
- [ ] 无重叠、无丢帧。
- [ ] 累计舍入不产生末段漂移。
- [ ] Shot 严格递增。
- [ ] 非法 schema 提供明确字段错误。
- [ ] JSON round-trip。

### 验收门槛

- [ ] 能从 fixture 生成稳定的 Story Plan。
- [ ] Story Plan 不依赖 ComfyUI 和 GPU 即可测试。

---

## 阶段 3：统一 LLM Backend

### 新增文件

```text
llm_backend.py
llm_backend_local.py
llm_backend_api.py
tests/test_llm_backend.py
```

### 任务

- [ ] 定义 `LLMBackend` protocol。
- [ ] 定义 `LLMResponse`。
- [ ] 为当前 disposable local worker 添加 adapter。
- [ ] 添加 OpenAI-compatible API adapter。
- [ ] 可选添加 Anthropic adapter。
- [ ] 可选添加 Gemini adapter。
- [ ] API 配置写入 ComfyUI user 目录。
- [ ] API Key 不进入 workflow、日志和导出数据。
- [ ] 统一 seed、max tokens、timeout 和错误类型。
- [ ] 本地 LLM 必须在 H3/CLIP/VAE 卸载后运行。

### 测试

- [ ] mock local backend 请求和响应。
- [ ] mock HTTP 正常响应。
- [ ] 401、429、timeout、坏 JSON。
- [ ] API Key 不出现在序列化结果或日志中。
- [ ] worker 非零退出码能够传播。

### 验收门槛

- [ ] 不改动现有 Gemma/Qwen director 的生成结果。
- [ ] 剧本处理器可以只依赖通用接口。

---

## 阶段 4：HR Story Processor

### 新增文件/目录

```text
story_processor.py
nodes_story.py
story_prompts/
story_styles/
tests/test_story_processor.py
```

### 从 JZL 迁移

- [ ] `presets/script.py` 的纯 prompt 构建逻辑。
- [ ] `sheding/mode_instructions.py`。
- [ ] `sheding/decompose_rules.py`。
- [ ] `sheding/h3_shot_rules.py`。
- [ ] `sheding/story_styles.py`。
- [ ] `sheding/styles/*.md`。

### 新节点

```text
HR Story Processor
```

### 支持模式

- [ ] 故事拆解。
- [ ] 故事扩展。
- [ ] 穿透生成。
- [ ] 仅提示词输出。

### 输入

- 原始故事；
- 故事名称；
- 风格；
- 目标故事段数；
- 每段时长；
- 素材说明；
- LLM backend；
- seed；
- 自定义规则。

### 输出

- `HR_STORY_PLAN`；
- 兼容 JZL 文本；
- 调试报告。

### 测试

- [ ] 所有风格模板可加载。
- [ ] 模板占位符完整。
- [ ] 固定 seed 请求稳定。
- [ ] 对白、歌词和可视文字保留原语言。
- [ ] 资产槽位不得引用未声明素材。
- [ ] 模型返回坏格式时提供明确修复/错误路径。

### 验收门槛

- [ ] 节点不连接 H3 模型也能完成 Story Plan。
- [ ] 节点不执行视频采样。

---

## 阶段 5：HR H3 Prompt Enhancer

### 新增文件

```text
prompt_enhancer.py
nodes_prompt_enhancer.py
tests/test_prompt_enhancer.py
```

### 新节点

```text
HR H3 Prompt Enhancer
```

### 任务

- [ ] 从 JZL 提取增强 prompt 构建逻辑。
- [ ] 将增强目标限制为单个 segment 的 `detailed_description`。
- [ ] 支持单段增强。
- [ ] 支持全部段批量增强。
- [ ] 单段失败时保留该段原文并聚合错误。
- [ ] 明确增强顺序：Story Processor 之后、chunk director 之前。

### 不可修改字段

- 素材绑定；
- Story Segment 帧范围；
- Shot 编号和时间；
- 对白原文；
- subject identity；
- 调度 JSON。

### 测试

- [ ] 只修改 `detailed_description`。
- [ ] marker 保持。
- [ ] 对白不翻译。
- [ ] 调度数据不变。
- [ ] 多段顺序不变。
- [ ] 部分失败不破坏其他段。

### 验收门槛

- [ ] 增强后的计划仍通过 Story Plan 校验。

---

## 阶段 6：资产 Manifest 和资产管理器

### 新增文件

```text
story_assets.py
nodes_assets.py
web/story_assets.js
tests/test_story_assets.py
```

### 新节点

```text
HR Story Asset Manager
```

### 功能

- [ ] 图片上传和预览。
- [ ] 视频上传和播放。
- [ ] 音频上传和试听。
- [ ] 资产启用/禁用。
- [ ] 自定义名称、类型和说明。
- [ ] 资产库导入/导出。
- [ ] 重名文件自动编号。
- [ ] 使用 ComfyUI input 下的相对路径。

### 目录建议

```text
input/hr_endless_assets/image/
input/hr_endless_assets/video/
input/hr_endless_assets/audio/
```

### 禁止迁移的 JZL 状态

```python
JZL_ASSET_POOL
JZL_BUS_POOL
JZL_SLOT_MAP
```

运行时状态应通过 Story Plan、显式输入或 execution-scoped cache 传递。

### 安全测试

- [ ] 拒绝 `..` 路径。
- [ ] 拒绝 input/output 根目录外路径。
- [ ] 符号链接逃逸测试。
- [ ] 扩展名白名单。
- [ ] 上传大小限制。
- [ ] 导入 schema/version 校验。
- [ ] API Key 不得进入资产导出。
- [ ] 原子写入配置。
- [ ] Windows/POSIX 路径 round-trip。

### 验收门槛

- [ ] 两个并发 workflow 不共享或串用资产 tensor。
- [ ] 工作流可在另一台机器通过相对路径恢复。

---

## 阶段 7：确定性素材调度器

### 新增文件

```text
story_dispatch.py
nodes_dispatch.py
tests/test_story_dispatch.py
```

### 新节点

```text
HR Story Dispatcher
```

### 匹配优先级

1. stable asset ID；
2. 完整槽位名；
3. 完整资产名；
4. 带警告的兼容模糊匹配。

### 输出结构

```python
ChunkAssetBinding(
    pictures=[...],
    videos=[...],
    video_audios=[...],
    audios=[...],
)
```

### 任务

- [ ] 场景、角色、道具图片调度。
- [ ] 参考视频调度。
- [ ] 视频同步音轨配对。
- [ ] 独立音频调度。
- [ ] 未知、重复和歧义槽位诊断。
- [ ] 按 Story Segment 生成素材绑定。
- [ ] 支持 physical chunk 与多个 segment 求交。
- [ ] 按首次出现顺序合并，按 asset ID 去重。

### H3 上限

- 图片 ≤ 9；
- 视频 ≤ 3；
- 独立音频 ≤ 3；
- 总引用数量遵守实际 H3 约束。

首版遇到跨段合并超限时必须明确报错，不做静默裁剪。

### 测试

- [ ] 顺序稳定。
- [ ] 同名不同 ID 不误合并。
- [ ] 相同 ID 跨段去重。
- [ ] 模糊匹配发出警告。
- [ ] 视频和同步音轨保持配对。
- [ ] 超限错误包含 chunk、segment 和素材信息。

### 验收门槛

- [ ] 相同输入始终生成相同素材绑定和标签顺序。

---

## 阶段 8：`ref_scale` 面积倍率

### 新增/调整

- [ ] 提取统一参考尺寸策略函数。
- [ ] `ref_scale` 范围 1.0–5.0，默认 1.0。
- [ ] 仅 `ref_image_size == "match"` 时生效。
- [ ] `max` 模式忽略 `ref_scale`。
- [ ] 禁止放大原始小图。
- [ ] 宽高对齐 32。
- [ ] tokenizer presentation 与 H3 latent 使用同一目标尺寸。
- [ ] replay fingerprint 包含 `ref_scale`。
- [ ] 日志输出实际参考尺寸和估算成本。

### 公式

```python
scale = min(
    1.0,
    math.sqrt(ref_scale * target_width * target_height / source_area),
)
```

### 测试

- [ ] `1.0` 与当前行为等价。
- [ ] 1–5 单调增大。
- [ ] 永不放大原图。
- [ ] 32 对齐。
- [ ] 极窄图、极小图。
- [ ] `max` 模式不受影响。
- [ ] token presentation 和 latent 尺寸一致。

### 验收门槛

- [ ] 旧工作流默认值的 conditioning 不变。

---

## 阶段 9：HR Story Plan Compiler

### 新增文件

```text
story_compiler.py
nodes_story_compiler.py
tests/test_story_compiler.py
```

### 新节点

```text
HR Story Plan Compiler
```

### 输出

- 全局 H3 prompt；
- 单一长 nested AV latent；
- 验证后的 `HR_STORY_PLAN`；
- 编译诊断报告。

### 任务

- [ ] 计算整剧总帧数。
- [ ] 创建单一长 video/audio latent。
- [ ] 将段内镜头转换为全局 `[Shot N]`。
- [ ] 保存 Story Segment/Shot 全局帧边界。
- [ ] 保存每段素材绑定。
- [ ] 生成供 sampler 使用的计划 fingerprint。

### 禁止行为

- 不输出 `positive[]`；
- 不输出 `latent[]`；
- 不逐段调用 sampler；
- 不逐段拼接独立生成的视频。

### 测试

- [ ] 单段和多段。
- [ ] Story Segment 跨多个 physical chunks。
- [ ] 一个 physical chunk 跨多个 Story Segments。
- [ ] 总 AV shape 精确。
- [ ] 全局 Shot 时间严格递增。

### 验收门槛

- [ ] 一部多段故事最终只需要一次 HR sampler execution。

---

## 阶段 10：HR Endless Sampler 接入 Story Plan

### 节点变化

为 `HREndlessSampler` 添加可选输入：

```text
story_plan: HR_STORY_PLAN
```

### 兼容要求

- [ ] 未连接 `story_plan` 时，旧 workflow schema 和结果不变。
- [ ] 原 `prompt`、`images`、`source_images` 路径继续可用。
- [ ] 不改变现有输出顺序。

### Story Plan 模式任务

- [ ] 使用计划中的全局 prompt 和 shot ranges。
- [ ] 每个 physical chunk 与 segments/shots 求交。
- [ ] 只加载当前 chunk 所需素材。
- [ ] 为当前 chunk 重建连续的 `<Picture N>/<Video N>/<Audio N>`。
- [ ] 同步改写 `subject_definitions`。
- [ ] 同步改写 `retention_analysis`。
- [ ] 同步改写 `detailed_description`。
- [ ] tokenizer presentation 与 `minimax_refs` 顺序一致。
- [ ] 基础 refs 完成后追加现有 Video1/Audio1 continuation。
- [ ] Story Plan 和资产 fingerprint 写入 replay。

### 实现约束

不要把所有故事素材一次编码进原始 conditioning。这样会：

- 增加所有 chunk 的 token 和显存；
- 将无关素材注入当前 chunk；
- 导致引用编号冲突；
- 破坏 12GB 目标。

### 测试

- [ ] 当前 chunk 不加载无关素材。
- [ ] 标签、tokenizer 和 latent refs 顺序完全一致。
- [ ] continuation 标签不与基础素材冲突。
- [ ] 跨段 chunk 素材去重。
- [ ] 修改素材或 Story Plan 会使 replay fingerprint 失效。
- [ ] 旧 sampler 全套测试通过。

### 验收门槛

- [ ] 12GB 环境中仍是 operation-local director 和 H3 严格串行。
- [ ] 不长期保留全剧素材 tensor 或 refs。

---

## 阶段 11：Timeline、Preview 和 Replay 扩展

### Timeline 三层结构

```text
Story Segment
Source Shot
Physical Chunk
```

### 任务

- [ ] Timeline 增加 `segments`。
- [ ] 每个 chunk 记录相交 segment IDs。
- [ ] 每个 chunk 记录实际使用素材摘要。
- [ ] Replay 记录 Story Plan schema/version/fingerprint。
- [ ] Preview 增加故事段视觉层。
- [ ] Hover 显示故事段、素材、最终提示词和耗时。
- [ ] 浏览器刷新后恢复三层时间线。

### 测试

- [ ] Segment/Shot/Chunk 边界独立且一致。
- [ ] 保存/加载后结构不丢失。
- [ ] 旧 Timeline 能迁移或正常缺省。

### 验收门槛

- [ ] Preview、Save 和 Load 对同一 Timeline 的显示一致。

---

## 阶段 12：按故事段导出视频

JZL 的分段保存能力放在输出层实现，不放在采样层。

### 任务

- [ ] 完整视频照常保存。
- [ ] 可选根据 Timeline 的 Segment 边界导出子视频。
- [ ] 子视频音频保持同步。
- [ ] 每段保存 prompt/timeline sidecar。
- [ ] 支持仅导出指定 segment。
- [ ] 中途导出失败不损坏完整主视频。

### 原则

```text
一次连续生成 → 完成后按故事段切分保存
```

禁止：

```text
每段独立生成 → 再拼接
```

### 验收门槛

- [ ] 子视频边界与 Segment 帧范围一致。
- [ ] 合并主视频仍保持原连续结果。

---

## 阶段 13：可选 standalone llama-server backend

此阶段不阻塞前述功能。

### 任务

- [ ] 借鉴 JZL 安装器的固定 runtime、SHA256 和原子安装。
- [ ] 增加 `llm_backend_server.py`。
- [ ] 每次 operation 启停 server。
- [ ] 增加端口竞争和残留进程处理。
- [ ] 完整映射当前 Qwen/Gemma 运行参数。
- [ ] 对 Qwen3.8 text、MTMD、MTP 分别做等价性测试。
- [ ] Windows CUDA 和 Linux CUDA 分别验证。

### 验收门槛

- [ ] 同一 fixture 的结构化输出通过相同 validator。
- [ ] server 退出后显存回落至可测基线。
- [ ] 不降低当前 MTP fallback 的可靠性。

---

## 阶段 14：文档、示例和一键导演台

### 任务

- [ ] 更新 README。
- [ ] 更新中文详细手册。
- [ ] 增加 Story workflow 示例。
- [ ] 增加 12GB 推荐参数。
- [ ] 记录经过验证的 Qwen3.6/3.8 UD-IQ2-MTP 文件组合。
- [ ] 增加从 JZL 文本导入的迁移说明。
- [ ] 所有独立节点稳定后，再开发可选“一键导演台”包装节点。

### 验收门槛

- [ ] 用户可以只按文档完成安装、模型配置、资产导入和完整生成。

---

# 6. 推荐新增节点

| 节点 | 职责 |
|---|---|
| `HR Story Processor` | 故事拆解、扩展和镜头规划 |
| `HR H3 Prompt Enhancer` | 增强 `detailed_description` |
| `HR Story Asset Manager` | 管理图片、视频、视频音轨和独立音频 |
| `HR Story Dispatcher` | 将素材确定性绑定到故事段 |
| `HR Story Plan Compiler` | 编译单一长视频任务 |
| `HR Endless Sampler` | 低显存连续 physical chunk 采样 |
| `HR Endless Sampler Preview` | 展示 Segment/Shot/Chunk |
| `HR Endless Sampler Save Video` | 保存全片和按故事段导出 |

首版采用职责清晰的多个节点；稳定后再做一键包装节点。

---

# 7. 端到端验收矩阵

## 7.1 纯文本中文故事

- [ ] 至少 3 个 Story Segments。
- [ ] 每段包含多个 Shots。
- [ ] 使用 Qwen3.8。
- [ ] 12GB VRAM。
- [ ] 至少一个 Story Segment 跨多个 physical chunks。
- [ ] 中文对白保留，H3 视觉/动作/镜头描述符合项目语言契约。

## 7.2 图片 Ref2VA

- [ ] 不同 Segment 使用不同角色和场景图。
- [ ] physical chunk 跨 Segment。
- [ ] 当前 chunk 仅加载相关图片。
- [ ] 图片标签连续重编号。
- [ ] 对比 `ref_scale=1.0` 与 `2.0`。

## 7.3 视频和音频参考

- [ ] 参考视频及同步音轨。
- [ ] 独立人物音频。
- [ ] `<Video N>/<Audio N>` 编号正确。
- [ ] continuation 的 Video1/Audio1 编号无冲突。

## 7.4 Replay

- [ ] 中断后自动恢复。
- [ ] 修改一个后续 Segment，不重做不受影响的前缀。
- [ ] 替换素材文件使 fingerprint 失效。
- [ ] 切换 Qwen3.6/3.8 不复用旧导演文本。
- [ ] 修改 `ref_scale` 使相关缓存失效。

## 7.5 输出

- [ ] 完整视频连续。
- [ ] Segment 子视频边界正确。
- [ ] 音频同步。
- [ ] Preview/Save/Load 显示同一 Segment/Shot/Chunk 信息。

---

# 8. 风险登记

| 风险 | 严重度 | 应对 |
|---|---:|---|
| 动态素材引用与 H3 refs 顺序不一致 | 高 | 单一 binding 对象同时驱动 prompt、tokenizer 和 latent refs；强测试 |
| 跨 Segment chunk 引用超限 | 高 | 首版明确报错，不静默删素材 |
| 两套 LLM runtime 同时占用 GPU | 高 | 所有本地 LLM operation-scoped，进入 H3 前强制退出 |
| JZL 全局资产池并发串料 | 高 | 不迁移全局池，使用显式 manifest/execution state |
| Prompt Enhancer 和 chunk director 互相覆盖 | 中 | 固定执行顺序和字段修改权限 |
| 秒到帧累计舍入漂移 | 中 | 统一累计边界算法，以全局帧为权威 |
| UI 单节点过度复杂 | 中 | 首版多个独立节点，最后再包装 |
| llama.cpp MTP 原生 abort | 高 | 保留 disposable worker 和 operation-local non-MTP fallback |
| 外部 API Key 泄露 | 高 | 仅存 user 配置，日志/工作流/导出全面脱敏 |
| JZL 代码无测试 | 高 | 先做 characterization tests，再迁实现 |

---

# 9. 推荐提交拆分

1. `test: add JZL story format characterization fixtures`
2. `feat: add structured HR story plan`
3. `feat: add reusable LLM backend protocol`
4. `feat: add HR story processor`
5. `feat: add H3 prompt enhancer`
6. `feat: add story asset manifest and manager`
7. `feat: add deterministic story asset dispatcher`
8. `feat: add reference area scaling`
9. `feat: add story plan compiler`
10. `feat: connect story plan to chunk sampler`
11. `feat: expose story segments in timeline and preview`
12. `feat: export timeline segments as individual videos`
13. `feat: add optional operation-scoped llama-server backend`
14. `docs: document integrated story workflow`

每个提交必须在修改下一个功能前完成对应验证。

---

# 10. 第一个开发里程碑

第一里程碑只建立不采样的数据链：

```text
中文故事
  → HR Story Processor
  → HR H3 Prompt Enhancer
  → HR Story Asset Manager
  → HR Story Dispatcher
  → HR Story Plan Compiler
  → 全局 H3 prompt + HR_STORY_PLAN
```

第一里程碑完成前不修改 sampler 的核心采样循环。

### 里程碑验收

- [ ] 能解析真实 JZL 输出。
- [ ] 能生成经过校验的 Story Plan。
- [ ] 能登记和调度图片/视频/音频素材。
- [ ] 能生成全局 Shot 时间和完整长 latent 规格。
- [ ] 不加载 H3 也能完成全部计划阶段。
- [ ] 无 GPU 环境可运行核心纯逻辑测试。

第一里程碑通过后，再执行阶段 10，将 Story Plan 接入 HR Endless Sampler。

---

# 11. 执行记录

后续每完成一项，在本节追加：

```text
日期：YYYY-MM-DD
阶段：阶段 N
提交：<commit>
变更：<摘要>
验证：<命令、退出码、测试数量或断言>
风险/遗留：<内容>
```

当前记录：

```text
日期：2026-09-03
阶段：规划
变更：建立 JZL 功能整合实施计划
状态：已开始阶段 0 和阶段 1

日期：2026-09-03
阶段：阶段 0 / 阶段 1（部分完成）
提交：未提交
变更：确认 MIT→Apache-2.0 移植边界；新增 story_format.py，移植 JZL 四段格式、故事块、slots 解析与规范化纯逻辑；新增测试。
验证：python tests/test_story_format.py，退出码 0，Ran 6 tests，OK；python -m compileall -q story_format.py tests/test_story_format.py，退出码 0；git diff --check，退出码 0。
风险/遗留：真实 JZL fixture、六字段序列化、重复字段和 round-trip 测试尚未完成；GitHub latest release API 本次触发 rate limit，仍需确认最新 llama-cpp-python release 并更新 dependency.md。

日期：2026-09-03 16:59（Asia/Singapore）
阶段：MVP M1-M7（代码完成，等待真实 ComfyUI 验证）
提交：未提交
变更：新增 HR MiniMax H3 Storyboard Planner；扩展 Qwen disposable worker 的 storyboard operation；按 H3 网格对齐总帧数；验证连续分镜和图片编号；程序化编译六字段 H3 prompt；注册节点。
验证：python tests/test_story_format.py，退出码 0，Ran 10 tests，OK；python tests/test_qwen35.py，退出码 0，Ran 26 tests，OK；python -m compileall -q __init__.py storyboard.py story_format.py qwen35.py，退出码 0；git diff --check，退出码 0。
风险/遗留：普通工具 Python 缺少 folder_paths，test_director_backend.py 被环境依赖阻断；尚未使用真实 Qwen GGUF/mmproj 执行 Planner，也未完成 Planner→H3 conditioning→两 chunk GPU 测试。

日期：2026-09-03 16:59（Asia/Singapore）
阶段：统一 LLM 调度
提交：未提交
变更：新增 HR Qwen3.8 Director Config 自定义配置节点；Storyboard Planner 改为必须接收共享配置；HR Endless Sampler 增加可选 director_config 输入并在连接时覆盖旧导演 widgets。Planner 和 chunk director 统一使用同一 Qwen3.8 GGUF/mmproj/MTP/reasoning/MoE/debug 配置，同时保留操作级 disposable worker 和显存释放。
验证：python -m compileall -q __init__.py director_config.py storyboard.py story_format.py qwen35.py nodes.py，退出码 0；python tests/test_story_format.py，Ran 10 tests，OK；python tests/test_qwen35.py，Ran 26 tests，OK；git diff --check，退出码 0。
风险/遗留：尚未在真实 ComfyUI 中确认 HR_DIRECTOR_CONFIG 端口显示和旧 sampler widget 覆盖行为；需要用实际 Qwen3.8 进行 Planner→Conditioning→Sampler 两 chunk 测试。

日期：2026-09-03 17:12（Asia/Singapore）
阶段：MVP 单次素材接线
提交：未提交
变更：新增 HR MiniMax H3 Reference Set 和 HR MiniMax H3 Reference Conditioning；Reference Set 打包 H3 规定的最多 9 图、3 视频、3 视频同步音轨、3 独立音频及 ref_scale；Conditioning 按官方 Ref2VA 顺序构建 tokenizer media 和 minimax_refs，并透传 Reference Set；Planner 改为从 Reference Set 读取图片；Sampler 增加可选 Reference Set 输入并可重建图片/视频/音频语义 presentation。
验证：python -m compileall -q __init__.py director_config.py storyboard.py story_format.py qwen35.py nodes.py reference_set.py，退出码 0；python tests/test_story_format.py，Ran 10 tests，OK；python tests/test_qwen35.py，Ran 26 tests，OK；git diff --check，退出码 0。
风险/遗留：Planner 当前只视觉分析 Reference Set 中的图片，视频/音频仍会进入 H3 Conditioning 和 Sampler，但尚未送入分镜 VLM；需在真实 ComfyUI 验证 Autogrow、audio_vae、不同尺寸图片和视频+同步音轨的实际协议；视频音轨首版按连接顺序与视频一一配对。
```
