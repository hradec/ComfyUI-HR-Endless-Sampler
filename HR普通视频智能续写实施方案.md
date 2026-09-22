# HR MiniMax H3 普通视频智能续写实施方案

> 制定时间：2026-09-12 18:43（Asia/Singapore）
> 状态：实施前方案
> 目标项目：`ComfyUI-MiniMax-H3-Sampler-Unlimited`

## 一、目标

新增一个只服务于 MiniMax H3 的独立节点：

```text
HR MiniMax H3 Video Continuation
```

用户指定任意本地普通视频、新续拍意图和可选的新参考媒体。节点解码源视频结尾，使用现有本地 Qwen/Gemma 多模态导演分析最后 22 帧的真实画面状态，再把模型分析、新提示词和参考媒体编译为 H3 continuation conditioning。后续采样必须从源视频结尾自然接续，不能只根据用户文字猜测。

本功能不支持 LTX、Wan 或其他模型，不修改 ComfyUI 核心，不建立第二套 replay/checkpoint 生命周期。

## 二、必须满足的行为

1. 读取用户明确选择的视频，而不是最近一次 HR replay。
2. 从视频末尾提取最多 22 个连续原始时间帧；不足 22 帧时使用全部可用帧。
3. LLM 必须实际接收这些尾帧的视觉信息，并输出结构化末态分析。
4. 分析至少包含：
   - 每个可见人物或主体的身份线索、位置、朝向、姿态和动作阶段；
   - 关键物体、遮挡、接触关系和空间拓扑；
   - 镜头景别、角度、焦段感、运动方向及运动是否仍在继续；
   - 光线方向、色温、天气、时间和背景动态；
   - 最后一帧的可见状态，以及下一动作合理起点；
   - 源视频有音频时的声音事件、对白状态和环境声延续要求。
5. H3 提示词必须把源视频末态作为第一约束，把用户的新意图作为后续动作目标。
6. 新参考图片只能补充或锁定身份、服装、物体和风格，不能无故覆盖尾帧中已确定的姿态、位置和动作相位。
7. 使用 H3 原生 video keyframe/continuation latent，不把 22 张图片当作普通 Ref2VA 图片。
8. 源视频原声存在时支持尾部音频 continuation；没有音频时走明确的无源音频路径。
9. 输出续拍片段，并提供可选的原视频与续拍片段拼接信息；不在预处理节点中偷偷保存最终视频。
10. 分析失败、结构缺失或尾帧不可解码时明确报错，不用静态模板伪造末态。

## 三、架构选择

### 3.1 不复制完整 Sampler

不新增第二份 H3 sampling loop。新节点负责普通视频导入、末态分析和 continuation 数据准备，现有 `HR Endless Sampler` 继续拥有：

- H3 physical chunk 采样；
- Qwen/Gemma worker 生命周期；
- reference conditioning；
- replay、retake、历史和中断恢复；
- timeline、preview 和输出拼接。

### 3.2 新增显式数据类型

新增：

```text
HR_H3_EXTERNAL_CONTINUATION
```

对象只在一次工作流执行中传递，不持久保存大 tensor，不写全局缓存。建议结构：

```python
{
    "type": "HR_H3_EXTERNAL_CONTINUATION",
    "version": 1,
    "source": {
        "path": "受控解析后的本地视频标识",
        "fps": 24.0,
        "frame_count": 240,
        "width": 1920,
        "height": 1080,
        "duration": 10.0,
    },
    "tail": {
        "requested_frames": 22,
        "effective_frames": 22,
        "images": tail_images,
        "audio": tail_audio_or_none,
        "analysis_images": sampled_analysis_images,
    },
    "director": {
        "backend": "qwen3.5",
        "analysis": structured_end_state,
        "continuation_prompt": final_h3_prompt,
        "raw_response": raw_response,
    },
    "reference_set": normalized_reference_set_or_none,
}
```

`images`、`audio` 是执行期数据；如果未来写入 workflow JSON，必须只序列化小型文本字段，绝不能序列化 tensor。

## 四、节点接口

### 4.1 节点名称

```text
HR MiniMax H3 Video Continuation
```

节点分类：

```text
model/sampling/custom
```

### 4.2 输入

按现有接口风格建议：

1. `video`
   - 使用 ComfyUI `VIDEO` 输入，优先复用 `HREndlessSamplerLoadVideo` 或标准视频加载节点。
   - 不接受未经 containment 检查的任意字符串路径。
2. `clip`
   - H3 Qwen3-VL CLIP，用于最终 H3 conditioning。
3. `vae`
   - MiniMax H3 Video VAE，用于尾帧编码。
4. `audio_vae`
   - 可选；源视频有音频且用户选择延续音频时需要。
5. `prompt`
   - 用户的新续拍意图，多行 STRING。
6. `length`
   - 新续拍片段目标帧数；使用 H3 合法网格，默认 124，步长 17，最小 5。
7. `reference_set`
   - 可选 `HR_MINIMAX_H3_REFERENCE_SET`，复用现有参考图片、视频和音频。
8. `director_backend`
   - 复用现有 `gemma4`、`qwen3.5`、`qwen3.8`选项及模型选择规则。
9. `director_model`、`director_mmproj`、`director_config`
   - 与 HR Endless Sampler 当前导演配置一致，不创造平行配置。
10. `tail_frames`
    - 默认 22；首版固定允许 H3 已知安全值 5/22，不把任意值伪装成有效 continuation。
11. `audio_mode`
    - `continue`：延续源视频尾部音频；
    - `mute`：新段以静音条件开始；
    - `new`：只使用 `reference_set` 中的新音频参考。
12. `audio_continuation_frames`
    - 默认 24；0 表示跟随兼容旧行为。遵循《HR整合H3MotionContext实施方案》的独立音频窗口约定。

### 4.3 输出

1. `positive`：H3 CONDITIONING；
2. `latent`：新续拍目标的空 H3 AV LATENT；
3. `continuation`：`HR_H3_EXTERNAL_CONTINUATION`，供 Sampler 获取源视频尾部及导演状态；
4. `prompt`：LLM 生成的最终 H3 STRING，便于检查和手工编辑；
5. `source_images`：源视频完整或受控输出，仅用于后续拼接节点；如果 VIDEO接口可保持视频对象，则优先输出 VIDEO而不是把完整视频常驻为IMAGE batch；
6. `analysis_json`：结构化末态分析，便于用户确认。

首版不输出不属于节点所有权的模型、VAE或参考集透传值。

## 五、源视频读取

### 5.1 复用现有视频边界

优先扩展 `video_io.py` 的受控媒体解析，而不是在新节点中自行接受磁盘路径。应复用：

- `_resolve_input_path()` 的路径解析和 containment语义；
- PyAV解码；
- `_probe_video()`；
- `HREndlessSamplerLoadVideo` 已有VIDEO/IMAGE/AUDIO约定。

### 5.2 尾帧选择

必须按时间轴选择最后22个输出帧，而不是仅依据容器声明的 `stream.frames`：

1. 读取实际视频流FPS与time base；
2. 解码尾部时间窗；无法可靠seek时顺序解码并只保留容量22的环形缓冲；
3. 将VFR视频重采样到H3原生24FPS；
4. 保留末尾连续22帧；
5. 记录源PTS、重采样时间戳和有效帧数；
6. 不足5帧时报错，因为H3 video continuation无法形成有效条件；
7. 5–21帧不伪造内容，使用实际帧并按H3 VAE最低帧网格做最小尾部复制或明确降级为5帧，具体策略必须通过编码测试确定。

### 5.3 尺寸处理

- 输出续拍画布默认跟随源视频宽高比；
- 使用现有 `_video_canvas()` 和32像素画布对齐；
- H3 VAE编码前使用现有Lanczos resize；
- 不裁剪主体，除非用户明确选择crop；
- 尾帧、目标latent和continuation keyframe必须使用同一目标宽高；
- 记录原始及调整后尺寸。

## 六、LLM末态分析

### 6.1 分析帧不能只取最后一张

LLM需要看到运动过程。22帧全部编码给多模态模型可能增加上下文和内存，因此使用两级数据：

- H3 continuation：完整22帧；
- LLM observation：从22帧中按时间均匀抽取建议8帧，并始终包含第1帧、倒数第2帧和最后1帧。

如当前backend能可靠消费22帧，可在实测后增加“完整22帧分析”模式，但默认不以增加显存为代价。

### 6.2 新增专用operation

在现有Qwen/Gemma worker协议中新增：

```text
operation = external_video_continuation
```

不得把它冒充现有chunk observation，因为它没有上一chunk replay state。

请求包含：

```json
{
  "operation": "external_video_continuation",
  "user_prompt": "新的续拍意图",
  "source": {"fps": 24, "tail_frames": 22, "duration": 10.0},
  "reference_summary": "新参考媒体标签和顺序",
  "required_schema": "固定JSON schema",
  "images": "按现有多模态worker机制传输"
}
```

### 6.3 响应schema

```json
{
  "confidence": "high|medium|low",
  "observed_end_state": {
    "subjects": [
      {
        "name": "可见身份或Subject标签",
        "position": "画面位置和空间关系",
        "pose": "最后姿态",
        "facing": "朝向/视线",
        "motion_phase": "动作进行阶段",
        "appearance": "只写可见且与连续性有关的特征"
      }
    ],
    "objects": [],
    "environment": "空间、天气、光线和背景动态",
    "camera": "景别、角度、运动方向和速度",
    "audio": "对白、环境声、音乐和未完成声音事件",
    "last_visible_event": "最后可见事件",
    "must_continue": ["不可跳变的状态"],
    "must_not_assume": ["尾帧无法证明的内容"]
  },
  "transition_plan": {
    "first_action": "新段第一动作，必须从末态开始",
    "camera_bridge": "镜头如何接续",
    "identity_mapping": [],
    "reference_usage": []
  },
  "h3_prompt": "完整可执行H3续拍提示词"
}
```

### 6.4 提示词规则

System prompt必须明确：

- 只报告图像中可观察事实；
- 不能因新参考图改变源视频中同一人物的当前空间位置；
- 新段第一个动作必须继承最后动作相位；
- 不允许“重新站好”“重新入场”“镜头切到完全不同场景”等无依据重置；
- 如用户意图与尾帧物理状态冲突，先设计可见过渡，再执行新意图；
- 第一个H3 Shot必须明确continuation boundary；
- 输出语言遵循用户提示语言或现有导演语言设置；
- 返回JSON，不返回检查清单或分析报告式正文。

### 6.5 校验与一次修复

校验：

1. JSON对象存在；
2. `observed_end_state`、`transition_plan`、`h3_prompt`非空；
3. `h3_prompt`包含H3要求字段和Shot；
4. 首个动作、镜头和末态主体均被引用；
5. 新参考图的标签不超过实际连接数量；
6. 不出现模型未观察到却断言为事实的字段模板；
7. 分析置信度低时仍可继续，但UI和日志必须显示警告。

结构错误时使用同一worker生命周期修复一次。修复仍失败则明确停止，不生成静态兜底H3提示词。

## 七、H3 continuation conditioning

### 7.1 视频尾部编码

1. 将最后22帧resize到目标画布；
2. 使用H3 VAE编码；
3. 保持checkpoint参数dtype/device无关，原始参数在使用时按ComfyUI既有cast规则处理；
4. 构造一个普通video keyframe：

```python
{
    "resolved_frame_index": -22,
    "latent": encoded_tail,
}
```

实际 `resolved_frame_index` 必须根据当前PackedLayout中目标timeline origin和H3 frame/token换算验证，不能直接把示例值写死。

### 7.2 Layout contract

实施前先完成或复用 `h3_layout_contract.py`，至少证明：

- 22帧编码后的latent steps正确；
- reference blocks存在时keyframe仍相对target origin正确；
- 视频cond rows等于layout中 `~img_update` 对应的视频条件行；
- 新参考图片/视频和video continuation可以共存；
- 尾部音频keyframe允许负数和小数位置；
- 不触发SelfLift中曾出现的cond row shape mismatch。

### 7.3 新参考媒体

复用 `normalize_reference_set()` 和 `HRMiniMaxH3ReferenceConditioning` 的编码规则：

- 图片按reference_set顺序作为Picture标签；
- 视频和音频参考按现有slot配对；
- 先加入源视频continuation keyframe，再保留独立 `minimax_refs`；
- 不把源视频尾部也重复加入Ref2VA refs；
- LLM使用的reference编号必须与CLIP tokenization一致。

### 7.4 音频尾部

按《HR整合H3MotionContext实施方案》：

- 视频上下文默认22帧；
- 音频上下文独立，默认建议24帧对应1秒/40个H3 audio latent steps；
- 从源视频末尾解码音频，重采样到audio VAE采样率；
- 精确计算previous overhang；
- `resolved_frame_index = end_frame - audio_steps / FRAME_RESCALE`；
- `continue`模式构造audio keyframe；
- `mute`模式不伪造环境声，目标audio latent按现有H3空音频路径；
- `new`模式不使用源音频尾部，只使用新reference audio。

## 八、与HR Endless Sampler集成

### 8.1 新增可选输入

在Sampler现有输入末尾追加：

```text
external_continuation: HR_H3_EXTERNAL_CONTINUATION (optional)
```

不改变已有参数顺序，不删除旧continuation_plan。

互斥规则：

- `external_continuation`不能与`continuation_plan`同时连接；
- 不能与retake_plan同时连接；
- 普通无续写工作流保持原行为；
- external continuation只影响第一个physical chunk的初始条件，后续chunk继续使用当前HR串行continuation。

### 8.2 Sampler第一chunk

当存在external continuation：

1. 使用其LLM生成的最终H3 prompt作为第一段导演上下文；
2. 第一chunk加入源视频尾部video keyframe；
3. 加入独立audio tail keyframe（如启用）；
4. 保留新的reference_set；
5. 采样结束后按普通HR流程形成`previous_video`和`previous_audio`；
6. 第二chunk开始完全回到现有`_conditioning_for_chunk()`路径。

不得把外部源视频写成“Chunk 0 replay”，否则会污染当前revision和retake语义。

### 8.3 Replay fingerprint和metadata

fingerprint加入小型字段：

```json
{
  "external_continuation": {
    "source_identity": "路径、size、mtime的hash",
    "source_tail_sha256": "22帧内容摘要",
    "tail_frames": 22,
    "audio_mode": "continue",
    "audio_continuation_frames": 24,
    "analysis_sha256": "LLM结构化分析摘要"
  }
}
```

每chunk metadata只在Chunk 1记录外部来源和分析摘要，不保存重复tensor。

## 九、拼接策略

新增或扩展一个小型assemble节点：

```text
HR MiniMax H3 External Continuation Assemble
```

输入：

- 原VIDEO或原IMAGE/AUDIO；
- 新续拍视频/音频；
- continuation metadata。

行为：

- 原视频保留全部帧；
- 新段若生成路径包含22帧条件前缀，只裁掉明确属于conditioning、非新生成内容的前缀；
- 不做像素级重复Trim；
- 音频按准确时间边界拼接；
- 输出timeline包含source和continuation两个区间；
- 不改变原视频编码，最终保存仍交给Save Video节点。

首版若Sampler输出天然只包含新目标段，则assemble不裁视频，只拼接。

## 十、文件修改计划

### 10.1 新增 `external_continuation.py`

包含：

- `ExternalH3Continuation`类型；
- 输入对象规范化；
- 尾帧和音频窗口几何纯函数；
- LLM响应schema校验；
- `HRMiniMaxH3VideoContinuation`节点；
- 可选assemble节点。

### 10.2 修改 `video_io.py`

- 暴露受控视频尾部解码helper；
- 支持VFR到24FPS时间采样；
- 返回尾帧、尾音频和源媒体元数据；
- 保持Load/Save Video现有接口不变。

### 10.3 修改 `qwen35.py`

- 新增external continuation请求和响应dataclass；
- 使用现有worker启动、日志和错误模型；
- 不改变普通chunk请求。

### 10.4 修改 `qwen36_38.py`与`qwen38_worker.py`

- 同步新增operation；
- 保持Qwen3.5与Qwen3.6/3.8 worker分离；
- 不增加重复多模态调用。

### 10.5 修改 `gemma4.py`

- 新增相同结构的external continuation operation；
- 复用现有图像data URL和生命周期；
- 保持MTP/非MTP现有路径。

### 10.6 修改 `reference_set.py`

- 提取可复用的reference编码helper；
- 避免新节点复制图片/视频/audio ref编码；
- 保持现有节点返回契约。

### 10.7 修改 `nodes.py`

- 在输入末尾追加可选`external_continuation`；
- 增加互斥检查；
- 第一chunk注入外部video/audio keyframe和末态提示词；
- fingerprint、metadata和日志增加外部来源摘要；
- 后续chunk逻辑不变。

### 10.8 修改 `__init__.py`

仅注册新节点及显示名。

### 10.9 前端

如VIDEO输入已有标准选择器，不新写文件浏览器。仅在必要时新增：

- 尾帧预览；
- LLM分析JSON折叠视图；
- 低置信度警告；
- “重新分析”按钮，但不得后台联网。

## 十一、测试计划

### 11.1 视频解码

1. CFR 24FPS，准确取末22帧；
2. 30FPS转24FPS，末时刻一致；
3. VFR按PTS选帧；
4. 5帧、21帧、22帧和超过22帧；
5. 少于5帧明确失败；
6. 无音频视频；
7. 单声道/双声道和不同采样率；
8. 非法路径、越界路径和损坏容器。

### 11.2 LLM

1. 请求确实包含尾帧图像；
2. 返回完整末态和H3 prompt；
3. 空JSON、仅analysis、无h3_prompt均失败；
4. 一次修复成功/失败；
5. 新参考图映射与实际slot一致；
6. 用户意图与末态冲突时存在过渡动作；
7. 不产生检查清单式输出。

### 11.3 Conditioning

1. 22帧video tail latent steps正确；
2. 源video keyframe与图片refs共存；
3. 源video keyframe与video/audio refs共存；
4. cond rows和PackedLayout完全相等；
5. 24帧独立audio window为40steps；
6. 无音频、mute、new三种路径；
7. 正负conditioning布局一致；
8. 不改变目标latent shape。

### 11.4 Sampler

1. external continuation只注入第一chunk；
2. 第二chunk恢复普通HR continuation；
3. 与continuation_plan、retake_plan连接时明确拒绝；
4. 中断恢复不会重新分析或改变已缓存末态；
5. replay fingerprint随源视频、尾帧、提示词或参考变化而失效；
6. 最近5次历史可保存此类完整生成。

### 11.5 拼接

1. 原视频末帧只出现一次；
2. 新片段没有误裁22帧有效生成内容；
3. 音频时长与视频边界一致；
4. timeline全局帧偏移正确；
5. Save/Load后source/continuation区间仍存在。

## 十二、真实GPU验收

至少准备四类源视频：

1. 人物走动尚未停下；
2. 镜头正在平移或推进；
3. 人物正在说话或做连续手部动作；
4. 持续音乐或环境声。

每类测试：

- 无新参考图；
- 加同人物新参考图；
- 加新物体参考图；
- 用户意图与当前动作自然一致；
- 用户意图需要先过渡再执行。

验收观察：

- 第一个新动作是否从源末态开始；
- 人物位置、朝向、服装和身份是否跳变；
- 镜头方向和速度是否突变；
- 是否重演源视频最后动作；
- 是否冻结22帧；
- 光线和背景是否突然重置；
- 音频是否断裂、重复或错位；
- 12GB环境下峰值VRAM/RAM及LLM时间。

## 十三、完成标准

1. 新节点只针对MiniMax H3；
2. 能选择普通本地视频；
3. LLM确实分析最后22帧而非仅读提示词；
4. 输出结构化末态分析和可执行H3续拍提示词；
5. H3使用实际尾部video latent作为continuation；
6. 新参考图片与尾帧状态共同生效且不互相覆盖；
7. 可选延续源音频尾部；
8. 第一chunk自然接续，后续chunk继续使用HR原有机制；
9. 不修改ComfyUI核心；
10. 不引入第二套采样、replay或continuation生命周期；
11. 单元、schema、JavaScript、工作流及真实GPU测试通过；
12. 在真实视觉和音频验收通过前，不宣称“无缝续写”。
