"""
MiniMax-H3 剧本与镜头处理器 — System Prompt 零件库
====================================================
节点 JZL_MiniMax_ScriptProcessor 的后台零件库。

故事风格/镜头数量选项在此维护（fallback），正式设定词在 shed/ 目录。
"""

import re

# ═══════════════════════════════════════════════════════════════
#  零件库 2: {Story_Style} — 故事风格
# ═══════════════════════════════════════════════════════════════

STORY_STYLES = {
    "热血战斗": (
        "风格要素: 高张力对抗, 力量碰撞, 速度感, 技能释放的视觉奇观。"
        "色调倾向: 高饱和暖色(红/橙/金), 强烈的明暗对比, 粒子特效密集。"
        "节奏: 快切镜头, 短爆发, 冲击力优先。"
        "角色关系: 正邪对决、师徒传承、同伴羁绊。"
    ),
    "悬疑推理": (
        "风格要素: 线索的视觉暗示, 人物微表情, 环境细节的慢慢揭示, 反转瞬间。"
        "色调倾向: 低饱和冷色(蓝灰/墨绿), 单一光源, 阴影中隐藏信息。"
        "节奏: 缓慢推进, 关键帧停留, 细节点放大。"
        "角色关系: 侦探与嫌疑人、目击者与追踪者。"
    ),
    "温馨日常": (
        "风格要素: 生活中的微小美好, 人与环境的和谐互动, 情感的自然流露。"
        "色调倾向: 暖中性色(奶油/淡木/薄荷绿/天空蓝), 柔和的漫反射光, 金色时刻。"
        "节奏: 舒缓从容, 给情感留白, 环境空镜穿插。"
        "角色关系: 家人、朋友、宠物、邻里。"
    ),
    "奇幻冒险": (
        "风格要素: 宏大世界观, 奇异生物, 魔法效果, 史诗级场景。"
        "色调倾向: 高饱和丰富色盘, 发光粒子, 魔法光芒, 异世界质感。"
        "节奏: 场景跳跃大, 奇观展示与角色反应交替。"
        "角色关系: 勇者与伙伴、冒险小队、跨种族联盟。"
    ),
    "科幻未来": (
        "风格要素: 高科技环境, 全息投影, 赛博格, 空间站/未来城市, AI 交互。"
        "色调倾向: 冷峻金属色(银/铬/深蓝), 霓虹点缀(青/品红), 全息扫描线。"
        "节奏: 科技细节展示, 人机交互, 未来感运镜。"
        "角色关系: 人类与AI、太空探索者、未来市民。"
    ),
    "古风武侠": (
        "风格要素: 中国古典意境, 轻功身法, 剑气内功, 山水场景。"
        "色调倾向: 水墨感(青绿/素白/墨色/朱砂), 留白构图, 飘落花瓣。"
        "节奏: 动静结合, 招式拆解, 意境空镜。"
        "角色关系: 侠客与江湖、师徒传承、正邪对立。"
    ),
    "都市情感": (
        "风格要素: 现代都市生活, 咖啡馆/办公室/街道, 情感细腻互动。"
        "色调倾向: 现实主义色彩, 自然窗光, 城市夜景的暖黄与冷蓝交织。"
        "节奏: 对话聚焦, 反应镜头, 环境氛围渲染。"
        "角色关系: 恋人、同事、老朋友、偶遇的陌生人。"
    ),
    "恐怖惊悚": (
        "风格要素: 心理恐惧, 未知威胁, 空间压迫感, 突如其来的惊吓。"
        "色调倾向: 极暗调, 去饱和, 单光源(手电/烛光), 诡异的色彩偏移。"
        "节奏: 缓慢累积→突然爆发, 荷兰角, 前景遮挡制造不安。"
        "角色关系: 受害者与未知存在、孤立无援的个体。"
    ),
}


# ═══════════════════════════════════════════════════════════════
#  零件库 3: {Shot_Count} — 故事长度选项
# ═══════════════════════════════════════════════════════════════

SEGMENT_COUNT_OPTIONS = {
    "4段": 4,
    "6段": 6,
    "9段": 9,
    "12段": 12,
    "16段": 16,
    "20段": 20,
    "24段": 24,
}


def _resolve_segment_count(label):
    """分段数解析：标准标签（「6段」）→ SEGMENT_COUNT_OPTIONS；数字字符串（「6」/「1」）→ 精确整数；否则 4。

    管理器把 video_count 的任意 1-48 精确传给剧本处理器（不限于 4/6/9/12/16/20/24）。
    """
    if isinstance(label, str) and label in SEGMENT_COUNT_OPTIONS:
        return SEGMENT_COUNT_OPTIONS[label]
    if isinstance(label, str) and label.strip().isdigit():
        return max(1, min(48, int(label.strip())))
    if isinstance(label, (int, float)):
        return max(1, min(48, int(label)))
    return 4


# ═══════════════════════════════════════════════════════════════
#  V2 融合骨架（剧本与镜头处理器专用）
#  占位符: {Mode_Instruction} {Story_Style} {Segment_Count}
#          {Decompose_Rules} {Reference_Intro} {H3_Shot_Rules} {User_Story}
# ═══════════════════════════════════════════════════════════════

SCRIPT_SKELETON_V2 = '''# Role: 顶级短视频编剧 & MiniMax H3 分段提示词工程师
你是专业的短视频编剧和 MiniMax H3 视频提示词撰写专家。你的任务分两步：
第一步：把用户故事按「情节」拆解为恰好 {Segment_Count} 个分段——每个分段是一段 {Segment_Duration} 秒的独立视频片段，对应故事的一个完整情节段落（不是单个动作；一个分段内可以包含多个连续动作）。
第二步：把每个分段当成一段独立视频去润色，写出一条可直接送入 MiniMax H3 模型生成视频的完整提示词，详细描述要覆盖这段视频从头到尾的全部内容。

## 0. 任务模式
{Mode_Instruction}

## 1. 故事风格
{Story_Style}
{User_Tags}

## 2. 分段数量铁律
必须恰好生成 {Segment_Count} 个分段，不多不少。每个分段都是一段独立的视频片段（对应一个情节段落）。

## 3. 拆解规则
{Decompose_Rules}

{Reference_Intro}
{H3_Shot_Rules}

## 用户故事：
{User_Story}'''


def _build_schedule_rules(lang, enable_scene, enable_props, enable_video, enable_audio):
    """根据 4 个调度开关生成约束文本（注入 H3_Shot_Rules 的 {Schedule_Rules} 占位符）。"""
    if lang == "zh":
        lines = ["### 4.5 调度开关约束（必须遵守）"]
        if enable_scene:
            lines.append("- 场景调度已启用：SCENE_INSTRUCTION 的 slots 中包含「场景:槽位名」元素（槽位名与素材节点标题一致），提示词用对应的 <Picture N> 引用背景图。")
        else:
            lines.append("- 场景调度已关闭：分段信息「**场景**」写「无」，slots 不包含「场景:」元素，提示词用纯文本描述环境、不写背景参考标签。")
        if enable_props:
            lines.append("- 道具调度已启用：SCENE_INSTRUCTION 的 slots 中包含「道具:槽位名」元素，提示词用对应的 <Picture N> 引用道具图。")
        else:
            lines.append("- 道具调度已关闭：分段信息「**道具**」写「无」，slots 不包含「道具:」元素，提示词用纯文本描述道具、不写道具参考标签。")
        if enable_video:
            lines.append("- 视频调度已启用：每个镜头输出 ===VIDEO_INSTRUCTION=== 段，slots 中含「视频:槽位名」元素，提示词用 <Video N> 引用。")
        else:
            lines.append("- 视频调度已关闭：不输出 ===VIDEO_INSTRUCTION=== 段，不写 <Video N> 标签，动作/运镜直接用纯文本描述。")
        if enable_audio:
            lines.append("- 音频调度已启用：每个镜头输出 ===AUDIO_INSTRUCTION=== 段，slots 中含「音频:槽位名」元素，提示词用 <Audio N> 引用。")
        else:
            lines.append("- 音频调度已关闭：不输出 ===AUDIO_INSTRUCTION=== 段，不写 <Audio N> 标签，对话直接用 (S1)/(S2) 纯文本描述。")
    else:
        lines = ["### 4.5 Scheduling Toggle Rules (MUST follow)"]
        if enable_scene:
            lines.append("- Scene scheduling ENABLED: SCENE_INSTRUCTION slots include \"scene:<slotName>\" elements (slot name matches the material node title); reference the background via the matching <Picture N>.")
        else:
            lines.append("- Scene scheduling DISABLED: write \"none\" in the **Scene** field; slots contain no \"scene:\" element; describe the environment in plain text without any background label.")
        if enable_props:
            lines.append("- Prop scheduling ENABLED: SCENE_INSTRUCTION slots include \"prop:<slotName>\" elements; reference props via matching <Picture N>.")
        else:
            lines.append("- Prop scheduling DISABLED: write \"none\" in the **Props** field; slots contain no \"prop:\" element; describe props in plain text without any prop label.")
        if enable_video:
            lines.append("- Video scheduling ENABLED: output ===VIDEO_INSTRUCTION=== with \"video:<slotName>\" slots; use <Video N> in the prompt.")
        else:
            lines.append("- Video scheduling DISABLED: do NOT output ===VIDEO_INSTRUCTION=== nor use <Video N>; describe action/camera in plain text.")
        if enable_audio:
            lines.append("- Audio scheduling ENABLED: output ===AUDIO_INSTRUCTION=== with \"audio:<slotName>\" slots; use <Audio N> in the prompt.")
        else:
            lines.append("- Audio scheduling DISABLED: do NOT output ===AUDIO_INSTRUCTION=== nor use <Audio N>; describe dialogue with (S1)/(S2) in plain text.")
    return "\n".join(lines)


def _extract_style_directing(style_text):
    """提取风格设定里的「核心导演语法」段落（写动作时必须逐条执行的部分）。"""
    if not style_text:
        return ""
    m = re.search(r'## 核心导演语法(.*?)(?=(?<!#)## )', style_text, re.DOTALL)
    if not m:
        return ""
    return ("## 核心导演语法" + m.group(1)).rstrip()


def _extract_detail_length(preference_text):
    """从偏好设定词里提取「详细描述字数」的数字范围（如 800-1200），默认 350-500。"""
    if not preference_text:
        return "350-500"
    m = re.search(r'详细描述字数[:：][^\n]*\((\d+-\d+)\s*[字词]', preference_text)
    if m:
        return m.group(1)
    return "350-500"


def _build_material_table(ref_image_intro, ref_video_intro, ref_audio_intro):
    """解析用户素材描述，生成「标签对照表」，供 LLM 严格按槽位名写 slots。

    用户描述每行格式：「类型+字母 = 素材名（描述）」，如「角色A = 孙悟空（橙色武道服）」。
    返回对照表文本；无素材时返回空串。
    """
    blocks = []
    for kind, intro in (("图片", ref_image_intro), ("视频", ref_video_intro), ("音频", ref_audio_intro)):
        if not intro or not intro.strip():
            continue
        rows = []
        for line in str(intro).splitlines():
            line = line.strip()
            if not line:
                continue
            m = re.match(r'^(角色|场景|道具|视频|音频|分镜|音效|音乐|其他)\s*([A-Za-z])\s*[=＝:：]\s*(.+)', line)
            if not m:
                continue
            typ, slot, rest = m.group(1), m.group(2).upper(), m.group(3).strip()
            if not rest:
                continue
            rows.append(f"- {typ}{slot} = {rest}")
        if rows:
            blocks.append(f"【{kind}】\n" + "\n".join(rows))
    if not blocks:
        return ""
    return ("## 3.5 素材标签对照表（CRITICAL — 唯一素材来源，slots 只能从这里取「类型:槽位名」）\n"
            + "\n".join(blocks)
            + "\n\n## 3.6 素材白名单（CRITICAL — 顶部字段与调度严禁引用对照表之外的素材）\n"
            "- 每段顶部元数据（**角色**/**场景**/**道具**/**视频**/**音频**/**音效**/**音乐**/**其他**）里列出的名称，必须全部来自上方对照表中的「素材名」或「类型:槽位名」（如 场景:场景B、角色:角色A），严禁自造或新增。\n"
            "- 剧情需要但对照表中没有的素材（如某道具/场景/角色），一律不得写入顶部字段或调度 slots：该字段写「无」，只允许在 detailed_description 剧情正文中作环境描写。\n"
            "- 调度指令 slots 写「类型:槽位名」，如 场景:场景A、角色:角色A、道具:道具A、视频:视频A、音频:音频A，严禁写素材名（孙悟空）或自造槽位。\n"
            "- subject_definitions 写素材名（<Subject N> 是 <Picture N> 中的孙悟空），括号里的外貌描述原样复述。\n"
            "- 音频槽位对应说话人音色：本段谁开口说话，AUDIO_INSTRUCTION.slots 就写对应角色的「音频:音频X」，无对话分段音频 slots 写空。")

def build_shot_prompt(
    user_story: str,
    mode: str = "拆解模式 (Decompose)",
    story_style: str = "热血战斗",
    segment_count_label: str = "4段",
    lang: str = "zh",
    segment_duration: int = 8,
    ref_image_intro: str = "",
    ref_video_intro: str = "",
    ref_audio_intro: str = "",
    enable_scene: bool = True,
    enable_props: bool = True,
    enable_video: bool = True,
    enable_audio: bool = True,
    user_tags: str = "",
    preference: str = "",
    custom_rules: str = "",
) -> str:
    """剧本与镜头处理器 — 拼装一次成型的 System Prompt（融合拆解 + 六段 Ref2VA 规范）。

    lang: "zh" / "en"；segment_duration: 每段视频时长(秒)，约束时间戳范围。
    """
    from ..sheding.mode_instructions import MODE_INSTRUCTIONS as _mi
    from ..sheding.story_styles import STORY_STYLES as _ss
    from ..sheding.decompose_rules import DECOMPOSE_RULES as _dr
    from ..sheding.h3_shot_rules import H3_SHOT_RULES_ZH, H3_SHOT_RULES_EN

    mode_instruction = _mi.get(mode, list(_mi.values())[0] if _mi else "")
    style = _ss.get(story_style, list(_ss.values())[0] if _ss else "")
    segment_count = _resolve_segment_count(segment_count_label)
    segment_duration = max(4, min(15, int(segment_duration or 8)))

    schedule_rules = _build_schedule_rules(lang, enable_scene, enable_props, enable_video, enable_audio)

    # 用 replace 而非 format：rules 文本内含 {视觉描述} 等示意大括号，不能走 format
    rules = (H3_SHOT_RULES_ZH if lang == "zh" else H3_SHOT_RULES_EN)
    rules = rules.replace("{Segment_Count}", str(segment_count))
    rules = rules.replace("{Segment_Duration}", str(segment_duration))
    rules = rules.replace("{Schedule_Rules}", schedule_rules)
    rules = rules.replace("{Style_Directing}", _extract_style_directing(style))
    rules = rules.replace("{Detail_Length}", _extract_detail_length(preference))
    rules = rules.replace("{Preference_Directing}", (preference or "").strip() or "- （无额外镜头语言偏好，按故事风格与标准规则执行）")
    rules = rules.replace("{Custom_Rules_Directing}", (custom_rules or "").strip() or "- （无自定义规则）")

    # 参考素材说明：生成「标签对照表」（槽位名 → 素材名 → 描述），LLM 严格按表写 slots
    reference_intro = _build_material_table(ref_image_intro, ref_video_intro, ref_audio_intro)

    return SCRIPT_SKELETON_V2.format(
        Mode_Instruction=mode_instruction.format(Segment_Count=segment_count),
        Story_Style=style,
        Segment_Count=segment_count,
        Segment_Duration=segment_duration,
        Decompose_Rules=_dr,
        Reference_Intro=reference_intro,
        H3_Shot_Rules=rules,
        User_Story=user_story,
        User_Tags=user_tags,
    )
