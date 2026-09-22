from .continuation import HREndlessContinuationAssemble, HREndlessContinuationCheckpoint, HREndlessContinuationPlan
from .director_config import HRQwen38DirectorConfig
from .external_continuation import (
    HRMiniMaxH3VideoContinuationAnalyzer,
    HRMiniMaxH3VideoContinuationApply,
)
from .nodes import HREndlessSampler
from .preview import HREndlessSamplerPreview
from .prompt_skill_node import HRH3PromptSkillCompiler
from .retake_director import HREndlessRetakeAssemble, HREndlessSegmentRetakeDirector
from .reference_set import HRMiniMaxH3ReferenceConditioning, HRMiniMaxH3ReferenceSet
from .storyboard import HRMiniMaxH3StoryboardPlanner
from .video_io import HREndlessSamplerLoadVideo, HREndlessSamplerSaveVideo
from .video_bridge import HRVideoBridgeAssemble, HRVideoBridgeConditioning, HRVideoBridgeDirector, HRVideoBridgeExtract
from .jzl_storyboard import HRMiniMaxH3JZLStoryboard, HRMiniMaxH3JZLSegmentDispatcher

__version__ = "0.9.0"


NODE_CLASS_MAPPINGS = {
    "HREndlessSampler": HREndlessSampler,
    "HREndlessSamplerPreview": HREndlessSamplerPreview,
    "HREndlessSamplerSaveVideo": HREndlessSamplerSaveVideo,
    "HREndlessSamplerLoadVideo": HREndlessSamplerLoadVideo,
    "HREndlessSegmentRetakeDirector": HREndlessSegmentRetakeDirector,
    "HREndlessRetakeAssemble": HREndlessRetakeAssemble,
    "HREndlessContinuationCheckpoint": HREndlessContinuationCheckpoint,
    "HREndlessContinuationPlan": HREndlessContinuationPlan,
    "HREndlessContinuationAssemble": HREndlessContinuationAssemble,
    "HRMiniMaxH3VideoContinuationAnalyzer": HRMiniMaxH3VideoContinuationAnalyzer,
    "HRMiniMaxH3VideoContinuationApply": HRMiniMaxH3VideoContinuationApply,
    "HRMiniMaxH3StoryboardPlanner": HRMiniMaxH3StoryboardPlanner,
    "HRQwen38DirectorConfig": HRQwen38DirectorConfig,
    "HRMiniMaxH3ReferenceSet": HRMiniMaxH3ReferenceSet,
    "HRMiniMaxH3ReferenceConditioning": HRMiniMaxH3ReferenceConditioning,
    "HRMiniMaxH3JZLStoryboard": HRMiniMaxH3JZLStoryboard,
    "HRMiniMaxH3JZLSegmentDispatcher": HRMiniMaxH3JZLSegmentDispatcher,
    "HRVideoBridgeExtract": HRVideoBridgeExtract,
    "HRVideoBridgeDirector": HRVideoBridgeDirector,
    "HRVideoBridgeConditioning": HRVideoBridgeConditioning,
    "HRVideoBridgeAssemble": HRVideoBridgeAssemble,
    "HRH3PromptSkillCompiler": HRH3PromptSkillCompiler,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "HREndlessSampler": "HR Endless Sampler",
    "HREndlessSamplerPreview": "HR Endless Sampler Preview",
    "HREndlessSamplerSaveVideo": "HR Endless Sampler Save Video",
    "HREndlessSamplerLoadVideo": "HR Endless Sampler Load Video",
    "HREndlessSegmentRetakeDirector": "HR Endless Segment Retake Director",
    "HREndlessRetakeAssemble": "HR Endless Retake Assemble",
    "HREndlessContinuationCheckpoint": "HR Endless Continuation Checkpoint",
    "HREndlessContinuationPlan": "HR Endless Continuation Plan",
    "HREndlessContinuationAssemble": "HR Endless Continuation Assemble",
    "HRMiniMaxH3VideoContinuationAnalyzer": "HR MiniMax H3 Continuation Analyzer",
    "HRMiniMaxH3VideoContinuationApply": "HR MiniMax H3 Continuation Apply",
    "HRMiniMaxH3StoryboardPlanner": "HR MiniMax H3 Storyboard Planner",
    "HRQwen38DirectorConfig": "HR Qwen Director Config",
    "HRMiniMaxH3ReferenceSet": "HR MiniMax H3 Reference Set",
    "HRMiniMaxH3ReferenceConditioning": "HR MiniMax H3 Reference Conditioning",
    "HRMiniMaxH3JZLStoryboard": "HR MiniMax H3 JZL Storyboard",
    "HRMiniMaxH3JZLSegmentDispatcher": "HR MiniMax H3 JZL Segment Dispatcher",
    "HRVideoBridgeExtract": "HR Video Bridge Extract",
    "HRVideoBridgeDirector": "HR Video Bridge Director",
    "HRVideoBridgeConditioning": "HR Video Bridge Conditioning",
    "HRVideoBridgeAssemble": "HR Video Bridge Assemble",
    "HRH3PromptSkillCompiler": "HR H3 Prompt Skill Compiler",
}

WEB_DIRECTORY = "./web"

__all__ = ["__version__", "NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
