#!/usr/bin/env python3
"""Replay a captured Gemma chunk-director handoff without H3 sampling."""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


PLUGIN_ROOT = Path(__file__).resolve().parents[1]
COMFY_ROOT = PLUGIN_ROOT.parents[1]
sys.path.insert(0, str(COMFY_ROOT))
sys.path.insert(0, str(PLUGIN_ROOT))

from gemma4 import replay_gemma_capture  # noqa: E402


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Replay one exact captured chunk-directing request using the current gemma4_prompts.txt."
    )
    parser.add_argument("capture", type=Path, help="prompt_NNN_chunk_NNN capture directory")
    parser.add_argument("--debug", action="store_true", help="enable verbose llama.cpp logging")
    args = parser.parse_args()
    result = replay_gemma_capture(args.capture, debug=args.debug)
    print(
        json.dumps(
            {
                "confidence": result.confidence,
                "analysis": result.analysis,
                "detailed_description": result.detailed_description,
                "raw_json": result.raw_json,
            },
            ensure_ascii=False,
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
