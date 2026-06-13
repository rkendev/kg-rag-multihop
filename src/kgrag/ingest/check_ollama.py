"""Fail fast if Ollama is unreachable or a pinned model tag is missing.

Per project policy we never silently substitute a different (or paid) model: if a
required tag is absent we stop and report.
"""
from __future__ import annotations

import sys

from .. import config
from ..ollama_client import list_models


def main() -> int:
    required = [config.GEN_MODEL, config.JUDGE_MODEL]
    try:
        available = list_models()
    except Exception as exc:  # noqa: BLE001
        print(f"ERROR: cannot reach Ollama at {config.OLLAMA_HOST}: {exc}")
        return 1

    missing = [m for m in required if m not in available]
    if missing:
        print(f"ERROR: missing Ollama model tags: {missing}")
        print(f"available: {available}")
        print("Pull them (e.g. `ollama pull qwen2.5:7b-instruct`) — do not substitute.")
        return 1

    print(f"ollama ok — generator={config.GEN_MODEL} judge={config.JUDGE_MODEL}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
