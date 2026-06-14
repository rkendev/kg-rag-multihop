"""Minimal Ollama HTTP client used by the generator and the judge.

Deterministic by default (temperature 0). Kept dependency-light: just `requests`.
"""
from __future__ import annotations

import requests

from . import config


def generate(
    model: str,
    prompt: str,
    *,
    system: str | None = None,
    temperature: float = 0.0,
    num_predict: int = 256,
    timeout: int = 600,
    format: str | None = None,
) -> str:
    """Single-shot, non-streaming completion. Returns the response text.

    ``format="json"`` enables Ollama's constrained JSON output (used by P1 relation
    extraction so the reply is always parseable).
    """
    payload = {
        "model": model,
        "prompt": prompt,
        "stream": False,
        "options": {
            "temperature": temperature,
            "num_predict": num_predict,
            # seed fixed so temp-0 ties break identically across runs
            "seed": config.SEED % 2_000_000_000,
        },
    }
    if system is not None:
        payload["system"] = system
    if format is not None:
        payload["format"] = format
    resp = requests.post(
        f"{config.OLLAMA_HOST}/api/generate", json=payload, timeout=timeout
    )
    resp.raise_for_status()
    return resp.json().get("response", "")


def list_models() -> list[str]:
    resp = requests.get(f"{config.OLLAMA_HOST}/api/tags", timeout=30)
    resp.raise_for_status()
    return [m["name"] for m in resp.json().get("models", [])]
