"""Shared config-file loader for all pipeline stages (Data Preparation, Training, Evaluation).

A single namespaced JSON file (see `config.json`) holds one section per stage. Each stage's
script loads only its own section and treats CLI flags as overrides on top of it. A missing
file, or a config file missing the requested stage's section, is not an error — it just means
"use this script's built-in defaults" (see docs/srs/data-preparation.md, A3.8/IR-DP-4).
"""

from __future__ import annotations

import json
from pathlib import Path


def load_config(path: str | Path, stage: str) -> dict:
    path = Path(path)
    if not path.is_file():
        return {}
    with path.open() as f:
        config = json.load(f)
    return config.get(stage, {})
