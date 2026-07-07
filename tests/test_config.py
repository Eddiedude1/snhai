"""Unit tests for the shared config-file loader used by all pipeline stages.

Spec: docs/srs/data-preparation.md, A3.8 / IR-DP-4.
"""

import json
from pathlib import Path

from snhai import config


def test_load_config_returns_named_stage_section(tmp_path):
    """IR-DP-4: load_config returns only the requested stage's namespaced section."""
    config_path = tmp_path / "config.json"
    config_path.write_text(
        json.dumps({"data_preparation": {"seed": 7}, "training": {"lr": 0.1}})
    )
    assert config.load_config(config_path, "data_preparation") == {"seed": 7}
    assert config.load_config(config_path, "training") == {"lr": 0.1}


def test_load_config_missing_file_returns_empty_dict():
    """A3.8: a missing config file falls back to the script's built-in defaults, not an error."""
    assert (
        config.load_config(Path("/nonexistent/config.json"), "data_preparation") == {}
    )


def test_load_config_missing_stage_section_returns_empty_dict(tmp_path):
    """A3.8: a config file lacking the requested stage's section also falls back to defaults."""
    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps({"training": {"lr": 0.1}}))
    assert config.load_config(config_path, "data_preparation") == {}
