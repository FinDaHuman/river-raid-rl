"""Tests for the YAML configuration loader used by the unified trainer."""

import pathlib
import yaml


def test_rainbow_yaml_can_be_parsed():
    cfg_path = pathlib.Path(__file__).parents[1] / "configs" / "rainbow.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Basic sanity checks – ensure required keys exist
    assert "env" in cfg
    assert "agent" in cfg and cfg["agent"]["type"] == "rainbow"
    assert "training" in cfg
    # Verify that a numeric hyper‑parameter is present
    assert cfg["agent"]["params"]["hidden_dim"] > 0
