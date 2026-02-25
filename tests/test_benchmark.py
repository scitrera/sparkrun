"""Tests for standalone benchmark YAML handling."""

from __future__ import annotations

from pathlib import Path

import pytest
import yaml

from sparkrun.benchmark import BenchmarkSpec, BenchmarkError


def _write_yaml(path: Path, data: dict):
    path.write_text(yaml.safe_dump(data))


def test_benchmark_load_valid(tmp_path: Path):
    p = tmp_path / "bench.yaml"
    _write_yaml(p, {
        "recipe": "my-recipe",
        "model": "org/model",
        "benchmark": {
            "framework": "llama-benchy",
            "args": {
                "pp": [2048],
                "enable_prefix_caching": True,
                "format": "csv",
            },
        },
    })

    spec = BenchmarkSpec.load(p)
    assert spec.framework == "llama-benchy"
    assert spec.recipe == "my-recipe"
    assert spec.model == "org/model"
    assert spec.args["format"] == "csv"


def test_benchmark_load_missing_block(tmp_path: Path):
    p = tmp_path / "bench.yaml"
    _write_yaml(p, {"recipe": "x"})

    with pytest.raises(BenchmarkError, match="benchmark"):
        BenchmarkSpec.load(p)


def test_benchmark_build_command(tmp_path: Path):
    p = tmp_path / "bench.yaml"
    _write_yaml(p, {
        "recipe": "my-recipe",
        "model": "org/model",
        "benchmark": {
            "framework": "llama-benchy",
            "args": {
                "pp": [2048],
                "depth": [0, 10],
                "enable_prefix_caching": True,
                "format": "csv",
            },
        },
    })
    spec = BenchmarkSpec.load(p)
    cmd = spec.build_command({"format": "json"})

    assert cmd[0] == "llama-benchy"
    assert "--recipe" in cmd and "my-recipe" in cmd
    assert "--model" in cmd and "org/model" in cmd
    assert "--enable-prefix-caching" in cmd
    # List values are repeated flags
    assert cmd.count("--depth") == 2
    # Override applied
    assert "json" in cmd
