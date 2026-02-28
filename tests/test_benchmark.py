"""Tests for benchmarking framework and benchmark spec handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import Mock, patch
import shutil

import pytest
import yaml

from sparkrun.benchmarking.base import (
    BenchmarkSpec,
    BenchmarkError,
    render_args_as_flags,
    export_results,
)
from sparkrun.benchmarking.llama_benchy import LlamaBenchyFramework
from sparkrun.recipe import Recipe
from sparkrun.runtimes.base import RuntimePlugin
from sparkrun.runtimes.vllm_ray import VllmRayRuntime
from sparkrun.runtimes.sglang import SglangRuntime


def _write_yaml(path: Path, data: dict):
    path.write_text(yaml.safe_dump(data))


# ========== BenchmarkSpec Loading Tests ==========


def test_benchmark_load_valid(tmp_path: Path):
    """Test loading a valid benchmark YAML with nested args format."""
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
    """Test that loading fails when benchmark block is missing."""
    p = tmp_path / "bench.yaml"
    _write_yaml(p, {"recipe": "x"})

    with pytest.raises(BenchmarkError, match="benchmark"):
        BenchmarkSpec.load(p)


def test_benchmark_load_flat_format(tmp_path: Path):
    """Test sweep-unknown-keys: flat benchmark block sweeps keys into args."""
    p = tmp_path / "bench.yaml"
    _write_yaml(p, {
        "recipe": "my-recipe",
        "benchmark": {
            "framework": "llama-benchy",
            "pp": [2048],
            "depth": [0],
            "enable_prefix_caching": True,
        },
    })

    spec = BenchmarkSpec.load(p)
    assert spec.framework == "llama-benchy"
    # Unknown keys swept into args
    assert spec.args["pp"] == [2048]
    assert spec.args["depth"] == [0]
    assert spec.args["enable_prefix_caching"] is True


def test_benchmark_from_recipe(tmp_path: Path):
    """Test BenchmarkSpec.from_recipe() with a Recipe that has a benchmark block."""
    recipe = Recipe.from_dict({
        "name": "test-recipe",
        "model": "org/model",
        "runtime": "vllm",
        "benchmark": {
            "framework": "llama-benchy",
            "args": {"pp": [2048]},
        },
    })

    spec = BenchmarkSpec.from_recipe(recipe)
    assert spec is not None
    assert spec.framework == "llama-benchy"
    assert spec.recipe == "test-recipe"
    assert spec.model == "org/model"
    assert spec.args["pp"] == [2048]


def test_benchmark_from_recipe_no_block(tmp_path: Path):
    """Test that from_recipe returns None when no benchmark block exists."""
    recipe = Recipe.from_dict({
        "name": "test-recipe",
        "model": "org/model",
        "runtime": "vllm",
    })

    spec = BenchmarkSpec.from_recipe(recipe)
    assert spec is None


def test_benchmark_build_command(tmp_path: Path):
    """Test command building with list args and overrides."""
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


# ========== render_args_as_flags Tests ==========


def test_render_args_booleans():
    """Test that True adds flag, False skips."""
    args = {"enable_cache": True, "disable_warmup": False, "verbose": True}
    flags = render_args_as_flags(args)

    assert "--enable-cache" in flags
    assert "--verbose" in flags
    assert "--disable-warmup" not in flags


def test_render_args_lists():
    """Test that list values emit repeated flags."""
    args = {"pp": [2048, 4096], "depth": [0]}
    flags = render_args_as_flags(args)

    assert flags.count("--pp") == 2
    assert "2048" in flags and "4096" in flags
    assert flags.count("--depth") == 1
    assert "0" in flags


def test_render_args_scalars():
    """Test regular key=value pairs."""
    args = {"port": 8000, "model": "my-model", "timeout": 30.5}
    flags = render_args_as_flags(args)

    assert "--port" in flags
    assert "8000" in flags
    assert "--model" in flags
    assert "my-model" in flags
    assert "--timeout" in flags
    assert "30.5" in flags


# ========== LlamaBenchyFramework Tests ==========


def test_llama_benchy_framework_name():
    """Test framework_name is correct."""
    fw = LlamaBenchyFramework()
    assert fw.framework_name == "llama-benchy"


def test_llama_benchy_default_args():
    """Test default args include pp, depth, enable_prefix_caching."""
    fw = LlamaBenchyFramework()
    defaults = fw.get_default_args()

    assert "pp" in defaults
    assert defaults["pp"] == [2048]
    assert "depth" in defaults
    assert defaults["depth"] == [0]
    assert "enable_prefix_caching" in defaults
    assert defaults["enable_prefix_caching"] is True


def test_llama_benchy_build_command():
    """Test build_benchmark_command builds correct uvx command."""
    fw = LlamaBenchyFramework()
    args = {"pp": [2048], "depth": [0], "enable_prefix_caching": True}
    cmd = fw.build_benchmark_command(
        target_url="http://localhost:8000/v1",
        model="org/model",
        args=args,
        result_file="/tmp/results.csv",
    )

    assert cmd[0] == "uvx"
    assert cmd[1] == "llama-benchy"
    assert "--base-url" in cmd
    assert "http://localhost:8000/v1" in cmd
    assert "--model" in cmd
    assert "org/model" in cmd
    assert "--format" in cmd
    assert "csv" in cmd
    assert "--save-result" in cmd
    assert "/tmp/results.csv" in cmd


def test_llama_benchy_build_command_bool_flag():
    """Test boolean args become bare flags."""
    fw = LlamaBenchyFramework()
    args = {"enable_prefix_caching": True, "no_warmup": True, "no_cache": False}
    cmd = fw.build_benchmark_command(
        target_url="http://localhost:8000/v1",
        model="org/model",
        args=args,
    )

    assert "--enable-prefix-caching" in cmd
    assert "--no-warmup" in cmd
    assert "--no-cache" not in cmd


def test_llama_benchy_build_command_list_args():
    """Test list args emit space-separated values after single flag."""
    fw = LlamaBenchyFramework()
    args = {"pp": [2048, 4096], "tg": [32, 64]}
    cmd = fw.build_benchmark_command(
        target_url="http://localhost:8000/v1",
        model="org/model",
        args=args,
    )

    # llama-benchy format: --pp 2048 4096 --tg 32 64
    pp_idx = cmd.index("--pp")
    assert cmd[pp_idx + 1] == "2048"
    assert cmd[pp_idx + 2] == "4096"

    tg_idx = cmd.index("--tg")
    assert cmd[tg_idx + 1] == "32"
    assert cmd[tg_idx + 2] == "64"


def test_llama_benchy_interpret_arg_list():
    """Test comma-separated value for list arg becomes list."""
    fw = LlamaBenchyFramework()

    result = fw.interpret_arg("pp", "2048,4096,8192")
    assert isinstance(result, list)
    assert result == [2048, 4096, 8192]

    result = fw.interpret_arg("depth", "0,10,20")
    assert isinstance(result, list)
    assert result == [0, 10, 20]


def test_llama_benchy_interpret_arg_bool():
    """Test boolean string becomes bool."""
    fw = LlamaBenchyFramework()

    result = fw.interpret_arg("enable_prefix_caching", "true")
    assert result is True

    result = fw.interpret_arg("no_warmup", "false")
    assert result is False


def test_llama_benchy_interpret_arg_scalar():
    """Test regular value gets coerced."""
    fw = LlamaBenchyFramework()

    result = fw.interpret_arg("timeout", "30")
    assert result == 30

    # Note: "concurrency" is a known list arg, so it returns a list even for single values
    result = fw.interpret_arg("max_tokens", "8")
    assert result == 8


def test_llama_benchy_parse_results_csv():
    """Test CSV text parsed into rows via DictReader."""
    fw = LlamaBenchyFramework()
    csv_text = "pp,tg,depth\n2048,32,0\n4096,64,100\n"

    results = fw.parse_results(csv_text, "", result_file=None)

    assert "rows" in results
    assert len(results["rows"]) == 2
    assert results["rows"][0]["pp"] == "2048"
    assert results["rows"][0]["tg"] == "32"
    assert results["rows"][0]["depth"] == "0"
    assert results["rows"][1]["pp"] == "4096"
    assert results["rows"][1]["tg"] == "64"
    assert results["rows"][1]["depth"] == "100"


def test_llama_benchy_parse_results_empty():
    """Test empty input returns empty rows."""
    fw = LlamaBenchyFramework()
    results = fw.parse_results("", "", result_file=None)

    assert results["rows"] == []


def test_llama_benchy_check_prerequisites_with_uvx():
    """Test prerequisites check passes when uvx is available."""
    fw = LlamaBenchyFramework()

    with patch.object(shutil, "which", return_value="/usr/bin/uvx"):
        missing = fw.check_prerequisites()

    assert missing == []


def test_llama_benchy_check_prerequisites_without_uvx():
    """Test prerequisites check fails when uvx is not available."""
    fw = LlamaBenchyFramework()

    with patch.object(shutil, "which", return_value=None):
        missing = fw.check_prerequisites()

    assert len(missing) > 0
    assert "uvx" in missing[0]


# ========== strip_flags_from_command Tests ==========


def test_strip_valued_flag():
    """Test removing --served-model-name and its value."""
    command = "vllm serve org/model --port 8000 --served-model-name my-alias --tp 2"
    flag_map = {"port": "--port", "served_model_name": "--served-model-name", "tensor_parallel": "--tp"}

    result = RuntimePlugin.strip_flags_from_command(
        command, skip_keys={"served_model_name"}, flag_map=flag_map, bool_keys=set()
    )

    assert "--served-model-name" not in result
    assert "my-alias" not in result
    assert "--port 8000" in result
    assert "--tp 2" in result


def test_strip_bool_flag():
    """Test removing a boolean flag."""
    command = "vllm serve org/model --enable-prefix-caching --port 8000"
    flag_map = {"enable_prefix_caching": "--enable-prefix-caching", "port": "--port"}

    result = RuntimePlugin.strip_flags_from_command(
        command, skip_keys={"enable_prefix_caching"}, flag_map=flag_map, bool_keys={"enable_prefix_caching"}
    )

    assert "--enable-prefix-caching" not in result
    assert "--port 8000" in result


def test_strip_unknown_key_noop():
    """Test key not in flag_map is a no-op."""
    command = "vllm serve org/model --port 8000"
    flag_map = {"port": "--port"}

    result = RuntimePlugin.strip_flags_from_command(
        command, skip_keys={"unknown_key"}, flag_map=flag_map, bool_keys=set()
    )

    # Should be unchanged
    assert result == command


def test_strip_multiple_keys():
    """Test stripping multiple flags at once."""
    command = "vllm serve org/model --port 8000 --served-model-name alias --tp 2 --enable-prefix-caching"
    flag_map = {
        "port": "--port",
        "served_model_name": "--served-model-name",
        "tensor_parallel": "--tp",
        "enable_prefix_caching": "--enable-prefix-caching",
    }

    result = RuntimePlugin.strip_flags_from_command(
        command,
        skip_keys={"served_model_name", "enable_prefix_caching"},
        flag_map=flag_map,
        bool_keys={"enable_prefix_caching"},
    )

    assert "--served-model-name" not in result
    assert "alias" not in result
    assert "--enable-prefix-caching" not in result
    assert "--port 8000" in result
    assert "--tp 2" in result


# ========== export_results Tests ==========


def test_export_results_writes_yaml(tmp_path: Path):
    """Test export_results creates YAML with expected structure."""
    recipe = Recipe.from_dict({
        "name": "test-recipe",
        "model": "org/model",
        "runtime": "vllm",
    })

    output_path = tmp_path / "results.yaml"

    result = export_results(
        recipe=recipe,
        hosts=["host1", "host2"],
        tp=2,
        cluster_id="test-cluster-123",
        framework_name="llama-benchy",
        profile_name="default",
        args={"pp": [2048], "depth": [0]},
        results={"rows": [{"pp": "2048", "tg": "32"}]},
        output_path=output_path,
    )

    assert result == output_path
    assert output_path.exists()

    data = yaml.safe_load(output_path.read_text())
    assert "sparkrun_benchmark" in data

    bench = data["sparkrun_benchmark"]
    assert bench["version"] == "1"
    assert "timestamp" in bench

    assert bench["recipe"]["name"] == "test-recipe"
    assert bench["recipe"]["model"] == "org/model"
    # Note: Recipe runtime resolution converts "vllm" to "vllm-distributed"
    assert bench["recipe"]["runtime"] == "vllm-distributed"

    assert bench["cluster"]["hosts"] == ["host1", "host2"]
    assert bench["cluster"]["tp"] == 2
    assert bench["cluster"]["cluster_id"] == "test-cluster-123"

    assert bench["benchmark"]["framework"] == "llama-benchy"
    assert bench["benchmark"]["profile"] == "default"
    assert bench["benchmark"]["args"] == {"pp": [2048], "depth": [0]}

    assert bench["results"]["rows"] == [{"pp": "2048", "tg": "32"}]


# ========== skip_keys in generate_command Tests ==========


def test_vllm_generate_command_skip_keys():
    """Test VllmRayRuntime with skip_keys removes specified flags."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "org/model",
        "runtime": "vllm",
        "defaults": {
            "served_model_name": "my-alias",
            "port": 8000,
            "tensor_parallel": 2,
        },
    })

    runtime = VllmRayRuntime()
    cmd = runtime.generate_command(
        recipe, {}, is_cluster=False, skip_keys={"served_model_name"}
    )

    assert "--served-model-name" not in cmd
    assert "my-alias" not in cmd
    assert "--port" in cmd
    assert "8000" in cmd
    assert "-tp" in cmd
    assert "2" in cmd


def test_sglang_generate_command_skip_keys():
    """Test SglangRuntime with skip_keys removes specified flags."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "org/model",
        "runtime": "sglang",
        "defaults": {
            "served_model_name": "my-alias",
            "port": 8000,
            "tensor_parallel": 2,
        },
    })

    runtime = SglangRuntime()
    cmd = runtime.generate_command(
        recipe, {}, is_cluster=False, skip_keys={"served_model_name"}
    )

    assert "--served-model-name" not in cmd
    assert "my-alias" not in cmd
    assert "--port" in cmd
    assert "8000" in cmd
    assert "--tp-size" in cmd
    assert "2" in cmd
