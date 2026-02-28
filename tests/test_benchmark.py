"""Tests for benchmarking framework and benchmark spec handling."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import patch
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
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime
from sparkrun.runtimes.sglang import SglangRuntime
from sparkrun.runtimes.llama_cpp import LlamaCppRuntime


def _write_yaml(path: Path, data: dict):
    path.write_text(yaml.safe_dump(data))


# ========== BenchmarkSpec Loading Tests ==========


def test_benchmark_load_valid(tmp_path: Path):
    """Test loading a valid benchmark YAML with nested args format."""
    p = tmp_path / "bench.yaml"
    _write_yaml(p, {
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
    assert spec.args["format"] == "csv"


def test_benchmark_load_missing_block(tmp_path: Path):
    """Test that loading fails when benchmark block is missing."""
    p = tmp_path / "bench.yaml"
    _write_yaml(p, {"recipe": "x"})

    with pytest.raises(BenchmarkError, match="benchmark"):
        BenchmarkSpec.load(p)


def test_benchmark_load_standalone_profile(tmp_path: Path):
    """Test loading a standalone profile file (no benchmark: wrapper)."""
    p = tmp_path / "profile.yaml"
    _write_yaml(p, {
        "framework": "llama-benchy",
        "metadata": {"description": "Test profile"},
        "args": {
            "pp": [2048],
            "depth": [0, 4096],
            "concurrency": [1, 2, 5],
        },
    })

    spec = BenchmarkSpec.load(p)
    assert spec.framework == "llama-benchy"
    assert spec.args["pp"] == [2048]
    assert spec.args["depth"] == [0, 4096]
    assert spec.args["concurrency"] == [1, 2, 5]
    # metadata should NOT leak into args
    assert "metadata" not in spec.args


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
    assert spec.args["pp"] == [2048]


def test_benchmark_load_with_timeout(tmp_path: Path):
    """Test loading a benchmark profile with a timeout value."""
    p = tmp_path / "bench.yaml"
    _write_yaml(p, {
        "benchmark": {
            "framework": "llama-benchy",
            "timeout": 7200,
            "args": {"pp": [2048]},
        },
    })

    spec = BenchmarkSpec.load(p)
    assert spec.timeout == 7200
    assert spec.args["pp"] == [2048]
    # timeout should NOT leak into args
    assert "timeout" not in spec.args


def test_benchmark_load_standalone_with_timeout(tmp_path: Path):
    """Test loading a standalone profile with timeout."""
    p = tmp_path / "profile.yaml"
    _write_yaml(p, {
        "framework": "llama-benchy",
        "timeout": 1800,
        "args": {"pp": [4096]},
    })

    spec = BenchmarkSpec.load(p)
    assert spec.timeout == 1800
    assert spec.args["pp"] == [4096]
    assert "timeout" not in spec.args


def test_benchmark_load_no_timeout(tmp_path: Path):
    """Test that timeout defaults to None when not specified."""
    p = tmp_path / "bench.yaml"
    _write_yaml(p, {
        "benchmark": {
            "framework": "llama-benchy",
            "args": {"pp": [2048]},
        },
    })

    spec = BenchmarkSpec.load(p)
    assert spec.timeout is None


def test_benchmark_from_recipe_with_timeout(tmp_path: Path):
    """Test BenchmarkSpec.from_recipe() extracts timeout from recipe benchmark block."""
    recipe = Recipe.from_dict({
        "name": "test-recipe",
        "model": "org/model",
        "runtime": "vllm",
        "benchmark": {
            "framework": "llama-benchy",
            "timeout": 5400,
            "args": {"pp": [2048]},
        },
    })

    spec = BenchmarkSpec.from_recipe(recipe)
    assert spec is not None
    assert spec.timeout == 5400
    assert "timeout" not in spec.args


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


def test_llama_benchy_build_command_alias():
    """Test that shorthand aliases are resolved to canonical flag names."""
    fw = LlamaBenchyFramework()
    # Profile uses 'prefix_caching' shorthand instead of 'enable_prefix_caching'
    args = {"prefix_caching": True, "pp": [2048]}
    cmd = fw.build_benchmark_command(
        target_url="http://localhost:8000/v1",
        model="org/model",
        args=args,
    )

    assert "--enable-prefix-caching" in cmd
    assert "--prefix-caching" not in cmd


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


def test_strip_flag_alias():
    """Test stripping a short-form flag alias (e.g. -a for --alias)."""
    command = "llama-server -hf org/model -a my-alias --port 8000"
    flag_map = {"served_model_name": "--alias", "port": "--port"}
    flag_aliases = {"served_model_name": ["-a"]}

    result = RuntimePlugin.strip_flags_from_command(
        command, skip_keys={"served_model_name"}, flag_map=flag_map,
        bool_keys=set(), flag_aliases=flag_aliases,
    )

    assert "-a" not in result
    assert "my-alias" not in result
    assert "--port 8000" in result


def test_strip_flag_canonical_and_alias():
    """Test that both canonical --alias and short -a are stripped."""
    # Canonical form
    command1 = "llama-server -hf org/model --alias my-alias --port 8000"
    flag_map = {"served_model_name": "--alias", "port": "--port"}
    flag_aliases = {"served_model_name": ["-a"]}

    result1 = RuntimePlugin.strip_flags_from_command(
        command1, skip_keys={"served_model_name"}, flag_map=flag_map,
        bool_keys=set(), flag_aliases=flag_aliases,
    )
    assert "--alias" not in result1
    assert "my-alias" not in result1
    assert "--port 8000" in result1

    # Short form
    command2 = "llama-server -hf org/model -a my-alias --port 8000"
    result2 = RuntimePlugin.strip_flags_from_command(
        command2, skip_keys={"served_model_name"}, flag_map=flag_map,
        bool_keys=set(), flag_aliases=flag_aliases,
    )
    assert "-a" not in result2
    assert "my-alias" not in result2
    assert "--port 8000" in result2


def test_strip_multiline_continuation():
    """Test stripping a flag from a multi-line command with backslash continuations."""
    command = (
        "python3 -m sglang.launch_server \\\n"
        "    --model-path Qwen/Qwen3-1.7B \\\n"
        "    --served-model-name Qwen/Qwen3-1.7B \\\n"
        "    --mem-fraction-static 0.3 \\\n"
        "    --tp-size 1 \\\n"
        "    --host 0.0.0.0 \\\n"
        "    --port 8000"
    )
    flag_map = {
        "served_model_name": "--served-model-name",
        "port": "--port",
        "tp": "--tp-size",
    }

    result = RuntimePlugin.strip_flags_from_command(
        command,
        skip_keys={"served_model_name"},
        flag_map=flag_map,
        bool_keys=set(),
    )

    assert "--served-model-name" not in result
    assert "Qwen/Qwen3-1.7B" in result  # model-path still present
    assert "--mem-fraction-static 0.3" in result
    # No double backslash artifacts
    assert "\\ \\" not in result


# ========== export_results Tests ==========


def test_export_results_writes_yaml(tmp_path: Path):
    """Test export_results creates YAML with expected structure."""
    recipe = Recipe.from_dict({
        "name": "test-recipe",
        "model": "org/model",
        "runtime": "vllm",
        "metadata": {
            "model_dtype": "bfloat16",
        },
    })
    recipe.source_registry = "test-registry"
    recipe.source_registry_url = "https://github.com/example/recipes.git"

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

    # Recipe section
    assert bench["recipe"]["name"] == "unnamed"  # from_dict has no source_path
    assert bench["recipe"]["model"] == "org/model"
    assert bench["recipe"]["type"] == "sparkrun"
    # Note: Recipe runtime resolution converts "vllm" to "vllm-distributed"
    assert bench["recipe"]["runtime"] == "vllm-distributed"
    assert bench["recipe"]["registry"] == "test-registry"
    assert bench["recipe"]["registry_git"] == "https://github.com/example/recipes.git"
    assert bench["recipe"]["text"]  # non-empty YAML text
    assert len(bench["recipe"]["hash"]) == 64  # SHA-256 hex digest

    # Model section
    assert bench["model"]["dtype"] == "bfloat16"

    # Cluster section (hosts excluded for privacy)
    assert "hosts" not in bench["cluster"]
    assert bench["cluster"]["tp"] == 2
    assert bench["cluster"]["cluster_id"] == "test-cluster-123"

    # Benchmark section
    assert bench["benchmark"]["framework"] == "llama-benchy"
    assert bench["benchmark"]["profile"] == "default"
    assert bench["benchmark"]["args"] == {"pp": [2048], "depth": [0]}

    assert bench["results"]["rows"] == [{"pp": "2048", "tg": "32"}]


def test_export_results_model_revision(tmp_path: Path):
    """Test export_results includes model_revision when set."""
    recipe = Recipe.from_dict({
        "name": "test-recipe",
        "model": "org/model",
        "model_revision": "abc123",
        "runtime": "vllm",
        "metadata": {
            "model_dtype": "float16",
            "num_layers": 32,
            "num_kv_heads": 8,
            "head_dim": 128,
            "model_params": 7000000000,
        },
    })

    output_path = tmp_path / "results.yaml"
    export_results(
        recipe=recipe,
        hosts=["host1"],
        tp=1,
        cluster_id="cluster-1",
        framework_name="llama-benchy",
        profile_name=None,
        args={},
        results={"rows": []},
        output_path=output_path,
    )

    data = yaml.safe_load(output_path.read_text())
    model = data["sparkrun_benchmark"]["model"]
    assert model["revision"] == "abc123"
    assert model["dtype"] == "float16"
    assert model["num_layers"] == 32
    assert model["num_kv_heads"] == 8
    assert model["head_dim"] == 128
    assert model["params"] == 7000000000


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


# ========== skip_keys in generate_node_command Tests ==========
# These verify that native-cluster runtimes propagate skip_keys when
# regenerating the serve command internally (not using the pre-built
# serve_command string).


def test_sglang_generate_node_command_skip_keys():
    """skip_keys propagates through sglang generate_node_command."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "org/model",
        "runtime": "sglang",
        "defaults": {
            "served_model_name": "my-alias",
            "port": 8000,
        },
    })

    runtime = SglangRuntime()
    cmd = runtime.generate_node_command(
        recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=0,
        skip_keys={"served_model_name"},
    )

    assert "--served-model-name" not in cmd
    assert "my-alias" not in cmd
    assert "--port" in cmd
    assert "--nnodes 2" in cmd
    assert "--node-rank 0" in cmd


def test_vllm_distributed_generate_node_command_skip_keys():
    """skip_keys propagates through vllm-distributed generate_node_command."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "org/model",
        "runtime": "vllm-distributed",
        "defaults": {
            "served_model_name": "my-alias",
            "port": 8000,
        },
    })

    runtime = VllmDistributedRuntime()
    cmd = runtime.generate_node_command(
        recipe, {}, head_ip="10.0.0.1", num_nodes=2, node_rank=1,
        skip_keys={"served_model_name"},
    )

    assert "--served-model-name" not in cmd
    assert "my-alias" not in cmd
    assert "--port" in cmd
    assert "--nnodes 2" in cmd
    assert "--node-rank 1" in cmd
    assert "--headless" in cmd


def test_llama_cpp_build_rpc_head_command_skip_keys():
    """skip_keys propagates through llama-cpp _build_rpc_head_command."""
    recipe = Recipe.from_dict({
        "name": "test",
        "model": "org/model",
        "runtime": "llama-cpp",
        "defaults": {
            "served_model_name": "my-alias",
            "port": 8000,
        },
    })

    runtime = LlamaCppRuntime()
    config = recipe.build_config_chain({})
    cmd = runtime._build_rpc_head_command(
        recipe, config, worker_hosts=["10.0.0.2"],
        rpc_port=50052, skip_keys={"served_model_name"},
    )

    assert "--alias" not in cmd
    assert "my-alias" not in cmd
    assert "--port" in cmd
    assert "--rpc 10.0.0.2:50052" in cmd
