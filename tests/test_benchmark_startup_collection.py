"""Benchmark orchestration collects readiness before framework work and export."""

import json
import os
import subprocess
import sys
from contextlib import nullcontext
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

import pytest
import yaml

from sparkrun.api import BenchmarkOptions, ResumeMode, benchmark
from sparkrun.api._context import default_sctx
from sparkrun.api._errors import BenchmarkFailed
from sparkrun.benchmarking.base import BenchmarkingPlugin
from sparkrun.core.bootstrap import get_runtime
from sparkrun.core.launcher import LaunchResult, ServeReadiness
from sparkrun.core.recipe import Recipe
from sparkrun.core.timing import Timeline
from sparkrun.orchestration.startup import run_probe
from sparkrun.orchestration.executors.docker import DockerExecutor

# Share the local SSE fixture with the probe's contract tests.
from test_startup_observation import streaming_server as streaming_server


@pytest.fixture
def bench_env(tmp_path, monkeypatch):
    sctx = default_sctx()
    sctx.config._data["readiness"] = {"inference_timeout_s": 17, "inference_prompt": "global prompt"}
    recipe = Recipe(
        {
            "name": "startup-test",
            "model": "test/model",
            "runtime": "vllm-distributed",
            "container": "test/image:latest",
            "defaults": {"port": 8000},
            "readiness": {"inference_prompt": "recipe prompt"},
        }
    )
    runtime = get_runtime(recipe.runtime, sctx.variables)
    runtime.executor = DockerExecutor()
    launch = LaunchResult(
        rc=0,
        cluster_id="test-job",
        host_list=["localhost"],
        is_solo=True,
        runtime=runtime,
        recipe=recipe,
        overrides={},
        container_image=recipe.container,
        effective_cache_dir=str(tmp_path),
        serve_port=8000,
        config=sctx.config,
        timeline=Timeline(),
    )
    rows = {"request_ttft_seconds": 0.025}
    fw = Mock(spec=BenchmarkingPlugin)
    fw.framework_name = "test-bench"
    fw.primary_category = "performance"
    fw.passthrough_args = set()
    fw.get_default_args.return_value = {}
    fw.check_prerequisites.return_value = []
    fw.prepare_benchmark_args.return_value = {}
    fw.build_task_list.return_value = None
    fw.estimate_test_count.return_value = None
    fw.build_benchmark_command.return_value = [sys.executable, "-c", "print(" + repr(json.dumps(rows)) + ")"]
    fw.parse_results.side_effect = lambda stdout, stderr, **kw: json.loads(stdout)
    fw.measured_nothing.return_value = False
    monkeypatch.setattr("sparkrun.core.bootstrap.get_benchmarking_framework", lambda *a, **kw: fw)
    monkeypatch.setattr("sparkrun.core.resolve.load_recipe", lambda *a, **kw: (recipe, tmp_path / "recipe.yaml", None))
    monkeypatch.setattr("sparkrun.core.validation.validate_for_launch", lambda *a, **kw: ([], False))
    monkeypatch.setattr("sparkrun.api._hosts.resolve_host_list", lambda *a, **kw: ["localhost"])
    monkeypatch.setattr("sparkrun.api.plan", Mock(return_value=SimpleNamespace(host_list=["localhost"], is_solo=True)))
    run = Mock(return_value=SimpleNamespace(launch_result=launch, cluster_id=launch.cluster_id, serve_port=8000, serve_command=""))
    stop = Mock()
    monkeypatch.setattr("sparkrun.api.run", run)
    monkeypatch.setattr("sparkrun.api.stop", stop)
    monkeypatch.setattr("sparkrun.api._benchmark._resolve_running_deployment", lambda *a, **kw: (["localhost"], True, "test-job"))
    probe = Mock(
        return_value={
            "format": 1,
            "measurement": "sparkrun-rank0-v1",
            "observer": "rank0",
            "container_id": "current",
            "container_started_unix_ns": 1_000_000_000,
            "port_open_unix_ns": 2_000_000_000,
            "http_ready_unix_ns": 3_000_000_000,
            "first_token_unix_ns": 4_000_000_000,
            "first_token_field": "content",
            "inference_ready": True,
            "response_validated": False,
        }
    )
    monkeypatch.setattr("sparkrun.orchestration.startup.run_probe", probe)
    endpoint = Mock(return_value=ServeReadiness(True, "localhost", "127.0.0.1", 8000, "head", port_wait_s=12, health_wait_s=34))
    monkeypatch.setattr("sparkrun.core.launcher.wait_for_endpoint_ready", endpoint)
    options = BenchmarkOptions(
        recipe="startup-test",
        framework="test-bench",
        hosts=("localhost",),
        solo=True,
        resume=ResumeMode.FRESH,
        output_file=str(tmp_path / "benchmark.yaml"),
    )
    return SimpleNamespace(
        sctx=sctx,
        recipe=recipe,
        launch=launch,
        fw=fw,
        rows=rows,
        run=run,
        stop=stop,
        probe=probe,
        endpoint=endpoint,
        options=options,
    )


def _run(env, **changes):
    from dataclasses import replace

    result = benchmark(replace(env.options, **changes), sctx=env.sctx)
    exported = yaml.safe_load(Path(result.outputs["yaml"]).read_text())["sparkrun_benchmark"]
    assert result.results == exported["results"] == env.rows
    return result, exported


@pytest.mark.parametrize("mode", ["normal", "coldsnap", "post_hook", "endpoint_only"])
def test_benchmark_collects_one_startup_observation_and_exports_it(bench_env, mode):
    env = bench_env
    obs = env.probe.return_value
    if mode in {"coldsnap", "post_hook"}:
        env.launch.startup_observation = dict(obs)
        if mode == "coldsnap":
            env.launch.startup_observation.update(measurement="rank0-acceptance-v1", response_validated=True)
    elif mode == "endpoint_only":
        env.recipe.readiness["inference"] = False
        obs.update(inference_requested=False, endpoint_ready=True, inference_ready=False)
        del obs["first_token_unix_ns"], obs["first_token_field"]

    # Framework work starts only after the launch has its accepted observation.
    def command(**kwargs):
        assert env.launch.startup_observation
        return [sys.executable, "-c", "print(" + repr(json.dumps(env.rows)) + ")"]

    env.fw.build_benchmark_command.side_effect = command
    result, exported = _run(env)
    meta = exported["timing"]["startup"]
    assert meta == result.metadata["timing"]["startup"]
    assert meta["ttr_port_open_s"] == 1 and meta["ttr_http_ready_s"] == 2
    assert meta["ttft_status"] == ("not_applicable" if mode == "endpoint_only" else "measured")
    if mode != "endpoint_only":
        assert meta["ttft_s"] == 3
    else:
        assert "ttft_s" not in meta
    if mode in {"normal", "endpoint_only"}:
        env.probe.assert_called_once()
        config = env.probe.call_args.args[1]
        assert config["prompt"] == "recipe prompt" and config["inference_timeout_s"] == 17
        assert config["inference"] is (mode != "endpoint_only")
    else:
        env.probe.assert_not_called()
        assert meta["response_validated"] is (mode == "coldsnap")
    env.endpoint.assert_not_called()
    env.stop.assert_called_once()


@pytest.mark.parametrize("mode", ["skip", "legacy_coldsnap", "missing_launch", "other_runtime"])
def test_benchmark_does_not_invent_startup_metrics(bench_env, mode):
    env = bench_env
    if mode == "legacy_coldsnap":
        env.launch.runtime_info["inference_readiness"] = "accepted"
    elif mode == "missing_launch":
        env.run.return_value.launch_result = None
    elif mode == "other_runtime":
        env.launch.runtime = SimpleNamespace(get_family=lambda: "llama-cpp")
    result, exported = _run(env, skip_run=mode == "skip")
    assert "timing" not in result.metadata and "timing" not in exported
    env.probe.assert_not_called()
    if mode == "skip":
        env.run.assert_not_called()
        env.stop.assert_not_called()
        env.endpoint.assert_not_called()
    else:
        env.endpoint.assert_called_once()


def test_failed_startup_stops_before_framework_work_or_export(bench_env):
    env = bench_env
    env.probe.side_effect = RuntimeError("empty inference stream")
    with pytest.raises(BenchmarkFailed, match="startup readiness failed.*inference"):
        benchmark(env.options, sctx=env.sctx)
    env.fw.build_benchmark_command.assert_not_called()
    env.fw.parse_results.assert_not_called()
    env.stop.assert_called_once()
    assert not Path(env.options.output_file).exists()


@pytest.mark.parametrize("inference", [True, False])
def test_real_docker_probe_through_benchmark_yaml(bench_env, streaming_server, monkeypatch, inference):
    """CPU-only contract: real Docker clock, local HTTP/SSE, subprocess, YAML."""
    image = os.environ.get("SPARKRUN_STARTUP_TEST_DOCKER_IMAGE")
    if not image:
        pytest.skip("set SPARKRUN_STARTUP_TEST_DOCKER_IMAGE for the local Docker contract test")
    env = bench_env
    port, posts, _ = streaming_server
    container = subprocess.check_output(["docker", "run", "-d", "--network=host", image, "sleep", "60"], text=True).strip()
    try:
        env.recipe.readiness["inference"] = inference
        env.launch.serve_port = env.run.return_value.serve_port = port
        monkeypatch.setattr(env.launch.runtime, "get_head_container_name", lambda *a, **kw: container)
        monkeypatch.setattr("sparkrun.orchestration.startup.run_probe", run_probe)
        monkeypatch.setattr("sparkrun.orchestration.ssh.should_run_locally", lambda *a, **kw: True)
        result, exported = _run(env)
        meta = exported["timing"]["startup"]
        obs = env.launch.startup_observation
        assert obs["container_id"] == container
        assert len(posts) == int(inference)
        assert 0 <= meta["ttr_port_open_s"] <= meta["ttr_http_ready_s"]
        assert meta["ttft_status"] == ("measured" if inference else "not_applicable")
        if inference:
            assert meta["ttft_s"] == round((obs["first_token_unix_ns"] - obs["container_started_unix_ns"]) / 1e9, 6)
            assert meta["first_token_field"] == "reasoning_content"
            assert posts[0][0]["messages"][0]["content"] == "recipe prompt"
        else:
            assert "ttft_s" not in meta
        assert result.metadata["timing"]["startup"] == meta
    finally:
        subprocess.run(["docker", "rm", "-f", container], check=True, capture_output=True)


@pytest.mark.parametrize("prior_completed", [0, 1, 2])
def test_scheduled_export_omits_startup_when_any_results_are_reused(bench_env, monkeypatch, prior_completed):
    from sparkrun.benchmarking.run_state import BenchmarkRunState
    from sparkrun.benchmarking.scheduler import BenchTask, ScheduleRunResult

    env = bench_env
    env.fw.build_task_list.return_value = [BenchTask(0, "first"), BenchTask(1, "second")]
    env.fw.detect_version.return_value = None
    monkeypatch.setattr("sparkrun.benchmarking.progress_ui.BenchmarkProgressUI", lambda **kw: nullcontext())
    monkeypatch.setattr("sparkrun.orchestration.primitives.resolve_image_sha", lambda *a, **kw: None)
    if prior_completed:
        monkeypatch.setattr(
            "sparkrun.benchmarking.run_state.BenchmarkRunState.load",
            lambda bid, cache: BenchmarkRunState(
                benchmark_id=bid,
                cluster_id="test-job",
                recipe_qualified_name=env.recipe.qualified_name,
                framework=env.fw.framework_name,
                profile=None,
                base_args={},
                schedule=[{}, {}],
                host_list=["localhost"],
                completed_indices=list(range(prior_completed)),
                updated_at="2026-01-01T00:00:00+00:00",
            ),
        )

    def schedule(**kwargs):
        assert bool(env.launch.startup_observation) is (prior_completed == 0)
        state = kwargs["state"]
        assert len(state.completed_indices) == prior_completed
        return ScheduleRunResult(True, 2, 0, state, env.rows)

    monkeypatch.setattr("sparkrun.benchmarking.scheduler.run_schedule", schedule)
    result, exported = _run(env, resume=ResumeMode.IF_EXISTS)
    assert result.resumed is bool(prior_completed)
    if prior_completed:
        env.probe.assert_not_called()
        env.endpoint.assert_called_once()
        assert "timing" not in exported and "timing" not in result.metadata
        assert exported["benchmark"]["measured_at"] == "2026-01-01T00:00:00+00:00"
    else:
        env.probe.assert_called_once()
        assert exported["timing"]["startup"]["ttft_s"] == 3
