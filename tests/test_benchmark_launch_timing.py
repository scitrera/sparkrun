"""Launch-stage timing in the benchmark provenance artifact.

The numbers land nested under ``metadata["timing"]``, beside the existing
benchmark ``start``/``end``/``duration`` — additively, because that mapping
is the Spark Arena submission and consumers key off the existing shape.
"""

from __future__ import annotations

from dataclasses import replace
from datetime import datetime, timezone
from unittest.mock import MagicMock

import pytest
import yaml

from sparkrun.benchmarking.base import BenchmarkResult, export_results, startup_timing_metadata
from sparkrun.core.launcher import ServeReadiness
from sparkrun.core.timing import Timeline


def _recipe() -> MagicMock:
    recipe = MagicMock()
    recipe.name = "r"
    recipe.qualified_name = "@reg/r"
    recipe.container = "img:latest"
    recipe.model = "org/model"
    recipe.runtime = "vllm-distributed"
    recipe.metadata = {}
    recipe.model_revision = None
    recipe.source_registry = "reg"
    recipe.source_registry_url = ""
    recipe.export.return_value = "yaml"
    return recipe


def _result(*, timeline=None, readiness=None, resumed=False) -> BenchmarkResult:
    br = BenchmarkResult()
    now = datetime.now(timezone.utc)
    br.start_time = now
    br.end_time = now
    br.recipe = _recipe()
    br.overrides = {}
    br.cluster_id = "cid"
    br.host_list = ["h1"]
    br.container_image = "img:latest"
    br.framework = MagicMock()
    br.framework.framework_name = "llama-benchy"
    br.profile = "default"
    br.benchmark_args = {}
    br.resumed = resumed
    br.readiness = readiness
    if timeline is not None:
        launch_result = MagicMock()
        launch_result.builder = None
        launch_result.timeline = timeline
        # generate_metadata prefers launch_result fields when present.
        launch_result.recipe = br.recipe
        launch_result.overrides = {}
        launch_result.cluster_id = "cid"
        launch_result.host_list = ["h1"]
        launch_result.container_image = "img:latest"
        launch_result.runtime_info = {}
        br.launch_result = launch_result
    return br


def _readiness(port_s: float, health_s: float) -> ServeReadiness:
    return ServeReadiness(
        ready=True,
        head_host="h1",
        head_ip="10.0.0.1",
        port=8000,
        container="cid_node_0",
        port_wait_s=port_s,
        health_wait_s=health_s,
    )


def test_timing_carries_the_full_span_list_and_serve_ready_breakdown():
    tl = Timeline()
    with tl.span("launch"):
        with tl.span("launch.distribute", mode="push"):
            pass

    meta = _result(timeline=tl, readiness=_readiness(41.5, 88.25)).generate_metadata()
    timing = meta["timing"]

    # Existing keys are untouched — this is an additive change to a
    # published artifact.
    assert {"start", "end", "duration"} <= set(timing)

    assert timing["serve_ready"] == {"port_open_s": 41.5, "health_ok_s": 88.25, "total_s": 129.75}

    names = [s["name"] for s in timing["launch"]["spans"]]
    assert names == ["launch", "launch.distribute"]
    assert timing["launch"]["spans"][1]["attrs"]["mode"] == "push"
    assert "wall_origin" in timing["launch"]


def test_nothing_launched_omits_the_keys_rather_than_zeroing_them():
    """``--skip-run`` measured a workload it did not start.

    A recorded ``total_s: 0.0`` would read as an instantaneous launch.
    """
    timing = _result().generate_metadata()["timing"]
    assert "serve_ready" not in timing
    assert "launch" not in timing


def test_resumed_run_does_not_report_launch_timings():
    """A resumed run re-emits recorded results; its launch was a different one.

    Same reasoning as ``measured_at`` for the benchmark numbers themselves
    (issue #267) — provenance must not attribute a stale launch to this
    result.
    """
    tl = Timeline()
    with tl.span("launch"):
        pass

    timing = _result(timeline=tl, readiness=_readiness(1.0, 2.0), resumed=True).generate_metadata()["timing"]
    assert "serve_ready" not in timing
    assert "launch" not in timing
    # The benchmark's own timing keys still apply to this invocation.
    assert "duration" in timing


def _observation(**changes):
    return {
        "format": 1,
        "measurement": "sparkrun-rank0-v1",
        "observer": "rank0",
        "container_id": "current-container",
        "container_started_unix_ns": 1_000_000_000,
        "observer_started_unix_ns": 1_125_000_000,
        "port_open_unix_ns": 2_250_000_000,
        "http_ready_unix_ns": 3_500_000_000,
        "first_token_unix_ns": 4_750_000_000,
        "first_token_field": "reasoning_content",
        "inference_ready": True,
        "response_validated": False,
        "http_ready_path": "/health",
        "prompt_sha256": "a" * 64,
        "max_tokens": 64,
        "temperature": 0,
        "request_ttft_seconds": 0.125,
        **changes,
    }


def _export(br, path):
    export_results(
        recipe=br.recipe,
        hosts=br.host_list,
        tp=1,
        cluster_id=br.cluster_id,
        framework_name="llama-benchy",
        profile_name=None,
        args={},
        results={"request_ttft": 0.1},
        output_path=path,
        readiness=br.readiness,
        resumed=br.resumed,
    )
    return yaml.safe_load(path.read_text())["sparkrun_benchmark"]


def test_yaml_arena_and_api_export_identical_startup_metrics(tmp_path):
    from sparkrun.api import BenchmarkOptions
    from sparkrun.api._benchmark import _build_result

    readiness = replace(_readiness(91, 92), startup_observation=_observation())
    br = _result(readiness=readiness)
    exported = _export(br, tmp_path / "benchmark.yaml")
    arena = br.generate_metadata()
    api_result = _build_result(BenchmarkOptions(recipe="r"), br)
    startup = exported["timing"]["startup"]
    assert startup == arena["timing"]["startup"] == api_result.metadata["timing"]["startup"]
    assert startup == {
        "format": 1,
        "measurement": "sparkrun-rank0-v1",
        "observer": "rank0",
        "start_boundary": "docker.State.StartedAt",
        "ttft_status": "measured",
        "inference_requested": True,
        "inference_ready": True,
        "ttr_port_open_s": 1.25,
        "ttr_http_ready_s": 2.5,
        "ttft_s": 3.75,
        "observer_start_delay_s": 0.125,
        "first_token_field": "reasoning_content",
        "response_validated": False,
        "http_ready_path": "/health",
        "prompt_sha256": "a" * 64,
        "max_tokens": 64,
        "temperature": 0,
        "request_ttft_s": 0.125,
    }
    # Legacy endpoint waits and framework request timings retain their meanings.
    assert arena["timing"]["serve_ready"] == {"port_open_s": 91, "health_ok_s": 92, "total_s": 183}
    assert exported["results"] == {"request_ttft": 0.1}
    assert exported["version"] == "1"


def test_endpoint_only_exports_ttr_with_ttft_not_applicable(tmp_path):
    obs = _observation(inference_requested=False, endpoint_ready=True, inference_ready=False)
    del obs["first_token_unix_ns"], obs["first_token_field"]
    readiness = replace(_readiness(1, 2), startup_observation=obs)
    meta = _export(_result(readiness=readiness), tmp_path / "benchmark.yaml")["timing"]["startup"]
    assert meta["ttft_status"] == "not_applicable"
    assert meta["ttr_port_open_s"] == 1.25 and meta["ttr_http_ready_s"] == 2.5
    assert meta["inference_requested"] is False and meta["inference_ready"] is False
    assert not {"ttft_s", "request_ttft_s", "prompt_sha256", "max_tokens", "temperature", "first_token_field"} & meta.keys()


def test_coldsnap_acceptance_keeps_profile_without_inventing_optional_metrics():
    obs = _observation(measurement="rank0-acceptance-v1", response_validated=True)
    for key in ("port_open_unix_ns", "http_ready_unix_ns", "temperature", "request_ttft_seconds"):
        del obs[key]
    meta = startup_timing_metadata(replace(_readiness(0, 0), startup_observation=obs))
    assert meta["measurement"] == "rank0-acceptance-v1" and meta["response_validated"] is True
    assert meta["ttft_s"] == 3.75
    assert not {"ttr_port_open_s", "ttr_http_ready_s", "temperature", "request_ttft_s"} & meta.keys()


@pytest.mark.parametrize("case", ["skip", "resumed", "legacy", "failed", "invalid", "unknown", "malformed"])
def test_unavailable_observations_omitted_everywhere(tmp_path, case):
    from sparkrun.api import BenchmarkOptions
    from sparkrun.api._benchmark import _build_result

    obs = _observation()
    if case == "invalid":
        obs["first_token_unix_ns"] = 0
    elif case == "unknown":
        obs["measurement"] = "future-profile"
    elif case == "malformed":
        obs["measurement"] = []
    elif case == "legacy":
        obs = {}
    readiness = replace(_readiness(12, 34), ready=case != "failed", startup_observation=obs)
    br = _result(readiness=None if case == "skip" else readiness, resumed=case == "resumed")
    assert "timing" not in _export(br, tmp_path / "benchmark.yaml")
    assert "startup" not in br.generate_metadata()["timing"]
    assert "timing" not in _build_result(BenchmarkOptions(recipe="r"), br).metadata


def test_startup_metadata_allowlist_excludes_private_or_invalid_optional_fields(tmp_path):
    obs = _observation(
        prompt="private prompt",
        content="private response",
        api_key="private key",
        host="192.168.10.4",
        container_name="private container",
        future_field={"secret": "private"},
        prompt_sha256="private hash",
        max_tokens=True,
        temperature=float("nan"),
        request_ttft_seconds=-1,
        observer_started_unix_ns=0,
        http_ready_path="/private/path",
        response_validated="private",
    )
    br = _result(readiness=replace(_readiness(1, 2), startup_observation=obs))
    meta = _export(br, tmp_path / "benchmark.yaml")["timing"]["startup"]
    text = yaml.safe_dump(meta)
    assert "private" not in text and "192.168.10.4" not in text and "current-container" not in text
    assert (
        not {
            "prompt_sha256",
            "max_tokens",
            "temperature",
            "request_ttft_s",
            "observer_start_delay_s",
            "http_ready_path",
            "response_validated",
        }
        & meta.keys()
    )
