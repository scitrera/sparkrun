# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Recipe defaults project to API-specific profiles without new workloads."""

from copy import deepcopy
from types import SimpleNamespace
from unittest.mock import patch
import pytest

from sparkrun.core.recipe import Recipe
from sparkrun.plugins.sparkroute.recipe_config import parse_sparkroute, SparkrouteRecipeError
from sparkrun.plugins.sparkroute.projection import ProjectedBinding, ProjectionError, resolve_bindings, dedupe_bindings, build_sparkrun_set
from sparkrun.plugins.sparkroute.engine import SparkrouteEngine, SparkrouteConfigError
from sparkrun.plugins.sparkroute.operations import _catalog
from sparkrun.plugins.sparkroute.protocol import Request, ProtocolError
from sparkrun.runtimes.vllm_distributed import VllmDistributedRuntime

PROFILES = {
    "low": {
        "chat_completions": {"chat_template_kwargs": {"enable_thinking": False}, "temperature": 0.2},
        "responses": {"reasoning": {"effort": "low"}},
    }
}
SETTINGS = {"capabilities": ["vision"], "request_profiles": PROFILES}


def binding(revision="one", profiles=None, model="coding"):
    return ProjectedBinding(
        recipe="@test/" + revision,
        recipe_revision=revision,
        model=model,
        virtual_model=model,
        cluster_candidates=["lab"],
        request_profiles=profiles or {},
    )


def test_recipe_defaults_project_and_do_not_change_deployment_or_binding_identity():
    recipe = Recipe({"model": "test/model", "runtime": "vllm", "container": "nightly:latest", "defaults": {"served_model_name": "coding"}})
    with (
        patch("sparkrun.api.resolve_catalog_recipe", return_value=(recipe, {})),
        patch("sparkrun.api._resolve.resolve_runtime", return_value=VllmDistributedRuntime()),
    ):
        before = build_sparkrun_set(resolve_bindings([{"recipe": "@test/coder"}]))
        recipe.sparkroute = deepcopy(SETTINGS)
        after = build_sparkrun_set(resolve_bindings([{"recipe": "@test/coder"}]), {"code": "coding"})
    deployment = after["deployments"][0]
    assert len(after["deployments"]) == 1
    assert deployment["capabilities"] == ["responses", "vision"]
    assert deployment["endpoint_source"] == before["deployments"][0]["endpoint_source"]
    models = {model["name"]: model for model in after["virtual_models"]}
    assert models["coding:low"]["aliases"] == ["code:low"]
    assert models["coding:low"]["pools"] == models["coding"]["pools"]
    assert models["coding:low"]["request_overrides"] == PROFILES["low"]
    assert "request_overrides" not in models["coding"]
    assert "required_capabilities" not in models["coding"]  # declaring vision does not require it on every request
    models["coding:low"]["request_overrides"]["responses"]["reasoning"]["effort"] = "high"
    assert recipe.sparkroute == SETTINGS


def test_profile_routes_only_to_recipes_that_declare_it_and_matching_definitions_merge():
    one, two, three = binding("one", PROFILES), binding("two"), binding("three", PROFILES)
    document = build_sparkrun_set([three, two, one])
    models = {model["name"]: model for model in document["virtual_models"]}
    assert len(models["coding"]["pools"][0]["targets"]) == 3
    assert [target["deployment"] for target in models["coding:low"]["pools"][0]["targets"]] == ["sparkrun:one", "sparkrun:three"]
    assert document == build_sparkrun_set([one, two, three])


@pytest.mark.parametrize("same_identity", [False, True])
def test_conflicting_recipe_profiles_fail_instead_of_selecting_arbitrary_parameters(same_identity):
    one = binding("one", PROFILES)
    other = binding("one" if same_identity else "two", {"low": {"chat_completions": {"temperature": 1}}})
    with pytest.raises(ProjectionError, match="conflicting request profile"):
        build_sparkrun_set(dedupe_bindings([one, other]))


@pytest.mark.parametrize("alias", [False, True])
def test_profiles_cannot_shadow_an_existing_model_or_alias(alias):
    entries = [binding(profiles=PROFILES)]
    if not alias:
        entries.append(binding("two", model="coding:low"))
    with pytest.raises(ProjectionError, match="conflicts with existing"):
        build_sparkrun_set(entries, {"coding:low": "coding"} if alias else {})


@pytest.mark.parametrize(
    "value",
    [
        None,
        [],
        {"unknown": True},
        {"capabilities": "vision"},
        {"capabilities": ["vision", "vision"]},
        {"capabilities": ["Vision"]},
        {"capabilities": ["unknown"]},
        {"request_profiles": []},
        {"request_profiles": {":low": PROFILES["low"]}},
        {"request_profiles": {"low": {}}},
        {"request_profiles": {"low": {"unsupported": {"temperature": 0}}}},
        {"request_profiles": {"low": {"chat_completions": "{}"}}},
    ],
)
def test_malformed_recipe_configuration_is_rejected(value):
    with pytest.raises(SparkrouteRecipeError, match="sparkroute"):
        parse_sparkroute(value)


@pytest.mark.parametrize("parameter", ["model", "messages", "stream", "input", "system", "previous_response_id", "tools", "extra_body"])
def test_profiles_cannot_replace_request_structure(parameter):
    with pytest.raises(SparkrouteRecipeError, match="protected"):
        parse_sparkroute({"request_profiles": {"low": {"responses": {parameter: "bad"}}}})


@pytest.mark.parametrize("value", [float("nan"), float("inf"), {1: "numeric key"}, {"too_large": "x" * (256 * 1024)}, {"set"}, b"bytes"])
def test_yaml_values_must_be_bounded_json(value):
    with pytest.raises(SparkrouteRecipeError):
        parse_sparkroute({"request_profiles": {"low": {"messages": {"thinking": value}}}})


def test_bridge_preview_enriches_capabilities_and_reports_field_errors():
    details = {"name": "@test/coder", "capabilities": ["responses"], "sparkroute": SETTINGS}
    with patch("sparkrun.api.get_recipe_details", return_value=details):
        preview = _catalog(Request("test", "catalog_resolve", arguments={"reference": "@test/coder"}), None)
    assert preview["capabilities"] == ["responses", "vision"]
    assert preview["sparkroute"] == SETTINGS
    details["sparkroute"] = {"request_profiles": {"low": {"responses": {"model": "bad"}}}}
    with patch("sparkrun.api.get_recipe_details", return_value=details), pytest.raises(ProtocolError, match="protected"):
        _catalog(Request("test", "catalog_resolve", arguments={"reference": "@test/coder"}), None)


def test_discovery_preserves_only_common_profiles_for_a_shared_upstream(tmp_path):
    engine = SparkrouteEngine(host="127.0.0.1", port=8000, state_dir=tmp_path / "proxy")
    endpoint = SimpleNamespace(healthy=True, actual_models=["coding"], native_protocols=["openai"], capabilities=[], sparkroute=SETTINGS)
    engine._persist_discovered_apis([endpoint])
    doc = build_sparkrun_set([], discovered_models=["coding"], discovered_apis=engine._read_discovered_apis())
    assert doc["deployments"][0]["capabilities"] == ["vision"]
    assert doc["virtual_models"][1]["request_overrides"] == PROFILES["low"]
    plain = SimpleNamespace(**{**vars(endpoint), "sparkroute": {}})
    engine._persist_discovered_apis([endpoint, plain])
    assert "request_profiles" not in engine._read_discovered_apis()["coding"]
    different = SimpleNamespace(**{**vars(endpoint), "sparkroute": {"request_profiles": {"low": {"responses": {"temperature": 1}}}}})
    with pytest.raises(SparkrouteConfigError, match="conflicting request profiles"):
        engine._persist_discovered_apis([endpoint, different])


def test_discovery_profile_conflicts_do_not_depend_on_endpoint_order(tmp_path):
    from itertools import permutations

    engine = SparkrouteEngine(host="127.0.0.1", port=8000, state_dir=tmp_path / "proxy")
    endpoints = [
        SimpleNamespace(healthy=True, actual_models=["coding"], sparkroute=settings)
        for settings in (SETTINGS, {}, {"request_profiles": {"low": {"chat_completions": {"temperature": 1}}}})
    ]
    for ordered in permutations(endpoints):
        with pytest.raises(SparkrouteConfigError, match="conflicting request profiles"):
            engine._persist_discovered_apis(list(ordered))
