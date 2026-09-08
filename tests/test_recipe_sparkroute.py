"""Gateway annotations survive recipe transport without affecting serve identity."""

from copy import deepcopy
import pytest
from sparkrun.core.recipe import Recipe, RecipeError
from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint

BASE = {"model": "test/model", "runtime": "vllm", "container": "test:latest"}
SETTINGS = {"capabilities": ["vision"], "request_profiles": {"low": {"chat_completions": {"temperature": 0.2}}}}


def test_annotations_round_trip_without_plugin_or_workload_identity_changes():
    recipe = Recipe({**BASE, "sparkroute": deepcopy(SETTINGS)})
    assert "sparkroute" not in recipe.runtime_config
    assert derive_recipe_fingerprint(recipe) == derive_recipe_fingerprint(Recipe(BASE))
    assert Recipe(recipe.to_dict()).sparkroute == SETTINGS
    restored = Recipe._deserialize_yaml(recipe._serialize_yaml())
    assert restored.sparkroute == SETTINGS
    assert derive_recipe_fingerprint(restored) == derive_recipe_fingerprint(recipe)
    exported = recipe.to_dict()
    exported["sparkroute"]["request_profiles"]["low"]["chat_completions"]["temperature"] = 1
    assert recipe.sparkroute == SETTINGS
    state = recipe.__getstate__()
    state["sparkroute"]["capabilities"].clear()
    assert recipe.sparkroute == SETTINGS


@pytest.mark.parametrize("value", [None, [], "vision", True])
def test_annotation_container_must_be_mapping(value):
    with pytest.raises(RecipeError, match="sparkroute must be a mapping"):
        Recipe({**BASE, "sparkroute": value})


def test_absent_annotations_do_not_pollute_export():
    assert "sparkroute" not in Recipe(BASE).to_dict()
