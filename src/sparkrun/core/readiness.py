"""Effective readiness policy: recipe > global config > built-in defaults."""

from __future__ import annotations

import math
from collections.abc import Mapping
from dataclasses import asdict, dataclass
from typing import Any

from scitrera_app_framework.api import EnvPlacement, Variables

# Engine initialization, weight loading and graph capture can all precede port
# binding. Generous budgets avoid mistaking a slow engine for a dead workload;
# the probes also check container liveness and honor cancellation.
DEFAULT_PORT_READY_TIMEOUT_S = 1800.0
DEFAULT_HEALTH_READY_TIMEOUT_S = 900.0


@dataclass(frozen=True)
class ReadinessSettings:
    port_timeout_s: float = DEFAULT_PORT_READY_TIMEOUT_S
    health_timeout_s: float = DEFAULT_HEALTH_READY_TIMEOUT_S
    inference: bool = True
    inference_timeout_s: float = 120.0
    inference_prompt: str = "Reply with exactly: sparkrun-ready"


def _normalize(key: str, value: Any) -> Any:
    if key == "inference":
        if type(value) is not bool:
            raise ValueError("readiness.inference must be a boolean")
        return value
    if key == "inference_prompt":
        if not isinstance(value, str) or not value.strip():
            raise ValueError("readiness.inference_prompt must be a non-empty string")
        return value
    if key in {"port_timeout_s", "health_timeout_s", "inference_timeout_s"}:
        try:
            seconds = float(value)
        except (TypeError, ValueError, OverflowError):
            seconds = math.nan
        if (
            isinstance(value, bool)
            or math.isnan(seconds)
            or (key == "inference_timeout_s" and (not math.isfinite(seconds) or seconds <= 0))
        ):
            raise ValueError(
                "readiness.%s must be %s" % (key, "a finite positive timeout" if key == "inference_timeout_s" else "a timeout")
            )
        return seconds if seconds > 0 else math.inf
    raise ValueError("unknown readiness setting %r" % key)


def parse_recipe_readiness(value: Any) -> dict[str, Any]:
    """Validate the explicit recipe layer without baking in inherited defaults."""
    if not isinstance(value, Mapping):
        raise ValueError("readiness must be a mapping")
    for key, setting in value.items():
        _normalize(key, setting)
    return dict(value)


def resolve_readiness_settings(*, config=None, recipe=None) -> ReadinessSettings:
    """Resolve one immutable policy without mutating the global or recipe layer.

    Existing global settings are permissive: invalid values fall back to the
    built-in default for that field. Explicit recipe errors fail at load time.
    Environment variables and runtime flag defaults do not enter this chain.
    """
    defaults = asdict(ReadinessSettings())
    get = getattr(config, "get", None)
    raw_global = get("readiness", {}) if callable(get) else {}
    global_layer = {}
    if isinstance(raw_global, Mapping):
        for key, value in raw_global.items():
            try:
                global_layer[key] = _normalize(key, value)
            except ValueError:
                continue
    raw_recipe = getattr(recipe, "readiness", {})
    recipe_layer = {key: _normalize(key, value) for key, value in parse_recipe_readiness(raw_recipe).items()}
    chain = Variables(sources=(recipe_layer, global_layer, defaults), env_placement=EnvPlacement.IGNORED)
    return ReadinessSettings(**{key: chain.get(key) for key in defaults})
