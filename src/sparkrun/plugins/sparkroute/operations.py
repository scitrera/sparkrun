# SPDX-FileCopyrightText: 2026 Scitrera LLC
# SPDX-FileCopyrightText: 2026 Fox Engine Ltd
# SPDX-License-Identifier: AGPL-3.0-only
# Additional permission under AGPLv3 section 7: see src/sparkrun/plugins/sparkroute/LICENSE_EXCEPTION.

"""Console-free Sparkrun operations for the gateway bridge."""

from __future__ import annotations

import logging
import time
from typing import Any

import sparkrun.api as api
from sparkrun.api._resolve import resolve_recipe
from sparkrun.orchestration.job_metadata import derive_recipe_fingerprint
from sparkrun.plugins.sparkroute.metadata import build_model_metadata
from sparkrun.plugins.sparkroute.protocol import (
    PROTOCOL_VERSION,
    SUPPORTED_OPERATIONS,
    Binding,
    ProtocolError,
    Request,
)
from sparkrun.proxy.discovery import discover_endpoints

logger = logging.getLogger(__name__)

MAX_DISCOVERED_ENDPOINTS = 256

# Readiness polling backs off: a launch that is still pulling weights will not
# become ready inside the first second, and every poll costs an HTTP probe.
READINESS_POLL_MIN_SECONDS = 0.5
READINESS_POLL_MAX_SECONDS = 5.0
READINESS_POLL_BACKOFF = 1.5

# Written into job metadata for every workload this bridge launches, and
# required before ``stop`` will tear one down.  Adoption is deliberately *not*
# gated on it: routing to someone else's healthy endpoint is harmless, while
# killing it is not.
GATEWAY_OWNER = "sparkroute"

#: Ownership marker carried on internal endpoint projections and stripped
#: before they are serialized.  The gateway decodes bridge results with Go's
#: ``DisallowUnknownFields``, so the projected key set *is* the wire contract:
#: an extra field is not ignored, it fails the whole decode.  Anything the
#: gateway should see has to be added to its ``Endpoint`` struct first.
_OWNED_KEY = "_owned"

#: Operations that change cluster state, and are therefore re-checked here.
#:
#: ``ensure_ready`` and ``stop`` let a non-interactive caller launch and tear
#: down cluster workloads.  In practice the flag is already on by the time this
#: runs — the command only exists because the plugin loaded, which the same
#: flag gates — so this is defence in depth for direct programmatic use of
#: :func:`execute`, not the primary gate.
GATED_OPERATIONS = frozenset({"ensure_ready", "stop"})

#: The single flag gating this whole integration, loading included; see
#: :mod:`sparkrun.plugins.sparkroute`.
REQUIRED_FEATURE_FLAG = "gateway.sparkroute"


def execute(request: Request) -> dict[str, Any]:
    if request.operation == "capabilities":
        from sparkrun import __version__

        return {
            "protocol_version": PROTOCOL_VERSION,
            "operations": list(SUPPORTED_OPERATIONS),
            "sparkrun_version": __version__,
        }

    if request.operation in GATED_OPERATIONS:
        _require_feature_enabled()

    sctx = api.default_sctx()
    binding = request.binding
    if request.operation == "discover" and binding is None:
        return {"endpoints": [_project(e) for e in _discover(sctx)]}
    if binding is None:  # Protocol parsing enforces this; retain a fail-closed seam.
        raise ProtocolError("invalid_request", "operation requires a binding")
    recipe, fingerprint = _resolve_binding(binding, sctx)

    if request.operation == "resolve":
        return {
            "recipe": getattr(recipe, "qualified_name", None) or getattr(recipe, "name", binding.recipe),
            "recipe_revision": fingerprint,
            "model": str(getattr(recipe, "model", "") or ""),
            "runtime": str(getattr(recipe, "runtime", "") or ""),
        }
    if request.operation == "discover":
        return {"endpoints": [_project(e) for e in _discover(sctx, fingerprint=fingerprint)]}
    if request.operation == "status":
        endpoint = _adoptable(_discover(sctx, fingerprint=fingerprint, cluster_id=request.cluster_id))
        return {
            "state": "ready" if endpoint else "offline",
            "endpoint": _project(endpoint),
        }
    if request.operation == "ensure_ready":
        return _ensure_ready(request, binding, recipe, fingerprint, sctx)
    if request.operation == "stop":
        return _stop(fingerprint, request.cluster_id, sctx)
    raise ProtocolError("unsupported_operation", "unsupported bridge operation")


def _require_feature_enabled() -> None:
    """Fail closed unless the integration is enabled."""
    from sparkrun.core.features import feature_gate_enabled

    if feature_gate_enabled(REQUIRED_FEATURE_FLAG):
        return
    raise ProtocolError(
        "feature_disabled",
        "the SparkRoute gateway integration is disabled; enable it with: sparkrun setup features enable %s" % REQUIRED_FEATURE_FLAG,
    )


def _resolve_binding(binding: Binding, sctx):
    try:
        recipe = resolve_recipe(binding.recipe, sctx=sctx, overrides=binding.overrides)
        fingerprint = derive_recipe_fingerprint(recipe, binding.overrides)
    except api.RecipeNotFound as exc:
        raise ProtocolError("recipe_not_found", "configured recipe could not be resolved") from exc
    except api.SparkrunError as exc:
        raise ProtocolError("resolve_failed", "configured recipe could not be resolved") from exc
    if binding.recipe_revision and binding.recipe_revision != fingerprint:
        raise ProtocolError("recipe_revision_mismatch", "configured recipe revision does not match the resolved recipe")
    return recipe, fingerprint


def _ensure_ready(request: Request, binding: Binding, recipe, fingerprint: str, sctx) -> dict[str, Any]:
    # Anchor the deadline before the launch, not after it.  ``api.run`` is
    # synchronous through model download and image distribution, so a deadline
    # started afterwards bounds only the tail of the operation and the caller's
    # own timeout is the one that actually fires — mid-launch, which is the
    # worst moment to be killed.  The deadline is still only *checked* at phase
    # boundaries: nothing here can interrupt a launch in flight.
    deadline = time.monotonic() + request.timeout_seconds

    existing = _adoptable(_discover(sctx, fingerprint=fingerprint))
    if existing:
        return {"state": "ready", "endpoint": _project(existing), "adopted": True}

    candidates: tuple[str | None, ...] = tuple(binding.cluster_candidates) or (None,)
    run_result = None
    last_capacity_error: Exception | None = None
    for candidate in candidates:
        try:
            run_result = api.run(
                api.RunOptions(
                    recipe=recipe,
                    cluster=candidate,
                    overrides=dict(binding.overrides),
                    follow=False,
                    detached=True,
                    trust=False,
                    owner=GATEWAY_OWNER,
                ),
                sctx=sctx,
            )
            break
        except api.InsufficientCapacity as exc:
            last_capacity_error = exc
            continue
        except api.TrustRejected as exc:
            raise ProtocolError("recipe_trust_rejected", "configured recipe requires explicit trust") from exc
        except api.HostsUnreachable as exc:
            raise ProtocolError("cluster_unreachable", "configured cluster is unreachable", retryable=True) from exc
        except api.SparkrunError as exc:
            raise ProtocolError("activation_failed", "Sparkrun could not activate the configured recipe", retryable=True) from exc
    if run_result is None:
        raise ProtocolError(
            "insufficient_capacity",
            "no configured cluster has capacity for the recipe",
            retryable=True,
        ) from last_capacity_error
    if run_result.rc != 0:
        raise ProtocolError("activation_failed", "Sparkrun reported a failed launch", retryable=True)

    # Poll on what the launch actually recorded rather than on what we derived
    # from the binding.  They agree today, and a mismatch would be a Sparkrun
    # bug — but the failure mode of matching on a stale fingerprint is a
    # readiness timeout followed by the caller launching the workload *again*,
    # so prefer the authoritative value.
    launched_fingerprint = run_result.recipe_fingerprint or fingerprint
    if launched_fingerprint != fingerprint:
        logger.warning("launched job recorded a different recipe fingerprint than the binding resolved")
    cluster_id = run_result.cluster_id

    if not request.wait:
        return {"state": "activating", "endpoint": None, "adopted": False}

    interval = READINESS_POLL_MIN_SECONDS
    while True:
        endpoints = _discover(sctx, fingerprint=launched_fingerprint, cluster_id=cluster_id)
        if endpoints:
            return {"state": "ready", "endpoint": _project(endpoints[0]), "adopted": False}
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise ProtocolError("readiness_timeout", "activated endpoint did not become ready in time", retryable=True)
        time.sleep(min(interval, remaining))
        interval = min(interval * READINESS_POLL_BACKOFF, READINESS_POLL_MAX_SECONDS)


def _project(endpoint: dict[str, Any] | None) -> dict[str, Any] | None:
    """Strip internal-only keys from an endpoint on its way into a response."""
    if endpoint is None:
        return None
    return {key: value for key, value in endpoint.items() if not key.startswith("_")}


def _adoptable(endpoints: list[dict[str, Any]]) -> dict[str, Any] | None:
    """Pick the endpoint to adopt, preferring one this bridge launched."""
    if not endpoints:
        return None
    for endpoint in endpoints:
        if endpoint.get(_OWNED_KEY):
            return endpoint
    return endpoints[0]


def _stop(fingerprint: str, cluster_id: str, sctx) -> dict[str, Any]:
    # A fingerprint says "same serve configuration", not "mine".  Two operators
    # — or an operator and this bridge — running the same recipe produce the
    # same digest, so stopping on the digest alone would let the gateway tear
    # down a workload a human launched and is using.  Teardown therefore
    # requires the ownership tag this bridge writes at launch.
    matching_ids, owned_ids = _matching_job_ids(sctx, fingerprint)
    if cluster_id:
        if cluster_id not in matching_ids:
            raise ProtocolError("job_not_found", "cluster does not belong to the configured recipe revision")
        if cluster_id not in owned_ids:
            raise ProtocolError("job_not_owned", "cluster was not launched by the gateway bridge")
        owned_ids = [cluster_id]
    stopped: list[str] = []
    for candidate in owned_ids:
        try:
            result = api.stop(cluster_id=candidate, sctx=sctx)
        except api.JobNotFound:
            continue
        except api.HostsUnreachable as exc:
            raise ProtocolError("cluster_unreachable", "configured cluster is unreachable", retryable=True) from exc
        except api.SparkrunError as exc:
            raise ProtocolError("stop_failed", "Sparkrun could not stop the configured workload", retryable=True) from exc
        if not result.success:
            raise ProtocolError("stop_failed", "Sparkrun could not confirm workload teardown", retryable=True)
        stopped.append(candidate)
    return {"state": "offline", "cluster_ids": stopped}


def _matching_job_ids(sctx, fingerprint: str) -> tuple[list[str], list[str]]:
    """Return ``(matching_ids, owned_ids)`` for *fingerprint*, in one pass."""
    matching: list[str] = []
    owned: list[str] = []
    for job in api.list_jobs(sctx=sctx):
        metadata = job.metadata or {}
        if str(metadata.get("recipe_fingerprint") or "") != fingerprint:
            continue
        matching.append(job.cluster_id)
        if str(metadata.get("owner") or "") == GATEWAY_OWNER:
            owned.append(job.cluster_id)
    return matching, owned


def _recipe_of(job) -> Any:
    """Rehydrate the launched recipe from job metadata, or ``None``.

    Job metadata embeds the full serialized recipe state, so this is the model
    card as it was *at launch* — which is the one the running workload is
    actually serving.  Re-resolving the recipe by name would read whatever the
    file says now, and would also reintroduce registry I/O onto the inference
    path.  A job written before recipe-state persistence simply has none.
    """
    if job is None:
        return None
    state = (getattr(job, "metadata", None) or {}).get("recipe_state")
    if not state:
        return None
    try:
        from sparkrun.core.recipe import Recipe

        return Recipe._deserialize(state)
    except Exception:  # noqa: BLE001 - advisory metadata only
        logger.debug("Could not rehydrate the recipe for endpoint metadata", exc_info=True)
        return None


def _discover(sctx, *, fingerprint: str = "", cluster_id: str = "") -> list[dict[str, Any]]:
    jobs = {job.cluster_id: job for job in api.list_jobs(sctx=sctx)}
    allowed_ids: set[str] | None = None
    if fingerprint:
        allowed_ids = {job_id for job_id, job in jobs.items() if str((job.metadata or {}).get("recipe_fingerprint") or "") == fingerprint}
    if cluster_id:
        allowed_ids = {cluster_id} if allowed_ids is None else allowed_ids & {cluster_id}

    try:
        # Push the filter into discovery rather than applying it to the
        # results: each candidate costs an HTTP probe, and the readiness poll
        # calls this repeatedly while watching a single known job.
        discovered = discover_endpoints(check_health=True, cluster_ids=allowed_ids, sctx=sctx)
    except Exception as exc:
        # Do not render raw exceptions into the *response*: discovery may use
        # credentials retained in owner-only job metadata, so the structured
        # reply carries only a stable error class.  stderr is local-only and is
        # the operator's sole diagnostic, so keep the traceback there.
        logger.error("gateway bridge discovery failed", exc_info=True)
        raise ProtocolError("discovery_failed", "Sparkrun endpoint discovery failed", retryable=True) from exc

    result: list[dict[str, Any]] = []
    for endpoint in discovered:
        if allowed_ids is not None and endpoint.cluster_id not in allowed_ids:
            continue
        job = jobs.get(endpoint.cluster_id)
        metadata = job.metadata if job is not None else {}
        # The gateway treats this as an authorization input, so use only the
        # model IDs returned by the live /v1/models probe. Job metadata is not
        # authoritative enough to assert what a reused host:port serves.
        served_models = sorted({str(model) for model in endpoint.actual_models if model})
        if not served_models:
            continue
        # Advisory public model card.  Omitted entirely when nothing is known,
        # because the Go struct marks it `omitempty` and an empty object would
        # claim "reported, all unknown" rather than "not reported".
        model_metadata = build_model_metadata(_recipe_of(job), served_models)
        result.append(
            {
                "state": "ready",
                "cluster_id": endpoint.cluster_id,
                "job_id": endpoint.cluster_id,
                "host": endpoint.host,
                "port": endpoint.port,
                "protocol": "openai",
                "served_models": served_models,
                "recipe": endpoint.recipe_name,
                "recipe_revision": str((metadata or {}).get("recipe_fingerprint") or ""),
                "runtime": endpoint.runtime,
                # Internal only — stripped by _project() before this reaches a
                # response.  The gateway decodes results with
                # DisallowUnknownFields, so every projected key is part of the
                # wire contract and an extra one is a hard decode failure.
                _OWNED_KEY: str((metadata or {}).get("owner") or "") == GATEWAY_OWNER,
            }
        )
        if model_metadata:
            result[-1]["model_metadata"] = model_metadata
        if len(result) >= MAX_DISCOVERED_ENDPOINTS:
            break
    result.sort(key=lambda value: (value["cluster_id"], value["host"], value["port"]))
    return result
