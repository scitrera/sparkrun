# Startup readiness and timing

On `develop-next`, watched normal Docker launches for vLLM and SGLang use one
streaming chat request as the final readiness signal. A listening port or a
successful health response alone does not establish that a model can generate.
Other runtime families and executors retain their existing endpoint checks.

The readiness observer runs on the head host (rank 0), using host Python 3,
Docker inspection, and the local inference endpoint. It records:

| Metric | Boundary measured from the serving container's Docker `State.StartedAt` |
| --- | --- |
| TTR port-open | First observed TCP LISTEN state on the serving port |
| TTR HTTP-ready | First observed HTTP 200 from `/health` |
| Startup TTFT | First non-empty `content`, `reasoning`, or `reasoning_content` streaming delta |

The first two are distinct from inference readiness. Empty role, usage, and
keepalive events do not count as generated text. The observer selects the served
model ID from `/v1/models`, sends a single temperature-zero, max-64-token chat
request, and closes it after receiving first text. Ordinary readiness does not
require an exact final reply. A timed-out, empty, or failed stream does not
signal ready. Recipe/override API keys use the runtime's existing key resolver;
keys travel over stdin, not command-line arguments, and are excluded from
the observation and logs.

All timestamps used in the startup duration come from the head host, avoiding
control-node clock skew. The probe verifies container identity and start time
again before accepting its observation. Port detection is passive via
`/proc/net/tcp{,6}`, not a TCP connection. The probe disables proxy environment
handling for its local HTTP calls. Detectable clock steps during inference
invalidate the observation.

The normal launch log prints the three Docker-start metrics when inference is
ready. They also appear as overlapping `serve.startup_port_open`,
`serve.startup_http_ready`, and `serve.startup_ttft` spans in the launch timeline.
Do not sum these spans or confuse them with total CLI wall time. Images/model
distribution and any preparation before container start remain separate phases.
`ServeReadiness.port_wait_s` and `health_wait_s` retain their wait-duration
meaning; Docker-start timestamps are in `startup_observation`.

## Configuration

In `~/.config/sparkrun/config.yaml`:

```yaml
readiness:
  inference: true
  inference_timeout_s: 120
  inference_prompt: "Reply with exactly: sparkrun-ready"
```

The inference timeout is a separate bounded budget after port and HTTP health
readiness. Existing `port_timeout_s` and `health_timeout_s` still apply. Set
`inference: false` to retain endpoint-only readiness, for example with an
embedding-only model or a custom server without chat support. The probe needs
Python 3 and Docker access on the head host; it does not install software there.

Default log-following launches and post-launch hooks use this readiness path.
`--no-follow` retains its fast return after the existing boot-liveness check;
it does not block for inference or promise a TTFT. Ctrl+C detaches from the
workload and cancels the local probe/SSH process. Probe operations also have
bounded timeouts; cancelling observation does not stop the serving container.

## Execution strategies

An execution strategy can return `ActivationResult.startup_observation` with
its successful inference observation. `LaunchResult` passes it to readiness,
which reuses it without sending a second inference. The optional mapping is
empty for existing strategies. Its format-1 contract includes measurement
profile, rank-0 observer, container identity/start, first-token timestamp/field,
and `inference_ready: true`; port/HTTP timestamps and full-response validation
are additional provenance. The current supported profiles are
`sparkrun-rank0-v1` and ColdSnap's `rank0-acceptance-v1`.

ColdSnap's canonical development plugin supplies its already-validated full
acceptance response timing. Older compatible ColdSnap controllers can mark
`runtime_info.inference_readiness` as `accepted` without supplying timestamps;
this suppresses a duplicate inference but does not invent TTFT. Updating this
upstream code does not itself update the vendored ColdSnap plugin or controller.

## Comparison and qualification

These are first-observed event timestamps, not kernel event tracing. The
observer begins when the post-launch readiness wait starts, and can therefore
observe a fast/already-ready endpoint late. The observation retains
`observer_started_unix_ns` to expose that delay. Port/HTTP readiness may precede
first inference by much more than a second.

For controlled qualification, `run_probe` can take an `expected` final response.
It still issues one inference request and timestamps first text, but then
requires the complete stream, finish reason, and exact final content before
returning `response_validated: true`. ColdSnap's maintained harness uses this
mode for normal-launch controls and reads ColdSnap acceptance for restore
cases. Match prompts, token limits, topology, engine, and cache policy. Keep
profile labels; do not mix rank-local samples with historical control-node
observer measurements. This feature is measurement infrastructure, not new
GPU qualification or published performance data.
