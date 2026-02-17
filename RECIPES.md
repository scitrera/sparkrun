# Recipe File Format

A sparkrun recipe is a YAML file that describes everything needed to launch an inference workload: the model, the
container image, the runtime engine, default parameters, and the serve command. Recipes are the central abstraction
in sparkrun — they let you capture a known-good configuration once and replay it with a single command.

```bash
sparkrun run my-recipe --solo          # use defaults
sparkrun run my-recipe -H host1,host2  # override hosts
sparkrun run my-recipe -o port=9000    # override any default
```

## Minimal Recipe

The smallest useful recipe needs only `model`, `runtime`, `container`, and `command`:

```yaml
model: Qwen/Qwen3-1.7B
runtime: vllm
container: scitrera/dgx-spark-vllm:0.16.0-t5

defaults:
  port: 8000
  host: 0.0.0.0

command: |
  vllm serve {model} --host {host} --port {port}
```

## Full Recipe Example

```yaml
model: nvidia/NVIDIA-Nemotron-3-Nano-30B-A3B-NVFP4
runtime: vllm
min_nodes: 1
container: scitrera/dgx-spark-vllm:0.16.0-t5

metadata:
  description: NVIDIA Nemotron 3 Nano 30B (upstream NVFP4) -- cluster or solo
  maintainer: scitrera.ai <open-source-team@scitrera.com>
  model_params: 30B
  model_dtype: nvfp4

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 1
  gpu_memory_utilization: 0.8
  max_model_len: 200000
  served_model_name: nemotron3-30b-a3b
  tool_call_parser: qwen3_coder

env:
  VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"

command: |
  vllm serve \
      {model} \
      --served-model-name {served_model_name} \
      --max-model-len {max_model_len} \
      --gpu-memory-utilization {gpu_memory_utilization} \
      -tp {tensor_parallel} \
      --host {host} \
      --port {port} \
      --enable-auto-tool-choice \
      --tool-call-parser {tool_call_parser} \
      --trust-remote-code
```

## Field Reference

### Core Fields

#### `model` (required)

The HuggingFace model identifier. This is the model that will be served.

```yaml
# Standard HuggingFace model
model: Qwen/Qwen3-1.7B

# GGUF model with quantization variant (llama-cpp runtime)
model: Qwen/Qwen3-1.7B-GGUF:Q8_0
```

For GGUF models, the colon syntax (`repo:quant`) selects a specific quantization variant. sparkrun uses this to
download only the matching quant files rather than the entire repository.

The model value is injected into the command template as `{model}` and is also used for model pre-sync (downloading
and distributing weights to target hosts before launch). (Models are downloaded to the local machine and then
transferred to the cluster via `rsync` so you don't need to repeatedly download them from the Internet on a cluster.)

#### `runtime` (required)

Which inference engine to use. Determines how sparkrun launches and manages the workload.

| Value | Engine | Clustering | Notes |
|-------|--------|------------|-------|
| `vllm` | vLLM | Ray | First-class. Solo and multi-node via Ray. |
| `sglang` | SGLang | Native | First-class. Solo and multi-node via SGLang's built-in distribution. |
| `llama-cpp` | llama.cpp | N/A | Solo mode. GGUF quantized models via `llama-server`. |
| `eugr-vllm` | vLLM (eugr) | Delegated | Compatibility runtime. Delegates to eugr/spark-vllm-docker scripts. |

If omitted, defaults to `vllm`. Recipes with `recipe_version: "1"` or eugr-specific fields (`build_args`, `mods`)
are automatically detected as `eugr-vllm`.

#### `container` (required)

The Docker/OCI container image to run. This should be a fully qualified image reference. Since we're using docker, 
docker.io is assumed by default (that's part of docker). You can also use a local image or private registry, but you're
responsible for making sure it's configured on your local system. The image will be pushed out to the cluster from the
local machine to avoid repeated downloads from the Internet.

```yaml
container: scitrera/dgx-spark-vllm:0.16.0-t5
container: scitrera/dgx-spark-sglang:0.5.8-t5
container: scitrera/dgx-spark-llama-cpp:b8076-cu131
```

If not specified, the runtime will attempt to construct a default from its `default_image_prefix`, but it is
strongly recommended to pin a specific image and tag.

#### `command` (recommended)

The shell command to execute inside the container. Uses `{placeholder}` syntax for template substitution — any key
from `defaults` (or CLI overrides) can be referenced.

```yaml
command: |
  vllm serve \
      {model} \
      --served-model-name {served_model_name} \
      --gpu-memory-utilization {gpu_memory_utilization} \
      -tp {tensor_parallel} \
      --host {host} \
      --port {port}
```

Substitution is iterative: if a placeholder resolves to a string containing another `{placeholder}`, it will be
expanded in a second pass. This continues until no more substitutions are possible.

If `command` is omitted, the runtime's `generate_command()` method builds a command from structured defaults.
Providing an explicit command template is preferred because it gives recipe authors full control over flag ordering,
runtime-specific flags, and optional features.

### Topology Fields

These fields control how many nodes the recipe can run on and how sparkrun decides between solo and cluster mode.

#### `mode`

Explicit topology mode. Usually not needed — sparkrun infers it from `min_nodes` and `max_nodes`.

| Value | Meaning |
|-------|---------|
| `auto` | (default) sparkrun decides based on node counts and CLI flags |
| `solo` | Forces single-node. Sets `min_nodes = max_nodes = 1`. |
| `cluster` | Forces multi-node. Requires 2+ hosts. |

Automatic inference rules:
- If `min_nodes > 1`, mode becomes `cluster` regardless of the `mode` field.
- If `max_nodes == 1`, mode becomes `solo`.
- Otherwise, `auto` means the recipe can run either way depending on how many hosts the user provides.

#### `min_nodes`

Minimum number of nodes required. Defaults to `1`.

```yaml
min_nodes: 1   # can run solo or cluster
min_nodes: 2   # requires at least 2 nodes (implies cluster mode)
min_nodes: 4   # requires at least 4 nodes
```

On DGX Spark, each node has one GPU, so `min_nodes` is effectively the minimum GPU count. A recipe with
`min_nodes: 2` and `tensor_parallel: 2` means the model needs 2 GPUs to fit.

#### `max_nodes`

Maximum number of nodes supported. Defaults to `None` (unlimited).

```yaml
max_nodes: 1   # solo only (e.g. GGUF models that don't support distribution)
max_nodes: 4   # up to 4-node tensor parallel
```

Set `max_nodes: 1` for models or runtimes that do not support multi-node inference (e.g. llama-cpp in solo mode).

#### `solo_only` / `cluster_only`

Boolean shorthand flags. These are an alternative to setting `mode` directly.

```yaml
solo_only: true     # equivalent to max_nodes: 1, mode: solo
cluster_only: true  # equivalent to min_nodes: 2, mode: cluster
```

### Configuration Fields

#### `defaults`

A flat dictionary of default parameter values. These are the knobs that control the inference server and can be
overridden at launch time via CLI flags or `-o key=value`.

```yaml
defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 1
  gpu_memory_utilization: 0.8
  max_model_len: 200000
  served_model_name: nemotron3-30b-a3b
```

Every key in `defaults` is available as a `{placeholder}` in the command template. sparkrun also recognizes certain
well-known keys and maps them to dedicated CLI flags:

| Default key | CLI flag | Description |
|-------------|----------|-------------|
| `port` | `--port` | Serve port |
| `tensor_parallel` | `--tp` | Tensor parallelism degree |
| `gpu_memory_utilization` | `--gpu-mem` | GPU memory fraction (0.0–1.0) |
| `max_model_len` | `-o max_model_len=N` | Maximum sequence length |
| `served_model_name` | `-o served_model_name=X` | Model name exposed by the API |
| `host` | `-o host=X` | Bind address |

Any key can appear in defaults — there is no fixed schema. Runtime-specific parameters (like `attention_backend`,
`tool_call_parser`, `ctx_size`, `n_gpu_layers`) are passed through as-is and substituted into the command template.

**Config chain precedence** (highest wins):
1. CLI overrides (`--port 9000`, `-o key=value`)
2. User config (future: per-user config file)
3. Recipe defaults

#### `env`

Environment variables injected into the container at launch time.

```yaml
env:
  VLLM_ALLOW_LONG_MAX_MODEL_LEN: "1"
  HF_TOKEN: "${HF_TOKEN}"
```

Values must be strings. These are passed to `docker run` via `-e` flags and are also set inside `docker exec` when
the serve command runs.

Shell-style variable references (`${VAR}` or `$VAR`) are expanded from the **control machine's** environment
when the recipe is loaded. If a variable is not set, the reference is left as-is. This lets you forward
secrets like `HF_TOKEN` without hardcoding them in the recipe file.

Use `env` for runtime knobs that are controlled via environment variables rather than CLI flags (e.g.
`VLLM_ALLOW_LONG_MAX_MODEL_LEN`, NCCL tuning variables).

### Metadata Fields

#### `metadata`

A dictionary of descriptive and VRAM-estimation fields. Metadata does not affect how the workload runs — it provides
information for `sparkrun show`, `sparkrun recipe vram`, and recipe search/listing.

```yaml
metadata:
  description: NVIDIA Nemotron 3 Nano 30B (upstream NVFP4) -- cluster or solo
  maintainer: scitrera.ai <open-source-team@scitrera.com>
  model_params: 30B
  model_dtype: nvfp4
```

##### Descriptive fields

| Field | Purpose |
|-------|---------|
| `description` | Human-readable summary shown in `sparkrun list` and `sparkrun show`. |
| `maintainer` | Contact info for the recipe author. |
| `name` | Override the recipe display name (defaults to filename stem). |

##### VRAM estimation fields

These fields feed sparkrun's VRAM estimator, which tells you whether a model fits on DGX Spark before you launch.
sparkrun auto-detects most of these from HuggingFace model configs, but metadata values take precedence when
provided — useful for quantized models or architectures that HF auto-detection doesn't cover well.

| Field | Type | Description |
|-------|------|-------------|
| `model_params` | string | Parameter count. Accepts shorthand: `"1.7B"`, `"30B"`, `"397B"`. |
| `model_dtype` | string | Weight dtype: `bf16`, `fp16`, `fp8`, `nvfp4`, `int8`, `int4`, `q4_k_m`, `q8_0`, etc. |
| `kv_dtype` | string | KV cache dtype (defaults to `bfloat16` if not specified). |
| `num_layers` | int | Number of transformer layers. |
| `num_kv_heads` | int | Number of key-value attention heads. |
| `head_dim` | int | Dimension per attention head. |
| `model_vram` | float | Override: total model weight VRAM in GB (skips param-based calculation). |
| `kv_vram_per_token` | float | Override: KV cache bytes per token (skips layer-based calculation). |

The estimator combines these with `tensor_parallel`, `max_model_len`, and `gpu_memory_utilization` from defaults
to produce a per-GPU VRAM breakdown:

```
VRAM Estimation:
  Model weights:    19.56 GB
  KV cache:         9.92 GB (max_model_len=200,000)
  Tensor parallel:  1
  Per-GPU total:    29.48 GB
  DGX Spark fit:    YES
```

### Runtime-Specific Fields

#### `runtime_config`

A dictionary for runtime-specific configuration that doesn't belong in `defaults`. Used primarily by the `eugr-vllm`
runtime.

```yaml
runtime_config:
  mods: [my-custom-mod]
  build_args: [--some-flag]
```

Any top-level key in the YAML that isn't part of the known recipe schema is automatically swept into
`runtime_config`. This means you can write:

```yaml
mods: [my-custom-mod]
build_args: [--some-flag]
```

...instead of nesting them under `runtime_config`, and sparkrun will handle it the same way. The explicit
`runtime_config` key is preferred for clarity.

### Version Fields [NOT FINALIZED]

#### `sparkrun_version` / `recipe_version`

Declares the recipe format version. The current format is version `2` (the default).

```yaml
sparkrun_version: "2"   # current format
recipe_version: "1"     # eugr v1 format — triggers eugr-vllm runtime
```

Version `1` recipes are automatically migrated: the runtime is set to `eugr-vllm` and eugr-specific fields are
preserved in `runtime_config`.

## GGUF Recipes (llama.cpp)

GGUF recipes target the `llama-cpp` runtime and use colon syntax in the model field to select a quantization
variant:

```yaml
model: Qwen/Qwen3-1.7B-GGUF:Q8_0
runtime: llama-cpp
max_nodes: 1
container: scitrera/dgx-spark-llama-cpp:b8076-cu131

metadata:
  description: Qwen3 1.7B (Q8_0 GGUF) -- small test model via llama.cpp
  maintainer: scitrera.ai <open-source-team@scitrera.com>
  model_params: 1.7B
  model_dtype: q8_0

defaults:
  port: 8000
  host: 0.0.0.0
  n_gpu_layers: 99
  ctx_size: 8192

command: |
  llama-server \
      -hf {model} \
      --host {host} \
      --port {port} \
      --n-gpu-layers {n_gpu_layers} \
      --ctx-size {ctx_size} \
      --flash-attn on \
      --jinja \
      --no-webui
```

Key differences from vLLM/SGLang recipes:

- **Model field** uses `repo:quant` syntax (e.g. `Qwen/Qwen3-1.7B-GGUF:Q8_0`). sparkrun parses this to download
  only the matching quantization files.
- **`max_nodes: 1`** — llama-cpp solo mode does not support multi-node distribution.
- **`ctx_size`** replaces `max_model_len` — the context window size in tokens.
- **Model pre-sync** downloads only the matching GGUF quant files locally, distributes them to target hosts via
  rsync, and rewrites `-hf` to `-m` with the resolved container cache path so the container serves from the local
  copy.

## Command Template Substitution

The `command` field supports `{placeholder}` substitution. Any key available in the config chain (defaults + CLI
overrides) can be referenced:

```yaml
defaults:
  port: 8000
  served_model_name: my-model

command: |
  vllm serve {model} --port {port} --served-model-name {served_model_name}
```

With `sparkrun run recipe -o port=9000`, this renders as:

```
vllm serve Qwen/Qwen3-1.7B --port 9000 --served-model-name my-model
```

The special placeholder `{model}` is always available — it is injected from the top-level `model` field into the
config chain automatically.

Substitution is iterative, so nested references work:

```yaml
defaults:
  base_url: "http://localhost:{port}"
  port: 8000
```

## Recipe Discovery

sparkrun searches for recipes in this order:

1. **Exact/relative file path** — if the argument is a path to an existing file.
2. **Bundled recipes** — shipped with sparkrun in the `recipes/` directory.
3. **Registry paths** — from configured custom registries.
4. **Registry stem matching** — searches registry files by filename stem.

Filenames are matched with or without `.yaml`/`.yml` extensions, so `sparkrun run my-recipe` finds
`my-recipe.yaml`.

## Validation

Run `sparkrun recipe validate <recipe>` to check a recipe for issues:

- Missing required fields (`model`, `runtime`)
- Invalid `mode` values
- `min_nodes` / `max_nodes` consistency
- `metadata.model_params` is a valid parameter count
- `metadata.model_dtype` and `metadata.kv_dtype` are recognized dtypes
- Runtime-specific validation (provided by each runtime plugin)

## Writing Your Own Recipes

1. **Start from an existing recipe** that uses the same runtime. Copy it and modify.
2. **Set `model`** to the HuggingFace model ID you want to serve.
3. **Set `container`** to a known-good image for your runtime.
4. **Tune `defaults`** — start conservative with `gpu_memory_utilization: 0.5` and `max_model_len` low, then
   increase after confirming the model loads.
5. **Add `metadata`** with `model_params` and `model_dtype` so VRAM estimation works.
6. **Validate**: `sparkrun recipe validate your-recipe.yaml`
7. **Test**: `sparkrun run your-recipe.yaml --solo --dry-run` to see what would be executed.

### Tips

- Use `--dry-run` liberally. It shows the exact Docker commands without executing anything.
- The `sparkrun show <recipe>` command displays rendered defaults plus VRAM estimation — use it to sanity-check
  before launching.
- For models that barely fit in VRAM, lower `max_model_len` first. KV cache scales with sequence length.
- `tensor_parallel` on DGX Spark maps 1:1 to node count. `--tp 2` means 2 hosts, each contributing 1 GPU.
- GGUF recipes should set `max_nodes: 1` unless using the experimental RPC multi-node backend.
