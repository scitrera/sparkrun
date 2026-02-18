---
name: registry
description: Manage recipe registries and create inference recipes
---

<Purpose>
Provides complete reference for managing sparkrun recipe registries, browsing and searching recipes, validating recipes, and understanding the recipe file format for NVIDIA DGX Spark inference workloads.
</Purpose>

<Use_When>
- User wants to add, remove, or update recipe registries
- User wants to browse or search for available recipes
- User wants to create or edit a recipe YAML file
- User wants to validate a recipe or check VRAM requirements
- User asks about recipe format or fields
</Use_When>

<Do_Not_Use_When>
- User wants to run, stop, or monitor workloads -- use the run skill instead
- User wants to install sparkrun or set up clusters -- use the setup skill instead
</Do_Not_Use_When>

<Steps>

## Registry Commands

```bash
# List configured registries
sparkrun recipe registries

# Add a custom registry (git repo with recipe YAML files)
sparkrun recipe add-registry <name> --url <git_url> [--branch <branch>] [--path <subdir>]

# Remove a registry
sparkrun recipe remove-registry <name>

# Update all registries from git (fetches latest recipes)
sparkrun recipe update
```

## Browsing Recipes

```bash
# List all recipes across all registries
sparkrun list
sparkrun list <query>

# Search by name, model, runtime, or description
sparkrun search <query>

# Show full recipe details + VRAM estimation
sparkrun show <recipe> [--tp N]
```

## Validating Recipes

```bash
# Check a recipe for issues
sparkrun recipe validate <recipe>

# Estimate VRAM usage with overrides
sparkrun recipe vram <recipe> [--tp N] [--max-model-len 32768] [--gpu-mem 0.9]
```

## Recipe File Format

Recipes are YAML files defining an inference workload:

```yaml
model: org/model-name                  # HuggingFace model ID (required)
runtime: vllm | sglang | llama-cpp     # Inference runtime (required)
container: registry/image:tag          # Docker image (required)
min_nodes: 1                           # Minimum hosts needed
max_nodes: 4                           # Maximum hosts (optional)
model_revision: abc123                 # Pin to specific HF revision (optional)

metadata:
  description: Human-readable description
  maintainer: name <email>
  model_params: 7B
  model_dtype: fp16

defaults:
  port: 8000
  host: 0.0.0.0
  tensor_parallel: 2
  gpu_memory_utilization: 0.9
  max_model_len: 32768
  served_model_name: my-model
  tokenizer_path: org/base-model       # Required for GGUF models on SGLang

# Optional: explicit command template (overrides auto-generation)
command: |
  python3 -m vllm.entrypoints.openai.api_server \
      --model {model} \
      --tensor-parallel-size {tensor_parallel} \
      --port {port}

# Optional: environment variables passed to the container
env:
  NCCL_DEBUG: INFO
```

**Key fields:**
- `{placeholder}` in `command:` templates are substituted from defaults + CLI overrides
- `model_revision` pins downloads to a specific HuggingFace commit/tag
- `tokenizer_path` is required for GGUF models on SGLang (points to base non-GGUF model)
- `min_nodes` / `max_nodes` control cluster size validation
- Shell variable references like `${HF_TOKEN}` in `env:` are expanded from the control machine's environment

</Steps>

<Tool_Usage>
All sparkrun commands are executed via the Bash tool. No MCP tools are required.
</Tool_Usage>

<Important_Notes>
- Run `sparkrun recipe update` periodically to get the latest community recipes
- Use `sparkrun recipe validate` before publishing custom recipes
- Use `sparkrun recipe vram` to check if a model fits on DGX Spark before trying to run it
- When creating GGUF + SGLang recipes, always set `tokenizer_path` in defaults
- Custom command templates should include all relevant `{placeholder}` references to pick up defaults and CLI overrides
- Registries are cached at `~/.cache/sparkrun/registries/` and updated with `sparkrun recipe update`
- sparkrun ships with built-in recipes; additional registries point to any git repo containing `.yaml` recipe files
</Important_Notes>

Task: {{ARGUMENTS}}
