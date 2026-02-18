# /sparkrun:list

Browse and search available inference recipes.

## Usage

```
/sparkrun:list [query]
```

## Examples

```
/sparkrun:list
/sparkrun:list qwen3
/sparkrun:list llama-cpp
```

## Behavior

When this command is invoked:

1. List or search recipes:

```bash
# List all recipes
sparkrun list

# Search by name, model, runtime, or description
sparkrun list <query>
```

2. If the user wants details on a specific recipe:

```bash
sparkrun show <recipe>
sparkrun show <recipe> --tp <N>   # include VRAM estimate for N nodes
```

3. If the user wants to validate or check VRAM for a recipe:

```bash
sparkrun recipe validate <recipe>
sparkrun recipe vram <recipe> --tp <N>
```

## Notes

- Recipes come from built-in and custom registries
- Run `sparkrun recipe update` to fetch the latest recipes from remote registries
- Use `sparkrun recipe registries` to see configured registries
