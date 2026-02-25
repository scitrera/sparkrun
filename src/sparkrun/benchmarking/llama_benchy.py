# TODO: implement llama-bency framework implementation based on base.py models/abstract classes (mimic runtime pattern)

# TODO: analogous to runtimes? benchmarking operators

# pre-req check
# llama-benchy benchmarking should:
#  ensure uv installed (likely to be installed since primary sparkrun install means is uvx)
#  use uvx llama-benchy to be able to launch latest llama-benchy
# run llama-benchy from **this node** using **mgmt interface** to reach spark cluster/node(s)
#  if same host -- ensure use of 127.0.0.1

# implicit --> that benchmarking means RUN, BENCHMARK, STOP
#       better/optional/future --> check status and benchmark on existing running matching model recipe if available??? [kinda complicated]

# ok spec thoughts for discussion...
#
# 1) we can have a default benchmarking profile in recipes -- keeping in mind what you were contributing to eugr's repo -- I think it might make sense for some people and I think it's fair to keep that coupled provided that it's entirely optional -- default recipe benchmark profiles would essentially set the default benchmarking for that recipe when no profile information provided for triggering a benchmark
#
# ```yaml
# # Optional benchmark configuration -- embedded in recipes
# benchmark:
#   framework: llama-benchy                 # default llama-benchy
#   args:
#     pp: [2048]                            # default [2048]
#     depth: [0]                            # default [0]
#     enable_prefix_caching: true           # default true
#     save_result: benchmark_{model}.yaml   # optional -- if provided as part of yaml, then it must support arg substitution
# ```
#
# similar to how sparkrun handles recipes, any "unknown fields" should be swept into args, meaning that the following is equivalent (operating for the moment on "framework" being the only known field):
#
# ```yaml
# # Optional benchmark configuration -- embedded in recipes
# benchmark:
#   framework: llama-benchy               # default llama-benchy
#   pp: [2048]                            # default [2048]
#   depth: [0]                            # default [0]
#   enable_prefix_caching: true           # default true
#   save_result: benchmark_{model}.yaml   # optional -- if provided as part of yaml, then it must support arg substitution
# ```
#
# Similar to previous discussion in eugr repo, model should not be specified as part of the profile/configuration. It should always be able to be inferred from the calling environment. It'll be important to decide which approach should be the official standard. Obviously less nesting is simpler for people to write, so is that preferred? If everything were not just in "args", I would prefer nesting, but if it's just going to be "args" one level deep and that's it -- it almost just seems redundant to make people write that every time. And since llama-benchy will be the default and doesn't need to be provided, it could make the recipe embedded format relatively clean like:
#
# ```yaml
# benchmark:
#   depth: [0,2048,4096]
#   tg: [32,128]
#   runs: 5
# ```
#
# 2) benchmarking profiles can be personal preference (and stored locally) and/or stored in registries (registry exact details to be worked out as we go); benchmarking profiles should use compatible syntax with the benchmark block above -- (and 100% share code with it for interpreting/processing benchmarking)
#
# 3) that also means that -- if we have a "spark-arena" recipe registry -- it can have its own benchmarking profiles as part of the registry -- to try to enforce consistency in benchmarking and allow versioning/updates to benchmarking profiles
#
# 4) benchmarking export should be a yaml file (or something else if it makes sense) that includes metadata for recipe, model, cluster size, tp, benchmarking configuration -- basically **everything you should need to reproduce the results** plus the markdown or json results themselves.
#
# 5) Each benchmarking framework implementation (analogous to runtimes in current design) should provide default usage args if there is no profile OR configured defaults. As well as handling functions required to properly interpret arguments that may be given as strings at CLI.
#
# 6) Usage examples from CLI:
#
# ```bash
#
# sparkrun benchmark <recipe> --tp 2  # run default benchmarking for given recipe on default cluster with --tp 2 ;; this is closest to original eugr repo PR submission in terms of UX
#
# sparkrun benchmark <recipe> --tp 2 --out xxx.yaml  # run default benchmarking for given recipe on default cluster with --tp 2 and use specified output file
#
# sparkrun benchmark <recipe> --tp 2 --profile spark-arena-v1  # run spark-arena-v1 benchmark for given recipe on default cluster with --tp 2 ; spark-arena-v1 profile should come from registry
#
# sparkrun benchmark <recipe> --tp 2 --profile ./my-profile.yaml # run benchmark using a custom profile file in that directory for the given recipe on default cluster with --tp 2
#
# sparkrun benchmark <recipe> --tp 4 -o depth=0,2048,4096  # run default benchmarking for given recipe on default cluster with --tp4 and overriding arg "depth" with 0,2048,4096 ; TODO: requires that benchmarking args interpretation assumes comma separated strings are lists -- means that framework implementation classes should have custom interpret_arg fn to ensure that this is portable across frameworks and properly handles which args may need or not need that treatment;
#
# ```
#
# 7) profiles should have full tab completion support following the same pattern as recipes.