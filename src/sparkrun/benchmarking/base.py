# TODO: define models, base classes, etc.

# TODO: base class for benchmarking framework should also extend from scitrera app framework Plugin
#         and use EXT_BENCHMARKING_FRAMEWORKS so that it can be auto-registered

# TODO: base class for a benchmarking framework would need to be able to handle:
#       - check and install pre-reqs
#       - resolve details from recipe + inputs
#       - create command to run benchmark (potentially on control node, potentially with mgmt interface)
#           (base implementation could assume args become e.g.: --arg-name from arg_name) OR have util fn to implement that approach
#       - execute benchmark command on this node (using 127.0.0.1 OR mgmt interface for head node as target)
#       - parse output and return results (potentially saving metadata for recipe, --tp, n_hosts, benchmarking_config, benchmarking_results)
#       - optionally support custom commands like runtimes with arg substitution -- should be able to generate commands by default
