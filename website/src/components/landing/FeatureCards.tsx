const features = [
  {
    icon: '⚡',
    title: 'One Command Launch',
    description: 'Pick a recipe, run it. sparkrun handles container orchestration, model distribution, and networking automatically.',
  },
  {
    icon: '🔗',
    title: 'Multi-Node Tensor Parallel',
    description: 'Scale across DGX Sparks with --tp. Each Spark contributes one GPU — sparkrun handles InfiniBand/RDMA and NCCL configuration.',
  },
  {
    icon: '📋',
    title: 'Recipe System',
    description: 'YAML configs capture model, container, runtime, and defaults. Override anything at launch time — no config files to hunt down.',
  },
  {
    icon: '📦',
    title: 'Recipe Registries',
    description: 'Share and collaborate on recipes via git registries. Add community or private registries and search across all of them.',
  },
  {
    icon: '📊',
    title: 'VRAM Estimation',
    description: 'Auto-detects model architecture from HuggingFace. Know whether your config fits on a single DGX Spark or how many you need before launching.',
  },
  {
    icon: '🔄',
    title: 'Multiple Runtimes',
    description: 'First-class support for vLLM, SGLang, and llama.cpp. Same CLI, same recipe format, different engines under the hood.',
  },
  {
    icon: '🤖',
    title: 'Claude Code Plugin',
    description: 'AI-assisted inference management. Claude learns your cluster and helps run, monitor, and stop workloads conversationally.',
  },
  {
    icon: '⌨️',
    title: 'CLI Tab Completion',
    description: 'Rich shell completions for Bash, Zsh, and Fish. Tab-complete commands, recipe names, cluster names, and options instantly.',
  },
];

export default function FeatureCards() {
  return (
    <div className="not-content feature-grid">
      {features.map((f) => (
        <div className="feature-card" key={f.title}>
          <span className="feature-icon">{f.icon}</span>
          <h3>{f.title}</h3>
          <p>{f.description}</p>
        </div>
      ))}
    </div>
  );
}
