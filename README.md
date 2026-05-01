# NeuralDiff

> Visual diff between AI model checkpoints. See what changed between your fine-tune runs.

## Demo

```bash
# With TUI scanner (interactive)
$ neuraldiff diff

# Direct comparison
$ neuraldiff diff base.safetensors finetuned.safetensors

# JSON export
$ neuraldiff diff base.safetensors finetuned.safetensors --json > diff.json
```

```
NEURALDIFF v0.1.0    model_a.safetensors → model_b.safetensors    111K params

┌─────────────────────────────────────────────────────────────────┐
│ Layer: layers.3.mlp  |  Type: MLP                               │
│ L2: 0.365213  |  Params: 8.35K  |  Tensors: 4  |  Changed: 4/4  │
├─────────────────────────────────────────────────────────────────┤
│   Tensor Name             Shape       L2     Cosine   Max Delta │
│ ─────────────────────────────────────────────────────────────── │
│ ▶ transformer.h.3.mlp...  [32, 128]  0.6412   0.8924   0.033812 │
│   transformer.h.3.mlp...  [128]      0.1215   0.0000   0.023842 │
│   transformer.h.3.mlp...  [128, 32]  0.6377   0.8908   0.043178 │
│   transformer.h.3.mlp...  [32]       0.0605   0.0000   0.026041 │
├─────────────────────────────────────────────────────────────────┤
│ Distribution: Low(0)  Med(0)  High(4)                           │
└─────────────────────────────────────────────────────────────────┘

[↑↓/jk] Navigate  [←→/hl] Tensor  [Enter] Heatmap  [b] Back
[s] Sort  [f] Filter  [J] JSON  [?] Help  [q] Quit
```

## Features

- **Auto Scanner** - Automatically finds `.safetensors` models on your system
- **Model Selection UI** - Interactive TUI to pick Model A and Model B with `[A]`/`[B]` indicators
- **Layer Comparison** - Side-by-side comparison with aligned columns (Name, Shape, L2, Cosine, Max Delta, Status)
- **Color Coding** - Per-column colors: L2 by magnitude, Cosine by similarity, Status by change
- **Anomaly Detection** - Highlights layers with unusually high delta using z-score analysis
- **Distribution Bars** - Visual bars showing low/med/high change distribution per layer
- **JSON Export** - Export full diff to JSON with `--json` flag or `[J]` key
- **Parallel Processing** - Uses rayon for fast tensor-level diff computation
- **Memory Efficient** - Memory-mapped file I/O for handling multi-GB models
- **Offline First** - No network requests, everything stays local
- **Weighted Metrics** - Layer L2 distance weighted by parameter count (not simple average)

## Install

```bash
cargo install neuraldiff

# Or from source
git clone https://github.com/Zertannax/neuraldiff
cd neuraldiff
cargo build --release
```

## Usage

```bash
# Interactive mode (scans for models)
neuraldiff diff

# Direct comparison
neuraldiff diff base.safetensors finetuned.safetensors

# JSON export
neuraldiff diff base.safetensors finetuned.safetensors --json

# Inspect a single model
neuraldiff inspect model.safetensors

# Scan and list all discovered models
neuraldiff scan
```

### Keyboard Shortcuts

| Key     | Action                              |
|---------|-------------------------------------|
| `↑/k`   | Navigate up in layer list           |
| `↓/j`   | Navigate down in layer list         |
| `←/h`   | Previous tensor                     |
| `→/l`   | Next tensor                         |
| `Enter` | Toggle heatmap / Enter detail view  |
| `b`     | Back to summary view                |
| `s`     | Cycle sort mode (L2 → Index → Anom) |
| `f`     | Toggle filter (All → Changed only)  |
| `J`     | Export JSON to diff.json            |
| `?`     | Show help                           |
| `q`     | Quit                                |

## Metrics Explained

- **L2 Distance** - Magnitude of changes (0 = identical, >1 = drastic)
- **Cosine Similarity** - Direction similarity (-1 = opposite, 1 = identical)
- **Max Delta** - Largest absolute change in any single parameter
- **Z-Score** - How unusual the change is compared to other layers
- **Anomaly** - Flagged when z-score > 2.0

## Architecture

```
CLI args → SafetensorsLoader → DiffEngine → LayerMapper → TuiRenderer
```

- `loader.rs` - Parses .safetensors, extracts metadata + mmap
- `diff.rs` - Computes L2, cosine, max delta per tensor (parallel)
- `mapper.rs` - Groups tensors into logical layers (LLaMA/Qwen/GPT-2/Falcon)
- `tui.rs` - Interactive terminal UI with legends and comparison view
- `scanner.rs` - Auto-discovers models in home/downloads/.cache

## Supported Models

Naming conventions for automatic layer grouping:

- **LLaMA/Qwen** - `model.layers.N.{self_attn,mlp,input_layernorm}`
- **GPT-2** - `transformer.h.N.{attn,mlp,ln_1,ln_2}`
- **Falcon** - `transformer.h.N.{self_attention,mlp,ln_attn,ln_mlp}`

## Roadmap

See [TODO.md](TODO.md) for the complete roadmap.

### v0.1.1 (Current)
- Heatmap visualization
- Inspect command
- Filter by layer type
- Progress bar for large models

### v0.2.0
- Web 3D visualization (Three.js, no CDN)
- Directory comparison
- Configurable thresholds

### v0.3.0
- Streaming for models > 10GB
- Cache system
- Theme support

## Development

```bash
# Build (TUI only)
cargo build

# Build with web feature
cargo build --features web

# Tests
cargo test

# Clippy
cargo clippy --all-targets --all-features

# Format check
cargo fmt --check
```

### Windows Build Note

On Windows, debug builds may fail with PDB/linker errors due to disk space. Use sequential compilation:
```bash
cargo test --jobs 1
cargo build --release --jobs 1
```

## License

MIT

## Contributors

- NeuralDiff Contributors
