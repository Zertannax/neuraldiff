# NeuralDiff

> Visual diff between AI model checkpoints. See what changed between your fine-tune runs.

```
$ neuraldiff diff base.safetensors finetuned.safetensors

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  NEURALDIFF  v0.1.0   base.safetensors → finetuned.safetensors
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
  Models     :  Qwen2.5-3B  (3.09B params each)
  Layers     :  36 transformer layers
  Changed    :  34 layers  (94.4%)
  Unchanged  :  2 layers   (embedding + lm_head)

  Top changed layers (by L2 distance):
  ████████████████████  layers.22.mlp          Δ = 0.847
  ███████████████░░░░░  layers.31.self_attn     Δ = 0.723
  ██████████████░░░░░░  layers.18.mlp           Δ = 0.691
  ...

  [t] TUI heatmap   [w] Web 3D view   [j] JSON export   [q] quit
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
```

## What it does

You fine-tuned a model. Something changed. But **what** changed, and **how much**?

NeuralDiff compares two `.safetensors` checkpoints and shows:
- Which layers changed (L2 distance, cosine similarity)
- How much each layer changed (heatmap by layer × component)
- Distribution of weight deltas (histogram)
- Anomaly detection: layers with unusually large shifts

Two views:
- **TUI** (default): ratatui-based, works in any terminal
- **Web 3D** (optional): Three.js interactive, better for deep exploration

## Install

```bash
cargo install neuraldiff

# Or from source
git clone https://github.com/yourname/neuraldiff
cd neuraldiff
cargo build --release
```

## Usage

```bash
# Basic diff (TUI)
neuraldiff diff base.safetensors finetuned.safetensors

# 2D heatmap mode (faster, less GPU)
neuraldiff diff base.safetensors finetuned.safetensors --mode 2d

# 3D interactive web view
neuraldiff diff base.safetensors finetuned.safetensors --mode 3d

# JSON export
neuraldiff diff base.safetensors finetuned.safetensors --json > diff.json

# Compare multiple checkpoints (timeline)
neuraldiff timeline checkpoint_*.safetensors

# Inspect a single model
neuraldiff inspect model.safetensors
```

## Use cases

- **Fine-tuning:** Understand which layers your training actually touched
- **Merging:** Before merging two models, see how different they are layer-by-layer
- **Debugging:** Training loss plateaued — which layers stopped changing?
- **Research:** Compare LoRA-merged vs full fine-tune on same base

## Stack

- **Core:** Rust + `safetensors` crate
- **TUI:** `ratatui` + `crossterm`
- **3D web view:** Three.js (served locally via embedded HTTP server)
- **Math:** BLAS via `ndarray` + `ndarray-linalg`

## License

MIT
