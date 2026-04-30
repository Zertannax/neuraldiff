# NeuralDiff

> Visual diff between AI model checkpoints. See what changed between your fine-tune runs.

## Demo

ash
$ neuraldiff diff base.safetensors finetuned.safetensors

NEURALDIFF v0.1.0    base.safetensors ? finetuned.safetensors    3.09B params

+--------------+  +--------------+  +--------------+  +--------------+
¦  3.09B       ¦  ¦     36       ¦  ¦     34       ¦  ¦      2       ¦
¦  Parameters  ¦  ¦   Layers     ¦  ¦   Changed    ¦  ¦  Unchanged   ¦
+--------------+  +--------------+  +--------------+  +--------------+

Top Changed Layers:
#1 layers.22.mlp    [¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦]  0.847 [CRIT]
#2 layers.31.attn   [¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦]  0.723 [HIGH]
#3 layers.18.mlp    [¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦]  0.691 [HIGH]
#4 layers.05.attn   [¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦]  0.534 [MED]
#5 layers.11.mlp    [¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦¦]  0.498 [MED]

[R] Anomalies Detected
  layers.22.mlp (z-score: 3.24)

Press ? or Enter to explore individual layers

## What it does

You fine-tuned a model. Something changed. But **what** changed, and **how much**?

NeuralDiff compares two .safetensors checkpoints and shows:
- Which layers changed (L2 distance, cosine similarity)
- How much each layer changed (colored bars by layer)
- Anomaly detection: layers with unusually large shifts

## Install

ash
cargo install neuraldiff

# Or from source
git clone https://github.com/yourname/neuraldiff
cd neuraldiff
cargo build --release

## Usage

ash
# Basic diff (TUI)
neuraldiff diff base.safetensors finetuned.safetensors

# JSON export
neuraldiff diff base.safetensors finetuned.safetensors --json > diff.json

# Inspect a single model
neuraldiff inspect model.safetensors

## Features

- **TUI Interface**: Dark-themed terminal UI with navigation, sorting, and filtering
- **Layer Grouping**: Automatically groups tensors by logical layer (attention, MLP, norm, embedding, head)
- **Anomaly Detection**: Highlights layers with unusually high delta using z-score analysis
- **Parallel Processing**: Uses rayon for fast tensor-level diff computation
- **Memory Efficient**: Memory-mapped file I/O for handling multi-GB models
- **Offline First**: No network requests, everything stays local

## Stack

- **Core**: Rust + safetensors crate
- **TUI**: atatui + crossterm
- **Math**: 
darray + ayon
- **Serialization**: serde + serde_json

## License

MIT
