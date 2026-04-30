# NeuralDiff — Architecture

## Overview

```
CLI args (files, mode)
        │
        ▼
┌────────────────────┐
│   SafetensorsLoader│  Parse .safetensors, extract tensor metadata + data
└────────┬───────────┘
         │  (ModelSnapshot × 2)
         ▼
┌────────────────────┐
│   DiffEngine       │  Compute distances per tensor/layer
└────────┬───────────┘
         │  (DiffResult)
         ▼
┌────────────────────┐
│   LayerMapper      │  Group tensors by logical layer (attn/mlp/norm/embed)
└────────┬───────────┘
         │  (LayerDiff[])
         ▼
    ┌────┴────────────┐
    │                 │
    ▼                 ▼
TuiRenderer      WebRenderer
(ratatui)        (Three.js via local HTTP)
```

---

## Core data types

```rust
/// A loaded model checkpoint
pub struct ModelSnapshot {
    pub path: PathBuf,
    pub tensors: HashMap<String, TensorMeta>,
    pub total_params: u64,
}

pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data: Arc<Vec<f32>>,  // always converted to f32
}

/// Result of comparing two tensors
pub struct TensorDiff {
    pub name: String,
    pub l2_distance: f32,
    pub cosine_similarity: f32,
    pub max_delta: f32,
    pub mean_delta: f32,
    pub std_delta: f32,
    pub changed: bool,  // true if l2 > threshold
}

/// Grouped by logical layer
pub struct LayerDiff {
    pub layer_index: usize,
    pub layer_type: LayerType,  // Attention | MLP | Norm | Embedding | Head
    pub tensors: Vec<TensorDiff>,
    pub aggregate_l2: f32,      // mean L2 across layer's tensors
    pub anomaly_score: f32,     // z-score vs other layers
}

pub struct DiffResult {
    pub model_a: ModelSnapshot,
    pub model_b: ModelSnapshot,
    pub layers: Vec<LayerDiff>,
    pub summary: DiffSummary,
}

pub struct DiffSummary {
    pub total_layers: usize,
    pub changed_layers: usize,
    pub unchanged_layers: usize,
    pub top_changed: Vec<LayerDiff>,  // top 5 by l2
    pub anomalies: Vec<LayerDiff>,    // z-score > 2.0
}
```

---

## Module breakdown

### `src/loader.rs` — SafetensorsLoader

**Responsibility:** Parse `.safetensors` files efficiently.

The safetensors format is:
```
[8 bytes: header length N] [N bytes: JSON header] [raw tensor data]
```

Header JSON contains tensor names, dtypes, shapes, and byte offsets.

Key decisions:
- **Memory map** large files instead of loading fully (models can be 3–30GB)
- **Convert all dtypes to f32** at load time (BF16, F16, F32 → F32). Simplifies all downstream math.
- **Lazy loading**: only load tensor data when needed by DiffEngine

```rust
pub fn load(path: &Path) -> Result<ModelSnapshot>
pub fn load_tensor(snapshot: &ModelSnapshot, name: &str) -> Result<Vec<f32>>
```

---

### `src/diff.rs` — DiffEngine

**Responsibility:** Compute distance metrics between corresponding tensors.

For each tensor name present in both models:

```rust
fn diff_tensor(a: &[f32], b: &[f32]) -> TensorDiff {
    let delta: Vec<f32> = a.iter().zip(b).map(|(x, y)| y - x).collect();
    
    TensorDiff {
        l2_distance:        l2_norm(&delta),
        cosine_similarity:  cosine_sim(a, b),
        max_delta:          delta.iter().cloned().fold(f32::NEG_INFINITY, f32::max),
        mean_delta:         delta.iter().sum::<f32>() / delta.len() as f32,
        std_delta:          std_dev(&delta),
        changed:            l2_norm(&delta) > CHANGE_THRESHOLD,
    }
}
```

`CHANGE_THRESHOLD` default: `1e-6` (catches float precision noise)

Parallelized with `rayon`:
```rust
tensors.par_iter().map(|(name, _)| diff_tensor(...)).collect()
```

---

### `src/mapper.rs` — LayerMapper

**Responsibility:** Group flat tensor names into logical layers.

Transformer models use naming conventions. Examples:
```
model.layers.0.self_attn.q_proj.weight   → Layer 0, Attention, Q projection
model.layers.0.self_attn.k_proj.weight   → Layer 0, Attention, K projection
model.layers.0.mlp.gate_proj.weight      → Layer 0, MLP, Gate
model.embed_tokens.weight                → Embedding
lm_head.weight                           → Head
```

The mapper uses regex patterns to extract layer index and component type. Handles:
- LLaMA / Mistral / Qwen naming (`model.layers.N.`)
- GPT-2 naming (`transformer.h.N.`)
- Falcon naming (`transformer.h.N.`)
- Fallback: group by common prefix

```rust
pub fn map(tensors: &[TensorDiff]) -> Vec<LayerDiff>
```

---

### `src/tui.rs` — TuiRenderer

**Responsibility:** ratatui-based interactive terminal view.

Layout:
```
┌──────────────────────────────────────────────────────────────┐
│ NEURALDIFF  base.safetensors → finetuned.safetensors         │
├────────────────────────┬─────────────────────────────────────┤
│ Layer List             │ Layer Detail                         │
│                        │                                     │
│ ▶ 22.mlp    ████ 0.847 │  layer.22.mlp                      │
│   31.attn   ███  0.723 │  ┌──────────────────────────┐      │
│   18.mlp    ███  0.691 │  │ Heatmap (2D)              │      │
│   05.attn   ██   0.534 │  │ [rendered with braille    │      │
│   ...                  │  │  unicode blocks]           │      │
│                        │  └──────────────────────────┘      │
│                        │                                     │
│                        │  gate_proj  ████████  0.912        │
│                        │  up_proj    ███████░  0.834        │
│                        │  down_proj  █████░░░  0.623        │
├────────────────────────┴─────────────────────────────────────┤
│ [↑↓] navigate  [h] heatmap  [w] open web 3D  [j] JSON  [q] quit │
└──────────────────────────────────────────────────────────────┘
```

Heatmap rendering in terminal:
- Divide tensor into a grid (e.g., 64×32 cells)
- Average absolute delta per cell
- Render with braille unicode characters (`⣿`, `⣶`, `⣤`, `⣀`, ` `)
- Color: green (no change) → yellow → red (large change)

---

### `src/web.rs` — WebRenderer

**Responsibility:** Spawn a local HTTP server, serve Three.js app, open browser.

On `--mode 3d`:
1. Serialize `DiffResult` to JSON
2. Start `tokio` HTTP server on `localhost:7070`
3. Serve embedded `index.html` (Three.js app compiled into binary via `include_str!`)
4. Open browser automatically
5. Shut down when browser disconnects or user presses Q in terminal

Three.js visualization:
- Each layer = a disc in 3D space, stacked vertically
- Color = aggregate L2 distance (green → red gradient)
- Size = number of parameters in layer
- Click layer → explode into tensor components
- Orbit controls for rotation

---

### `src/metrics.rs` — Math utilities

```rust
pub fn l2_norm(v: &[f32]) -> f32
pub fn cosine_sim(a: &[f32], b: &[f32]) -> f32
pub fn std_dev(v: &[f32]) -> f32
pub fn z_scores(values: &[f32]) -> Vec<f32>
pub fn percentile(v: &[f32], p: f32) -> f32
```

No external ML libraries needed. Pure Rust + `ndarray` for matrix ops on large tensors.

---

## Performance considerations

| Model size | Load time | Diff time |
|-----------|-----------|-----------|
| 3B params | ~8s | ~12s |
| 7B params | ~20s | ~30s |
| 13B params | ~40s | ~60s |

Target: diff a 7B model in under 60s total on consumer hardware.

Key optimizations:
- Memory-mapped file I/O (avoid loading 14GB into RAM twice)
- `rayon` parallelism for tensor-level diff
- BF16 → F32 conversion on the fly, not pre-loaded
- L2 norm computed via BLAS (`ndarray-linalg`) when available

---

## Project structure

```
neuraldiff/
├── src/
│   ├── main.rs
│   ├── cli.rs         # clap argument parsing
│   ├── loader.rs      # safetensors parsing
│   ├── diff.rs        # diff engine
│   ├── mapper.rs      # layer grouping
│   ├── metrics.rs     # math utilities
│   ├── tui.rs         # ratatui renderer
│   ├── web.rs         # Three.js server
│   └── types.rs       # shared data types
├── web/
│   └── index.html     # Three.js app (embedded into binary)
├── tests/
│   ├── fixtures/      # small .safetensors test files
│   └── diff_tests.rs
├── ARCHITECTURE.md
├── ROADMAP.md
├── SPEC_TUI.md
├── SPEC_VIZ.md
├── Cargo.toml
└── README.md
```

---

## Dependencies (`Cargo.toml`)

```toml
[dependencies]
# CLI
clap = { version = "4", features = ["derive"] }

# Terminal UI
ratatui = "0.27"
crossterm = "0.27"

# Safetensors parsing
safetensors = "0.4"
memmap2 = "0.9"      # memory-mapped files

# Math
ndarray = "0.15"
rayon = "1.10"       # parallelism

# Web server (3D mode)
tokio = { version = "1", features = ["full"] }
axum = "0.7"

# Serialization
serde = { version = "1", features = ["derive"] }
serde_json = "1"

# Misc
anyhow = "1"
half = "2"           # BF16/F16 conversion
```
