# NeuralDiff — TUI Specification

> Detailed spec for the ratatui interface. Reference this when building `src/tui.rs`.

---

## Layout (80×24 minimum terminal size)

```
┌─ NEURALDIFF v0.1.0 ──────────────────────────────────────────────────────────┐
│ A: base.safetensors  →  B: finetuned.safetensors   3.09B params  36 layers   │
├──────────────────────────┬───────────────────────────────────────────────────┤
│ LAYERS              (34) │ DETAIL: layers.22.mlp                             │
│                          │                                                   │
│ ▶ 22  mlp    ████  0.847 │  Tensors:                                        │
│   31  attn   ███░  0.723 │  gate_proj.weight  [4096, 11008]  Δ=0.912       │
│   18  mlp    ███░  0.691 │  up_proj.weight    [4096, 11008]  Δ=0.834       │
│   05  attn   ██░░  0.534 │  down_proj.weight  [11008, 4096]  Δ=0.623       │
│   11  mlp    ██░░  0.498 │                                                   │
│   27  attn   ██░░  0.467 │  Heatmap (gate_proj.weight):                     │
│   03  mlp    █░░░  0.312 │  ⣿⣿⣿⣷⣶⣦⣤⣄⣀⣀⣀⣀⣄⣤⣦⣶⣷⣿⣿⣿⣿⣿⣷⣶⣦⣤⣄⣀⣀⣀   │
│   14  norm   ░░░░  0.012 │  ⣿⣿⣷⣶⣦⣤⣄⣀⣀⣀⣀⣀⣄⣤⣦⣶⣷⣿⣿⣿⣿⣷⣶⣦⣤⣄⣀⣀⣀⣀   │
│   00  embed  ░░░░  0.000 │  ⣷⣶⣦⣤⣄⣀⣀⣀⣀⣀⣀⣀⣄⣤⣦⣶⣷⣿⣿⣷⣶⣦⣤⣄⣀⣀⣀⣀⣀⣀   │
│   ...                    │  ⣶⣦⣤⣄⣀⣀⣀⣀⣀⣀⣀⣀⣄⣤⣦⣶⣷⣿⣷⣶⣦⣤⣄⣀⣀⣀⣀⣀⣀⣀   │
│                          │  ⣤⣄⣀⣀⣀⣀⣀⣀⣀⣀⣀⣀⣄⣤⣦⣶⣷⣿⣶⣦⣤⣄⣀⣀⣀⣀⣀⣀⣀⣀   │
│                          │                                                   │
│                          │  ⬛ no change  🟨 moderate  🟥 large delta       │
├──────────────────────────┴───────────────────────────────────────────────────┤
│ [↑↓/jk] layer  [←→/hl] tensor  [h] heatmap  [w] 3D web  [j] JSON  [q] quit │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Color scheme

| Element | Color |
|---------|-------|
| Title bar | Bold white on dark bg |
| Selected layer | Cyan background |
| L2 bar: low (< 0.3) | Green `█` |
| L2 bar: medium (0.3–0.6) | Yellow `█` |
| L2 bar: high (> 0.6) | Red `█` |
| Unchanged layers | Dim gray |
| Anomaly layers | Bold red + `⚠` prefix |
| Heatmap: 0.0 | ` ` (space) |
| Heatmap: 0.0–0.2 | `⣀` dim |
| Heatmap: 0.2–0.4 | `⣤` gray |
| Heatmap: 0.4–0.6 | `⣶` yellow |
| Heatmap: 0.6–0.8 | `⣷` orange |
| Heatmap: 0.8–1.0 | `⣿` red |

---

## Interactions

| Key | Action |
|-----|--------|
| `↑` / `k` | Move up in layer list |
| `↓` / `j` | Move down in layer list |
| `←` / `h` | Previous tensor in detail panel |
| `→` / `l` | Next tensor in detail panel |
| `Enter` | Toggle heatmap for selected tensor |
| `w` | Open 3D web view in browser |
| `J` (shift) | Export full diff to `diff.json` in cwd |
| `s` | Sort layers by: L2 (default) / layer index / anomaly score |
| `f` | Filter: show only changed / show all |
| `?` | Toggle help overlay |
| `q` / `Ctrl-C` | Quit |

---

## States

```rust
pub struct AppState {
    pub diff: DiffResult,
    pub selected_layer: usize,
    pub selected_tensor: usize,
    pub show_heatmap: bool,
    pub sort_mode: SortMode,
    pub filter_mode: FilterMode,
    pub show_help: bool,
    pub status_message: Option<String>,  // e.g. "JSON exported to diff.json"
}

pub enum SortMode { L2Desc, LayerIndex, AnomalyScore }
pub enum FilterMode { All, ChangedOnly }
```

---

## Heatmap rendering

The heatmap shows the absolute delta matrix of a 2D weight tensor.

For a tensor of shape `[rows, cols]`:
1. Compute `delta = |B - A|` element-wise
2. Normalize to `[0.0, 1.0]` using 99th percentile (not max, to avoid outlier dominance)
3. Downsample to terminal cell grid: `target_width = panel_width - 4`, `target_height = 10`
4. Map each cell value to braille character + color

```rust
fn render_heatmap(delta: &Array2<f32>, width: u16, height: u16) -> Vec<Vec<(char, Color)>>
```

For 1D tensors (bias vectors, norms): render as a single row.
For tensors with > 2 dimensions: reshape to 2D by flattening all dims except last two.

---

## Progress indicator

Large models take time to load and diff. Show progress:

```
Loading base.safetensors...     ████████████████████████░░░░  87%
```

Use `ratatui::widgets::Gauge` during load phase, then transition to main layout.

---

## Summary panel (shown on startup before layer selection)

```
┌─ SUMMARY ───────────────────────────────────────────────────────────────────┐
│                                                                              │
│  Models:   Qwen2.5-3B (3.09B params)                                        │
│  Layers:   36 total  ·  34 changed  ·  2 unchanged                         │
│                                                                              │
│  Most changed layers:                                                        │
│  layers.22.mlp      ████████████████████  Δ = 0.847                        │
│  layers.31.attn     ████████████████░░░░  Δ = 0.723                        │
│  layers.18.mlp      ██████████████░░░░░░  Δ = 0.691                        │
│                                                                              │
│  ⚠ Anomalies detected:  layers.22.mlp  (z-score: 3.2)                      │
│                                                                              │
│  Press ↓ or Enter to explore layers                                         │
│                                                                              │
└──────────────────────────────────────────────────────────────────────────────┘
```

---

## Edge cases

| Case | Behavior |
|------|----------|
| Models have different architectures | Error with diff of tensor names: "X tensors in A not in B" |
| Tensor shape mismatch | Show warning, skip that tensor in diff |
| Terminal < 80 cols | Show "Terminal too narrow" message |
| Terminal < 24 rows | Collapse heatmap panel |
| Model too large to memory-map | Error with OOM suggestion |
| Identical models | Show "No differences found" summary |
