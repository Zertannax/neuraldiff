# 🔬 NeuralDiff — Full Analysis & Monster Features Roadmap

> *"What if diffing model weights was just the beginning?"*
>
> Generated 2026-05-04 by Silas · For Claude Code evening session

---

## 📊 Part 1 — Current State Analysis

### What NeuralDiff Is Today

A **Rust CLI tool** that compares two AI model checkpoints (`.safetensors` format) and produces:
- Per-tensor metrics: L2 distance, cosine similarity, max/mean/std delta
- Layer-level aggregation (attention, MLP, norm, embedding, head)
- Anomaly detection (z-score > 2.0)
- Interactive TUI with heatmaps (ratatui + braille unicode)
- Optional web 3D viewer (Three.js)
- Auto-scan for models on the filesystem
- JSON/CSV export

### Architecture — What's Strong

| Component | Quality | Notes |
|---|---|---|
| **Modular design** | ✅ Excellent | `types.rs`, `loader.rs`, `diff.rs`, `mapper.rs`, `tui.rs`, `scanner.rs` — separation of concerns is clean |
| **Memory efficiency** | ✅ Good | `memmap2` for memory-mapped files, converts to f32 on load |
| **Parallelism** | ✅ Good | `rayon` for parallel tensor iteration |
| **TUI** | ✅ Very Good | ratatui with proper TerminalGuard lifecycle, unified flow (scan → pick → diff → explore) |
| **Layer mapping** | ✅ Solid | `mapper.rs` maps raw tensor names to logical layers for LLaMA/Mistral/Qwen, GPT-2, Falcon |
| **Test coverage** | ⚠️ Partial | 24 tests, mostly for scanner and mapping. No tests for diff computation or TUI rendering |

### Current Limitations

1. **Only `.safetensors`** — No GGUF, no PyTorch `.bin`/`.pt`, no ONNX
2. **Only 2-model comparison** — Can't compare a timeline of checkpoints (training run evolution)
3. **Metrics are basic** — L2 and cosine are fine, but miss structural information (rank, spectral properties, weight distribution shape)
4. **No "merge preview"** — Can't simulate SLERP/TIES/DARE merge before executing it
5. **No architecture inference** — Mapper needs manual regex patterns. Can't auto-detect "this is a GQA model with SwiGLU"
6. **No HuggingFace Hub integration** — Can't `neuraldiff diff user/model-a user/model-b` directly
7. **Web viewer is empty** — `web.rs` is a stub (8 lines)

---

## 🧠 Part 2 — Monster Features (Plot Twists)

### Feature 1: 🕰 "Chronos" — Temporal Checkpoint Analysis

**The gap:** All tools diff A vs B. Nobody diffs A → B → C → D across a training run.

**What it does:**
- Feed NeuralDiff a **directory of checkpoints** (or HF repo with multiple revisions)
- It computes the **evolution trajectory** of every layer across time
- Shows: which layers converge first, which oscillate, which "die" (stop learning after epoch N)

**Visual output in TUI:**
```
Layer                  Epoch 1    Epoch 5    Epoch 10   Epoch 50   Status
attention.q_proj       0.842      0.210      0.045      0.003      ✅ Converged
attention.k_proj         0.801      0.195      0.038      0.002      ✅ Converged
mlp.gate_proj            0.765      0.340      0.280      0.275      ⚠️ Oscillating
norm.weight              0.120      0.115      0.118      0.119      💀 Dead (no learning)
```

**Why it's a monster:**
- Training runs produce dozens of checkpoints. This turns them into **actionable intelligence**
- Detect "dead layers" before they waste GPU time
- Spot "unstable layers" that might cause divergence
- **No competitor does this** — not mergekit, not model-diffing, not git-lfs diff

**Implementation sketch:**
```rust
// src/chronos/mod.rs
pub struct TimelineAnalyzer {
    checkpoints: Vec<PathBuf>,
    baseline: ModelSnapshot,
}

impl TimelineAnalyzer {
    pub fn compute_trajectories(&self) -> Vec<LayerTrajectory> {
        // Parallel over layers, sequential over checkpoints
        // Store deltas per epoch, fit convergence curves
    }
    
    pub fn detect_dead_layers(&self, threshold: f32) -> Vec<LayerTrajectory> {
        // Layer whose delta variance < threshold across all checkpoints
    }
}
```

---

### Feature 2: 🔮 "Merge Oracle" — Predict Merge Conflicts Before They Happen

**The gap:** Model merging (SLERP, TIES, DARE, Task Arithmetic) is trial-and-error. You merge, you test, you pray.

**What it does:**
- Select two models and a merge method
- NeuralDiff **simulates** the merge and predicts:
  - Which layers will "interfere" (deltas in opposite directions)
  - Which layers are "compatible" (deltas aligned)
  - Predicted final metrics without running inference

**Visual output:**
```
Merge Preview: base + LoRA_A (SLERP, t=0.5)

Compatible layers (smooth merge):     87%
Conflicting layers (destructive):      8%  ← attention.o_proj, mlp.down_proj
Orthogonal layers (neutral):           5%

Predicted output quality: 0.73 (vs 0.71 base, 0.69 A)
Recommendation: ✅ SAFE to merge
```

**Why it's a monster:**
- Saves hours of GPU inference testing bad merges
- Makes mergekit-style workflows **interactive and predictive**
- Could become the standard pre-merge check

**Implementation sketch:**
```rust
// src/merge_oracle/mod.rs
pub enum MergeMethod { Slerp, Ties, Dare, TaskArithmetic }

pub struct MergeOracle;

impl MergeOracle {
    pub fn predict(base: &ModelSnapshot, a: &ModelSnapshot, method: MergeMethod) -> MergePrediction {
        // Compute merge without materializing full tensors
        // Use directional analysis: cosine of delta vectors
        // Predict interference score per layer
    }
}
```

---

### Feature 3: 🎼 "Spectral Fingerprint" — SVD-Based Structural Analysis

**The gap:** L2 and cosine are scalar metrics. They tell you "how different" but not **"in what way"**.

**What it does:**
- For every weight matrix, compute **SVD** and compare the singular value spectra
- Detect if a fine-tune increased or decreased the **effective rank** of a layer
- Show if quantization destroyed high-frequency components

**Visual output:**
```
Layer: attention.q_proj (768×768)

Base model:    effective rank = 312 / 768  (41% capacity used)
Fine-tuned:    effective rank = 287 / 768  (37% capacity used)  ← rank collapsed!

Singular value decay:
Base:    ████████████████████░░░░░░░░░░
Tuned:   ███████████████░░░░░░░░░░░░░░░  ← faster decay = information loss

Interpretation: Fine-tuning narrowed the representational capacity.
                Possible overfitting or catastrophic forgetting.
```

**Why it's a monster:**
- Reveals **why** models differ, not just **how much**
- Detects rank collapse (common in LoRA over-training)
- Quantization-aware: show which spectral components are lost
- **Zero inference needed** — pure weight analysis

**Implementation sketch:**
```rust
// src/spectral/mod.rs
use ndarray_linalg::SVD;

pub fn effective_rank(singular_values: &[f32]) -> f32 {
    // Nuclear norm / spectral norm = sum(s_i) / max(s_i)
    // Or use entropy-based: -sum(p_i * log(p_i)) where p_i = s_i / sum(s)
}

pub fn compare_spectra(base: Array2<f32>, tuned: Array2<f32>) -> SpectralDiff {
    let svd_base = base.svd(true, true).unwrap();
    let svd_tuned = tuned.svd(true, true).unwrap();
    // Compare singular value distributions, subspace angles, etc.
}
```

---

### Feature 4: 🏛 "Architecture Archaeologist" — Auto-Detect Model Family

**The gap:** Mapper needs manual regexes per architecture. User must know what model they're diffing.

**What it does:**
- Load any model → auto-detect architecture from tensor names, shapes, and weight patterns
- Output: *"This is a LLaMA-3-8B variant with GQA (4 KV heads), SwiGLU MLP, and tied embeddings"*
- Handle "frankenmodels" — detect when someone mixed architectures

**Visual output:**
```
Model Autopsy: mystic-model-v2.gguf

Detected architecture: LLaMA-3 derivative (custom)
- Hidden dim: 4096
- Layers: 32
- Attention: GQA (8 query groups)
- MLP: SwiGLU (gate_up + down, no bias)
- Norm: RMSNorm (no bias)
- Embeddings: UNTIED (vocab 128256 × 4096)

⚠️ Anomaly detected: layer 15 has attention head count = 16 (expected 32)
   → Possible MoE routing layer or architecture modification
```

**Why it's a monster:**
- No more manual regex maintenance
- Handle the explosion of custom architectures (Yi, InternLM, CodeLlama, etc.)
- Detect architecture drift between base and fine-tuned models

---

### Feature 5: 🌐 "Hub Diff" — HuggingFace Integration

**The gap:** Must download models manually. No `neuraldiff diff meta-llama/Llama-3-8B-Instruct mistralai/Mistral-7B-Instruct-v0.3`.

**What it does:**
- Native HuggingFace Hub integration (via `hf-hub` crate)
- Cache management, resume downloads, API token support
- Compare models without manual download steps

**CLI:**
```bash
neuraldiff diff "meta-llama/Llama-3-8B" "meta-llama/Llama-3-8B-Instruct" --hub
neuraldiff diff "Qwen/Qwen2-7B" "Qwen/Qwen2-7B-Instruct" --hub --token $HF_TOKEN
```

---

## 🔩 Part 3 — GGUF Support Deep Dive

### Why GGUF Matters

GGUF is the **dominant format** for local inference:
- llama.cpp, Ollama, LM Studio, KoboldCpp — all use GGUF
- TheBloke, bartowski, Qwen official — publish in GGUF
- Quantization is built-in (Q4_K_M, Q5_K_M, Q6_K, Q8_0, IQ quants)

**Without GGUF, NeuralDiff misses 80% of models people actually run locally.**

### Technical Analysis

#### GGUF Format Structure

```
GGUF File:
├── Header (magic "GGUF", version, tensor_count, metadata_kv_count)
├── Metadata (JSON-like key-value: architecture, context_length, quantization, etc.)
├── Tensor Info (name, dimensions, type, offset in file)
└── Tensor Data (raw bytes, quantized)
```

#### Available Rust Crates

| Crate | Use Case | Maturity | Recommendation |
|---|---|---|---|
| `gguf-parser` | Header/metadata only, no weight loading | ✅ Stable | Good for inspection |
| `gguf-rs-lib` | Read/write full GGUF | ⚠️ Early | Possible but limited docs |
| `pmetal-gguf` | PMetal/MLX ecosystem | ⚠️ Niche | Not general purpose |
| `llama-gguf` | Full inference engine | ✅ Mature | Too heavy for diffing |
| `mlmf` | Multi-format (GGUF+Safetensors+ONNX) | ⚠️ Early | Interesting but unproven |
| `ggml-quant` (llama.cpp bindings) | Dequantization | ✅ Mature | Best for accurate dequant |

**Recommendation for NeuralDiff:**

Use `gguf-parser` for metadata + **custom dequantization** for tensor comparison.

Why custom? Because:
1. Dequantizing a full 70B model to f32 = ~280GB RAM. NeuralDiff only needs **per-tensor** dequant → compare → discard.
2. For diffing, we can sometimes compare **without full dequant** (compare scales, compare block distributions).
3. We want **quantization-aware metrics**, not just "pretend it's f32".

#### Implementation Roadmap

**Phase 1: GGUF Inspection (Week 1)**
```rust
// src/loader/gguf.rs
use gguf_parser::GgufFile;

pub struct GgufLoader;

impl GgufLoader {
    pub fn inspect(path: &Path) -> Result<GgufMetadata> {
        let file = File::open(path)?;
        let mut reader = BufReader::new(file);
        let gguf = GgufFile::parse(&mut reader)?;
        
        Ok(GgufMetadata {
            architecture: gguf.architecture(),
            quantization: gguf.quantization_name(),
            context_length: gguf.context_length(),
            tensor_count: gguf.header.tensor_count,
            metadata: gguf.metadata().clone(),
        })
    }
}
```

**Phase 2: Tensor Extraction + Dequantization (Week 2-3)**
```rust
// Support all GGML quant types
pub enum GgmlType {
    F32, F16, BF16, Q4_0, Q4_1, Q5_0, Q5_1, Q8_0,
    Q2_K, Q3_K_S, Q3_K_M, Q3_K_L, Q4_K_S, Q4_K_M,
    Q5_K_S, Q5_K_M, Q6_K, IQ1_S, IQ1_M, IQ2_XXS, /* ... */
}

pub fn dequantize_tensor(
    tensor_info: &TensorInfo,
    raw_bytes: &[u8],
    target_dtype: DType,
) -> Result<ArrayD<f32>> {
    match tensor_info.ggml_type {
        GgmlType::F32 => Ok(parse_f32(raw_bytes, &tensor_info.shape)),
        GgmlType::F16 => Ok(parse_f16(raw_bytes, &tensor_info.shape)),
        GgmlType::Q4_0 => dequantize_q4_0(raw_bytes, &tensor_info.shape),
        GgmlType::Q4_K_M => dequantize_q4_k_m(raw_bytes, &tensor_info.shape),
        GgmlType::Q8_0 => dequantize_q8_0(raw_bytes, &tensor_info.shape),
        // ... all 30+ quant types
    }
}
```

**Dequantization approach:**
- For small tensors: dequantize fully to f32, then diff normally
- For large tensors: dequantize block-by-block ( streaming ), compute running metrics
- For quantization-aware diff: compare quant parameters (scales, zero-points, importance matrix)

**Phase 3: Unified Loader Abstraction (Week 4)**
```rust
// src/loader/mod.rs
pub trait ModelLoader {
    fn load(&self, path: &Path) -> Result<ModelSnapshot>;
    fn inspect(&self, path: &Path) -> Result<ModelMetadata>;
}

pub struct SafetensorsLoader;
pub struct GgufLoader;

impl ModelLoader for SafetensorsLoader { /* existing code */ }
impl ModelLoader for GgufLoader { /* new code */ }

// Auto-detect format from file extension/magic
pub fn auto_load(path: &Path) -> Result<ModelSnapshot> {
    match detect_format(path)? {
        ModelFormat::Safetensors => SafetensorsLoader.load(path),
        ModelFormat::Gguf => GgufLoader.load(path),
        ModelFormat::PyTorch => Err("PyTorch .bin support planned for v0.4"),
    }
}
```

**Phase 4: Quantization-Aware Metrics (Week 5-6)**

Instead of "dequantize everything to f32 and pretend", add metrics that are **native to quantized models**:

```rust
pub struct QuantizationDiff {
    pub dequantized_l2: f32,        // Traditional metric
    pub scale_divergence: f32,      // How different are the quantization scales
    pub zero_point_shift: f32,      // Zero-point drift
    pub block_variance_ratio: f32,  // Per-block variance comparison
    pub imatrix_divergence: f32,    // If importance matrix available
}
```

This is **differentiating** — no other tool shows quantization-native metrics.

---

## 🔬 Part 4 — Analysis "Like a Transformers Library"

### What Transformers Libraries Do That NeuralDiff Doesn't (Yet)

| Capability | Transformers (HF) | NeuralDiff Today | Gap |
|---|---|---|---|
| **Auto-config detection** | `AutoConfig.from_pretrained()` auto-detects architecture | Manual regex in `mapper.rs` | ❌ |
| **Tokenizer awareness** | Load tokenizer, compare vocab embeddings | Only compares raw tensors | ❌ |
| **Model class mapping** | `AutoModel`, `AutoModelForCausalLM` | Flat tensor list | ❌ |
| **Attention pattern analysis** | GQA vs MHA vs MLA detection | Basic layer grouping | ⚠️ |
| **Weight tying detection** | `tie_word_embeddings` | Not explicitly handled | ⚠️ |
| **Bias detection** | `use_bias` per layer | Bias is just another tensor | ✅ (implicitly works) |
| **Quantization config** | `BitsAndBytesConfig`, `GPTQConfig` | No quant awareness | ❌ |

### What NeuralDiff Should Steal From Transformers

**1. `AutoConfig` equivalent**
```rust
// src/archaeologist/mod.rs
pub fn detect_architecture(tensors: &HashMap<String, Tensor>) -> Architecture {
    // Heuristic-based detection from tensor names + shapes
    if tensors.contains_key("model.embed_tokens.weight") 
        && tensors.contains_key("model.layers.0.self_attn.q_proj.weight") {
        // Check for GQA: k_proj shape vs q_proj shape
        let gqa = is_gqa(tensors);
        // Check for SwiGLU: gate_up vs up_proj naming
        let mlp_type = detect_mlp_type(tensors);
        Architecture::LlamaVariant { gqa, mlp_type, ... }
    }
}
```

**2. Weight tying detection**
```rust
// Compare embedding and lm_head — are they the same tensor?
pub fn detect_weight_tying(a: &ModelSnapshot, b: &ModelSnapshot) -> bool {
    // Check if embed.weight and lm_head.weight are identical
    // Common in GPT-2, uncommon in LLaMA-3
}
```

**3. Attention head analysis**
```rust
// Detect GQA, MHA, MLA from tensor shapes
pub fn analyze_attention(tensors: &HashMap<String, Tensor>) -> AttentionConfig {
    let q_shape = tensors["self_attn.q_proj.weight"].shape;
    let k_shape = tensors["self_attn.k_proj.weight"].shape;
    let num_heads = q_shape[0] / head_dim;
    let num_kv_heads = k_shape[0] / head_dim;
    AttentionConfig {
        type_: if num_kv_heads == num_heads { MHA }
               else if num_kv_heads == 1 { MQA }
               else { GQA(num_kv_heads) },
    }
}
```

---

## 🎯 Recommended Priority Order

If I had to pick the sequence that maximizes impact with your existing architecture:

| Priority | Feature | Effort | Impact | Why First |
|---|---|---|---|---|
| **1** | **GGUF Support** | Medium-High | 🔥🔥🔥🔥🔥 | 80% of local models are GGUF. Without this, the tool is niche. |
| **2** | **Hub Diff** | Low | 🔥🔥🔥🔥 | Quick win. `hf-hub` crate is mature. One week of work. |
| **3** | **Architecture Archaeologist** | Medium | 🔥🔥🔥🔥 | Eliminates manual mapper maintenance. Enables all future features. |
| **4** | **Chronos (Timeline)** | High | 🔥🔥🔥🔥🔥 | The true differentiator. Nobody else has this. |
| **5** | **Spectral Fingerprint** | Medium | 🔥🔥🔥 | Reveals "why", not just "how much". Needs `ndarray-linalg`. |
| **6** | **Merge Oracle** | Medium-High | 🔥🔥🔥🔥 | Positions NeuralDiff as a merge workflow tool, not just a diff tool. |
| **7** | **Quantization-Aware Metrics** | High | 🔥🔥🔥🔥 | Unique value for GGUF models. No competitor does this. |

---

## 🔗 Links & Context

- Repo: `Zertannax/neuraldiff`
- Current version: `0.2.0`
- Stack: Rust 1.88, safetensors, ndarray, rayon, ratatui, memmap2
- Target: Model developers, fine-tuners, merge experimenters, quantization researchers

---

## 📝 Implementation Notes for Claude Code

### GGUF Dequantization — Where to Start

The reference implementation is in `llama.cpp`'s `ggml-quants.c`. For Rust:

1. **Option A — Bindings**: Use `llama-cpp-rs` or `ggml` bindings for dequant. Accurate but heavy dependency.
2. **Option B — Pure Rust**: Port the dequant functions. ~2000 LOC but zero dependencies. Full control.
3. **Option C — Hybrid**: Use `gguf-parser` for metadata, implement dequant for the 5 most common types first (F32, F16, Q4_0, Q4_K_M, Q8_0), add others incrementally.

**My recommendation: Option C.** Start with the 80% case (Q4_K_M is ~60% of GGUF downloads), iterate.

### Key Crates to Add

```toml
[dependencies]
# GGUF support
gguf-parser = "0.3"           # Header/metadata parsing
# OR pure-rust dequant (implement yourself)
# half = "2.4"                # Already have this — use for F16 parsing

# For spectral analysis (Feature 3)
ndarray-linalg = "0.16"       # SVD, eigenvalue analysis

# For HuggingFace Hub (Feature 5)
hf-hub = "0.4"                # Download models from HF

# For serialization (Chronos timeline data)
serde = { version = "1.0", features = ["derive"] }  # Already have
ron = "0.8"                   # Human-readable config/timeline format
```

### Testing Strategy

1. Download 3 real GGUF models (small, medium, large)
   - `Qwen/Qwen2.5-0.5B-Instruct-GGUF` (tiny, fast tests)
   - `bartowski/Llama-3.2-1B-Instruct-GGUF` (medium)
   - `TheBloke/Mistral-7B-Instruct-v0.2-GGUF` (large, stress test)

2. For each, test:
   - Metadata extraction matches `gguf-inspect` CLI
   - Tensor shapes match after dequantization
   - Memory usage stays bounded (no OOM on large models)
   - Diff results are consistent with safetensors equivalent (if available)

---

> *"The best diff tool doesn't just show differences. It tells you what they mean."*
>
> — Silas, 2026-05-04
