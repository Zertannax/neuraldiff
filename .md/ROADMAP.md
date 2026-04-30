# NeuralDiff — Roadmap

---

## v0.1.0 — Hackathon MVP (ce weekend)

**Goal:** Working TUI diff, publishable on GitHub, demo GIF ready.

### Must have
- [ ] SafetensorsLoader (parse + dtype conversion)
- [ ] DiffEngine (L2 + cosine per tensor, parallelized)
- [ ] LayerMapper (LLaMA/Qwen/Mistral naming support)
- [ ] TUI: layer list + summary panel
- [ ] TUI: tensor detail panel
- [ ] TUI: heatmap (braille chars, 2D only)
- [ ] CLI: `neuraldiff diff A.safetensors B.safetensors`
- [ ] `--json` export
- [ ] README with demo GIF (asciinema)
- [ ] MIT license

### Nice to have (if time allows)
- [ ] Web 3D view (basic — tower of discs, no explode)
- [ ] `neuraldiff inspect` (single model stats)
- [ ] Progress bar during load

---

## v0.2.0 — Post-hackathon

- [ ] Web 3D view fully interactive (click to explode, hover tooltip)
- [ ] Timeline mode (`neuraldiff timeline *.safetensors`)
- [ ] GPT-2 / Falcon naming support in LayerMapper
- [ ] Anomaly detection + highlighting
- [ ] Sort/filter in TUI (by L2, anomaly, layer type)
- [ ] Homebrew formula
- [ ] Pre-built binaries for Linux/macOS/Windows

---

## v0.3.0

- [ ] GGUF format support (llama.cpp models)
- [ ] LoRA diff: compare LoRA adapter to base effect
- [ ] Merge preview: estimate final model weights before merging
- [ ] Export heatmap as PNG

---

## v1.0.0

- [ ] Multi-model comparison (A vs B vs C)
- [ ] Layer importance scoring (combine diff with gradient norms)
- [ ] Plugin system for custom metrics
- [ ] Benchmarks published

---

## Explicitly out of scope

- Training pipeline integration — stay as analysis tool, not training tool
- Cloud backend — offline-first, stays local
- Paid tier — MIT license, always free

---

## Hackathon timeline

| Time | Task |
|------|------|
| Vendredi soir 21h | `cargo new neuraldiff`, Cargo.toml deps, types.rs |
| Vendredi soir 22h–00h | loader.rs (safetensors parse + f32 conversion) |
| Samedi matin 9h–11h | diff.rs (L2 + cosine, rayon parallel) |
| Samedi 11h–13h | mapper.rs (LLaMA/Qwen naming) + metrics.rs |
| Samedi 13h–16h | tui.rs (layer list + summary panel) |
| Samedi soir | Rave. |
| Dimanche 10h–12h | tui.rs (heatmap, tensor detail) |
| Dimanche 12h–15h | web.rs + index.html (basic 3D si motivé, sinon skip v0.1) |
| Dimanche 15h–17h | README, asciinema démo, release binaires |
| Dimanche 17h | Push. Done. |
