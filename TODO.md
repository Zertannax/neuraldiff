# NeuralDiff — Roadmap to publication

> Last updated: 2026-05-01
> Current state: v0.1.x feature-complete, 33 tests green, working on real models (Qwen3-0.6B Base vs Instruct verified)

---

## ✅ Done — session 2026-05-01

- RAII `TerminalGuard` — terminal never wedged on error (C10/C11/C12)
- UTF-8-safe truncation + reset selection on filter (C8/C9/C17)
- Typed `LayerKey` refactor — distinct LayerDiff per norm component (C2/C3/C4)
- Heatmap caching via `Arc<ModelSnapshot>` in AppState (C5)
- WSL-aware scanner + skip-list for huge dirs (AppData, etc.)
- CLI cleaned: `--help` lisible, version dynamic, default subcommand, tracing→stderr
- TUI polish: branded header, sober footer, smooth-gradient half-block heatmap
- Extended CLI: `summary`, `inspect --top/--json`, `scan --root/--json`, `diff --output`
- Unified TUI flow: one command, single terminal session, no flashing
- Scanning loading screen (no more black screen on startup)

---

## 🚀 Phase 1 — Publishable (target: v0.2.0 release-ready)

**Goal:** GitHub public release with confidence. Estimated total: ~6h.

### Critical (must-have before tagging)

- [ ] **LICENSE file** — `LICENSE` MIT in repo root (Cargo.toml says MIT but no file exists). _2 min_
- [ ] **README rewrite** — current README has a v0.1.0 demo, no screenshot of new branded TUI, no mention of `summary`/`scan --root`. _30 min_
  - Add screenshot of summary view + heatmap on Qwen3 diff
  - Update keybindings table
  - Document the unified `neuraldiff` (no-args) flow as the primary entry
  - Drop the "v0.1.0" placeholders
- [ ] **NaN / Inf detection (C14)** — currently NaN deltas propagate silently into JSON/CSV. Add `summary.nan_count` / `summary.inf_count`, surface in TUI. _30 min_
- [ ] **Multi-shard model support** — open the `model.safetensors.index.json` if present; load all shards under one logical `ModelSnapshot`. Required to diff Gemma 31B / Llama 70B. _2-4h_
  - New `loader::load_sharded(dir)` that detects index.json
  - Update CLI to accept either a file or a directory
  - Test on the local Gemma fixtures

### Important (should-have)

- [ ] **`.gitignore`** — exclude `target/`, `*.json`, `*.csv`, `/tmp/*`. _5 min_
- [ ] **Critical bugs `--threshold` actually filters** in TUI/CSV/JSON (currently accepted but pass-through). _15 min_
- [ ] **Selected_tensor bounds clamp (C19)** — small UX bug after layer change. _10 min_
- [ ] **CSV/JSON output paths** — currently hardcoded to CWD; honor `--output` everywhere. _15 min_
- [ ] **`cargo fmt` pass** — pre-existing non-conformance in diff.rs/lib.rs. _10 min_
- [ ] **Clippy zero-warnings** — bring count from 22 down to 0 via let-chains, doc-comments. _45 min_

---

## 🎁 Phase 2 — Public release (v0.2.0 tagged)

**Goal:** binary distribution, crates.io, GitHub release. Estimated: ~3h.

- [ ] **CI: GitHub Actions** — `.github/workflows/ci.yml` runs `cargo test` + `cargo clippy` + `cargo fmt --check` on push/PR. _30 min_
- [ ] **Pre-built binaries** — `cargo dist` (or `cargo binstall`) for Linux x86_64, macOS arm64, Windows x86_64. Attach to GitHub release. _1h_
- [ ] **crates.io publish** — `cargo publish` after cleaning `Cargo.toml` (description, keywords, repository, categories). _15 min_
- [ ] **GitHub release v0.2.0** — git tag, release notes, screenshots/GIF, link to binaries. _30 min_
- [ ] **Demo GIF or screencast** — 30s of `neuraldiff` running on Qwen Base→Instruct. Embed in README. _30 min_
- [ ] **Open GitHub issues** for known Critical bugs (transparency: C1, C6/C7, C13, C15, C16, C18). _15 min_

---

## 🧪 Phase 3 — Hardening (v0.3.0)

**Goal:** correctness on edge cases, broader test coverage. Estimated: ~6h.

- [ ] **Test fixtures per dtype** — F16, BF16, I64, I32, I8, U8, Bool. Currently only F32 is exercised (T6). _2h_
- [ ] **Test `compute_diff` failure modes** — shape mismatch, missing tensor, dtype mismatch (T7). _1h_
- [ ] **F64 accumulators in metrics** — `l2_norm`, `aggregate_l2`, `compute_summary::mean_delta` (C15). _30 min_
- [ ] **Loader overflow check (C1)** — `start.checked_add(data_len)`. _15 min_
- [ ] **Skip integer tensors in diff (C13)** — token-id tensors give meaningless L2; emit warning, exclude from aggregate. _30 min_
- [ ] **Bool decoder strict (C18)** — assert byte ∈ {0,1}, warn otherwise. _15 min_
- [ ] **Heatmap math (C6/C7)** — bail on shape > 2D, fix downsample off-by-one. _45 min_
- [ ] **Per-layer-type anomaly z-score** — currently mixes embed/head/MLP in one pool, dominates anomalies (I7). _30 min_
- [ ] **Scrolling for long layer lists (I32/I33)** — ratatui::List + ListState. _45 min_

---

## 🌟 Phase 4 — Stretch (v1.0.0)

- [ ] **Web 3D viz** (Three.js, no CDN) — `--web` opens a local server with 3D tensor delta viewer
- [ ] **GGUF support** — many local models ship as `.gguf`, not safetensors
- [ ] **LoRA diff mode** — `neuraldiff lora base.safetensors lora_A.safetensors lora_B.safetensors`
- [ ] **Streaming for >10GB models** — process tensors in batches
- [ ] **PNG export** of heatmap (for slides / blog posts)
- [ ] **Theme support** — light mode, configurable palette
- [ ] **Config file** — `~/.config/neuraldiff/config.toml` for default scan paths, threshold, theme
- [ ] **Shell completions** — bash/zsh/fish via `clap_complete`

---

## Known Critical bugs not yet fixed

Track with one GitHub issue each (Phase 2 task above):

- **C1** loader.rs:90 — `start + data_len` overflow not checked
- **C6** tui.rs:308 — heatmap collapses 3D+ tensors meaninglessly (fixed by half-block?  verify)
- **C7** tui.rs:336 — downsample_grid bord off-by-one
- **C13** loader.rs:108 — i64/i32 → f32 silent precision loss
- **C14** diff.rs:97 — NaN propagates into max/mean/std
- **C15** mapper.rs:23 — f32 accumulator overflows on 70B+ models
- **C16** tui.rs render_l2_bar — saturating cast undocumented
- **C18** loader.rs:113 — Bool decoder accepts any nonzero byte
- **C19** tui.rs — selected_tensor not bounds-clamped after layer change

---

## Order of attack (recommended)

**Today (~3h):**
1. LICENSE file (2 min)
2. .gitignore (5 min)
3. README rewrite (30 min)
4. NaN detection (30 min)
5. C19 + threshold (25 min)
6. Multi-shard support (2h core, can defer to tomorrow)

**Then v0.2.0 push (~3h):** CI, cargo dist, crates.io, GitHub release.

**Then iterate:** Phase 3 hardening, Phase 4 stretch as time permits.
