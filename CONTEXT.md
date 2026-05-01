# NeuralDiff — Session context

> Last updated: 2026-05-01
> Repo: **public** at https://github.com/Zertannax/neuraldiff
> Latest release: **v0.2.0**
> Current branch: `master`, in sync with `origin/master`

---

## TL;DR — pick up here

The tool **works end-to-end on real models** (Qwen3-0.6B Base vs Instruct verified, 596M params, 8s diff). The repo is **public on GitHub**, tagged **v0.2.0**, with a complete README, brand kit, CI, and roadmap. Next milestones tracked in `TODO.md`.

To continue work: `cd /mnt/c/Users/remic/neuraldiff` and look at the **"What's next"** section below.

---

## What got done (session 2026-05-01)

15 commits, 33 tests green, 7 Critical bugs fixed, full publication setup.

### Code fixes
- **C2/C3/C4** — typed `LayerComponent` enum replaces stringly-typed round-trip in `mapper.rs`. Distinct LayerDiff per norm component (was collapsing `input_layernorm` + `post_attention_layernorm`).
- **C5** — `Arc<ModelSnapshot>` cached in `AppState`; heatmap no longer re-mmaps multi-GB files on every Enter.
- **C8/C9** — selection state reset after sort/filter changes.
- **C10/C11/C12** — RAII `TerminalGuard` in `src/terminal.rs`; terminal always restored on error or panic.
- **C17** — UTF-8-safe `truncate_path` / `truncate_str` (no panic on accented paths).

### Features
- **Unified TUI flow**: `neuraldiff` (no args) → scan → pick → diff → detail in one terminal session. No flash, no re-entry.
- **WSL-aware scanner**: detects WSL via `/proc/version`, walks `/mnt/<drive>/Users/$USER/`, skips AppData/Windows/etc.
- **Loading-screen during scan**: no more black screen at startup.
- **Smooth gradient heatmap**: half-block (▀) chars, RGB ramp, doubled vertical resolution, stats inline (min/p50/mean/p95/max).
- **Sober TUI footer**: keys + labels, active filters highlighted with green chip.
- **Branded UI**: ◆ NEURALDIFF logo, ◐A / ◑B model markers (cyan / pink), aligned palette.
- **Extended CLI**: `summary`, `inspect --top N`, `inspect --json`, `scan --root <dir>`, `scan --json`, `diff --output <file>`, `diff --threshold` (accepted but pass-through).

### Publication
- LICENSE MIT in repo root
- README rewritten with hero, screenshots from real Qwen diff, full CLI reference
- Brand kit in `assets/` (logo, icon, hero, social-card, divider)
- TODO.md = phased roadmap (Phase 1 → v0.2.0, Phase 2 → release infra, Phase 3 → hardening, Phase 4 → stretch)
- .gitignore expanded
- GitHub Actions CI (Linux/macOS/Windows build + test, advisory clippy) — passing
- v0.2.0 git tag + GitHub release with notes
- Cargo.toml metadata polished (description, keywords, categories, exclude)
- Personal email scrubbed from history via `git filter-repo` → `Zertannax@users.noreply.github.com`
- Repo switched **private → public**

---

## Current architecture

```
src/
├── main.rs          — CLI dispatch, summary printer, csv writer, format helpers
├── cli.rs           — Clap definitions (Diff, Summary, Inspect, Scan + aliases d/i/s)
├── lib.rs           — module re-exports
├── types.rs         — ModelSnapshot, DiffResult, AppState, LayerType, LayerTypeFilter
├── loader.rs        — safetensors load via Arc<Mmap>, dtype decoding
├── diff.rs          — compute_diff (rayon-parallel), compute_summary
├── mapper.rs        — typed LayerComponent + LayerKey enums; map_layers + anomaly z-score
├── metrics.rs       — l2_norm, cosine_similarity
├── scanner.rs       — recursive .safetensors discovery, model-selection TUI
├── terminal.rs      — RAII TerminalGuard for raw mode + alt screen
├── tui.rs           — main TUI, run_unified, loading screen, scanning screen,
│                      heatmap, summary view, detail view, footer, help
└── web.rs           — stub for v1.0.0 (currently no-op behind `web` feature)
```

### Key decisions
- **`LayerType` schema is frozen** — public JSON keeps `embedding|attention|mlp|norm|head|other`. Fine-grained variants live in private `LayerComponent` (mapper.rs only).
- **`Arc<ModelSnapshot>` everywhere** — load once, share between diff worker and TUI heatmap.
- **One TerminalGuard per session** — `run_unified` owns it across scanning/loading/detail. No re-entry.
- **Tracing → stderr** — keeps `--json` stdout clean.
- **Half-block heatmap** — `▀` paints fg+bg per cell, doubling vertical resolution.

### Cargo features
- `default = ["tui"]` — enables ratatui + crossterm
- `web` — stub (tokio + axum, not implemented)

### Test fixtures
- `tests/fixtures/tiny_model_a.safetensors` — minimal LLaMA-style, 19 tensors
- `tests/fixtures/tiny_model_b.safetensors` — same shape, modified weights

---

## Local test setup (user's machine)

The user has these models on disk for end-to-end testing:

```
/mnt/c/Users/remic/Qwen3-0.6B/model.safetensors          (1.4 GB, instruct)
/mnt/c/Users/remic/Qwen3-0.6B-Base/model.safetensors     (1.1 GB, base)
~/.cache/huggingface/hub/models--google--gemma-4-31b-it/ (58 GB, multi-shard — needs multi-shard support to diff)
```

Qwen3 Base vs Instruct is the canonical happy-path demo: 596M params, 114 layers, 93.9% changed, `embed_tokens` flagged as anomaly (z=8.09).

Convenience script (gitignored): `./run-qwen.sh` runs the diff on those two paths.

---

## What's next — pick up here

Roadmap is in `TODO.md`. Phase order:

### Phase 1 — finish v0.2.x (a few hours)
- **Multi-shard support** — read `model.safetensors.index.json`, load all shards under one logical `ModelSnapshot`. Unblocks Llama 70B / Gemma 31B. **2-4h, biggest impact.**
- **NaN / Inf detection (C14)** — currently propagates silently into JSON/CSV. Add `summary.nan_count`. **30 min.**
- **C19** — `selected_tensor` bounds clamp after layer change. **10 min.**
- **`--threshold` actually filters** — wired in CLI but pass-through. **15 min.**
- **clippy zero-warnings** — currently 22 advisory warnings, mostly let-chain candidates. **1h.**

### Phase 2 — release infrastructure
- **`cargo dist`** — auto-build Linux/macOS/Windows binaries on each tag, attach to GitHub release. **45 min.**
- **More Releases** — v0.2.1, v0.2.2 as fixes land.

### Phase 3 — hardening
- Per-dtype tests (F16, BF16, I64, etc.)
- F64 accumulators in metric reductions for huge models
- Heatmap math (C6/C7) — bail on shape > 2D, fix downsample off-by-one
- Per-LayerType anomaly z-score pools (currently mixes embed/head/mlp)
- Scrolling for long layer / tensor lists

### Phase 4 — stretch (v1.0.0)
- Web 3D viz (Three.js, no CDN)
- GGUF support
- LoRA diff mode
- Streaming for >10GB models
- PNG export of heatmaps

---

## Known issues (not yet fixed)

Tracked Critical bugs from the original audit, not yet addressed:

| Bug | File | Symptom |
|-----|------|---------|
| C1  | loader.rs:90 | `start + data_len` overflow not checked |
| C6  | tui.rs ~308 | 3D+ tensors flattened arbitrarily for heatmap |
| C7  | tui.rs ~336 | downsample_grid edge off-by-one |
| C13 | loader.rs:108 | i64/i32 → f32 silent precision loss on token-id tensors |
| C14 | diff.rs:97 | NaN propagates into max/mean/std silently |
| C15 | mapper.rs:23 | f32 accumulator loses precision on 70B+ models |
| C16 | tui.rs render_l2_bar | saturating cast undocumented (acceptable but fragile) |
| C18 | loader.rs:113 | Bool decoder accepts any non-zero byte |
| C19 | tui.rs | selected_tensor not bounds-clamped after layer change |

Plus 38 Important + 14 Minor findings from the same audit, listed in the original review (not in this file — Phase 3 will surface them as GitHub issues for transparency).

---

## Quick commands cheat sheet

```bash
# Build
cd /mnt/c/Users/remic/neuraldiff
cargo build --release

# Run unified TUI (default)
./target/release/neuraldiff

# Direct diff on Qwen
./run-qwen.sh
# or
./target/release/neuraldiff diff /mnt/c/Users/remic/Qwen3-0.6B-Base/model.safetensors /mnt/c/Users/remic/Qwen3-0.6B/model.safetensors

# Text-only summary
./target/release/neuraldiff summary <a> <b> -n 10

# Tests + lint
cargo test --all-features
cargo clippy --all-targets --all-features

# GitHub via Windows gh CLI from WSL
gh.exe repo view Zertannax/neuraldiff
gh.exe run list --repo Zertannax/neuraldiff --limit 5
gh.exe release view v0.2.0 --repo Zertannax/neuraldiff

# Push (needs gh.exe token because git on WSL has no creds)
TOKEN=$(gh.exe auth token | tr -d '\r\n')
git push "https://x-access-token:${TOKEN}@github.com/Zertannax/neuraldiff.git" master
```

---

## Author identity

Commits are now signed as `Zertannax <Zertannax@users.noreply.github.com>` (personal email scrubbed via `git filter-repo`). Older author entries `NeuralDiff <neuraldiff@example.com>` and `Remic <remic@neuraldiff.dev>` remain in the rewritten history — both are non-personal placeholders.

When committing during a Claude session, the trailer `Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>` is added per the user's preference.
