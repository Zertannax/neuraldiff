# NeuralDiff — Contexte de développement

> Ce fichier résume l'état du projet, les décisions prises, et ce qui reste à faire.
> Mis à jour le 2026-05-01.

---

## Statut actuel

**v0.1.1** — feature-complete, pushé sur `master`.

Tous les items v0.1.1 sont cochés dans `TODO.md`.
La prochaine étape est **v0.2.0**.

---

## Ce qui a été fait (cette session)

### Review complète du code (commit `cc5b7ab`)

Problèmes identifiés et corrigés :

| Problème | Fix |
|---|---|
| `load_tensor_data` rouvrait le fichier pour chaque tenseur | `Arc<Mmap>` stocké dans `ModelSnapshot`, accès par offset |
| `regex_captures` — fausse regex (match sur le string pattern) | Vraies `Regex` compilées via `OnceLock` (crate `regex = "1"`) |
| `.unwrap()` dans le hot path de `compute_tensor_diff` | `.with_context()` avec messages explicites |
| `missing_tensors` calculés puis silencieusement ignorés | Exposés dans `DiffSummary.missing_tensors` + JSON export |
| Scanner sautait les modèles HuggingFace (trop superficiel) | Scan récursif avec depth configurable (`.cache/huggingface` → depth 6) |
| `HashMap<PathBuf, bool>` pour dédupliquer | `HashSet<PathBuf>` |
| `eprintln!` au lieu de tracing | `tracing::warn!` + subscriber initialisé dans `main` |
| Aucun test pour `metrics.rs` | 9 tests ajoutés dans `tests/metrics_tests.rs` |

### v0.1.1 features (commits `cd6c32a`, `99caae7`)

- **Loading screen** : `run_with_loading()` — spinner braille animé pendant `compute_diff` en thread background
- **Heatmap** : `[Enter]` en Detail view charge les deltas réels et affiche une grille ░▒▓█ GREEN→RED
- **Inspect** : `neuraldiff inspect model.safetensors` — tableau complet (nom, shape, dtype, params)
- **Filtre type** : `[t]` cycle `All → Attn → MLP → Norm → Embed → Head → Other`
- **Export CSV** : `[C]` écrit `diff.csv` avec tous les tenseurs et leurs métriques
- **Warnings** : imports inutilisés retirés, variables inutilisées corrigées

---

## Architecture

```
src/
├── main.rs        — CLI entry, inspect command, run_with_loading dispatch
├── cli.rs         — Clap: Diff { model_a, model_b, json }, Inspect { model }
├── types.rs       — ModelSnapshot (Arc<Mmap>), DiffResult, AppState, HeatmapData, LayerTypeFilter
├── loader.rs      — load() → ModelSnapshot, load_tensor_data() via stored mmap offset
├── diff.rs        — compute_diff() parallelisé rayon, compute_summary()
├── mapper.rs      — map_layers() avec regex LLaMA/GPT-2/Falcon, anomaly z-score
├── metrics.rs     — l2_norm(), cosine_similarity()
├── scanner.rs     — scan récursif avec depth limits, TUI de sélection de modèles
└── tui.rs         — run_with_loading(), draw_ui(), draw_heatmap(), export_csv()
```

### Features Cargo

- `default = ["tui"]` — compile avec ratatui + crossterm
- `web` — ajoute tokio + axum (stub, v0.2.0)

### Dépendances clés

- `safetensors = "0.4"` + `memmap2 = "0.9"` — lecture zero-copy
- `rayon = "1.10"` — parallélisme tenseurs
- `regex = "1.11"` — matching architecture modèles
- `half = "2.4"` — conversion F16/BF16 → F32
- `ratatui = "0.29"` + `crossterm = "0.28"` — TUI

---

## Décisions de design

- **`model_a/model_b` sont des `String`** (pas `Option<String>`) dans `DiffResult` — ils sont toujours présents
- **`Arc<Mmap>` dans `ModelSnapshot`** — évite de rouvrir le fichier pour chaque tenseur lors du diff
- **Heatmap chargée à la demande** — pas stockée dans `DiffResult` pour économiser la mémoire
- **Scanner récursif max depth 5** (6 pour `.cache/huggingface`) — HuggingFace hub path est profond
- **`compute_diff` dans un thread séparé** — la TUI reste réactive pendant le calcul

---

## Prochaine étape : v0.2.0

### Priorité recommandée

1. **Comparaison de répertoires** (`neuraldiff diff dir_a/ dir_b/`) — forte valeur pratique
2. **Seuil configurable** (`--threshold 0.01`) — simple à implémenter
3. **Web 3D** (Three.js sans CDN) — ambitieux, voir `SPEC_VIZ.md`
4. **Diff partiel** (`--layers 0,1,2`) — utile pour les gros modèles

### Tests manquants

- `tests/diff_tests.rs` : tests end-to-end sur les fixtures LLaMA (`tiny_model_a/b`)
- `tests/mapper_tests.rs` : couvre LLaMA et anomalies, pas GPT-2/Falcon
- `tests/scanner_tests.rs` : aucun test

---

## Fixtures de test

```
tests/fixtures/tiny_model_a.safetensors   — architecture LLaMA miniature (19 tenseurs)
tests/fixtures/tiny_model_b.safetensors   — même architecture, poids modifiés
models/model_a.safetensors                — modèle de démo
models/model_b.safetensors                — modèle de démo
```
