# NeuralDiff — Contexte de développement

> Ce fichier résume l'état du projet, les décisions prises, et ce qui reste à faire.
> Mis à jour le 2026-05-01.

---

## Statut actuel

**v0.1.1** — feature-complete, compilable, testable, pushé sur `master`.

Tous les items v0.1.1 sont cochés dans `TODO.md`.
Commit actuel : `74cefcb`.
La prochaine étape est **v0.2.0**.

---

## Ce qui a été fait (sessions précédentes)

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

## Ce qui a été fait (session du 2026-05-01)

### Corrections de compilation (commit `74cefcb`)

Le code récupéré depuis GitHub ne compilait pas. 3 erreurs corrigées :

| Fichier | Erreur | Correction |
|---|---|---|
| `src/loader.rs:99` | `cannot infer type of type parameter B` | Ajout explicite `: Vec<f32>` sur le match dtype |
| `src/tui.rs:66-67` | `borrow of moved value: path_a/path_b` | Clonage des `PathBuf` avant passage au thread |
| `Cargo.toml:50` | `EnvFilter` not found | Ajout feature `env-filter` sur `tracing-subscriber` |

**Tests** : 20/20 passent (4 scanner + 2 diff + 3 loader + 2 mapper + 9 metrics).

### Commande `scan` ajoutée

Nouvelle sous-commande CLI :
```bash
neuraldiff scan
```
Affiche la liste de tous les modèles `.safetensors` trouvés sur le système.

### Améliorations du scanner

- Scan du répertoire **home** lui-même (depth 2) — trouve les repos git clone directement dans home
- Ajout de chemins courants : `Desktop`, `checkpoints`, `weights`, `huggingface`, `transformers`
- Augmentation des profondeurs max (4→5 pour la plupart)

### Correction sensibilité TUI

Problème : les flèches du clavier déclenchaient plusieurs événements (Press + Repeat + Release), causant des sauts de sélection.

Solution : filtrage des événements `KeyEventKind` dans `handle_selection_key()` — seul `Press` est traité.

### Tests scanner ajoutés

4 tests unitaires dans `scanner.rs` :
- `test_scan_recursive` — trouve un modèle à 2 niveaux de profondeur
- `test_scan_respects_max_depth` — respecte la limite de profondeur
- `test_scan_skips_hidden_dirs` — ignore les dossiers cachés (sauf `.cache`)
- `test_scan_allows_cache_dir` — autorise explicitement `.cache/huggingface`

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

### Tests

- `tests/diff_tests.rs` : 2 tests end-to-end sur les fixtures LLaMA ✅
- `tests/loader_tests.rs` : 3 tests (load fixture, tensor data f32, consistency) ✅
- `tests/mapper_tests.rs` : 2 tests (LLaMA grouping, anomaly detection) — manque GPT-2/Falcon
- `tests/metrics_tests.rs` : 9 tests (L2 norm, cosine similarity edge cases) ✅
- `tests/scanner_tests.rs` : 4 tests unitaires dans `scanner.rs` ✅

### Limitations connues

- **Format support** : Seuls les fichiers `.safetensors` sont supportés. Les modèles `.gguf` (LM Studio, Ollama) et `.pt`/`.pth` (PyTorch) ne sont pas lisibles.
- **Fichiers incomplets** : Un fichier `.safetensors` incomplet (téléchargement interrompu) provoque `MetadataIncompleteBuffer`. Solution : re-télécharger le modèle.
- **Windows build** : Compilation debug peut échouer par manque d'espace disque ou erreurs PDB. Utiliser `cargo test --jobs 1` ou `--release`.

---

## Fixtures de test

```
tests/fixtures/tiny_model_a.safetensors   — architecture LLaMA miniature (19 tenseurs)
tests/fixtures/tiny_model_b.safetensors   — même architecture, poids modifiés
models/model_a.safetensors                — modèle de démo
models/model_b.safetensors                — modèle de démo
```
