# NeuralDiff v0.1.0 - Guide d'utilisation

## Compilation

`ash
cd C:\Users\remic\neuraldiff

# Build debug (rapide, pour le dev)
cargo build

# Build release (optimisé, pour les tests)
cargo build --release

# Tests
cargo test
`

## Commandes CLI

### Diff de deux modèles (mode TUI)
`ash
neuraldiff diff model_a.safetensors model_b.safetensors
`
? Lance l'interface terminal interactive

### Export JSON
`ash
neuraldiff diff model_a.safetensors model_b.safetensors --json > diff.json
`
? Sort le résultat au format JSON (stdout)

### Inspect (v0.2.0)
`ash
neuraldiff inspect model.safetensors
`
? Non implémenté pour l'instant

## Navigation dans le TUI

### Écran de résumé (démarrage)
`
+- NEURALDIFF v0.1.0 -------------------------------------------------------+
¦ base.safetensors ? finetuned.safetensors    3.09B params                  ¦
+--------------+  +--------------+  +--------------+  +--------------+     ¦
¦  3.09B       ¦  ¦     36       ¦  ¦     34       ¦  ¦      2       ¦     ¦
¦  Parameters  ¦  ¦   Layers     ¦  ¦   Changed    ¦  ¦  Unchanged   ¦     ¦
+--------------+  +--------------+  +--------------+  +--------------+     ¦
¦                                                                           ¦
¦  #1 layers.22.mlp  [¦¦¦¦...]  0.847 [CRIT]                                ¦
¦  #2 layers.31.attn [¦¦¦¦...]  0.723 [HIGH]                                ¦
¦  ...                                                                      ¦
¦  [R] Anomalies: layers.22.mlp (z-score: 3.24)                             ¦
¦                                                                           ¦
¦  Press ? or Enter to explore layers                                       ¦
+---------------------------------------------------------------------------+
`

### Touches

| Touche | Action |
|--------|--------|
| **? / j** | Descendre dans la liste / Passer en mode détail |
| **? / k** | Monter dans la liste |
| **? / l** | Tensor suivant (dans le détail) |
| **? / h** | Tensor précédent (dans le détail) |
| **Enter** | Basculer le heatmap / Entrer en mode détail |
| **s** | Changer le tri : L2? ? Index ? Anomalie |
| **f** | Filtrer : Tout ? Changés uniquement |
| **J** (majuscule) | Exporter JSON vers diff.json |
| **?** | Afficher l'aide |
| **q** / **Ctrl+C** | Quitter |

### Écran de détail (après ? ou Enter)
`
+- Layers [L2?] --------- Detail: layers.22.mlp ---------------------------+
¦ #0  attn  [¦¦¦] 0.534¦ Layer: layers.22.mlp                               ¦
¦ #1  mlp   [¦¦¦¦]0.847¦ Type:  mlp                                         ¦
¦ #2  norm  [¦]  0.012 ¦ L2:    0.847                                       ¦
¦ ...                  ¦ Params: 134M                                       ¦
¦                      ¦                                                    ¦
¦                      ¦ Tensors:                                           ¦
¦                      ¦ > gate_proj.weight  [4096,11008] L2=0.912          ¦
¦                      ¦   up_proj.weight    [4096,11008] L2=0.834          ¦
¦                      ¦   down_proj.weight  [11008,4096] L2=0.623          ¦
+---------------------------------------------------------------------------+
¦ [??/jk] navigate  [Enter] heatmap  [s] sort  [f] filter  [J] JSON  [q] quit¦
+----------------------------------------------------------------------------+
`

## Schéma JSON

`json
{
  "model_a": "base.safetensors",
  "model_b": "finetuned.safetensors",
  "total_params": 3090000000,
  "layers": [
    {
      "layer_index": 22,
      "layer_name": "layers.22.mlp",
      "layer_type": "MLP",
      "aggregate_l2": 0.847,
      "anomaly_score": 3.24,
      "param_count": 134000000,
      "tensors": [
        {
          "name": "model.layers.22.mlp.gate_proj.weight",
          "shape": [4096, 11008],
          "l2_distance": 0.912,
          "cosine_similarity": 0.834,
          "max_delta": 0.0152,
          "mean_delta": 0.0008,
          "changed": true
        }
      ]
    }
  ],
  "summary": {
    "total_layers": 36,
    "changed_layers": 34,
    "unchanged_layers": 2,
    "change_ratio_percent": 94.4,
    "mean_delta": 0.342,
    "max_delta": 0.847,
    "top_changed_indices": [22, 31, 18, 5, 11],
    "anomalies": [
      {
        "layer_index": 22,
        "layer_name": "layers.22.mlp",
        "z_score": 3.24,
        "reason": "Unusually high L2 distance compared to other layers"
      }
    ]
  }
}
`

## Architecture supportée

### Formats de nommage
- **LLaMA / Qwen / Mistral** : model.layers.N.COMPONENT
- **GPT-2** : 	ransformer.h.N.COMPONENT
- **Falcon** : 	ransformer.h.N.COMPONENT
- **Fallback** : Regroupement par préfixe commun

### Types de couches détectés
- embed : Embedding
- ttn : Attention
- mlp : MLP (Multi-Layer Perceptron)
- 
orm : Normalisation
- head : LM Head

## Dtypes supportés

Tous les dtypes sont convertis en **f32** au chargement :
- F32 (direct)
- F16, BF16 (via half crate)
- I64, I32, I16, I8 (cast)
- U8 (cast)
- Bool (0.0 ou 1.0)

## Performance

| Taille | Temps de chargement | Temps de diff |
|--------|-------------------|--------------|
| 3B params | ~8s | ~12s |
| 7B params | ~20s | ~30s |
| 13B params | ~40s | ~60s |

*Tests sur hardware consommateur, avec memory-map et parallélisation rayon*

## Troubleshooting

### "Terminal too narrow"
? Le terminal doit faire au moins **80 colonnes × 24 lignes**

### "Application blocked by Device Guard" (Windows)
? Ajouter une exception dans la politique WDAC ou exécuter sur une machine sans restriction

### Models have different architectures
? Les tensors non-correspondants sont **skippés** (pas d'erreur bloquante)
Un warning s'affiche dans la console

## Développement

`ash
# Mode watch (recompile auto)
cargo watch -x test

# Clippy (linting)
cargo clippy --all-targets --all-features

# Format
cargo fmt --check
`

## Prochaines versions

- **v0.2.0** : Vue 3D interactive (Three.js embarqué), mode inspect, timeline
- **v0.3.0** : Support GGUF, diff LoRA, merge preview
- **v1.0.0** : Multi-model comparison, benchmarks
