# NeuralDiff - TODO v0.1.1

## Status: v0.1.0 RELEASED

---

## PRIORITE HAUTE (v0.1.1)

### TUI - Fonctionnalites Manquantes

- [x] **Heatmap** - draw_heatmap: grille downsampleée avec palette ░▒▓█ et couleurs GREEN→RED, chargement à la demande
- [x] **Inspect** - Affiche nom, shape, dtype, param count de tous les tenseurs
- [ ] **Filtre par type** - Ajouter filtre pour n'afficher que les couches de type specifique (attn/mlp/norm/embed)
- [x] **Barre de progression** - Loading screen animé (spinner braille + elapsed) via run_with_loading
- [ ] **Export CSV** - Ajouter touche `[C]` pour exporter en CSV en plus du JSON

### Corrections

- [x] **Warnings** - Supprimés : imports Cell/Row/Table retirés, variable diff inutilisée corrigée
- [ ] **Tests integration** - Tester end-to-end avec les fixtures LLaMA (tiny_model_a/b)
- [ ] **Scan recursif** - Le scanner ne scanne qu'un niveau de profondeur, faire recursif

---

## PRIORITE MOYENNE (v0.2.0)

### Web 3D (SPEC_VIZ.md)

- [ ] **Serveur web** - Feature flag `web` avec tokio + axum
- [ ] **Vue 3D** - Three.js embarque (pas de CDN) pour visualiser les tenseurs en 3D
- [ ] **Heatmap 3D** - Representation en volume des changements

### CLI Ameliore

- [ ] **Comparaison recursive** - `neuraldiff diff dir_a/ dir_b/` pour comparer des repertoires
- [ ] **Seuil configurable** - `--threshold 0.01` pour changer le seuil de detection
- [ ] **Diff partiel** - `--layers 0,1,2` pour ne comparer que certaines couches

---

## PRIORITE BASSE (v0.3.0)

### Performance

- [ ] **Streaming** - Traiter les tenseurs par batch pour les modeles > 10GB
- [ ] **Cache** - Cacher les donnees f32 deja converties
- [ ] **GPU** - Utiliser GPU pour le calcul des deltas (optional)

### UX

- [ ] **Themes** - Support pour theme clair/sombre automatique
- [ ] **Configuration** - Fichier de config `~/.config/neuraldiff/config.toml`
- [ ] **Shell completion** - Auto-completion pour bash/zsh/fish

---

## BUGS CONNUS (a verifier)

- [ ] **Terminal < 80 colonnes** - SPEC_TUI.md mentionne fallback, a implementer
- [ ] **Fichiers > 2GB sur Windows** - Verifier memmap2 avec grands fichiers

---

## ARCHITECTURE - Ameliorations Futures

- [ ] **Plugin system** - Permettre des custom metrics (MAE, KL-divergence)
- [ ] **Format support** - Support pickle (.pt, .pth) en plus de safetensors
- [ ] **Diff patch** - Generer un patch minimal entre deux modeles

---

## DATE CIBLE

- **v0.1.1** : Cette semaine
- **v0.2.0** : Dans 2 semaines (Web 3D)
- **v0.3.0** : Dans 1 mois (Performance + UX)

---

> Derniere mise a jour : 2026-05-01
