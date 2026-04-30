# NeuralDiff — 3D Visualization Spec

> Spec for the Three.js web view. Reference when building `web/index.html` and `src/web.rs`.

---

## Concept

A model is a **tower of layers**. Each layer is a disc. The color and size of each disc encodes how much that layer changed.

You orbit around the tower. You click a disc. It explodes into its component tensors.

Simple. Dense. Useful.

---

## Visual layout

```
                    ↑
              [lm_head]         ← small disc, gray (unchanged)
        [layers.35.norm]        ← tiny disc, dim
       [layers.35.self_attn]    ← medium disc, yellow
          [layers.35.mlp]       ← medium disc, orange
              ...
       [layers.22.self_attn]    ← large disc, yellow
          [layers.22.mlp]       ← LARGE disc, RED ← highest delta
              ...
          [layers.0.mlp]        ← small disc, green
       [layers.0.self_attn]     ← small disc, green
        [embed_tokens]          ← large disc, gray (unchanged)
                    ↓
```

---

## Three.js scene setup

```javascript
// Scene
const scene = new THREE.Scene();
scene.background = new THREE.Color(0x060a0f);  // Vektor void color — fits the aesthetic

// Camera
const camera = new THREE.PerspectiveCamera(60, width/height, 0.1, 1000);
camera.position.set(0, 0, 80);

// Lights
const ambient = new THREE.AmbientLight(0xffffff, 0.3);
const point   = new THREE.PointLight(0x3b82f6, 2, 200);  // electric blue accent
point.position.set(0, 50, 50);

// Controls
const controls = new OrbitControls(camera, renderer.domElement);
controls.enableDamping = true;
controls.dampingFactor = 0.05;
```

---

## Disc geometry per layer

```javascript
function createLayerDisc(layer) {
    const radius    = 5 + (layer.param_count / max_params) * 10;  // 5–15 units
    const thickness = 0.8;
    const segments  = 64;

    const geo  = new THREE.CylinderGeometry(radius, radius, thickness, segments);
    const mat  = new THREE.MeshPhongMaterial({
        color:     deltaToColor(layer.aggregate_l2),
        emissive:  deltaToEmissive(layer.aggregate_l2),
        shininess: 80,
        transparent: true,
        opacity:   0.85,
    });

    const mesh = new THREE.Mesh(geo, mat);
    mesh.position.y = layer.layer_index * 2.5;  // stack vertically, 2.5 units apart
    mesh.userData = { layer };

    return mesh;
}
```

---

## Color mapping

```javascript
function deltaToColor(l2) {
    // 0.0 → green, 0.5 → yellow, 1.0 → red
    if (l2 < 0.001) return new THREE.Color(0x1e293b);  // near-black for unchanged
    if (l2 < 0.3)   return new THREE.Color(0x22c55e);  // green
    if (l2 < 0.6)   return new THREE.Color(0xeab308);  // yellow
    return               new THREE.Color(0xef4444);    // red
}

function deltaToEmissive(l2) {
    const intensity = Math.min(l2, 1.0) * 0.3;
    return deltaToColor(l2).multiplyScalar(intensity);
}
```

---

## Interaction: click to explode

When user clicks a disc:
1. Animate disc expanding (scale tween)
2. Spawn smaller discs around it representing individual tensors
3. Each tensor disc: size = param count, color = tensor L2 delta
4. Click anywhere else: collapse back

```javascript
raycaster.setFromCamera(mouse, camera);
const hits = raycaster.intersectObjects(layerDiscs);
if (hits.length > 0) {
    const layer = hits[0].object.userData.layer;
    explodeLayer(layer);
}
```

---

## Hover tooltip

On hover over any disc:
```
┌──────────────────────┐
│ layers.22.mlp        │
│ Δ L2  = 0.847        │
│ Params = 134M        │
│ Tensors: 3           │
│                      │
│ Click to expand      │
└──────────────────────┘
```

Implemented as a floating `<div>` over the canvas, positioned via `projectVector`.

---

## Timeline mode (`neuraldiff timeline *.safetensors`)

When multiple checkpoints are passed, the web view shows a **timeline slider**.

- Scrub through checkpoints
- Discs animate color changes between steps
- Useful for watching training progress

Data format: array of `DiffResult` JSON objects.

```javascript
let checkpointIndex = 0;
slider.addEventListener('input', () => {
    checkpointIndex = slider.value;
    updateScene(diffs[checkpointIndex]);
});

function updateScene(diff) {
    diff.layers.forEach((layer, i) => {
        const disc = layerDiscs[i];
        // Tween color from current to new
        gsap.to(disc.material.color, { ...deltaToColorRGB(layer.aggregate_l2), duration: 0.3 });
    });
}
```

---

## Data contract (Rust → JS)

The Rust server serves this JSON at `/api/diff`:

```json
{
  "model_a": "base.safetensors",
  "model_b": "finetuned.safetensors",
  "total_params": 3090000000,
  "layers": [
    {
      "layer_index": 0,
      "layer_name": "embed_tokens",
      "layer_type": "Embedding",
      "aggregate_l2": 0.000,
      "anomaly_score": 0.1,
      "param_count": 131072000,
      "tensors": [
        {
          "name": "model.embed_tokens.weight",
          "shape": [32000, 4096],
          "l2_distance": 0.000,
          "cosine_similarity": 1.000,
          "max_delta": 0.000001,
          "changed": false
        }
      ]
    }
  ],
  "summary": {
    "total_layers": 36,
    "changed_layers": 34,
    "top_changed": [22, 31, 18],
    "anomalies": [22]
  }
}
```

---

## HTML file structure (embedded in binary)

```html
<!DOCTYPE html>
<html>
<head>
  <style>/* minimal, dark, monospace UI */</style>
</head>
<body>
  <div id="ui-overlay">
    <div id="tooltip"></div>
    <div id="legend">
      <span class="unchanged">■ unchanged</span>
      <span class="low">■ low Δ</span>
      <span class="mid">■ medium Δ</span>
      <span class="high">■ high Δ</span>
    </div>
    <div id="layer-info"></div>
  </div>

  <canvas id="canvas"></canvas>

  <script src="https://cdnjs.cloudflare.com/ajax/libs/three.js/r128/three.min.js"></script>
  <script>/* all app code inline */</script>
</body>
</html>
```

The entire HTML file is embedded in the Rust binary at compile time:
```rust
const WEB_HTML: &str = include_str!("../web/index.html");
```

Served by axum at `GET /`.
Diff data served at `GET /api/diff`.

---

## Performance

| Layers | Disc count | Target FPS |
|--------|-----------|-----------|
| 36 | 36 | 60 fps |
| 80 | 240 (exploded) | 30 fps |
| 200 | 200 | 30 fps |

Use `InstancedMesh` if layer count > 100.
Dispose geometry/material on collapse to free GPU memory.
