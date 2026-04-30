#!/usr/bin/env python3
"""
Regenerate test models properly:
- Model A: base model with fixed seed
- Model B: copy of A with selective modifications
"""

import numpy as np
from safetensors.numpy import save_file
import os

def create_tiny_model():
    """Create base model with fixed seed for reproducibility."""
    np.random.seed(12345)  # Fixed seed!
    
    # LLaMA-style naming (same as fixtures)
    tensors = {
        "model.embed_tokens.weight": np.random.randn(100, 64).astype(np.float32) * 0.01,
        "model.layers.0.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "model.layers.0.self_attn.k_proj.weight": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "model.layers.0.self_attn.v_proj.weight": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "model.layers.0.self_attn.o_proj.weight": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "model.layers.0.mlp.gate_proj.weight": np.random.randn(64, 128).astype(np.float32) * 0.01,
        "model.layers.0.mlp.up_proj.weight": np.random.randn(64, 128).astype(np.float32) * 0.01,
        "model.layers.0.mlp.down_proj.weight": np.random.randn(128, 64).astype(np.float32) * 0.01,
        "model.layers.0.input_layernorm.weight": np.ones(64, dtype=np.float32),
        "model.layers.1.self_attn.q_proj.weight": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "model.layers.1.self_attn.k_proj.weight": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "model.layers.1.self_attn.v_proj.weight": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "model.layers.1.self_attn.o_proj.weight": np.random.randn(64, 64).astype(np.float32) * 0.01,
        "model.layers.1.mlp.gate_proj.weight": np.random.randn(64, 128).astype(np.float32) * 0.01,
        "model.layers.1.mlp.up_proj.weight": np.random.randn(64, 128).astype(np.float32) * 0.01,
        "model.layers.1.mlp.down_proj.weight": np.random.randn(128, 64).astype(np.float32) * 0.01,
        "model.layers.1.input_layernorm.weight": np.ones(64, dtype=np.float32),
        "model.norm.weight": np.ones(64, dtype=np.float32),
        "lm_head.weight": np.random.randn(100, 64).astype(np.float32) * 0.01,
    }
    return tensors

def create_model_b_from_a(tensors_a):
    """Create model B by selectively modifying model A."""
    # Deep copy all tensors
    tensors_b = {k: v.copy() for k, v in tensors_a.items()}
    
    np.random.seed(99999)  # Different seed for modifications
    
    # Modify only specific layers (simulating fine-tuning on MLP)
    # Layer 0 MLP: large change
    tensors_b["model.layers.0.mlp.gate_proj.weight"] += np.random.randn(64, 128).astype(np.float32) * 0.5
    tensors_b["model.layers.0.mlp.up_proj.weight"] += np.random.randn(64, 128).astype(np.float32) * 0.3
    tensors_b["model.layers.0.mlp.down_proj.weight"] += np.random.randn(128, 64).astype(np.float32) * 0.4
    
    # Layer 1 attention: moderate change
    tensors_b["model.layers.1.self_attn.q_proj.weight"] += np.random.randn(64, 64).astype(np.float32) * 0.2
    tensors_b["model.layers.1.self_attn.k_proj.weight"] += np.random.randn(64, 64).astype(np.float32) * 0.15
    
    # Embedding and lm_head: NO CHANGE (identical to model A)
    # Norm layers: tiny change
    tensors_b["model.layers.0.input_layernorm.weight"] += np.random.randn(64).astype(np.float32) * 0.001
    
    return tensors_b

def main():
    models_dir = os.path.join(os.path.dirname(__file__), "..", "models")
    os.makedirs(models_dir, exist_ok=True)
    
    # Model A (base)
    model_a = create_tiny_model()
    save_file(model_a, os.path.join(models_dir, "model_a.safetensors"))
    print(f"Created model_a.safetensors ({len(model_a)} tensors)")
    
    # Model B (fine-tuned from A)
    model_b = create_model_b_from_a(model_a)
    save_file(model_b, os.path.join(models_dir, "model_b.safetensors"))
    print(f"Created model_b.safetensors ({len(model_b)} tensors)")
    
    # Verify: show which tensors are identical vs changed
    print("\nVerification - Tensors that should be IDENTICAL:")
    identical = []
    changed = []
    for name in sorted(model_a.keys()):
        is_same = np.allclose(model_a[name], model_b[name], atol=1e-7)
        if is_same:
            identical.append(name)
            print(f"  ✓ {name}")
        else:
            changed.append(name)
            l2 = np.linalg.norm(model_b[name] - model_a[name])
            print(f"  ✗ {name} (L2={l2:.4f})")
    
    print(f"\nSummary: {len(identical)} identical, {len(changed)} changed")

if __name__ == "__main__":
    main()
