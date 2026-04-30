#!/usr/bin/env python3
\"\"\"
Generate minimal .safetensors fixtures for testing NeuralDiff.
Creates two tiny models with LLaMA-style naming for integration tests.
\"\"\"

import numpy as np
from safetensors.numpy import save_file
import os

# Create tiny weights (small shapes for fast tests)
def create_tiny_model_a():
    tensors = {
        \"model.embed_tokens.weight\": np.random.randn(100, 64).astype(np.float32) * 0.01,
        \"model.layers.0.self_attn.q_proj.weight\": np.random.randn(64, 64).astype(np.float32) * 0.01,
        \"model.layers.0.self_attn.k_proj.weight\": np.random.randn(64, 64).astype(np.float32) * 0.01,
        \"model.layers.0.self_attn.v_proj.weight\": np.random.randn(64, 64).astype(np.float32) * 0.01,
        \"model.layers.0.self_attn.o_proj.weight\": np.random.randn(64, 64).astype(np.float32) * 0.01,
        \"model.layers.0.mlp.gate_proj.weight\": np.random.randn(64, 128).astype(np.float32) * 0.01,
        \"model.layers.0.mlp.up_proj.weight\": np.random.randn(64, 128).astype(np.float32) * 0.01,
        \"model.layers.0.mlp.down_proj.weight\": np.random.randn(128, 64).astype(np.float32) * 0.01,
        \"model.layers.0.input_layernorm.weight\": np.ones(64, dtype=np.float32),
        \"model.layers.1.self_attn.q_proj.weight\": np.random.randn(64, 64).astype(np.float32) * 0.01,
        \"model.layers.1.self_attn.k_proj.weight\": np.random.randn(64, 64).astype(np.float32) * 0.01,
        \"model.layers.1.self_attn.v_proj.weight\": np.random.randn(64, 64).astype(np.float32) * 0.01,
        \"model.layers.1.self_attn.o_proj.weight\": np.random.randn(64, 64).astype(np.float32) * 0.01,
        \"model.layers.1.mlp.gate_proj.weight\": np.random.randn(64, 128).astype(np.float32) * 0.01,
        \"model.layers.1.mlp.up_proj.weight\": np.random.randn(64, 128).astype(np.float32) * 0.01,
        \"model.layers.1.mlp.down_proj.weight\": np.random.randn(128, 64).astype(np.float32) * 0.01,
        \"model.layers.1.input_layernorm.weight\": np.ones(64, dtype=np.float32),
        \"model.norm.weight\": np.ones(64, dtype=np.float32),
        \"lm_head.weight\": np.random.randn(100, 64).astype(np.float32) * 0.01,
    }
    return tensors

def create_tiny_model_b():
    \"\"\"Same structure as A but with some weights modified to simulate fine-tuning.\"\"\"
    np.random.seed(42)
    tensors = create_tiny_model_a()
    
    # Modify some layers to simulate fine-tuning
    # Layer 0 MLP: large change
    tensors[\"model.layers.0.mlp.gate_proj.weight\"] += np.random.randn(64, 128).astype(np.float32) * 0.5
    tensors[\"model.layers.0.mlp.up_proj.weight\"] += np.random.randn(64, 128).astype(np.float32) * 0.3
    tensors[\"model.layers.0.mlp.down_proj.weight\"] += np.random.randn(128, 64).astype(np.float32) * 0.4
    
    # Layer 1 attention: moderate change
    tensors[\"model.layers.1.self_attn.q_proj.weight\"] += np.random.randn(64, 64).astype(np.float32) * 0.2
    tensors[\"model.layers.1.self_attn.k_proj.weight\"] += np.random.randn(64, 64).astype(np.float32) * 0.15
    
    # Embedding and lm_head: no change (keep as-is)
    # Norm layers: tiny change
    tensors[\"model.layers.0.input_layernorm.weight\"] += np.random.randn(64).astype(np.float32) * 0.001
    
    return tensors

def main():
    fixtures_dir = os.path.join(os.path.dirname(__file__), \"..\", \"tests\", \"fixtures\")
    os.makedirs(fixtures_dir, exist_ok=True)
    
    # Model A (base)
    model_a = create_tiny_model_a()
    save_file(model_a, os.path.join(fixtures_dir, \"tiny_model_a.safetensors\"))
    print(f\"Created tiny_model_a.safetensors ({len(model_a)} tensors)\")
    
    # Model B (fine-tuned)
    model_b = create_tiny_model_b()
    save_file(model_b, os.path.join(fixtures_dir, \"tiny_model_b.safetensors\"))
    print(f\"Created tiny_model_b.safetensors ({len(model_b)} tensors)\")
    
    # Print info
    for name, path in [(\"A\", \"tiny_model_a.safetensors\"), (\"B\", \"tiny_model_b.safetensors\")]:
        full_path = os.path.join(fixtures_dir, path)
        size = os.path.getsize(full_path)
        print(f\"  Model {name}: {size / 1024:.1f} KB\")

if __name__ == \"__main__\":
    main()
