use neuraldiff::loader::{load, load_tensor_data};

#[test]
fn test_load_fixture_a() {
    let snapshot = load("tests/fixtures/tiny_model_a.safetensors".as_ref())
        .expect("Failed to load fixture A");
    
    assert_eq!(snapshot.tensors.len(), 19);
    assert!(snapshot.total_params > 0);
    
    assert!(snapshot.tensors.contains_key("model.embed_tokens.weight"));
    assert!(snapshot.tensors.contains_key("model.layers.0.self_attn.q_proj.weight"));
    assert!(snapshot.tensors.contains_key("model.layers.0.mlp.gate_proj.weight"));
    assert!(snapshot.tensors.contains_key("lm_head.weight"));
}

#[test]
fn test_load_tensor_data_f32() {
    let snapshot = load("tests/fixtures/tiny_model_a.safetensors".as_ref())
        .expect("Failed to load fixture A");
    
    let data = load_tensor_data(&snapshot, "model.embed_tokens.weight")
        .expect("Failed to load tensor data");
    
    assert_eq!(data.len(), 6400);
}

#[test]
fn test_load_tensor_data_consistency() {
    let snapshot_a = load("tests/fixtures/tiny_model_a.safetensors".as_ref())
        .expect("Failed to load fixture A");
    let snapshot_b = load("tests/fixtures/tiny_model_b.safetensors".as_ref())
        .expect("Failed to load fixture B");
    
    let data_a = load_tensor_data(&snapshot_a, "model.layers.0.mlp.gate_proj.weight")
        .expect("Failed to load tensor A");
    let data_b = load_tensor_data(&snapshot_b, "model.layers.0.mlp.gate_proj.weight")
        .expect("Failed to load tensor B");
    
    assert_eq!(data_a.len(), data_b.len());
    
    let mut different = false;
    for (a, b) in data_a.iter().zip(data_b.iter()) {
        if (a - b).abs() > 1e-6 {
            different = true;
            break;
        }
    }
    assert!(different, "Expected model B to have different weights");
}
