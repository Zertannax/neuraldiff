use neuraldiff::mapper::map_layers;
use neuraldiff::types::{LayerType, TensorDiff};

fn create_test_tensor(name: &str, l2: f32) -> TensorDiff {
    TensorDiff {
        name: name.to_string(),
        shape: vec![64, 64],
        l2_distance: l2,
        cosine_similarity: 0.95,
        max_delta: l2 * 2.0,
        mean_delta: l2 * 0.5,
        std_delta: l2 * 0.3,
        changed: l2 > 1e-6,
    }
}

#[test]
fn test_map_layers_llama_style() {
    let tensors = vec![
        create_test_tensor("model.layers.0.self_attn.q_proj.weight", 0.5),
        create_test_tensor("model.layers.0.self_attn.k_proj.weight", 0.4),
        create_test_tensor("model.layers.0.mlp.gate_proj.weight", 0.8),
        create_test_tensor("model.layers.1.self_attn.q_proj.weight", 0.3),
        create_test_tensor("model.embed_tokens.weight", 0.01),
        create_test_tensor("lm_head.weight", 0.01),
    ];

    let layers = map_layers(&tensors);
    
    // Debug: print all layers
    for layer in &layers {
        println!("Layer: index={:?}, name={}, type={:?}, l2={}", 
            layer.layer_index, layer.layer_name, layer.layer_type, layer.aggregate_l2);
    }
    
    // Should group by layer
    assert!(!layers.is_empty());
    
    // Should have embedding layer
    let embed_layers: Vec<_> = layers.iter().filter(|l| l.layer_type == LayerType::Embedding).collect();
    assert_eq!(embed_layers.len(), 1);
    
    // Should have head layer
    let head_layers: Vec<_> = layers.iter().filter(|l| l.layer_type == LayerType::Head).collect();
    assert_eq!(head_layers.len(), 1);
    
    // Layer 0 should have higher aggregate L2 than layer 1
    let layer_0 = layers.iter().find(|l| l.layer_index == Some(0)).expect("Layer 0 not found");
    let layer_1 = layers.iter().find(|l| l.layer_index == Some(1)).expect("Layer 1 not found");
    assert!(layer_0.aggregate_l2 > layer_1.aggregate_l2, 
        "Layer 0 should have higher aggregate L2: {} vs {}", layer_0.aggregate_l2, layer_1.aggregate_l2);
}

#[test]
fn test_anomaly_detection() {
    let tensors = vec![
        create_test_tensor("model.layers.0.mlp.gate_proj.weight", 0.1),
        create_test_tensor("model.layers.1.mlp.gate_proj.weight", 0.1),
        create_test_tensor("model.layers.2.mlp.gate_proj.weight", 0.1),
        create_test_tensor("model.layers.3.mlp.gate_proj.weight", 0.9), // Anomaly
    ];

    let layers = map_layers(&tensors);
    
    let anomaly_layer = layers.iter().find(|l| l.layer_index == Some(3)).expect("Layer 3 not found");
    assert!(anomaly_layer.anomaly_score > 1.5, 
        "Expected high anomaly score for layer 3: {}", anomaly_layer.anomaly_score);
}
