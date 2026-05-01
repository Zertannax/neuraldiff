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

#[test]
fn test_llama_norms_are_distinguished() {
    // Regression for C2: input_layernorm and post_attention_layernorm
    // must produce two distinct LayerDiff entries, not collapse into one.
    let tensors = vec![
        create_test_tensor("model.layers.10.input_layernorm.weight", 0.2),
        create_test_tensor("model.layers.10.post_attention_layernorm.weight", 0.7),
    ];
    let layers = map_layers(&tensors);

    let norm_layers: Vec<_> = layers
        .iter()
        .filter(|l| l.layer_index == Some(10) && l.layer_type == LayerType::Norm)
        .collect();
    assert_eq!(
        norm_layers.len(),
        2,
        "input and post-attn norms must be separate LayerDiff entries"
    );

    let names: std::collections::HashSet<_> =
        norm_layers.iter().map(|l| l.layer_name.as_str()).collect();
    assert!(names.contains("layers.10.input_norm"));
    assert!(names.contains("layers.10.post_attn_norm"));
}

#[test]
fn test_llama_norm_l2_not_mixed_with_other_norm() {
    // Regression for C2: a high-L2 input_layernorm must not pollute
    // post_attention_layernorm's aggregate_l2 (or vice versa).
    let tensors = vec![
        create_test_tensor("model.layers.0.input_layernorm.weight", 0.9),
        create_test_tensor("model.layers.0.post_attention_layernorm.weight", 0.1),
    ];
    let layers = map_layers(&tensors);

    let input = layers
        .iter()
        .find(|l| l.layer_name == "layers.0.input_norm")
        .expect("input_norm not found");
    let postat = layers
        .iter()
        .find(|l| l.layer_name == "layers.0.post_attn_norm")
        .expect("post_attn_norm not found");
    assert!((input.aggregate_l2 - 0.9).abs() < 1e-6);
    assert!((postat.aggregate_l2 - 0.1).abs() < 1e-6);
}

#[test]
fn test_gpt2_layer_grouping_and_distinct_norms() {
    let tensors = vec![
        create_test_tensor("transformer.h.0.attn.c_attn.weight", 0.5),
        create_test_tensor("transformer.h.0.mlp.c_fc.weight", 0.3),
        create_test_tensor("transformer.h.0.ln_1.weight", 0.1),
        create_test_tensor("transformer.h.0.ln_2.weight", 0.4),
    ];
    let layers = map_layers(&tensors);

    let at_0: Vec<_> = layers.iter().filter(|l| l.layer_index == Some(0)).collect();
    assert_eq!(at_0.len(), 4, "GPT-2 should produce 4 distinct layers per block");

    let names: std::collections::HashSet<_> = at_0.iter().map(|l| l.layer_name.as_str()).collect();
    for expected in ["layers.0.attn", "layers.0.mlp", "layers.0.ln_1", "layers.0.ln_2"] {
        assert!(names.contains(expected), "missing {}", expected);
    }
}

#[test]
fn test_falcon_layer_grouping_and_distinct_norms() {
    let tensors = vec![
        create_test_tensor("transformer.h.0.self_attention.query_key_value.weight", 0.5),
        create_test_tensor("transformer.h.0.mlp.dense_h_to_4h.weight", 0.3),
        create_test_tensor("transformer.h.0.ln_attn.weight", 0.1),
        create_test_tensor("transformer.h.0.ln_mlp.weight", 0.2),
        create_test_tensor("transformer.h.0.input_layernorm.weight", 0.4),
    ];
    let layers = map_layers(&tensors);

    let at_0: Vec<_> = layers.iter().filter(|l| l.layer_index == Some(0)).collect();
    assert_eq!(at_0.len(), 5, "Falcon should produce 5 distinct layers per block");

    let names: std::collections::HashSet<_> = at_0.iter().map(|l| l.layer_name.as_str()).collect();
    for expected in [
        "layers.0.attn",
        "layers.0.mlp",
        "layers.0.ln_attn",
        "layers.0.ln_mlp",
        "layers.0.input_norm",
    ] {
        assert!(names.contains(expected), "missing {}", expected);
    }
}

#[test]
fn test_all_components_per_arch_have_distinct_keys() {
    let llama = [
        "model.layers.5.self_attn.q_proj.weight",
        "model.layers.5.mlp.gate_proj.weight",
        "model.layers.5.input_layernorm.weight",
        "model.layers.5.post_attention_layernorm.weight",
    ];
    let gpt2 = [
        "transformer.h.5.attn.c_attn.weight",
        "transformer.h.5.mlp.c_fc.weight",
        "transformer.h.5.ln_1.weight",
        "transformer.h.5.ln_2.weight",
    ];
    let falcon = [
        "transformer.h.5.self_attention.query_key_value.weight",
        "transformer.h.5.mlp.dense_h_to_4h.weight",
        "transformer.h.5.ln_attn.weight",
        "transformer.h.5.ln_mlp.weight",
        "transformer.h.5.input_layernorm.weight",
    ];

    for (arch, names, expected) in [
        ("llama", &llama[..], 4usize),
        ("gpt2", &gpt2[..], 4),
        ("falcon", &falcon[..], 5),
    ] {
        let tensors: Vec<_> = names.iter().map(|n| create_test_tensor(n, 0.1)).collect();
        let layers = map_layers(&tensors);
        let at_5: Vec<_> = layers.iter().filter(|l| l.layer_index == Some(5)).collect();
        assert_eq!(
            at_5.len(),
            expected,
            "{} should produce {} distinct layers",
            arch,
            expected
        );

        let unique_names: std::collections::HashSet<_> =
            at_5.iter().map(|l| l.layer_name.as_str()).collect();
        assert_eq!(
            unique_names.len(),
            expected,
            "{} layer_names must be distinct",
            arch
        );
    }
}

#[test]
fn test_extract_parse_round_trip_preserves_component() {
    // For each known component, a single tensor must map to exactly one
    // LayerDiff with the expected (index, type, name) triple.
    let cases: &[(&str, LayerType, &str)] = &[
        ("model.layers.7.self_attn.q_proj.weight", LayerType::Attention, "layers.7.attn"),
        ("model.layers.7.mlp.gate_proj.weight", LayerType::MLP, "layers.7.mlp"),
        ("model.layers.7.input_layernorm.weight", LayerType::Norm, "layers.7.input_norm"),
        ("model.layers.7.post_attention_layernorm.weight", LayerType::Norm, "layers.7.post_attn_norm"),
        ("transformer.h.7.attn.c_attn.weight", LayerType::Attention, "layers.7.attn"),
        ("transformer.h.7.ln_1.weight", LayerType::Norm, "layers.7.ln_1"),
        ("transformer.h.7.ln_2.weight", LayerType::Norm, "layers.7.ln_2"),
        ("transformer.h.7.self_attention.query_key_value.weight", LayerType::Attention, "layers.7.attn"),
        ("transformer.h.7.ln_attn.weight", LayerType::Norm, "layers.7.ln_attn"),
        ("transformer.h.7.ln_mlp.weight", LayerType::Norm, "layers.7.ln_mlp"),
    ];
    for (raw, expected_ty, expected_name) in cases {
        let layers = map_layers(&[create_test_tensor(raw, 0.5)]);
        assert_eq!(layers.len(), 1, "{}: should produce exactly 1 layer", raw);
        assert_eq!(layers[0].layer_index, Some(7), "{}: wrong index", raw);
        assert_eq!(layers[0].layer_type, *expected_ty, "{}: wrong type", raw);
        assert_eq!(layers[0].layer_name, *expected_name, "{}: wrong name", raw);
    }
}
