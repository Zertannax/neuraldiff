use crate::types::{LayerDiff, LayerType, TensorDiff};
use std::collections::HashMap;

pub fn map_layers(tensor_diffs: &[TensorDiff]) -> Vec<LayerDiff> {
    let mut layer_groups: HashMap<String, Vec<&TensorDiff>> = HashMap::new();

    for diff in tensor_diffs {
        let layer_key = extract_layer_key(&diff.name);
        layer_groups
            .entry(layer_key)
            .or_default()
            .push(diff);
    }

    let mut layers: Vec<LayerDiff> = layer_groups
        .into_iter()
        .map(|(key, diffs)| {
            let (layer_index, layer_type, layer_name) = parse_layer_key(&key);
            
            let aggregate_l2 = if !diffs.is_empty() {
                diffs.iter().map(|d| d.l2_distance).sum::<f32>() / diffs.len() as f32
            } else {
                0.0
            };
            
            let param_count = diffs.iter().map(|d| {
                d.shape.iter().product::<usize>() as u64
            }).sum();

            LayerDiff {
                layer_index,
                layer_name,
                layer_type,
                tensors: diffs.into_iter().cloned().collect(),
                aggregate_l2,
                anomaly_score: 0.0,
                param_count,
            }
        })
        .collect();

    layers.sort_by(|a, b| {
        match (a.layer_index, b.layer_index) {
            (Some(ai), Some(bi)) => ai.cmp(&bi),
            (Some(_), None) => std::cmp::Ordering::Less,
            (None, Some(_)) => std::cmp::Ordering::Greater,
            (None, None) => std::cmp::Ordering::Equal,
        }
    });

    if !layers.is_empty() {
        let l2_values: Vec<f32> = layers.iter().map(|l| l.aggregate_l2).collect();
        let mean = l2_values.iter().sum::<f32>() / l2_values.len() as f32;
        let variance = l2_values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / l2_values.len() as f32;
        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            for layer in &mut layers {
                layer.anomaly_score = (layer.aggregate_l2 - mean) / std_dev;
            }
        }
    }

    layers
}

fn extract_layer_key(name: &str) -> String {
    if let Some(caps) = regex_captures(r"model\.layers\.(\d+)\.(self_attn|mlp|input_layernorm|post_attention_layernorm)", name) {
        let layer_num = &caps[0];
        let component = &caps[1];
        let comp_short = match component.as_str() {
            "self_attn" => "attn",
            "mlp" => "mlp",
            "input_layernorm" | "post_attention_layernorm" => "norm",
            _ => component.as_str(),
        };
        return format!("layer_{}_{}", layer_num, comp_short);
    }

    if name.contains("embed_tokens") || name.contains("embed") {
        return "embedding".to_string();
    }

    if name.contains("lm_head") || name.contains("head") {
        return "head".to_string();
    }

    if name.contains("model.norm") || name.contains("final_layernorm") {
        return "final_norm".to_string();
    }

    if let Some(caps) = regex_captures(r"transformer\.h\.(\d+)\.(attn|mlp|ln_1|ln_2)", name) {
        let layer_num = &caps[0];
        let component = &caps[1];
        let comp_short = match component.as_str() {
            "attn" => "attn",
            "mlp" => "mlp",
            "ln_1" | "ln_2" => "norm",
            _ => component.as_str(),
        };
        return format!("layer_{}_{}", layer_num, comp_short);
    }

    if let Some(caps) = regex_captures(r"transformer\.h\.(\d+)\.(self_attention|mlp|ln_attn|ln_mlp|input_layernorm)", name) {
        let layer_num = &caps[0];
        let component = &caps[1];
        let comp_short = match component.as_str() {
            "self_attention" => "attn",
            "mlp" => "mlp",
            "ln_attn" | "ln_mlp" | "input_layernorm" => "norm",
            _ => component.as_str(),
        };
        return format!("layer_{}_{}", layer_num, comp_short);
    }

    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() >= 2 {
        format!("{}.{}", parts[0], parts[1])
    } else {
        name.to_string()
    }
}

fn parse_layer_key(key: &str) -> (Option<usize>, LayerType, String) {
    if key == "embedding" {
        return (None, LayerType::Embedding, "embed_tokens".to_string());
    }
    if key == "head" {
        return (None, LayerType::Head, "lm_head".to_string());
    }
    if key == "final_norm" {
        return (None, LayerType::Norm, "norm".to_string());
    }

    if key.starts_with("layer_") {
        let parts: Vec<&str> = key.split('_').collect();
        if parts.len() >= 3 {
            if let Ok(index) = parts[1].parse::<usize>() {
                let layer_type = match parts[2] {
                    "attn" => LayerType::Attention,
                    "mlp" => LayerType::MLP,
                    "norm" => LayerType::Norm,
                    _ => LayerType::Other,
                };
                let name = format!("layers.{}.{}", index, parts[2]);
                return (Some(index), layer_type, name);
            }
        }
    }

    (None, LayerType::Other, key.to_string())
}

fn regex_captures(pattern_str: &str, text: &str) -> Option<Vec<String>> {
    if pattern_str == r"model\.layers\.(\d+)\.(self_attn|mlp|input_layernorm|post_attention_layernorm)" {
        if text.starts_with("model.layers.") {
            let rest = &text[13..];
            if let Some(dot_pos) = rest.find('.') {
                let num_str = &rest[..dot_pos];
                let component_rest = &rest[dot_pos + 1..];
                let component = if let Some(next_dot) = component_rest.find('.') {
                    &component_rest[..next_dot]
                } else {
                    component_rest
                };
                if let Ok(num) = num_str.parse::<usize>() {
                    let valid_components = ["self_attn", "mlp", "input_layernorm", "post_attention_layernorm"];
                    if valid_components.contains(&component) {
                        return Some(vec![num.to_string(), component.to_string()]);
                    }
                }
            }
        }
        return None;
    }
    
    if pattern_str == r"transformer\.h\.(\d+)\.(attn|mlp|ln_1|ln_2)" {
        if text.starts_with("transformer.h.") {
            let rest = &text[14..];
            if let Some(dot_pos) = rest.find('.') {
                let num_str = &rest[..dot_pos];
                let component_rest = &rest[dot_pos + 1..];
                let component = if let Some(next_dot) = component_rest.find('.') {
                    &component_rest[..next_dot]
                } else {
                    component_rest
                };
                if let Ok(num) = num_str.parse::<usize>() {
                    let valid_components = ["attn", "mlp", "ln_1", "ln_2"];
                    if valid_components.contains(&component) {
                        return Some(vec![num.to_string(), component.to_string()]);
                    }
                }
            }
        }
        return None;
    }
    
    if pattern_str == r"transformer\.h\.(\d+)\.(self_attention|mlp|ln_attn|ln_mlp|input_layernorm)" {
        if text.starts_with("transformer.h.") {
            let rest = &text[14..];
            if let Some(dot_pos) = rest.find('.') {
                let num_str = &rest[..dot_pos];
                let component_rest = &rest[dot_pos + 1..];
                let component = if let Some(next_dot) = component_rest.find('.') {
                    &component_rest[..next_dot]
                } else {
                    component_rest
                };
                if let Ok(num) = num_str.parse::<usize>() {
                    let valid_components = ["self_attention", "mlp", "ln_attn", "ln_mlp", "input_layernorm"];
                    if valid_components.contains(&component) {
                        return Some(vec![num.to_string(), component.to_string()]);
                    }
                }
            }
        }
        return None;
    }
    
    None
}
