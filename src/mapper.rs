use crate::types::{LayerDiff, LayerType, TensorDiff};
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

pub fn map_layers(tensor_diffs: &[TensorDiff]) -> Vec<LayerDiff> {
    let mut layer_groups: HashMap<String, Vec<&TensorDiff>> = HashMap::new();

    for diff in tensor_diffs {
        let layer_key = extract_layer_key(&diff.name);
        layer_groups.entry(layer_key).or_default().push(diff);
    }

    let mut layers: Vec<LayerDiff> = layer_groups
        .into_iter()
        .map(|(key, diffs)| {
            let (layer_index, layer_type, layer_name) = parse_layer_key(&key);

            let aggregate_l2 = {
                let (weighted_sum, total_params) = diffs
                    .iter()
                    .map(|d| {
                        let params = d.shape.iter().product::<usize>() as f32;
                        (d.l2_distance * params, params)
                    })
                    .fold((0.0f32, 0.0f32), |(sl, sp), (l, p)| (sl + l, sp + p));
                if total_params > 0.0 { weighted_sum / total_params } else { 0.0 }
            };

            let param_count = diffs
                .iter()
                .map(|d| d.shape.iter().product::<usize>() as u64)
                .sum();

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

    layers.sort_by(|a, b| match (a.layer_index, b.layer_index) {
        (Some(ai), Some(bi)) => ai.cmp(&bi),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => std::cmp::Ordering::Equal,
    });

    if !layers.is_empty() {
        let l2_values: Vec<f32> = layers.iter().map(|l| l.aggregate_l2).collect();
        let mean = l2_values.iter().sum::<f32>() / l2_values.len() as f32;
        let variance =
            l2_values.iter().map(|v| (v - mean).powi(2)).sum::<f32>() / l2_values.len() as f32;
        let std_dev = variance.sqrt();

        if std_dev > 0.0 {
            for layer in &mut layers {
                layer.anomaly_score = (layer.aggregate_l2 - mean) / std_dev;
            }
        }
    }

    layers
}

fn component_short(raw: &str, attn_names: &[&str], mlp_names: &[&str]) -> &'static str {
    if attn_names.contains(&raw) { "attn" }
    else if mlp_names.contains(&raw) { "mlp" }
    else { "norm" }
}

// Compiled regex patterns — initialized once, reused for every tensor name.
fn re_llama() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^model\.layers\.(\d+)\.(self_attn|mlp|input_layernorm|post_attention_layernorm)",
        )
        .unwrap()
    })
}

fn re_gpt2() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(r"^transformer\.h\.(\d+)\.(attn|mlp|ln_1|ln_2)").unwrap()
    })
}

fn re_falcon() -> &'static Regex {
    static RE: OnceLock<Regex> = OnceLock::new();
    RE.get_or_init(|| {
        Regex::new(
            r"^transformer\.h\.(\d+)\.(self_attention|mlp|ln_attn|ln_mlp|input_layernorm)",
        )
        .unwrap()
    })
}

fn extract_layer_key(name: &str) -> String {
    // LLaMA / Qwen style: model.layers.N.{self_attn,mlp,layernorm}
    if let Some(caps) = re_llama().captures(name) {
        let num = caps[1].to_string();
        let comp = component_short(&caps[2], &["self_attn"], &["mlp"]);
        return format!("layer_{}_{}", num, comp);
    }

    // GPT-2 style: transformer.h.N.{attn,mlp,ln_1,ln_2}
    if let Some(caps) = re_gpt2().captures(name) {
        let num = caps[1].to_string();
        let comp = component_short(&caps[2], &["attn"], &["mlp"]);
        return format!("layer_{}_{}", num, comp);
    }

    // Falcon style: transformer.h.N.{self_attention,mlp,ln_*}
    if let Some(caps) = re_falcon().captures(name) {
        let num = caps[1].to_string();
        let comp = component_short(&caps[2], &["self_attention"], &["mlp"]);
        return format!("layer_{}_{}", num, comp);
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

    let parts: Vec<&str> = name.split('.').collect();
    if parts.len() >= 2 {
        format!("{}.{}", parts[0], parts[1])
    } else {
        name.to_string()
    }
}

fn parse_layer_key(key: &str) -> (Option<usize>, LayerType, String) {
    match key {
        "embedding" => return (None, LayerType::Embedding, "embed_tokens".to_string()),
        "head" => return (None, LayerType::Head, "lm_head".to_string()),
        "final_norm" => return (None, LayerType::Norm, "norm".to_string()),
        _ => {}
    }

    if key.starts_with("layer_") {
        let parts: Vec<&str> = key.splitn(3, '_').collect();
        if parts.len() == 3 {
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
