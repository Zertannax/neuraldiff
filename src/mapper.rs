use crate::types::{LayerDiff, LayerType, TensorDiff};
use regex::Regex;
use std::collections::HashMap;
use std::sync::OnceLock;

/// Fine-grained per-block component captured by the architecture regexes.
/// This is the single source of truth for the regex-capture ↔ LayerType ↔
/// display-suffix mapping. Adding a new variant forces every match below
/// to be updated, so component types cannot drift across functions.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash)]
enum LayerComponent {
    LlamaSelfAttn,
    LlamaMlp,
    LlamaInputNorm,
    LlamaPostAttnNorm,

    Gpt2Attn,
    Gpt2Mlp,
    Gpt2Ln1,
    Gpt2Ln2,

    FalconSelfAttention,
    FalconMlp,
    FalconLnAttn,
    FalconLnMlp,
    FalconInputNorm,
}

impl LayerComponent {
    fn from_llama_capture(s: &str) -> Option<Self> {
        match s {
            "self_attn" => Some(Self::LlamaSelfAttn),
            "mlp" => Some(Self::LlamaMlp),
            "input_layernorm" => Some(Self::LlamaInputNorm),
            "post_attention_layernorm" => Some(Self::LlamaPostAttnNorm),
            _ => None,
        }
    }

    fn from_gpt2_capture(s: &str) -> Option<Self> {
        match s {
            "attn" => Some(Self::Gpt2Attn),
            "mlp" => Some(Self::Gpt2Mlp),
            "ln_1" => Some(Self::Gpt2Ln1),
            "ln_2" => Some(Self::Gpt2Ln2),
            _ => None,
        }
    }

    fn from_falcon_capture(s: &str) -> Option<Self> {
        match s {
            "self_attention" => Some(Self::FalconSelfAttention),
            "mlp" => Some(Self::FalconMlp),
            "ln_attn" => Some(Self::FalconLnAttn),
            "ln_mlp" => Some(Self::FalconLnMlp),
            "input_layernorm" => Some(Self::FalconInputNorm),
            _ => None,
        }
    }

    /// Human-readable suffix used in `layer_name` (e.g. `layers.10.<suffix>`).
    /// Distinct per variant to avoid the C2 collision where two components
    /// previously collapsed onto the same `"norm"` suffix.
    fn display_suffix(self) -> &'static str {
        match self {
            Self::LlamaSelfAttn => "attn",
            Self::LlamaMlp => "mlp",
            Self::LlamaInputNorm => "input_norm",
            Self::LlamaPostAttnNorm => "post_attn_norm",

            Self::Gpt2Attn => "attn",
            Self::Gpt2Mlp => "mlp",
            Self::Gpt2Ln1 => "ln_1",
            Self::Gpt2Ln2 => "ln_2",

            Self::FalconSelfAttention => "attn",
            Self::FalconMlp => "mlp",
            Self::FalconLnAttn => "ln_attn",
            Self::FalconLnMlp => "ln_mlp",
            Self::FalconInputNorm => "input_norm",
        }
    }

    /// Coarse-grained roll-up exposed via the public `LayerType`.
    /// Exhaustive match — fixes C3 (no silent fallback to "norm").
    fn layer_type(self) -> LayerType {
        match self {
            Self::LlamaSelfAttn | Self::Gpt2Attn | Self::FalconSelfAttention => {
                LayerType::Attention
            }
            Self::LlamaMlp | Self::Gpt2Mlp | Self::FalconMlp => LayerType::MLP,
            Self::LlamaInputNorm
            | Self::LlamaPostAttnNorm
            | Self::Gpt2Ln1
            | Self::Gpt2Ln2
            | Self::FalconLnAttn
            | Self::FalconLnMlp
            | Self::FalconInputNorm => LayerType::Norm,
        }
    }
}

/// Typed HashMap key for layer grouping. Replaces the previous string round-trip
/// (extract -> "layer_10_norm" -> reparse) which lost information.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
enum LayerKey {
    Block { index: usize, component: LayerComponent },
    Embedding,
    Head,
    FinalNorm,
    /// Catch-all for tensors no architecture regex matched. Keeps the original
    /// path prefix so distinct unknowns don't collapse together.
    Other(String),
}

impl LayerKey {
    fn into_descriptor(self) -> (Option<usize>, LayerType, String) {
        match self {
            LayerKey::Block { index, component } => (
                Some(index),
                component.layer_type(),
                format!("layers.{}.{}", index, component.display_suffix()),
            ),
            LayerKey::Embedding => (None, LayerType::Embedding, "embed_tokens".to_string()),
            LayerKey::Head => (None, LayerType::Head, "lm_head".to_string()),
            LayerKey::FinalNorm => (None, LayerType::Norm, "norm".to_string()),
            LayerKey::Other(s) => (None, LayerType::Other, s),
        }
    }
}

pub fn map_layers(tensor_diffs: &[TensorDiff]) -> Vec<LayerDiff> {
    let mut layer_groups: HashMap<LayerKey, Vec<&TensorDiff>> = HashMap::new();

    for diff in tensor_diffs {
        let layer_key = extract_layer_key(&diff.name);
        layer_groups.entry(layer_key).or_default().push(diff);
    }

    let mut layers: Vec<LayerDiff> = layer_groups
        .into_iter()
        .map(|(key, diffs)| {
            let (layer_index, layer_type, layer_name) = key.into_descriptor();

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

    // Tie-break by layer_name so the order between distinct components at the
    // same index (e.g. layers.10.input_norm vs layers.10.post_attn_norm) is
    // deterministic across runs — HashMap iteration order otherwise leaks.
    layers.sort_by(|a, b| match (a.layer_index, b.layer_index) {
        (Some(ai), Some(bi)) => ai.cmp(&bi).then_with(|| a.layer_name.cmp(&b.layer_name)),
        (Some(_), None) => std::cmp::Ordering::Less,
        (None, Some(_)) => std::cmp::Ordering::Greater,
        (None, None) => a.layer_name.cmp(&b.layer_name),
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
    RE.get_or_init(|| Regex::new(r"^transformer\.h\.(\d+)\.(attn|mlp|ln_1|ln_2)").unwrap())
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

fn extract_layer_key(name: &str) -> LayerKey {
    if let Some(caps) = re_llama().captures(name)
        && let Ok(index) = caps[1].parse::<usize>()
        && let Some(component) = LayerComponent::from_llama_capture(&caps[2])
    {
        return LayerKey::Block { index, component };
    }

    if let Some(caps) = re_gpt2().captures(name)
        && let Ok(index) = caps[1].parse::<usize>()
        && let Some(component) = LayerComponent::from_gpt2_capture(&caps[2])
    {
        return LayerKey::Block { index, component };
    }

    if let Some(caps) = re_falcon().captures(name)
        && let Ok(index) = caps[1].parse::<usize>()
        && let Some(component) = LayerComponent::from_falcon_capture(&caps[2])
    {
        return LayerKey::Block { index, component };
    }

    if name.contains("embed_tokens") || name.contains("embed") {
        return LayerKey::Embedding;
    }

    if name.contains("lm_head") || name.contains("head") {
        return LayerKey::Head;
    }

    if name.contains("model.norm") || name.contains("final_layernorm") {
        return LayerKey::FinalNorm;
    }

    let parts: Vec<&str> = name.split('.').collect();
    LayerKey::Other(if parts.len() >= 2 {
        format!("{}.{}", parts[0], parts[1])
    } else {
        name.to_string()
    })
}
