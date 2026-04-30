use crate::loader::{load, load_tensor_data};
use crate::metrics::{cosine_similarity, l2_norm};
use crate::types::{AnomalyInfo, DiffResult, DiffSummary, LayerDiff, TensorDiff};
use anyhow::{Context, Result};
use rayon::prelude::*;
use std::path::Path;

const CHANGE_THRESHOLD: f32 = 1e-6;

pub fn compute_diff(model_a_path: &Path, model_b_path: &Path) -> Result<DiffResult> {
    let snapshot_a = load(model_a_path)
        .with_context(|| format!("Failed to load model A: {}", model_a_path.display()))?;
    let snapshot_b = load(model_b_path)
        .with_context(|| format!("Failed to load model B: {}", model_b_path.display()))?;

    let total_params = snapshot_a.total_params;

    // Find common tensor names
    let common_names: Vec<String> = snapshot_a
        .tensors
        .keys()
        .filter(|name| snapshot_b.tensors.contains_key(*name))
        .cloned()
        .collect();

    // Compute diff for each tensor in parallel
    let tensor_diffs: Vec<TensorDiff> = common_names
        .par_iter()
        .filter_map(|name| {
            match compute_tensor_diff(&snapshot_a, &snapshot_b, name) {
                Ok(diff) => Some(diff),
                Err(e) => {
                    eprintln!("Warning: Failed to diff tensor '{}': {}", name, e);
                    None
                }
            }
        })
        .collect();

    // Map to layers
    let layers = crate::mapper::map_layers(&tensor_diffs);

    // Compute summary
    // Find missing tensors (only in A or only in B)
    let missing_tensors = {
        let names_a: std::collections::HashSet<String> = snapshot_a.tensors.keys().cloned().collect();
        let names_b: std::collections::HashSet<String> = snapshot_b.tensors.keys().cloned().collect();
        let mut missing = Vec::new();
        for name in names_a.difference(&names_b) {
            missing.push(format!("Only in A: {}", name));
        }
        for name in names_b.difference(&names_a) {
            missing.push(format!("Only in B: {}", name));
        }
        missing
    };
    
    if !missing_tensors.is_empty() {
        eprintln!("Warning: {} tensors are not present in both models", missing_tensors.len());
        for msg in &missing_tensors {
            eprintln!("  {}", msg);
        }
    }
    
    let summary = compute_summary(&layers, total_params, missing_tensors);

    Ok(DiffResult {
        model_a: Some(model_a_path.to_string_lossy().to_string()),
        model_b: Some(model_b_path.to_string_lossy().to_string()),
        total_params,
        layers,
        summary,
    })
}

fn compute_tensor_diff(
    snapshot_a: &crate::types::ModelSnapshot,
    snapshot_b: &crate::types::ModelSnapshot,
    name: &str,
) -> Result<TensorDiff> {
    let meta_a = snapshot_a.tensors.get(name).unwrap();
    let meta_b = snapshot_b.tensors.get(name).unwrap();

    // Skip if shapes don't match
    if meta_a.shape != meta_b.shape {
        anyhow::bail!(
            "Shape mismatch for '{}': {:?} vs {:?}",
            name,
            meta_a.shape,
            meta_b.shape
        );
    }

    let data_a = load_tensor_data(snapshot_a, name)?;
    let data_b = load_tensor_data(snapshot_b, name)?;

    let delta: Vec<f32> = data_a
        .iter()
        .zip(data_b.iter())
        .map(|(a, b)| b - a)
        .collect();

    let l2_dist = l2_norm(&delta);
    let cos_sim = cosine_similarity(&data_a, &data_b);
    let max_delta = delta
        .iter()
        .map(|d| d.abs())
        .fold(0.0f32, f32::max);
    let mean_delta = delta.iter().sum::<f32>() / delta.len() as f32;
    let std_delta = std_dev(&delta);

    Ok(TensorDiff {
        name: name.to_string(),
        shape: meta_a.shape.clone(),
        l2_distance: l2_dist,
        cosine_similarity: cos_sim,
        max_delta,
        mean_delta,
        std_delta,
        changed: l2_dist > CHANGE_THRESHOLD,
    })
}

fn std_dev(v: &[f32]) -> f32 {
    if v.is_empty() {
        return 0.0;
    }
    let mean = v.iter().sum::<f32>() / v.len() as f32;
    let variance = v.iter().map(|x| (x - mean).powi(2)).sum::<f32>() / v.len() as f32;
    variance.sqrt()
}

fn compute_summary(layers: &[LayerDiff], _total_params: u64, _missing_tensors: Vec<String>) -> DiffSummary {
    let total_layers = layers.len();
    let changed_layers = layers.iter().filter(|l| l.tensors.iter().any(|t| t.changed)).count();
    let unchanged_layers = total_layers - changed_layers;
    let change_ratio = if total_layers > 0 {
        (changed_layers as f32 / total_layers as f32) * 100.0
    } else {
        0.0
    };

    let mean_delta = {
        let all_tensor_l2: Vec<f32> = layers.iter()
            .flat_map(|l| l.tensors.iter().map(|t| t.l2_distance))
            .collect();
        if !all_tensor_l2.is_empty() {
            all_tensor_l2.iter().sum::<f32>() / all_tensor_l2.len() as f32
        } else {
            0.0
        }
    };

    let max_delta = layers
        .iter()
        .map(|l| l.aggregate_l2)
        .fold(0.0f32, f32::max);

    // Top 5 changed layers
    let mut top_indices: Vec<usize> = (0..layers.len()).collect();
    top_indices.sort_by(|a, b| {
        layers[*b]
            .aggregate_l2
            .partial_cmp(&layers[*a].aggregate_l2)
            .unwrap()
    });
    let top_changed_indices = top_indices.into_iter().take(5).collect();

    // Anomalies: z-score > 2.0
    let anomalies = layers
        .iter()
        .filter(|l| l.anomaly_score > 2.0)
        .map(|l| AnomalyInfo {
            layer_index: l.layer_index,
            layer_name: l.layer_name.clone(),
            z_score: l.anomaly_score,
            reason: "Unusually high L2 distance compared to other layers".to_string(),
        })
        .collect();

    DiffSummary {
        total_layers,
        changed_layers,
        unchanged_layers,
        change_ratio_percent: change_ratio,
        mean_delta,
        max_delta,
        top_changed_indices,
        anomalies,
    }
}
