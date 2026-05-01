use memmap2::Mmap;
use serde::{Deserialize, Serialize};
use std::collections::HashMap;
use std::path::PathBuf;
use std::sync::Arc;

#[derive(Debug, Clone)]
pub struct ModelSnapshot {
    pub path: PathBuf,
    pub tensors: HashMap<String, TensorMeta>,
    pub total_params: u64,
    pub mmap: Arc<Mmap>,
}

#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data_offset: u64,
    pub data_len: u64,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum DType {
    F32, F16, BF16, I64, I32, I16, I8, U8, Bool,
}

impl DType {
    pub fn size_in_bytes(&self) -> usize {
        match self {
            DType::F32 => 4,
            DType::F16 => 2,
            DType::BF16 => 2,
            DType::I64 => 8,
            DType::I32 => 4,
            DType::I16 => 2,
            DType::I8 => 1,
            DType::U8 => 1,
            DType::Bool => 1,
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct TensorDiff {
    pub name: String,
    pub shape: Vec<usize>,
    pub l2_distance: f32,
    pub cosine_similarity: f32,
    pub max_delta: f32,
    pub mean_delta: f32,
    pub std_delta: f32,
    pub changed: bool,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum LayerType {
    Embedding, Attention, MLP, Norm, Head, Other,
}

impl std::fmt::Display for LayerType {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            LayerType::Embedding => write!(f, "embed"),
            LayerType::Attention => write!(f, "attn"),
            LayerType::MLP => write!(f, "mlp"),
            LayerType::Norm => write!(f, "norm"),
            LayerType::Head => write!(f, "head"),
            LayerType::Other => write!(f, "other"),
        }
    }
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LayerDiff {
    pub layer_index: Option<usize>,
    pub layer_name: String,
    pub layer_type: LayerType,
    pub tensors: Vec<TensorDiff>,
    pub aggregate_l2: f32,
    pub anomaly_score: f32,
    pub param_count: u64,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffResult {
    pub model_a: String,
    pub model_b: String,
    pub total_params: u64,
    pub layers: Vec<LayerDiff>,
    pub summary: DiffSummary,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DiffSummary {
    pub total_layers: usize,
    pub changed_layers: usize,
    pub unchanged_layers: usize,
    pub change_ratio_percent: f32,
    pub mean_delta: f32,
    pub max_delta: f32,
    pub top_changed_indices: Vec<usize>,
    pub anomalies: Vec<AnomalyInfo>,
    pub missing_tensors: Vec<String>,
}

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct AnomalyInfo {
    pub layer_index: Option<usize>,
    pub layer_name: String,
    pub z_score: f32,
    pub reason: String,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum Severity {
    Low, Medium, High, Critical,
}

impl Severity {
    pub fn from_l2(l2: f32) -> Self {
        if l2 < 0.001 { Severity::Low }
        else if l2 < 0.3 { Severity::Medium }
        else if l2 < 0.6 { Severity::High }
        else { Severity::Critical }
    }

    pub fn as_str(&self) -> &'static str {
        match self {
            Severity::Low => "[LOW]",
            Severity::Medium => "[MED]",
            Severity::High => "[HIGH]",
            Severity::Critical => "[CRIT]",
        }
    }
}

#[derive(Debug, Clone)]
pub struct ProgressState {
    pub current: usize,
    pub total: usize,
    pub message: String,
    pub start_time: std::time::Instant,
}

impl ProgressState {
    pub fn new(total: usize, message: impl Into<String>) -> Self {
        Self {
            current: 0,
            total,
            message: message.into(),
            start_time: std::time::Instant::now(),
        }
    }

    pub fn percent(&self) -> f32 {
        if self.total == 0 { 0.0 }
        else { (self.current as f32 / self.total as f32) * 100.0 }
    }

    pub fn eta_seconds(&self) -> Option<u64> {
        if self.current == 0 || self.current >= self.total { return None; }
        let elapsed = self.start_time.elapsed().as_secs_f64();
        let rate = self.current as f64 / elapsed;
        let remaining = (self.total - self.current) as f64 / rate;
        Some(remaining.ceil() as u64)
    }
}

#[derive(Debug, Clone)]
pub struct AppState {
    pub diff: Option<DiffResult>,
    pub selected_layer: usize,
    pub selected_tensor: usize,
    pub show_heatmap: bool,
    pub sort_mode: SortMode,
    pub filter_mode: FilterMode,
    pub show_help: bool,
    pub status_message: Option<String>,
    pub progress: Option<ProgressState>,
    pub view_mode: ViewMode,
}

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum SortMode { L2Desc, LayerIndex, AnomalyScore }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum FilterMode { All, ChangedOnly }

#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub enum ViewMode { Summary, Detail }

impl Default for AppState {
    fn default() -> Self {
        Self {
            diff: None,
            selected_layer: 0,
            selected_tensor: 0,
            show_heatmap: false,
            sort_mode: SortMode::L2Desc,
            filter_mode: FilterMode::All,
            show_help: false,
            status_message: None,
            progress: None,
            view_mode: ViewMode::Summary,
        }
    }
}
