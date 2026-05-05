use crate::types::{DType, ModelSnapshot, TensorMeta};
use anyhow::{Context, Result};
use half::{bf16, f16};
use memmap2::Mmap;
use safetensors::SafeTensors;
use std::collections::HashMap;
use std::fs::File;
use std::path::Path;
use std::sync::Arc;

pub fn load(path: &Path) -> Result<ModelSnapshot> {
    use crate::checkpoint::{resolve, CheckpointSource};
    match resolve(path)? {
        CheckpointSource::SingleFile(p) => load_single(&p),
        CheckpointSource::Sharded { index_path, root } => load_sharded(&index_path, &root),
    }
}

fn load_single(path: &Path) -> Result<ModelSnapshot> {
    let file = File::open(path)
        .with_context(|| format!("Failed to open file: {}", path.display()))?;
    let mmap = Arc::new(unsafe { Mmap::map(&file)? });
    let tensors = SafeTensors::deserialize(&mmap)
        .with_context(|| format!("Failed to parse safetensors: {}", path.display()))?;

    let mut tensor_map = HashMap::new();
    let mut total_params = 0u64;

    for (name, view) in tensors.tensors() {
        let shape = view.shape().to_vec();
        let numel = shape.iter().product::<usize>() as u64;
        total_params += numel;

        let dtype = decode_dtype(view.dtype());

        let data = view.data();
        let data_offset = data.as_ptr() as u64 - mmap.as_ptr() as u64;

        tensor_map.insert(
            name.to_string(),
            TensorMeta {
                name: name.to_string(),
                shape,
                dtype,
                data_offset,
                data_len: data.len() as u64,
                shard_index: 0,
            },
        );
    }

    Ok(ModelSnapshot {
        path: path.to_path_buf(),
        tensors: tensor_map,
        total_params,
        mmaps: vec![mmap],
    })
}

fn load_sharded(index_path: &Path, root: &Path) -> Result<ModelSnapshot> {
    use std::collections::BTreeMap;

    let index_bytes = std::fs::read(index_path)
        .with_context(|| format!("Failed to read index: {}", index_path.display()))?;
    let index: serde_json::Value = serde_json::from_slice(&index_bytes)
        .with_context(|| format!("Failed to parse index json: {}", index_path.display()))?;

    let weight_map = index
        .get("weight_map")
        .and_then(|v| v.as_object())
        .with_context(|| format!("index.json missing 'weight_map': {}", index_path.display()))?;

    // Group tensors by shard filename, sorted for stable order.
    let mut groups: BTreeMap<String, Vec<String>> = BTreeMap::new();
    for (tensor_name, shard_value) in weight_map {
        let shard_name = shard_value
            .as_str()
            .with_context(|| format!("weight_map['{tensor_name}'] not a string"))?;
        groups
            .entry(shard_name.to_string())
            .or_default()
            .push(tensor_name.clone());
    }

    let mut mmaps: Vec<Arc<Mmap>> = Vec::with_capacity(groups.len());
    let mut tensor_map: HashMap<String, TensorMeta> = HashMap::new();
    let mut total_params: u64 = 0;

    for (shard_idx, (shard_name, expected_tensors)) in groups.iter().enumerate() {
        if shard_idx > u16::MAX as usize {
            anyhow::bail!(
                "too many shards ({}): max supported is {}",
                groups.len(),
                u16::MAX
            );
        }
        let shard_path = root.join(shard_name);
        let file = File::open(&shard_path)
            .with_context(|| format!("missing shard: {}", shard_path.display()))?;
        let mmap = Arc::new(unsafe { Mmap::map(&file)? });
        let parsed = SafeTensors::deserialize(&mmap)
            .with_context(|| format!("Failed to parse shard: {}", shard_path.display()))?;

        let shard_tensors = parsed.tensors();

        // Verify expected tensors are present in this shard.
        let actual_names: std::collections::HashSet<&str> = shard_tensors
            .iter()
            .map(|(n, _)| n.as_str())
            .collect();
        for expected in expected_tensors {
            if !actual_names.contains(expected.as_str()) {
                anyhow::bail!(
                    "index references '{expected}' but it is not in shard {}",
                    shard_path.display()
                );
            }
        }
        // Warn on extra tensors not declared in the index.
        for (name, _) in &shard_tensors {
            if !expected_tensors.iter().any(|e| e == name) {
                tracing::warn!(
                    "tensor '{}' present in {} but absent from index.json — keeping anyway",
                    name,
                    shard_path.display()
                );
            }
        }

        for (name, view) in shard_tensors {
            if tensor_map.contains_key(&name) {
                anyhow::bail!(
                    "tensor name collision across shards: '{name}' in {}",
                    shard_path.display()
                );
            }
            let shape = view.shape().to_vec();
            let numel = shape.iter().product::<usize>() as u64;
            total_params += numel;
            let dtype = decode_dtype(view.dtype());
            let data = view.data();
            let data_offset = data.as_ptr() as u64 - mmap.as_ptr() as u64;

            tensor_map.insert(
                name.to_string(),
                TensorMeta {
                    name: name.to_string(),
                    shape,
                    dtype,
                    data_offset,
                    data_len: data.len() as u64,
                    shard_index: shard_idx as u16,
                },
            );
        }

        mmaps.push(mmap);
    }

    Ok(ModelSnapshot {
        path: root.to_path_buf(),
        tensors: tensor_map,
        total_params,
        mmaps,
    })
}

fn decode_dtype(dt: safetensors::Dtype) -> DType {
    match dt {
        safetensors::Dtype::F32 => DType::F32,
        safetensors::Dtype::F16 => DType::F16,
        safetensors::Dtype::BF16 => DType::BF16,
        safetensors::Dtype::I64 => DType::I64,
        safetensors::Dtype::I32 => DType::I32,
        safetensors::Dtype::I16 => DType::I16,
        safetensors::Dtype::I8 => DType::I8,
        safetensors::Dtype::U8 => DType::U8,
        safetensors::Dtype::BOOL => DType::Bool,
        _ => DType::F32,
    }
}

fn read_f32_le(data: &[u8]) -> f32 {
    let bits = u32::from_le_bytes([data[0], data[1], data[2], data[3]]);
    f32::from_bits(bits)
}

fn read_i64_le(data: &[u8]) -> i64 {
    i64::from_le_bytes([data[0], data[1], data[2], data[3], data[4], data[5], data[6], data[7]])
}

fn read_i32_le(data: &[u8]) -> i32 {
    i32::from_le_bytes([data[0], data[1], data[2], data[3]])
}

fn read_i16_le(data: &[u8]) -> i16 {
    i16::from_le_bytes([data[0], data[1]])
}

fn read_u16_le(data: &[u8]) -> u16 {
    u16::from_le_bytes([data[0], data[1]])
}

pub fn load_tensor_data(snapshot: &ModelSnapshot, name: &str) -> Result<Vec<f32>> {
    let meta = snapshot
        .tensors
        .get(name)
        .with_context(|| format!("Tensor '{}' not found in snapshot", name))?;

    let mmap = snapshot
        .mmaps
        .get(meta.shard_index as usize)
        .with_context(|| {
            format!(
                "Tensor '{}' references shard_index {} but snapshot has {} shards",
                name,
                meta.shard_index,
                snapshot.mmaps.len()
            )
        })?;

    let start = meta.data_offset as usize;
    let end = start + meta.data_len as usize;
    let data = mmap
        .get(start..end)
        .with_context(|| format!("Tensor '{}' data range [{start}..{end}] out of mmap bounds", name))?;

    let numel = meta.shape.iter().product::<usize>();

    let f32_data: Vec<f32> = match meta.dtype {
        DType::F32 => data.chunks_exact(4).map(read_f32_le).collect(),
        DType::F16 => data
            .chunks_exact(2)
            .map(|c| f16::from_bits(read_u16_le(c)).to_f32())
            .collect(),
        DType::BF16 => data
            .chunks_exact(2)
            .map(|c| bf16::from_bits(read_u16_le(c)).to_f32())
            .collect(),
        DType::I64 => data.chunks_exact(8).map(|c| read_i64_le(c) as f32).collect(),
        DType::I32 => data.chunks_exact(4).map(|c| read_i32_le(c) as f32).collect(),
        DType::I16 => data.chunks_exact(2).map(|c| read_i16_le(c) as f32).collect(),
        DType::I8  => data.iter().map(|&b| (b as i8) as f32).collect(),
        DType::U8  => data.iter().map(|&b| b as f32).collect(),
        DType::Bool => data.iter().map(|&b| if b != 0 { 1.0 } else { 0.0 }).collect(),
    };

    if f32_data.len() != numel {
        anyhow::bail!(
            "Data length mismatch for '{}': expected {} elements, got {}",
            name,
            numel,
            f32_data.len()
        );
    }

    Ok(f32_data)
}
