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

        let dtype = match view.dtype() {
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
        };

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
            },
        );
    }

    Ok(ModelSnapshot {
        path: path.to_path_buf(),
        tensors: tensor_map,
        total_params,
        mmap,
    })
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

    let start = meta.data_offset as usize;
    let end = start + meta.data_len as usize;
    let data = snapshot
        .mmap
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
