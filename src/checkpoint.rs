use anyhow::{anyhow, bail, Context, Result};
use regex::Regex;
use std::path::{Path, PathBuf};
use std::sync::OnceLock;

#[derive(Debug, Clone, PartialEq, Eq)]
pub enum CheckpointSource {
    SingleFile(PathBuf),
    Sharded { index_path: PathBuf, root: PathBuf },
}

const INDEX_FILENAME: &str = "model.safetensors.index.json";

fn shard_pattern() -> &'static Regex {
    static PATTERN: OnceLock<Regex> = OnceLock::new();
    PATTERN.get_or_init(|| Regex::new(r"^model-\d{5}-of-\d{5}\.safetensors$").unwrap())
}

fn is_shard_filename(name: &str) -> bool {
    shard_pattern().is_match(name)
}

pub fn resolve(path: &Path) -> Result<CheckpointSource> {
    if !path.exists() {
        bail!("path not found: {}", path.display());
    }

    if path.is_file() {
        return resolve_file(path);
    }

    if path.is_dir() {
        return resolve_dir(path);
    }

    bail!("unrecognised checkpoint format: {}", path.display())
}

fn resolve_file(path: &Path) -> Result<CheckpointSource> {
    let filename = path
        .file_name()
        .and_then(|n| n.to_str())
        .ok_or_else(|| anyhow!("non-utf8 filename: {}", path.display()))?;

    if filename == INDEX_FILENAME {
        let root = path
            .parent()
            .ok_or_else(|| anyhow!("index.json has no parent directory: {}", path.display()))?
            .to_path_buf();
        return Ok(CheckpointSource::Sharded {
            index_path: path.to_path_buf(),
            root,
        });
    }

    if filename.ends_with(".safetensors") {
        if is_shard_filename(filename) {
            // Look for sibling index.json
            if let Some(parent) = path.parent() {
                let candidate = parent.join(INDEX_FILENAME);
                if candidate.exists() {
                    return Ok(CheckpointSource::Sharded {
                        index_path: candidate,
                        root: parent.to_path_buf(),
                    });
                }
            }
            tracing::warn!(
                "treating shard {} as a standalone file — no sibling index.json found",
                path.display()
            );
        }
        return Ok(CheckpointSource::SingleFile(path.to_path_buf()));
    }

    bail!("unrecognised checkpoint format: {}", path.display())
}

fn resolve_dir(dir: &Path) -> Result<CheckpointSource> {
    let index_path = dir.join(INDEX_FILENAME);
    if index_path.exists() {
        return Ok(CheckpointSource::Sharded {
            index_path,
            root: dir.to_path_buf(),
        });
    }

    // No index — look for .safetensors files
    let mut safetensors_files: Vec<PathBuf> = std::fs::read_dir(dir)
        .with_context(|| format!("cannot read directory: {}", dir.display()))?
        .filter_map(|e| e.ok())
        .map(|e| e.path())
        .filter(|p| {
            p.is_file() && p.extension().is_some_and(|ext| ext == "safetensors")
        })
        .collect();

    safetensors_files.sort();

    match safetensors_files.len() {
        0 => bail!(
            "directory contains no safetensors files: {}",
            dir.display()
        ),
        1 => {
            let only = safetensors_files.into_iter().next().unwrap();
            if let Some(name) = only.file_name().and_then(|n| n.to_str()) {
                if is_shard_filename(name) {
                    tracing::warn!(
                        "directory {} contains only one shard ({}) and no index.json — looks like an incomplete sharded checkpoint",
                        dir.display(),
                        name
                    );
                }
            }
            Ok(CheckpointSource::SingleFile(only))
        }
        n => bail!(
            "directory has {} safetensors files but no index.json — pass one explicitly: {}",
            n,
            dir.display()
        ),
    }
}
