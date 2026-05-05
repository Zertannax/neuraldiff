# Multi-shard Support Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Allow `neuraldiff diff` to operate on safetensors checkpoints split across multiple shards (HuggingFace standard for models >10GB), unblocking Llama 70B / Gemma 31B diffs.

**Architecture:** Auto-detect input format via new `src/checkpoint.rs::resolve()` returning `SingleFile | Sharded` enum. Internal data model migrates `ModelSnapshot.mmap: Arc<Mmap>` → `mmaps: Vec<Arc<Mmap>>`, with `TensorMeta` gaining a `shard_index: u16` field. Single-file path is wrapped uniformly as a 1-shard model. Public CLI and JSON schema unchanged.

**Tech Stack:** Rust 1.88+, safetensors 0.4, memmap2 0.9, serde_json 1.0, tempfile 3.14, anyhow 1.0, tracing 0.1.

**Spec:** `docs/superpowers/specs/2026-05-05-multishard-support-design.md`

---

## File Structure

| File                                         | Action  | Responsibility                                                    |
|----------------------------------------------|---------|-------------------------------------------------------------------|
| `src/types.rs`                               | Modify  | Change `ModelSnapshot.mmap` → `mmaps`; add `TensorMeta.shard_index` |
| `src/loader.rs`                              | Modify  | Refactor `load()` into dispatcher; add `load_single()`/`load_sharded()`; update `load_tensor_data()` |
| `src/checkpoint.rs`                          | Create  | `CheckpointSource` enum + `resolve()` function                    |
| `src/lib.rs`                                 | Modify  | Add `pub mod checkpoint;`                                         |
| `src/scanner.rs`                             | Modify  | Surface sharded directories as one entry, skip shards inside      |
| `tests/checkpoint_tests.rs`                  | Create  | Unit tests for `resolve()` (8 tests)                              |
| `tests/multishard_tests.rs`                  | Create  | Integration tests for `load_sharded()` + fixture helper           |
| `CONTEXT.md`                                 | Modify  | Move multi-shard from "What's next" to "What got done"            |
| `Cargo.toml`                                 | Modify  | Bump version `0.2.0` → `0.2.1`                                    |

The split between `checkpoint.rs` (resolution / format detection) and `loader.rs` (mmap + parse) keeps each module focused on one job. `checkpoint::resolve()` is pure path-logic, easy to unit-test without filesystem fixtures for most cases (only the directory cases need temp dirs).

---

## Task 1: Migrate data model to Vec<Arc<Mmap>> + shard_index

**Goal:** Restructure `ModelSnapshot` and `TensorMeta` so the single-file path becomes a 1-shard case. No behavioural change. All 33 existing tests must remain green.

**Files:**
- Modify: `src/types.rs:7-22` (`ModelSnapshot` and `TensorMeta` structs)
- Modify: `src/loader.rs` (entire file — `load()` and `load_tensor_data()`)

- [ ] **Step 1: Update `ModelSnapshot` struct**

In `src/types.rs`, replace lines 7-13:

```rust
#[derive(Debug, Clone)]
pub struct ModelSnapshot {
    pub path: PathBuf,
    pub tensors: HashMap<String, TensorMeta>,
    pub total_params: u64,
    pub mmaps: Vec<Arc<Mmap>>,
}
```

- [ ] **Step 2: Add `shard_index` to `TensorMeta`**

In `src/types.rs`, replace lines 15-22:

```rust
#[derive(Debug, Clone)]
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data_offset: u64,
    pub data_len: u64,
    pub shard_index: u16,
}
```

- [ ] **Step 3: Run `cargo build` to surface all callers**

Run: `cargo build 2>&1 | head -30`
Expected: errors in `src/loader.rs` referencing `snapshot.mmap` and `TensorMeta { ... }` literal construction missing `shard_index`. Possibly errors in `src/tui.rs` or `src/diff.rs` if anything reads `mmap` directly.

- [ ] **Step 4: Update `load()` in `src/loader.rs`**

Replace the entire body of `pub fn load(path: &Path) -> Result<ModelSnapshot>` (lines 11-60) with:

```rust
pub fn load(path: &Path) -> Result<ModelSnapshot> {
    load_single(path)
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
```

- [ ] **Step 5: Update `load_tensor_data()` in `src/loader.rs`**

Replace lines 83-126 (the existing `pub fn load_tensor_data`) with:

```rust
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
```

- [ ] **Step 6: Run `cargo build` to confirm it compiles**

Run: `cargo build 2>&1 | tail -10`
Expected: `Finished` line, no errors. If errors mention other files (tui.rs, diff.rs, mapper.rs, scanner.rs), grep for `.mmap` (without trailing `s`) and replace remaining offending references — there should be none, but if there are, change `snapshot.mmap` to `snapshot.mmaps[0]` and report the file/line in the commit body.

- [ ] **Step 7: Run all tests to confirm zero regression**

Run: `cargo test --all-features 2>&1 | tail -20`
Expected: `test result: ok. 33 passed; 0 failed` (number may vary slightly with ratatui feature gating; key signal is **0 failed**).

- [ ] **Step 8: Commit**

```bash
git add src/types.rs src/loader.rs
git commit -m "$(cat <<'EOF'
refactor(loader): generalise ModelSnapshot to Vec<Arc<Mmap>> + per-tensor shard_index

Pre-requisite for multi-shard checkpoint support. Single-file load
now wraps as a 1-shard case (mmaps: vec![one], shard_index: 0).
No behavioural change; all 33 tests green.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 2: Create `src/checkpoint.rs` with `resolve()`

**Goal:** Pure path-logic module that decides whether an input is a single file or a sharded checkpoint. Eight unit tests, all behaviour-driven.

**Files:**
- Create: `src/checkpoint.rs`
- Modify: `src/lib.rs:9` (add `pub mod checkpoint;`)
- Create: `tests/checkpoint_tests.rs`

- [ ] **Step 1: Add module declaration**

In `src/lib.rs`, add after line 9 (`pub mod loader;`):

```rust
pub mod checkpoint;
```

- [ ] **Step 2: Create `src/checkpoint.rs` with the empty interface**

Write to `src/checkpoint.rs`:

```rust
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
```

- [ ] **Step 3: Create `tests/checkpoint_tests.rs` with all 8 tests (failing)**

Write to `tests/checkpoint_tests.rs`:

```rust
use neuraldiff::checkpoint::{resolve, CheckpointSource};
use std::fs;
use tempfile::TempDir;

#[test]
fn test_resolve_single_file() {
    let result = resolve("tests/fixtures/tiny_model_a.safetensors".as_ref())
        .expect("should resolve");
    match result {
        CheckpointSource::SingleFile(p) => {
            assert!(p.ends_with("tiny_model_a.safetensors"));
        }
        other => panic!("expected SingleFile, got {:?}", other),
    }
}

#[test]
fn test_resolve_directory_with_index() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("model.safetensors.index.json"), "{}").unwrap();
    fs::write(dir.path().join("model-00001-of-00002.safetensors"), b"x").unwrap();
    fs::write(dir.path().join("model-00002-of-00002.safetensors"), b"x").unwrap();

    let result = resolve(dir.path()).expect("should resolve");
    match result {
        CheckpointSource::Sharded { index_path, root } => {
            assert!(index_path.ends_with("model.safetensors.index.json"));
            assert_eq!(root, dir.path());
        }
        other => panic!("expected Sharded, got {:?}", other),
    }
}

#[test]
fn test_resolve_directory_single_safetensors_no_index() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("model.safetensors"), b"x").unwrap();

    let result = resolve(dir.path()).expect("should resolve");
    match result {
        CheckpointSource::SingleFile(p) => {
            assert!(p.ends_with("model.safetensors"));
        }
        other => panic!("expected SingleFile, got {:?}", other),
    }
}

#[test]
fn test_resolve_directory_multi_no_index_errors() {
    let dir = TempDir::new().unwrap();
    fs::write(dir.path().join("a.safetensors"), b"x").unwrap();
    fs::write(dir.path().join("b.safetensors"), b"x").unwrap();

    let err = resolve(dir.path()).expect_err("should error");
    let msg = format!("{}", err);
    assert!(
        msg.contains("2 safetensors files") && msg.contains("no index.json"),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn test_resolve_index_json_direct() {
    let dir = TempDir::new().unwrap();
    let index = dir.path().join("model.safetensors.index.json");
    fs::write(&index, "{}").unwrap();

    let result = resolve(&index).expect("should resolve");
    match result {
        CheckpointSource::Sharded { index_path, root } => {
            assert_eq!(index_path, index);
            assert_eq!(root, dir.path());
        }
        other => panic!("expected Sharded, got {:?}", other),
    }
}

#[test]
fn test_resolve_shard_pattern_with_parent_index() {
    let dir = TempDir::new().unwrap();
    let shard = dir.path().join("model-00001-of-00002.safetensors");
    fs::write(&shard, b"x").unwrap();
    fs::write(dir.path().join("model-00002-of-00002.safetensors"), b"x").unwrap();
    fs::write(dir.path().join("model.safetensors.index.json"), "{}").unwrap();

    let result = resolve(&shard).expect("should resolve");
    match result {
        CheckpointSource::Sharded { index_path, root } => {
            assert!(index_path.ends_with("model.safetensors.index.json"));
            assert_eq!(root, dir.path());
        }
        other => panic!("expected Sharded via parent index, got {:?}", other),
    }
}

#[test]
fn test_resolve_shard_pattern_no_parent_index_falls_back_to_single() {
    let dir = TempDir::new().unwrap();
    let shard = dir.path().join("model-00001-of-00002.safetensors");
    fs::write(&shard, b"x").unwrap();

    let result = resolve(&shard).expect("should resolve");
    match result {
        CheckpointSource::SingleFile(p) => assert_eq!(p, shard),
        other => panic!("expected SingleFile fallback, got {:?}", other),
    }
}

#[test]
fn test_resolve_missing_path() {
    let err = resolve("/nonexistent/path/foo.safetensors".as_ref()).expect_err("should error");
    assert!(format!("{}", err).contains("path not found"));
}
```

- [ ] **Step 4: Run the new tests — confirm they pass**

Run: `cargo test --test checkpoint_tests 2>&1 | tail -20`
Expected: `test result: ok. 8 passed; 0 failed`.

If the build fails because `regex` is not imported as a direct dependency at top-level — it is (see `Cargo.toml:48`); the import in checkpoint.rs is fine.

- [ ] **Step 5: Run the full test suite to confirm no regression**

Run: `cargo test --all-features 2>&1 | tail -5`
Expected: 41 tests passed (33 baseline + 8 new), 0 failed.

- [ ] **Step 6: Commit**

```bash
git add src/checkpoint.rs src/lib.rs tests/checkpoint_tests.rs
git commit -m "$(cat <<'EOF'
feat(checkpoint): add resolve() — auto-detect single-file vs sharded inputs

New module src/checkpoint.rs with CheckpointSource enum and resolve()
function. Handles 8 input variants (single file, directory ±index.json,
shard pattern, etc.) covered by 8 unit tests.

Not yet wired into loader::load — that's the next task.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 3: Add `load_sharded()` with fixture helper

**Goal:** Implement the actual sharded load. The fixture helper splits an existing tiny safetensors fixture into two shards in a temp dir, so tests don't need binary fixtures committed to the repo.

**Files:**
- Modify: `src/loader.rs` (add `load_sharded()` and dependencies)
- Create: `tests/multishard_tests.rs`

- [ ] **Step 1: Add `serde_json::Value` parsing for index.json — no new dep needed**

`serde_json` is already in Cargo.toml:45. No change.

- [ ] **Step 2: Write the fixture helper + first sharded test (failing)**

Write to `tests/multishard_tests.rs`:

```rust
use neuraldiff::loader::{load, load_tensor_data};
use neuraldiff::types::ModelSnapshot;
use safetensors::{tensor::TensorView, Dtype, SafeTensors};
use std::collections::HashMap;
use std::fs;
use std::path::PathBuf;
use tempfile::TempDir;

/// Build a sharded fixture from `tiny_model_a.safetensors` by splitting its tensors
/// alphabetically into two halves and emitting an index.json that points to them.
/// Returns (TempDir, root_path).
fn setup_sharded_fixture() -> (TempDir, PathBuf) {
    let raw = fs::read("tests/fixtures/tiny_model_a.safetensors")
        .expect("fixture must exist");
    let st = SafeTensors::deserialize(&raw).expect("parse fixture");

    let mut names: Vec<String> = st.names().into_iter().map(String::from).collect();
    names.sort();
    let mid = names.len() / 2;
    let (first, second) = names.split_at(mid);

    let dir = TempDir::new().unwrap();
    let root = dir.path().to_path_buf();

    // Build per-shard tensor lists. Each entry is (name, dtype, shape, raw bytes).
    // SafeTensors::serialize takes a HashMap with TensorView values that borrow data.
    let collect_views = |selected: &[String]| -> Vec<(String, (Dtype, Vec<usize>, Vec<u8>))> {
        selected
            .iter()
            .map(|n| {
                let v = st.tensor(n).expect("tensor exists");
                (
                    n.clone(),
                    (v.dtype(), v.shape().to_vec(), v.data().to_vec()),
                )
            })
            .collect()
    };

    let shard1_data = collect_views(first);
    let shard2_data = collect_views(second);

    let serialize_shard = |data: &[(String, (Dtype, Vec<usize>, Vec<u8>))], path: &PathBuf| {
        let mut map: HashMap<String, TensorView> = HashMap::new();
        for (name, (dtype, shape, bytes)) in data {
            map.insert(
                name.clone(),
                TensorView::new(*dtype, shape.clone(), bytes).expect("build view"),
            );
        }
        let bytes = safetensors::serialize(&map, &None).expect("serialize");
        fs::write(path, bytes).unwrap();
    };

    let shard1_path = root.join("model-00001-of-00002.safetensors");
    let shard2_path = root.join("model-00002-of-00002.safetensors");
    serialize_shard(&shard1_data, &shard1_path);
    serialize_shard(&shard2_data, &shard2_path);

    // Build index.json
    let mut weight_map: HashMap<String, String> = HashMap::new();
    for n in first {
        weight_map.insert(n.clone(), "model-00001-of-00002.safetensors".to_string());
    }
    for n in second {
        weight_map.insert(n.clone(), "model-00002-of-00002.safetensors".to_string());
    }
    let total_size: u64 = shard1_data
        .iter()
        .chain(shard2_data.iter())
        .map(|(_, (_, _, b))| b.len() as u64)
        .sum();
    let index = serde_json::json!({
        "metadata": { "total_size": total_size },
        "weight_map": weight_map,
    });
    fs::write(
        root.join("model.safetensors.index.json"),
        serde_json::to_vec_pretty(&index).unwrap(),
    )
    .unwrap();

    (dir, root)
}

#[test]
fn test_load_sharded_two_shards() {
    let (_dir, root) = setup_sharded_fixture();
    let snapshot: ModelSnapshot = load(&root).expect("load sharded");

    assert_eq!(snapshot.tensors.len(), 19, "all 19 tensors recovered");
    assert_eq!(snapshot.mmaps.len(), 2, "two shards mmapped");
    assert!(snapshot.total_params > 0);

    let shard_indices: Vec<u16> = snapshot
        .tensors
        .values()
        .map(|m| m.shard_index)
        .collect();
    assert!(
        shard_indices.contains(&0) && shard_indices.contains(&1),
        "tensors should span both shards"
    );
}

#[test]
fn test_load_sharded_matches_single_file() {
    let single = load("tests/fixtures/tiny_model_a.safetensors".as_ref())
        .expect("single load");
    let (_dir, root) = setup_sharded_fixture();
    let sharded = load(&root).expect("sharded load");

    assert_eq!(single.tensors.len(), sharded.tensors.len());
    assert_eq!(single.total_params, sharded.total_params);

    for (name, single_meta) in &single.tensors {
        let sharded_meta = sharded.tensors.get(name).expect("name in both");
        assert_eq!(single_meta.shape, sharded_meta.shape, "shape match for {name}");
        assert_eq!(single_meta.dtype, sharded_meta.dtype, "dtype match for {name}");

        let a = load_tensor_data(&single, name).unwrap();
        let b = load_tensor_data(&sharded, name).unwrap();
        assert_eq!(a, b, "tensor data match for {name}");
    }
}

#[test]
fn test_load_sharded_missing_shard_errors() {
    let (_dir, root) = setup_sharded_fixture();
    fs::remove_file(root.join("model-00002-of-00002.safetensors")).unwrap();

    let err = load(&root).expect_err("should error on missing shard");
    let msg = format!("{}", err);
    assert!(
        msg.contains("missing shard") || msg.contains("model-00002-of-00002"),
        "unexpected error: {}",
        msg
    );
}

#[test]
fn test_load_tensor_data_routes_to_correct_shard() {
    let (_dir, root) = setup_sharded_fixture();
    let snapshot = load(&root).expect("load sharded");

    // Pick one tensor from each shard and confirm load_tensor_data succeeds.
    let mut from_shard0: Option<String> = None;
    let mut from_shard1: Option<String> = None;
    for (name, meta) in &snapshot.tensors {
        match meta.shard_index {
            0 if from_shard0.is_none() => from_shard0 = Some(name.clone()),
            1 if from_shard1.is_none() => from_shard1 = Some(name.clone()),
            _ => {}
        }
        if from_shard0.is_some() && from_shard1.is_some() {
            break;
        }
    }
    let s0 = from_shard0.expect("at least one tensor in shard 0");
    let s1 = from_shard1.expect("at least one tensor in shard 1");

    let data0 = load_tensor_data(&snapshot, &s0).expect("read from shard 0");
    let data1 = load_tensor_data(&snapshot, &s1).expect("read from shard 1");
    assert!(!data0.is_empty());
    assert!(!data1.is_empty());
}

#[test]
fn test_total_params_sums_across_shards() {
    let single = load("tests/fixtures/tiny_model_a.safetensors".as_ref())
        .expect("single load");
    let (_dir, root) = setup_sharded_fixture();
    let sharded = load(&root).expect("sharded load");

    assert_eq!(single.total_params, sharded.total_params);
}
```

- [ ] **Step 3: Run the new test file — confirm it fails because `load_sharded()` is unwired and `load(&dir)` would currently call `load_single` which fails on a directory**

Run: `cargo test --test multishard_tests 2>&1 | tail -20`
Expected: tests fail because `load(&root)` (a directory) returns an error from `File::open` — exactly the gap we're filling.

- [ ] **Step 4: Implement `load_sharded()` in `src/loader.rs`**

Add at the bottom of `src/loader.rs`:

```rust
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

        // Verify expected tensors are present in this shard.
        let actual_names: std::collections::HashSet<&str> = parsed
            .tensors()
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
        for (name, _) in parsed.tensors() {
            if !expected_tensors.iter().any(|e| e == name) {
                tracing::warn!(
                    "tensor '{}' present in {} but absent from index.json — keeping anyway",
                    name,
                    shard_path.display()
                );
            }
        }

        for (name, view) in parsed.tensors() {
            if tensor_map.contains_key(name) {
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
```

- [ ] **Step 5: Wire `load()` to dispatch via `checkpoint::resolve()`**

In `src/loader.rs`, replace the current `load()` function (the one-liner that calls `load_single`):

```rust
pub fn load(path: &Path) -> Result<ModelSnapshot> {
    use crate::checkpoint::{resolve, CheckpointSource};
    match resolve(path)? {
        CheckpointSource::SingleFile(p) => load_single(&p),
        CheckpointSource::Sharded { index_path, root } => load_sharded(&index_path, &root),
    }
}
```

- [ ] **Step 6: Run multishard tests — confirm they pass**

Run: `cargo test --test multishard_tests 2>&1 | tail -10`
Expected: `test result: ok. 5 passed; 0 failed`.

- [ ] **Step 7: Run full suite — confirm zero regression**

Run: `cargo test --all-features 2>&1 | tail -5`
Expected: 46 tests passed (33 + 8 + 5), 0 failed.

- [ ] **Step 8: Smoke test on real Qwen3 — confirm single-file still works end-to-end**

Run: `cargo build --release 2>&1 | tail -3`
Expected: Finished release build.

Run:
```bash
./target/release/neuraldiff summary \
    /mnt/c/Users/remic/Qwen3-0.6B-Base/model.safetensors \
    /mnt/c/Users/remic/Qwen3-0.6B/model.safetensors \
    -n 5 2>&1 | tail -20
```
Expected: a normal summary output (top-5 changed layers). If it errors, single-file regression — investigate before commit.

- [ ] **Step 9: Commit**

```bash
git add src/loader.rs tests/multishard_tests.rs
git commit -m "$(cat <<'EOF'
feat(loader): multi-shard safetensors support

load_sharded() reads model.safetensors.index.json, mmaps each shard,
and merges tensors into a single ModelSnapshot. load() now dispatches
via checkpoint::resolve() so any of: directory, index.json, single
.safetensors, or shard file works as input.

Tests: 5 new in tests/multishard_tests.rs covering load equivalence
with single-file, missing-shard error, per-shard routing of
load_tensor_data, and total_params accounting.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 4: Update scanner to surface sharded directories

**Goal:** When the recursive scanner encounters a directory containing `model.safetensors.index.json`, list the directory itself as a model entry and skip its individual shards.

**Files:**
- Modify: `src/scanner.rs` (function `scan_dir_recursive`)

- [ ] **Step 1: Update `scan_dir_recursive` to detect index.json**

In `src/scanner.rs`, replace the body of `scan_dir_recursive` (lines 34-77) with:

```rust
fn scan_dir_recursive(
    dir: &Path,
    depth: usize,
    max_depth: usize,
    seen: &mut HashSet<PathBuf>,
    models: &mut Vec<ModelInfo>,
) {
    if depth > max_depth {
        return;
    }

    // If this dir is itself a sharded model, surface it as one entry and don't recurse.
    let index_file = dir.join("model.safetensors.index.json");
    if index_file.is_file() {
        if seen.insert(dir.to_path_buf()) {
            let total_size_mb = total_safetensors_size_mb(dir);
            let name = dir
                .file_name()
                .and_then(|n| n.to_str())
                .unwrap_or("unknown")
                .to_string();
            let location = format_location(dir.parent().unwrap_or(dir));
            models.push(ModelInfo {
                path: dir.to_path_buf(),
                name,
                size_mb: total_size_mb,
                location,
            });
        }
        return;
    }

    let Ok(entries) = std::fs::read_dir(dir) else { return };

    for entry in entries.flatten() {
        let path = entry.path();

        if path.is_dir() {
            let name = path.file_name().and_then(|n| n.to_str()).unwrap_or("");
            if name.starts_with('.') && name != ".cache" {
                continue;
            }
            if is_skip_dir(name) {
                continue;
            }
            scan_dir_recursive(&path, depth + 1, max_depth, seen, models);
        } else if path.extension().is_some_and(|ext| ext == "safetensors") {
            if seen.insert(path.clone()) {
                if let Ok(meta) = path.metadata() {
                    let size_mb = meta.len() as f64 / (1024.0 * 1024.0);
                    let name = path
                        .file_stem()
                        .and_then(|s| s.to_str())
                        .unwrap_or("unknown")
                        .to_string();
                    let location =
                        format_location(path.parent().unwrap_or(dir));
                    models.push(ModelInfo { path, name, size_mb, location });
                }
            }
        }
    }
}

/// Sum the size in MB of every .safetensors file at the top level of `dir`.
/// Used to display a meaningful size for sharded directories.
fn total_safetensors_size_mb(dir: &Path) -> f64 {
    let Ok(entries) = std::fs::read_dir(dir) else { return 0.0 };
    let mut total: u64 = 0;
    for entry in entries.flatten() {
        let path = entry.path();
        if path.is_file() && path.extension().is_some_and(|e| e == "safetensors") {
            if let Ok(meta) = path.metadata() {
                total += meta.len();
            }
        }
    }
    total as f64 / (1024.0 * 1024.0)
}
```

- [ ] **Step 2: Build to confirm it compiles**

Run: `cargo build 2>&1 | tail -3`
Expected: Finished.

- [ ] **Step 3: Run all tests — confirm zero regression**

Run: `cargo test --all-features 2>&1 | tail -5`
Expected: 46 tests passed, 0 failed.

- [ ] **Step 4: Manual verification on real disk (optional but recommended)**

Run:
```bash
./target/release/neuraldiff scan --root ~/.cache/huggingface --depth 8 --json 2>&1 \
  | python3 -c 'import json,sys; d=json.load(sys.stdin); [print(m["name"], m["path"]) for m in d if "gemma" in m["path"].lower()][:5]' 2>&1 | head -10
```
Expected: a single Gemma entry pointing at the directory (not 2 entries pointing at individual shards). If still listing per shard, the index.json detection is wrong — investigate.

- [ ] **Step 5: Commit**

```bash
git add src/scanner.rs
git commit -m "$(cat <<'EOF'
feat(scanner): surface sharded checkpoints as one entry

When a directory contains model.safetensors.index.json, list the
directory itself instead of each shard file. Total size is the sum
of all .safetensors files at that level.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

---

## Task 5: E2E smoke test on Gemma 31B + bump version + update CONTEXT.md

**Goal:** Validate the multi-shard load works on a real 31B-parameter model (self-diff = all-zero deltas), bump version to 0.2.1, document the milestone.

**Files:**
- Modify: `Cargo.toml:3` (`version = "0.2.0"` → `"0.2.1"`)
- Modify: `CONTEXT.md` (move multi-shard from "What's next" to "What got done")

- [ ] **Step 1: Manual smoke — self-diff Gemma 31B**

Run:
```bash
GEMMA=$(ls -d ~/.cache/huggingface/hub/models--google--gemma-4-31b-it/snapshots/*/ | head -1)
echo "Using: $GEMMA"
./target/release/neuraldiff summary "$GEMMA" "$GEMMA" -n 3 2>&1 | tail -25
```
Expected: a summary printed without error, with `change_ratio_percent ≈ 0%` and `mean_delta ≈ 0`. Total parameters reported should be in the 30-billion range. **Record the elapsed seconds reported by `tracing` (visible on stderr) for the commit message.**

If the command errors with "directory has N safetensors files but no index.json": the snapshot dir layout is non-standard — fall back to passing the index.json explicitly:
```bash
./target/release/neuraldiff summary "$GEMMA/model.safetensors.index.json" "$GEMMA/model.safetensors.index.json" -n 3
```

- [ ] **Step 2: Bump version to 0.2.1 in Cargo.toml**

Edit `Cargo.toml` line 3:

```toml
version = "0.2.1"
```

- [ ] **Step 3: Update CONTEXT.md to reflect new state**

In `CONTEXT.md`:

1. Top header: change `Latest release: **v0.2.0**` to `Latest release: **v0.2.1**` and `Last updated: 2026-05-01` to `Last updated: 2026-05-05`.

2. In the "What's next" section, **remove** the entire bullet:
   ```
   - **Multi-shard support** — read `model.safetensors.index.json`, ...
   ```

3. Add a new section just below "## What got done (session 2026-05-01)" titled `## What got done (session 2026-05-05)`:

   ```markdown
   ## What got done (session 2026-05-05)

   - **Multi-shard support** — `neuraldiff diff` now accepts directories,
     index.json files, or shard files as input. New `src/checkpoint.rs`
     resolves any of those to a `CheckpointSource::SingleFile | Sharded`,
     and `loader::load_sharded()` mmaps every shard, merging tensors into
     a unified `ModelSnapshot`.
   - **Data model**: `ModelSnapshot.mmap: Arc<Mmap>` → `mmaps: Vec<Arc<Mmap>>`,
     `TensorMeta` gains `shard_index: u16` (max 65535 shards).
   - **Scanner**: directories with `model.safetensors.index.json` are now
     listed as one entry instead of per-shard.
   - **Tests**: 13 new (8 in `tests/checkpoint_tests.rs`, 5 in
     `tests/multishard_tests.rs`). Total: 46.
   - **Verified end-to-end on Gemma 31B** (self-diff, ~30B params).
   ```

- [ ] **Step 4: Run the full test suite one last time**

Run: `cargo test --all-features 2>&1 | tail -5`
Expected: 46 passed, 0 failed.

- [ ] **Step 5: Commit version bump + CONTEXT.md update**

```bash
git add Cargo.toml Cargo.lock CONTEXT.md
git commit -m "$(cat <<'EOF'
chore: bump to v0.2.1 — multi-shard support

Validated on Gemma 31B self-diff (all-zero deltas as expected,
~30B params loaded across multiple shards). CONTEXT.md updated
with the 2026-05-05 session log.

Co-Authored-By: Claude Opus 4.7 (1M context) <noreply@anthropic.com>
EOF
)"
```

- [ ] **Step 6: Tag and push**

```bash
git tag -a v0.2.1 -m "v0.2.1 — multi-shard checkpoint support"
TOKEN=$(gh.exe auth token | tr -d '\r\n')
git push "https://x-access-token:${TOKEN}@github.com/Zertannax/neuraldiff.git" master
git push "https://x-access-token:${TOKEN}@github.com/Zertannax/neuraldiff.git" v0.2.1
```

Expected: both pushes succeed. CI on GitHub should kick off automatically; check at `gh.exe run list --repo Zertannax/neuraldiff --limit 3` after ~30s.

- [ ] **Step 7: (Optional) Create GitHub release**

```bash
gh.exe release create v0.2.1 \
  --repo Zertannax/neuraldiff \
  --title "v0.2.1 — Multi-shard checkpoint support" \
  --notes "$(cat <<'EOF'
## Highlights

- **Multi-shard checkpoint support.** `neuraldiff diff` now works on HuggingFace-format sharded models (Llama 70B, Gemma 31B, DeepSeek-R1, ...). Pass a directory, an `index.json`, or even a single shard file and the tool figures out the rest.
- 13 new tests, full backward compatibility with single-file inputs.
- Verified end-to-end on Gemma 31B.
EOF
)"
```

---

## Self-Review

**Spec coverage:**
- [x] CLI accepts 4 input variants (Task 2 — 8 unit tests cover all)
- [x] `ModelSnapshot.mmaps: Vec<Arc<Mmap>>` + `TensorMeta.shard_index: u16` (Task 1)
- [x] `src/checkpoint.rs::resolve()` with all resolution rules (Task 2)
- [x] `load_sharded()` reads index.json, mmaps shards, merges tensors (Task 3)
- [x] Error handling — missing shard, name collision, extra tensor warning (Task 3 step 4)
- [x] Scanner lists sharded dirs as one entry (Task 4)
- [x] Programmatic fixture helper, no binary fixtures committed (Task 3 step 2)
- [x] E2E smoke on Gemma 31B (Task 5 step 1)
- [x] Backward compat: single-file Qwen3 still works (Task 3 step 8)
- [x] CONTEXT.md updated (Task 5 step 3)
- [x] Version bump (Task 5 step 2)

**Placeholder scan:** all "Expected:" lines describe concrete output; no "TBD"/"TODO"/"add error handling".

**Type consistency:**
- `CheckpointSource::Sharded { index_path, root }` — same field names in checkpoint.rs (Task 2 step 2), tests (Task 2 step 3), and `load()` dispatch (Task 3 step 5). ✓
- `mmaps: Vec<Arc<Mmap>>` — same name in types.rs (Task 1 step 1), load_single (Task 1 step 4), load_sharded (Task 3 step 4). ✓
- `shard_index: u16` — same type and name everywhere it appears. ✓
- `decode_dtype` defined in Task 1 step 4, reused in Task 3 step 4. ✓

No issues found.

---

## Execution Handoff

Plan complete and saved to `docs/superpowers/plans/2026-05-05-multishard-support.md`. Two execution options:

**1. Subagent-Driven (recommended)** — I dispatch a fresh subagent per task, review between tasks, fast iteration.

**2. Inline Execution** — Execute tasks in this session using executing-plans, batch execution with checkpoints.

Which approach?
