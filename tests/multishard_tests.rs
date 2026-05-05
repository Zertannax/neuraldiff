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
