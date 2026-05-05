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
