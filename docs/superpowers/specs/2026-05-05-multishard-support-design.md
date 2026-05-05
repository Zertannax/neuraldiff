# Multi-shard checkpoint support — design

> Date: 2026-05-05
> Status: approved (pending implementation plan)
> Target version: v0.2.1
> Roadmap: Phase 1 (highest impact item — unblocks Llama 70B / Gemma 31B)

---

## Goal

Allow `neuraldiff diff` to operate on safetensors checkpoints split across multiple
shards (the standard HuggingFace layout for models > ~10GB). Today the loader
assumes a single `.safetensors` file backed by a single mmap; this design
generalises the data model to N shards while keeping the single-file path
backward compatible and the public CLI / JSON schema unchanged.

## Non-goals

- GGUF support (Phase 4)
- PyTorch pickle / `.bin` support (Phase 4)
- Streaming for models that exceed physical RAM (Phase 4 — mmap already handles
  this implicitly via OS paging)
- Validating SHA checksums in `index.json`
- Rich shard metadata in the scanner display ("X shards, Y params") — can be
  added later, kept out of v0.2.1 to keep scope tight

## User-visible behaviour

### CLI inputs accepted (auto-detect)

```bash
# Single file (current behaviour, backward compatible)
neuraldiff diff a.safetensors b.safetensors

# Directory containing index.json + shards
neuraldiff diff /path/Llama-70B/ /path/Llama-70B-ft/

# Index file pointed to directly
neuraldiff diff a/model.safetensors.index.json b/model.safetensors.index.json

# A single shard file — tool walks up to the parent and uses its index.json
neuraldiff diff /path/Llama-70B/model-00001-of-00012.safetensors ...
```

The CLI surface (`Diff`, `Summary`, `Inspect`, `Scan`) gains no new flags. Each
positional path is run through a `resolve()` step that picks the right loading
strategy.

### Public JSON / CSV schema

Unchanged. `model_a` / `model_b` will be the resolved root path:
- single-file: the `.safetensors` path (as today)
- sharded: the directory path (or the `index.json` path if the user passed it
  explicitly)

This is the only externally observable change. No keys added, no keys renamed.

### Scanner / TUI

The recursive scanner currently surfaces every `.safetensors` file as one
entry. After this change:

- A directory containing `model.safetensors.index.json` is surfaced as **one
  entry** (the directory itself); individual shards inside it are not listed.
- A directory with a single `.safetensors` and no index file behaves as today
  (the `.safetensors` is listed).
- Standalone shards outside a sharded directory (rare) are still listed
  individually.

Display label is the directory/file path as today. No "X shards" annotation in
v0.2.1 (defer to Phase 3 polish).

---

## Internal data model

### `ModelSnapshot` (in `src/types.rs`)

```rust
pub struct ModelSnapshot {
    pub path: PathBuf,                // root: directory for sharded, file for single
    pub tensors: HashMap<String, TensorMeta>,
    pub total_params: u64,
    pub mmaps: Vec<Arc<Mmap>>,        // CHANGED: was Arc<Mmap>
}
```

### `TensorMeta` (in `src/types.rs`)

```rust
pub struct TensorMeta {
    pub name: String,
    pub shape: Vec<usize>,
    pub dtype: DType,
    pub data_offset: u64,
    pub data_len: u64,
    pub shard_index: u16,             // NEW: index into ModelSnapshot.mmaps
}
```

`shard_index: u16` — chosen to keep `TensorMeta` small. `u16` allows up to
65 535 shards; the largest known public model (DeepSeek-R1 671B) ships in
~80 shards, so this ceiling is comfortable.

### Loader contract

`load_tensor_data(snapshot, name)` reads from
`snapshot.mmaps[meta.shard_index as usize].get(start..end)` rather than
`snapshot.mmap.get(...)`. All other consumers of `ModelSnapshot` (diff.rs,
mapper.rs, tui.rs) are unaffected — they only read `tensors` and `total_params`.

Single-file load returns `mmaps: vec![mmap]` and `shard_index: 0` for every
tensor → uniform code path with sharded.

---

## New module: `src/checkpoint.rs`

Owns the input-resolution logic. Defines:

```rust
pub enum CheckpointSource {
    SingleFile(PathBuf),
    Sharded { index_path: PathBuf, root: PathBuf },
}

pub fn resolve(path: &Path) -> Result<CheckpointSource>;
```

### Resolution rules

| Input                                              | Result                                                                          |
|----------------------------------------------------|---------------------------------------------------------------------------------|
| `foo.safetensors` (regular file)                   | `SingleFile(foo.safetensors)`                                                   |
| `model.safetensors.index.json`                     | `Sharded { index_path, root: parent }`                                          |
| Directory containing `model.safetensors.index.json`| `Sharded { index_path = dir/model.safetensors.index.json, root: dir }`          |
| Directory with exactly one `.safetensors`, no index, name does not match shard pattern | `SingleFile(that_one)`                                            |
| Directory with exactly one `.safetensors`, no index, name matches shard pattern | `SingleFile(that_one)` + `tracing::warn!("looks like an incomplete sharded checkpoint")` |
| Directory with multiple `.safetensors`, no index   | Error: `"directory has N safetensors files but no index.json — pass one explicitly"` |
| `model-00001-of-00012.safetensors` (shard pattern) | If parent has `index.json` → `Sharded { ..., root: parent }`; else SingleFile + warning |
| Path does not exist                                | Error: `"path not found: ..."`                                                  |
| Anything else (e.g. `.txt`, `.bin`)                | Error: `"unrecognised checkpoint format: ..."`                                  |

The shard-pattern detection uses regex `^model-\d{5}-of-\d{5}\.safetensors$`
(HuggingFace standard). Non-standard patterns are not auto-grouped.

---

## Loader changes (`src/loader.rs`)

Public entry point becomes a dispatcher:

```rust
pub fn load(path: &Path) -> Result<ModelSnapshot> {
    match crate::checkpoint::resolve(path)? {
        CheckpointSource::SingleFile(p) => load_single(&p),
        CheckpointSource::Sharded { index_path, root } => load_sharded(&index_path, &root),
    }
}
```

### `load_single` (refactor of current code)

Identical to today, except:
- builds `mmaps: vec![Arc::new(mmap)]`
- sets `shard_index: 0` on every TensorMeta

### `load_sharded` (new)

```text
1. Read index.json. Parse `weight_map: {tensor_name: shard_filename}`.
2. Group: shard_filename -> Vec<tensor_name>
3. For each unique shard_filename (sorted for stable output):
     - open root.join(shard_filename)
     - mmap it
     - SafeTensors::deserialize
     - for each tensor in this shard:
         - extract shape, dtype, offset within this shard's mmap
         - insert TensorMeta { ..., shard_index: i }
4. Sum total_params, return ModelSnapshot { path: root, tensors, total_params, mmaps }
```

### Error handling

| Condition                                             | Behaviour                                            |
|-------------------------------------------------------|------------------------------------------------------|
| `index.json` missing fields / invalid JSON            | `bail!` with file path + parse error                 |
| Shard file referenced in index but not on disk        | `bail!("missing shard: {filename}")`                 |
| Tensor in index but absent from its shard             | `bail!("index references {tensor} not in {shard}")` |
| Tensor present in shard but absent from index         | Ignore + warning via `tracing::warn!`                |
| `metadata.total_size` in index disagrees with reality | Ignored — recomputed from real shapes                |
| Shard file fails safetensors parse                    | `bail!` with shard path + parse error                |

---

## Tests

### New file `tests/multishard_tests.rs`

| Test                                              | Validates                                                |
|---------------------------------------------------|----------------------------------------------------------|
| `test_resolve_single_file`                        | `.safetensors` → `SingleFile`                            |
| `test_resolve_directory_with_index`               | dir with index.json → `Sharded`                          |
| `test_resolve_directory_single_safetensors`       | dir with one .safetensors, no index → `SingleFile`       |
| `test_resolve_directory_multi_no_index_errors`    | dir with N safetensors, no index → error                 |
| `test_resolve_index_json_direct`                  | path to index.json → `Sharded`                           |
| `test_resolve_shard_pattern_with_parent_index`    | shard file → `Sharded` via parent                        |
| `test_resolve_missing_path`                       | non-existent path → error                                |
| `test_load_sharded_two_shards`                    | load fixture with 2 shards, verify tensors / params      |
| `test_load_sharded_matches_single_file`           | same content single vs sharded → same tensor map         |
| `test_load_sharded_missing_shard_errors`          | delete one shard file → `load_sharded` bails             |
| `test_load_tensor_data_routes_to_correct_shard`   | tensor in shard 1 reads from `mmaps[1]`                  |
| `test_total_params_sums_across_shards`            | sum equals sum-of-shards                                 |

### Fixtures

Generated programmatically in a `setup_sharded_fixture()` helper (avoids
committing binary fixtures to the repo). Approach:

1. Read `tests/fixtures/tiny_model_a.safetensors` (existing).
2. Split its 19 tensors into two halves by name-sort.
3. Write each half as `tmp/sharded_<uuid>/model-0000{1,2}-of-00002.safetensors`.
4. Write `model.safetensors.index.json` with `weight_map` covering both halves
   and a `metadata.total_size` field.
5. Return the temp dir path; cleaned up via `tempfile::TempDir`.

This guarantees byte-exact equivalence between the single-file load and the
sharded load (same tensors, same dtypes, same data) — strong correctness signal.

### E2E manual smoke (not automated)

`/home/remic/.cache/huggingface/hub/models--google--gemma-4-31b-it/` contains
a real multi-shard Gemma 31B. After implementation, run `neuraldiff summary
<gemma_dir> <gemma_dir>` to validate:
- model loads (no panic, no missing shard error)
- diff against itself yields all-zero deltas (sanity check)
- timing is reasonable (target: < 60s for 31B params on the user's SSD)

A real two-model multi-shard diff is not testable today (only one sharded
model on disk). Will be validated by users in the field after release.

---

## Implementation order

Each step ends with `cargo build && cargo test` green. No step leaves the tree
in a broken state.

1. **Migrate ModelSnapshot to `Vec<Arc<Mmap>>` and TensorMeta to include
   `shard_index`.** Update `load_single` and `load_tensor_data`. All existing
   tests must stay green. → 33 tests pass, single-file behaviour unchanged.
2. **Add `src/checkpoint.rs` with `resolve()` + 7 unit tests.** No wiring yet
   into loader.
3. **Add `load_sharded()` + helper `setup_sharded_fixture()` + 5 fixture-based
   tests** in `tests/multishard_tests.rs`.
4. **Wire `loader::load()` to `checkpoint::resolve()`.** Smoke test against
   real Qwen3 checkpoint to confirm zero regression on single-file path.
5. **Scanner update**: skip individual shard files when the parent contains an
   `index.json`. Add scanner unit test if cheap.
6. **Manual smoke test on Gemma 31B** (self-diff). Document timing in commit
   message.
7. **Update `CONTEXT.md`**: move "Multi-shard support" from "What's next" to
   "What got done"; update test count.
8. **Single commit** `feat(loader): multi-shard safetensors support
   (Llama 70B / Gemma 31B unblocked)`. Optional: tag `v0.2.1`.

---

## Risks & mitigations

| ID  | Risk                                                                    | Mitigation                                                                       |
|-----|-------------------------------------------------------------------------|----------------------------------------------------------------------------------|
| R1  | `shard_index: u16` overflows                                            | 65 535 ceiling vs ~80 max in real models — non-issue                             |
| R2  | Linux mmap of 140GB virtual blows ulimit on small machines              | mmap is virtual; OS pages on demand; add fallback `bail!` if mmap fails per-shard|
| R3  | No second multi-shard model on disk for E2E diff testing                | Self-diff smoke on Gemma validates load; full diff covered by user reports later |
| R4  | `model_a` / `model_b` in JSON output now sometimes a directory path     | Documented in commit + CHANGELOG; no current consumer parses these as files      |
| R5  | Sharded fixture generation flaky on Windows CI (path separators, mmap)  | `tempfile::TempDir` is cross-platform; rely on `Path` API; CI matrix covers Win  |
| R6  | A consumer somewhere relies on `snapshot.mmap` (single field)           | Compile-time error after rename; grep + fix in step 1 — no silent breakage       |
| R7  | Tensor name collisions across shards (same name in shard A and B)       | HuggingFace guarantees unique names per index; assert + bail if collision found  |

---

## Open questions

None — all design decisions resolved during brainstorming. Implementation
plan to be drafted next via the `writing-plans` skill.
