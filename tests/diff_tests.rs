use neuraldiff::diff::compute_diff;

#[test]
fn test_compute_diff_fixtures() {
    let result = compute_diff(
        "tests/fixtures/tiny_model_a.safetensors".as_ref(),
        "tests/fixtures/tiny_model_b.safetensors".as_ref(),
    ).expect("Failed to compute diff");

    // Should have layers
    assert!(!result.layers.is_empty(), "Expected at least one layer");
    
    // Should have detected changes
    assert!(result.summary.changed_layers > 0, "Expected some layers to have changed");
    
    // Max delta should be significant (model B has large modifications)
    assert!(result.summary.max_delta > 0.1, "Expected significant max delta");
    
    // Should have model paths
    assert!(result.model_a.is_some());
    assert!(result.model_b.is_some());
}

#[test]
fn test_diff_summary_structure() {
    let result = compute_diff(
        "tests/fixtures/tiny_model_a.safetensors".as_ref(),
        "tests/fixtures/tiny_model_b.safetensors".as_ref(),
    ).expect("Failed to compute diff");

    assert!(result.summary.total_layers > 0);
    assert_eq!(
        result.summary.changed_layers + result.summary.unchanged_layers,
        result.summary.total_layers
    );
    
    // Top changed should not be empty if there are changes
    if result.summary.changed_layers > 0 {
        assert!(!result.summary.top_changed_indices.is_empty());
    }
}
