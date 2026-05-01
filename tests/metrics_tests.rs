use neuraldiff::metrics::{cosine_similarity, l2_norm};

#[test]
fn test_l2_norm_zero_vector() {
    assert_eq!(l2_norm(&[0.0, 0.0, 0.0]), 0.0);
}

#[test]
fn test_l2_norm_unit_vector() {
    let norm = l2_norm(&[1.0, 0.0, 0.0]);
    assert!((norm - 1.0).abs() < 1e-6, "Expected 1.0, got {norm}");
}

#[test]
fn test_l2_norm_known_value() {
    // sqrt(3^2 + 4^2) = 5
    let norm = l2_norm(&[3.0, 4.0]);
    assert!((norm - 5.0).abs() < 1e-6, "Expected 5.0, got {norm}");
}

#[test]
fn test_l2_norm_empty() {
    assert_eq!(l2_norm(&[]), 0.0);
}

#[test]
fn test_cosine_similarity_identical() {
    let v = vec![1.0, 2.0, 3.0];
    let sim = cosine_similarity(&v, &v);
    assert!((sim - 1.0).abs() < 1e-6, "Identical vectors should have cosine = 1.0, got {sim}");
}

#[test]
fn test_cosine_similarity_orthogonal() {
    let a = vec![1.0, 0.0, 0.0];
    let b = vec![0.0, 1.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!(sim.abs() < 1e-6, "Orthogonal vectors should have cosine = 0.0, got {sim}");
}

#[test]
fn test_cosine_similarity_opposite() {
    let a = vec![1.0, 0.0];
    let b = vec![-1.0, 0.0];
    let sim = cosine_similarity(&a, &b);
    assert!((sim + 1.0).abs() < 1e-6, "Opposite vectors should have cosine = -1.0, got {sim}");
}

#[test]
fn test_cosine_similarity_both_zero() {
    let sim = cosine_similarity(&[0.0, 0.0], &[0.0, 0.0]);
    assert_eq!(sim, 1.0, "Two zero vectors should be treated as identical");
}

#[test]
fn test_cosine_similarity_one_zero() {
    let sim = cosine_similarity(&[0.0, 0.0], &[1.0, 0.0]);
    assert_eq!(sim, 0.0, "One zero vector vs non-zero should give 0.0");
}
