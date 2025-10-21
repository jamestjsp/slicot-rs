//! Integration tests for MB03MY routine
//!
//! These tests verify the behavior of mb03my in a more realistic context,
//! including edge cases and comparison with expected SLICOT behavior.

use slicot_rs::mb::mb03my;

#[test]
fn test_integration_empty_array() {
    let data: Vec<f64> = vec![];
    assert_eq!(mb03my(&data), None);
}

#[test]
fn test_integration_standard_case() {
    // Test case based on typical usage in SLICOT routines
    // Example: Finding minimum diagonal element magnitude in a matrix
    let diagonal_elements = vec![5.2, -3.1, 0.8, -1.2, 4.5];
    let result = mb03my(&diagonal_elements);
    assert_eq!(result, Some(0.8));
}

#[test]
fn test_integration_all_zeros() {
    let data = vec![0.0, 0.0, 0.0, 0.0];
    assert_eq!(mb03my(&data), Some(0.0));
}

#[test]
fn test_integration_numerical_precision() {
    // Test with values that might cause precision issues
    let data = vec![1.0 + 1e-15, 1.0 - 1e-15, 1.0 + 2e-15, 1.0 - 2e-15];
    let result = mb03my(&data).unwrap();
    // The minimum should be close to 1.0 (with tiny offset)
    assert!((result - 1.0).abs() < 1e-14);
}

#[test]
fn test_integration_large_dataset() {
    // Test with a larger dataset (100 elements)
    let mut data: Vec<f64> = (1..=100).map(|i| i as f64).collect();
    // Insert a smaller value
    data.push(0.5);
    assert_eq!(mb03my(&data), Some(0.5));
}

#[test]
fn test_integration_alternating_signs() {
    // Alternating positive and negative values
    let data: Vec<f64> = (1..=10)
        .map(|i| if i % 2 == 0 { i as f64 } else { -(i as f64) })
        .collect();
    // Minimum absolute value should be 1.0 (from -1)
    assert_eq!(mb03my(&data), Some(1.0));
}

#[test]
fn test_integration_special_values() {
    // Test with very small and very large values
    let data = vec![1e-100, 1e100, 1e-50, 1e50];
    assert_eq!(mb03my(&data), Some(1e-100));
}

#[test]
fn test_integration_near_zero() {
    // Values very close to zero
    let data = vec![1e-14, -1e-15, 1e-13, -1e-16];
    assert_eq!(mb03my(&data), Some(1e-16));
}

#[test]
fn test_integration_identical_magnitudes() {
    // Multiple values with the same minimum magnitude
    let data = vec![2.0, -2.0, 3.0, -1.0, 1.0, 4.0];
    // Should return 1.0 (either from -1.0 or 1.0)
    assert_eq!(mb03my(&data), Some(1.0));
}

#[test]
fn test_integration_slice_from_larger_array() {
    // Test using a slice of a larger array (common use case)
    let large_array = vec![10.0, 20.0, 1.5, 30.0, 40.0, 2.5, 50.0];
    let slice = &large_array[2..6]; // [1.5, 30.0, 40.0, 2.5]
    assert_eq!(mb03my(slice), Some(1.5));
}

#[test]
fn test_integration_monotonic_increasing() {
    let data = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    assert_eq!(mb03my(&data), Some(1.0));
}

#[test]
fn test_integration_monotonic_decreasing() {
    let data = vec![5.0, 4.0, 3.0, 2.0, 1.0];
    assert_eq!(mb03my(&data), Some(1.0));
}

/// Test that mimics actual SLICOT usage from MB03MD routine
/// where mb03my is used to find minimum diagonal element
#[test]
fn test_integration_slicot_mb03md_usage() {
    // Simulating the call from MB03MD.f line 237:
    // THETA = MB03MY( N, Q, 1 )
    // where Q contains diagonal elements of a bidiagonal matrix
    let q = vec![4.5, 3.2, 1.8, 5.1, 2.3];
    let n = q.len();

    // In Fortran: MB03MY(N, Q, 1) means N elements, starting at Q, increment 1
    // In Rust: Just use the slice directly
    let theta = mb03my(&q[0..n]);

    assert_eq!(theta, Some(1.8));
}

#[test]
fn test_integration_negative_near_zero() {
    // Test case where negative value closest to zero
    let data = vec![5.0, 3.0, -0.1, 2.0, 4.0];
    assert_eq!(mb03my(&data), Some(0.1));
}

#[test]
fn test_integration_positive_near_zero() {
    // Test case where positive value closest to zero
    let data = vec![-5.0, -3.0, 0.1, -2.0, -4.0];
    assert_eq!(mb03my(&data), Some(0.1));
}
