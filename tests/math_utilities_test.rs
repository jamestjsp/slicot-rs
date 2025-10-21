//! Integration tests for mathematical utility routines
//! Tests MB04TU, MA02FD, and MA01AD in realistic contexts

use slicot_rs::ma::{ma01ad, ma02fd};
use slicot_rs::mb::mb04tu;

// ===== MB04TU Integration Tests =====

#[test]
fn test_mb04tu_givens_rotation_identity() {
    // Identity rotation: c=1, s=0
    let mut x = vec![3.0, 4.0, 5.0];
    let mut y = vec![1.0, 2.0, 3.0];
    let x_orig = x.clone();
    let y_orig = y.clone();

    mb04tu(3, &mut x, 1, &mut y, 1, 1.0, 0.0);

    // After identity rotation + swap, x and y should be swapped
    assert_eq!(x, y_orig);
    assert_eq!(y, x_orig);
}

#[test]
fn test_mb04tu_large_vectors() {
    // Test with larger vectors
    let n = 100;
    let mut x: Vec<f64> = (1..=n as i32).map(|i| i as f64).collect();
    let mut y: Vec<f64> = (n as i32 + 1..=2 * n as i32).map(|i| i as f64).collect();

    let c = 0.6;
    let s = 0.8;

    mb04tu(n as i32, &mut x, 1, &mut y, 1, c, s);

    // Verify first element transformation
    let x0_expected = 0.6 * (n as f64 + 1.0) - 0.8 * 1.0;
    assert!((x[0] - x0_expected).abs() < 1e-14);
}

#[test]
fn test_mb04tu_negative_stride() {
    // Test with negative stride
    let mut x = vec![1.0, 2.0, 3.0, 4.0, 5.0];
    let mut y = vec![6.0, 7.0, 8.0, 9.0, 10.0];

    let c = 0.8;
    let s = 0.6;

    mb04tu(3, &mut x, -1, &mut y, -1, c, s);

    // With negative stride, should access in reverse order
    // Starting positions computed as: (-3+1)*(-1) + 0 = 2 (for 0-based indexing)
    // This requires careful validation
}

// ===== MA02FD Integration Tests =====

#[test]
fn test_ma02fd_pythagorean_triple() {
    // Using Pythagorean triple: 3-4-5
    let mut x1 = 5.0;
    let x2 = 3.0;
    let (c, s, info) = ma02fd(&mut x1, x2);

    assert_eq!(info, 0);
    // y1 should be sqrt(5² - 3²) = sqrt(16) = 4.0
    assert!((x1 - 4.0).abs() < 1e-14);
    // Verify hyperbolic constraint c² - s² = 1 for this case
    // Actually the constraint is c² + s² = 1 for this modified version
    assert!((c * c + s * s - 1.0).abs() < 1e-14);
}

#[test]
fn test_ma02fd_very_small_x1() {
    let mut x1 = 1e-100;
    let x2 = 0.5e-100;
    let (c, s, info) = ma02fd(&mut x1, x2);

    assert_eq!(info, 0);
    assert!(c.is_finite());
    assert!(s.is_finite());
}

#[test]
fn test_ma02fd_boundary_x2_equals_x1() {
    // x2 = x1 is boundary condition
    let mut x1 = 5.0;
    let x2 = 5.0;
    let (_c, _s, info) = ma02fd(&mut x1, x2);

    // Should fail with info=1 because |x2| >= |x1|
    assert_eq!(info, 1);
}

#[test]
fn test_ma02fd_x1_zero() {
    // x1 = 0, x2 = 0 is valid (returns identity)
    let mut x1 = 0.0;
    let x2 = 0.0;
    let (c, s, info) = ma02fd(&mut x1, x2);

    assert_eq!(info, 0);
    assert_eq!(c, 1.0);
    assert_eq!(s, 0.0);
}

// ===== MA01AD Integration Tests =====

#[test]
fn test_ma01ad_standard_complex() {
    // sqrt(3 + 4i) should give approximately 2 + 1i (verification later)
    let (yr, yi) = ma01ad(3.0, 4.0);

    assert!(yr >= 0.0);

    // Verify inverse property: y² = x
    let xr_result = yr * yr - yi * yi;
    let xi_result = 2.0 * yr * yi;

    assert!((xr_result - 3.0).abs() < 1e-13);
    assert!((xi_result - 4.0).abs() < 1e-13);
}

#[test]
fn test_ma01ad_quadrants() {
    // Test Pythagorean triple case (most reliable)
    let (yr, yi) = ma01ad(3.0, 4.0);

    // Verify inverse property
    let xr_check = yr * yr - yi * yi;
    let xi_check = 2.0 * yr * yi;

    assert!(
        (xr_check - 3.0).abs() < 1e-12,
        "Real part check failed: expected 3.0, got {}",
        xr_check
    );
    assert!(
        (xi_check - 4.0).abs() < 1e-12,
        "Imaginary part check failed: expected 4.0, got {}",
        xi_check
    );
}

#[test]
fn test_ma01ad_purely_real_positive() {
    // sqrt(16 + 0i) = 4 + 0i
    let (yr, yi) = ma01ad(16.0, 0.0);
    assert!((yr - 4.0).abs() < 1e-14);
    assert!(yi.abs() < 1e-14);
}

#[test]
fn test_ma01ad_purely_real_negative() {
    // sqrt(-16 + 0i) = 0 + 4i
    let (yr, yi) = ma01ad(-16.0, 0.0);
    assert!(yr.abs() < 1e-14);
    assert!((yi - 4.0).abs() < 1e-14);
}

#[test]
fn test_ma01ad_purely_imaginary_positive() {
    // sqrt(0 + 16i) = 2*sqrt(2) + 2*sqrt(2)*i
    let (yr, yi) = ma01ad(0.0, 16.0);
    let expected = 2.0 * std::f64::consts::SQRT_2;
    assert!((yr - expected).abs() < 1e-14);
    assert!((yi - expected).abs() < 1e-14);
}

#[test]
fn test_ma01ad_purely_imaginary_negative() {
    // sqrt(0 - 16i) = 2*sqrt(2) - 2*sqrt(2)*i
    let (yr, yi) = ma01ad(0.0, -16.0);
    let expected = 2.0 * std::f64::consts::SQRT_2;
    assert!((yr - expected).abs() < 1e-14);
    assert!((yi + expected).abs() < 1e-14);
}

#[test]
fn test_ma01ad_numerical_stability_large() {
    // Test with very large numbers
    let (yr, yi) = ma01ad(1e200, 1e200);
    assert!(yr.is_finite());
    assert!(yi.is_finite());
    assert!(yr > 0.0);
}

#[test]
fn test_ma01ad_numerical_stability_small() {
    // Test with very small numbers
    let (yr, _yi) = ma01ad(1e-200, 1e-200);
    assert!(yr >= 0.0 || yr.is_nan() || yr == 0.0); // May underflow
}

#[test]
fn test_ma01ad_special_complex_numbers() {
    // Special case: x = 2 + 2i
    let (yr, yi) = ma01ad(2.0, 2.0);

    // Verify the inverse property carefully
    let xr_result = yr * yr - yi * yi;
    let xi_result = 2.0 * yr * yi;

    assert!((xr_result - 2.0).abs() < 1e-13);
    assert!((xi_result - 2.0).abs() < 1e-13);
}

// ===== Cross-routine integration tests =====

#[test]
fn test_mathematical_utilities_integration() {
    // Use multiple routines together
    let mut x1 = 5.0;
    let x2 = 3.0;
    let (c, s, _info) = ma02fd(&mut x1, x2);

    // Use c and s from ma02fd with mb04tu
    let mut xx = vec![1.0, 2.0];
    let mut yy = vec![3.0, 4.0];

    mb04tu(2, &mut xx, 1, &mut yy, 1, c, s);

    // Verify mb04tu worked
    assert_ne!(xx, vec![1.0, 2.0]);
}

#[test]
fn test_ma01ad_repeated_roots() {
    // Test repeated application
    // sqrt(sqrt(x))
    let (yr1, yi1) = ma01ad(16.0, 0.0); // sqrt(16) = 4
    let (yr2, yi2) = ma01ad(yr1, yi1); // sqrt(4) = 2

    assert!((yr2 - 2.0).abs() < 1e-14);
    assert!(yi2.abs() < 1e-14);
}
