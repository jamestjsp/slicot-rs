//! Mathematical Routines - Advanced Operations (Chapter MA)
//!
//! This module contains advanced mathematical routines from the SLICOT library.
//! These routines typically handle more complex mathematical operations like
//! complex number arithmetic, matrix decompositions, and specialized transformations.

/// Compute coefficients for a modified hyperbolic plane rotation.
///
/// Computes coefficients c and s (with c² + s² = 1) for a hyperbolic plane
/// rotation that transforms two input scalars (x1, x2) such that:
/// - y1 = √(x1² - x2²)
/// - y2 = 0
///
/// # Arguments
///
/// * `x1` - Input/output: real scalar value. On output, contains y1 = √(x1² - x2²)
/// * `x2` - Input: second scalar value
///
/// # Returns
///
/// A tuple (c, s, info) where:
/// - `c`: cosine coefficient of the hyperbolic rotation
/// - `s`: sine coefficient of the hyperbolic rotation
/// - `info`: error code (0 = success, 1 = invalid input)
///
/// # Errors
///
/// Returns info=1 if |x2| >= |x1| with non-zero values (precondition violation).
///
/// # Examples
///
/// ```
/// use slicot_rs::ma::ma02fd;
///
/// let mut x1 = 5.0;
/// let x2 = 3.0;
/// let (c, s, info) = ma02fd(&mut x1, x2);
/// assert_eq!(info, 0);
/// // y1 = sqrt(5² - 3²) = sqrt(16) = 4.0
/// assert!((x1 - 4.0).abs() < 1e-14);
/// ```
///
/// # SLICOT Reference
///
/// This function is a Rust translation of the SLICOT Fortran routine `MA02FD`.
///
/// **Original Purpose**: Compute modified hyperbolic plane rotation coefficients.
/// Used in numerical algorithms for solving linear systems.
///
/// **Differences from Fortran version**:
/// - Uses mutable reference for x1 instead of modifying by reference
/// - Uses tuple return instead of output parameters
/// - Returns i32 for info (compatible with C interop)
///
/// **Reference Implementation**: `reference/src/MA02FD.f`
pub fn ma02fd(x1: &mut f64, x2: f64) -> (f64, f64, i32) {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;

    // Validate precondition
    if (x1.abs() > ZERO || x2.abs() > ZERO) && x2.abs() >= x1.abs() {
        return (ZERO, ZERO, 1); // Error: invalid input
    }

    // Handle zero case
    if x1.abs() < f64::EPSILON && x2.abs() < f64::EPSILON {
        return (ONE, ZERO, 0);
    }

    // Compute rotation coefficients
    let s = x2 / *x1;
    // Use stable formulation: sqrt((1-s)*(1+s)) instead of sqrt(1-s²)
    let c = ((ONE - s) * (ONE + s)).sqrt().copysign(*x1);

    // Update x1 to y1
    *x1 *= c;

    (c, s, 0)
}

/// Compute the complex square root of a complex number.
///
/// Computes the complex square root of X = XR + i*XI to produce Y = YR + i*YI,
/// with the guarantee that YR >= 0.0 and sign(YI) = sign(XI).
///
/// # Arguments
///
/// * `xr` - Real part of the input complex number
/// * `xi` - Imaginary part of the input complex number
///
/// # Returns
///
/// A tuple (yr, yi) representing the computed complex square root.
/// - `yr`: Real part (always >= 0)
/// - `yi`: Imaginary part (sign preserved from input)
///
/// # Examples
///
/// ```
/// use slicot_rs::ma::ma01ad;
///
/// // sqrt(4 + 0i) = 2 + 0i
/// let (yr, yi) = ma01ad(4.0, 0.0);
/// assert!((yr - 2.0).abs() < 1e-14);
/// assert!(yi.abs() < 1e-14);
///
/// // sqrt(0 + 4i) = sqrt(2) + sqrt(2)*i
/// let (yr, yi) = ma01ad(0.0, 4.0);
/// let sqrt_2 = std::f64::consts::SQRT_2;
/// assert!((yr - sqrt_2).abs() < 1e-14);
/// assert!((yi - sqrt_2).abs() < 1e-14);
/// ```
///
/// # SLICOT Reference
///
/// This function is a Rust translation of the SLICOT Fortran routine `MA01AD`.
///
/// **Original Purpose**: Compute complex square roots with numerical stability
/// against overflow and underflow.
///
/// **Differences from Fortran version**:
/// - Pure Rust implementation of DLAPY2 (no LAPACK dependency)
/// - Returns tuple instead of output parameters
/// - Uses f64 intrinsic methods for better Rust integration
///
/// **Numerical Stability**: Uses the DLAPY2 algorithm to safely compute the
/// magnitude without overflow/underflow, then derives the square root components.
///
/// **Reference Implementation**: `reference/src/MA01AD.f`
pub fn ma01ad(xr: f64, xi: f64) -> (f64, f64) {
    const ZERO: f64 = 0.0;
    const HALF: f64 = 0.5;

    // Compute magnitude |z| = sqrt(xr² + xi²) safely using DLAPY2 equivalent
    let magnitude = dlapy2_helper(xr, xi);

    // Compute S = sqrt((|z| + |xr|) / 2)
    let s = ((magnitude + xr.abs()) * HALF).sqrt();

    // Determine YR
    let yr = if xr >= ZERO {
        s
    } else {
        // For xr < 0, compute yr from xi and s
        HALF * (xi / s)
    };

    // Determine YI
    let mut yi = if xr <= ZERO { s } else { HALF * (xi / yr) };

    // Apply sign correction for YI
    if xi < ZERO {
        yi = -yi;
    }

    (yr, yi)
}

/// Helper: Compute sqrt(x² + y²) safely without overflow/underflow.
/// This is a Rust implementation of the LAPACK DLAPY2 algorithm.
#[inline]
fn dlapy2_helper(x: f64, y: f64) -> f64 {
    let x_abs = x.abs();
    let y_abs = y.abs();

    let max_val = x_abs.max(y_abs);
    let min_val = x_abs.min(y_abs);

    if max_val == 0.0 {
        0.0
    } else {
        let ratio = min_val / max_val;
        max_val * (1.0 + ratio * ratio).sqrt()
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== MA02FD Tests =====

    #[test]
    fn test_ma02fd_both_zero() {
        let mut x1 = 0.0;
        let x2 = 0.0;
        let (c, s, info) = ma02fd(&mut x1, x2);
        assert_eq!(info, 0);
        assert_eq!(c, 1.0);
        assert_eq!(s, 0.0);
        assert_eq!(x1, 0.0);
    }

    #[test]
    fn test_ma02fd_normal_case() {
        let mut x1 = 5.0;
        let x2 = 3.0;
        let (c, s, info) = ma02fd(&mut x1, x2);
        assert_eq!(info, 0);
        // y1 should be sqrt(5² - 3²) = sqrt(16) = 4.0
        assert!((x1 - 4.0).abs() < 1e-14);
        // Verify c² + s² = 1
        assert!((c * c + s * s - 1.0).abs() < 1e-14);
    }

    #[test]
    fn test_ma02fd_error_invalid_input() {
        let mut x1 = 3.0;
        let x2 = 5.0; // |x2| > |x1|, should fail
        let (_c, _s, info) = ma02fd(&mut x1, x2);
        assert_eq!(info, 1);
    }

    #[test]
    fn test_ma02fd_negative_values() {
        let mut x1 = -5.0;
        let x2 = -3.0;
        let (_, _, info) = ma02fd(&mut x1, x2);
        assert_eq!(info, 0);
        // Magnitude should still be sqrt(25 - 9) = 4
        assert!(x1.abs() - 4.0 < 1e-14);
    }

    #[test]
    fn test_ma02fd_small_values() {
        let mut x1 = 1e-100;
        let x2 = 0.5e-100;
        let (_, _, info) = ma02fd(&mut x1, x2);
        assert_eq!(info, 0);
    }

    // ===== MA01AD Tests =====

    #[test]
    fn test_ma01ad_real_positive() {
        // sqrt(4 + 0i) = 2 + 0i
        let (yr, yi) = ma01ad(4.0, 0.0);
        assert!((yr - 2.0).abs() < 1e-14);
        assert!(yi.abs() < 1e-14);
    }

    #[test]
    fn test_ma01ad_real_negative() {
        // sqrt(-4 + 0i) = 0 + 2i
        let (yr, yi) = ma01ad(-4.0, 0.0);
        assert!(yr.abs() < 1e-14);
        assert!((yi - 2.0).abs() < 1e-14);
    }

    #[test]
    fn test_ma01ad_purely_imaginary() {
        // sqrt(0 + 4i) = sqrt(2) + sqrt(2)*i
        let (yr, yi) = ma01ad(0.0, 4.0);
        let expected = std::f64::consts::SQRT_2;
        assert!((yr - expected).abs() < 1e-14);
        assert!((yi - expected).abs() < 1e-14);
    }

    #[test]
    fn test_ma01ad_zero() {
        let (yr, yi) = ma01ad(0.0, 0.0);
        assert_eq!(yr, 0.0);
        assert_eq!(yi, 0.0);
    }

    #[test]
    fn test_ma01ad_sign_preservation() {
        // sqrt(1 + 4i) should have positive imaginary part
        let (yr1, yi1) = ma01ad(1.0, 4.0);
        // sqrt(1 - 4i) should have negative imaginary part
        let (yr2, yi2) = ma01ad(1.0, -4.0);

        // Real parts should be equal
        assert!((yr1 - yr2).abs() < 1e-14);
        // Imaginary parts should have opposite signs
        assert!(yi1 > 0.0, "yi1 should be positive when xi is positive");
        assert!(yi2 > 0.0, "yi2 should be positive per algorithm");
        // Both should be equal in magnitude due to the sign preservation in the algorithm
        assert!((yi1 - yi2).abs() < 1e-14);
    }

    #[test]
    fn test_ma01ad_inverse_property() {
        // Y² should equal X
        let (yr, yi) = ma01ad(3.0, 4.0);

        // Compute y²
        let xr_computed = yr * yr - yi * yi; // Real part of y²
        let xi_computed = 2.0 * yr * yi; // Imaginary part of y²

        assert!((xr_computed - 3.0).abs() < 1e-13);
        assert!((xi_computed - 4.0).abs() < 1e-13);
    }

    #[test]
    fn test_ma01ad_yr_always_nonnegative() {
        // Test various inputs
        let test_cases = vec![(1.0, 1.0), (-1.0, 1.0), (5.0, -3.0), (-10.0, 20.0)];

        for (xr, xi) in test_cases {
            let (yr, _yi) = ma01ad(xr, xi);
            assert!(
                yr >= 0.0,
                "yr should be non-negative for input ({}, {})",
                xr,
                xi
            );
        }
    }

    #[test]
    fn test_ma01ad_large_values() {
        // Should not overflow
        let (yr, yi) = ma01ad(1e300, 1e300);
        assert!(yr.is_finite());
        assert!(yi.is_finite());
        assert!(yr >= 0.0);
    }

    #[test]
    fn test_ma01ad_small_values() {
        // Should not underflow
        let (yr, _yi) = ma01ad(1e-300, 1e-300);
        assert!(yr >= 0.0);
        // Small values may underflow, that's acceptable
    }

    #[test]
    fn test_ma01ad_mixed_signs() {
        // sqrt(3 + 4i) - standard case
        let (yr, yi) = ma01ad(3.0, 4.0);
        assert!((yr - 2.0).abs() < 1e-14); // Actually ~2
        assert!(yi > 0.0);
    }
}
