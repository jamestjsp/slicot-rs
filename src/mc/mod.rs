//! Mathematical Routines - Control (Chapter MC)
//!
//! This module contains mathematical routines related to polynomial operations
//! and control theory computations from the SLICOT library.

/// Result type for polynomial stability check.
#[derive(Debug, Clone, PartialEq, Eq)]
pub struct StabilityResult {
    /// True if the polynomial is stable
    pub stable: bool,
    /// Number of unstable zeros (zeros in RHP for continuous-time,
    /// or outside unit circle for discrete-time)
    pub num_unstable: usize,
    /// Degree of polynomial after trimming trailing zeros
    pub actual_degree: usize,
    /// Warning: number of leading zero coefficients that were trimmed
    pub warning: usize,
}

/// Error type for polynomial operations
#[derive(Debug, Clone, PartialEq, Eq)]
pub enum PolynomialError {
    /// Invalid discretization type (must be 'C' or 'D')
    InvalidDico(char),
    /// Polynomial degree is negative
    NegativeDegree,
    /// Polynomial is identically zero
    ZeroPolynomial,
    /// Stability is inconclusive (zeros very close to boundary)
    InconclusiveStability,
}

impl std::fmt::Display for PolynomialError {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        match self {
            PolynomialError::InvalidDico(c) => {
                write!(f, "Invalid DICO parameter '{}': must be 'C' or 'D'", c)
            }
            PolynomialError::NegativeDegree => write!(f, "Polynomial degree cannot be negative"),
            PolynomialError::ZeroPolynomial => write!(f, "Polynomial is identically zero"),
            PolynomialError::InconclusiveStability => {
                write!(
                    f,
                    "Stability is inconclusive: zeros may be very close to stability boundary"
                )
            }
        }
    }
}

impl std::error::Error for PolynomialError {}

/// Determines whether a real polynomial is stable.
///
/// This function checks the stability of a polynomial P(x) with real coefficients,
/// either in the continuous-time or discrete-time sense:
///
/// - **Continuous-time** (DICO = 'C'): Stable if all zeros lie in the left half-plane
/// - **Discrete-time** (DICO = 'D'): Stable if all zeros lie inside the unit circle
///
/// # Arguments
///
/// * `dico` - Discretization type:
///   - `'C'` or `'c'`: Continuous-time stability test
///   - `'D'` or `'d'`: Discrete-time stability test
/// * `p` - Polynomial coefficients in **increasing** powers of x:
///   ```text
///   P(x) = p[0] + p[1]*x + p[2]*x² + ... + p[n]*xⁿ
///   ```
///
/// # Returns
///
/// * `Ok(StabilityResult)` - Contains:
///   - `stable`: true if polynomial is stable
///   - `num_unstable`: count of unstable zeros
///   - `actual_degree`: degree after trimming leading zeros
///   - `warning`: number of leading zero coefficients trimmed
/// * `Err(PolynomialError)` - If input is invalid or stability cannot be determined
///
/// # Algorithm
///
/// - **Continuous-time**: Uses the Routh stability criterion. Computes Routh
///   coefficients and counts sign changes to determine the number of zeros in
///   the right half-plane.
///
/// - **Discrete-time**: Uses the Schur-Cohn algorithm applied to the reciprocal
///   polynomial. Counts sign changes in the constant terms of Schur transforms
///   to determine zeros outside the unit circle.
///
/// # Examples
///
/// ```
/// use slicot_rs::mc::mc01td;
///
/// // Stable continuous-time polynomial: P(x) = 1 + 3x + 2x²
/// // Zeros at x = -1 and x = -2 (both in left half-plane)
/// let p = vec![1.0, 3.0, 2.0];
/// let result = mc01td('C', &p).unwrap();
/// assert!(result.stable);
/// assert_eq!(result.num_unstable, 0);
///
/// // Unstable continuous-time polynomial: P(x) = -1 + x
/// // Zero at x = 1 (right half-plane)
/// let p = vec![-1.0, 1.0];
/// let result = mc01td('C', &p).unwrap();
/// assert!(!result.stable);
/// assert_eq!(result.num_unstable, 1);
///
/// // Stable discrete-time polynomial: P(x) = 1 + 0.5x
/// // Zero at x = -0.5 (inside unit circle)
/// let p = vec![1.0, 0.5];
/// let result = mc01td('D', &p).unwrap();
/// assert!(result.stable);
/// ```
///
/// # Numerical Considerations
///
/// The algorithm is numerically stable. However, if Routh coefficients
/// (continuous-time) or Schur transform constants (discrete-time) are very
/// small relative to machine precision, the stability determination may be
/// incorrect. In such cases, `Err(PolynomialError::InconclusiveStability)`
/// is returned.
///
/// # SLICOT Reference
///
/// This function is a Rust translation of the SLICOT Fortran routine `MC01TD`.
///
/// **Original Purpose**: Determine stability of polynomials using classical
/// stability criteria (Routh for continuous-time, Schur-Cohn for discrete-time).
///
/// **Differences from Fortran version**:
/// - Returns `Result<StabilityResult, PolynomialError>` instead of INFO/IWARN codes
/// - Takes immutable slice instead of mutable array with degree parameter
/// - Polynomial degree inferred from slice length (degree = p.len() - 1)
/// - Uses Rust-idiomatic error handling and result types
///
/// **References**:
/// - [1] Gantmacher, F.R. "Applications of the Theory of Matrices"
/// - [2] Kucera, V. "Discrete Linear Control. The Algorithmic Approach"
/// - [3] Henrici, P. "Applied and Computational Complex Analysis (Vol. 1)"
///
/// **Reference Implementation**: `reference/src/MC01TD.f`
pub fn mc01td(dico: char, p: &[f64]) -> Result<StabilityResult, PolynomialError> {
    const ZERO: f64 = 0.0;

    // Normalize DICO to uppercase
    let dico_upper = dico.to_ascii_uppercase();

    // Validate DICO parameter
    if dico_upper != 'C' && dico_upper != 'D' {
        return Err(PolynomialError::InvalidDico(dico));
    }

    if p.is_empty() {
        return Err(PolynomialError::ZeroPolynomial);
    }

    // Trim leading zero coefficients to find actual degree
    let mut dp = p.len() - 1;
    let mut iwarn = 0;

    while dp > 0 && p[dp] == ZERO {
        dp -= 1;
        iwarn += 1;
    }

    // Check if polynomial is identically zero
    if dp == 0 && p[0] == ZERO {
        return Err(PolynomialError::ZeroPolynomial);
    }

    // Working array for Routh coefficients or Schur transforms
    let mut dwork = vec![ZERO; 2 * dp + 2];

    let num_unstable = if dico_upper == 'C' {
        // Continuous-time case: Routh algorithm
        routh_stability(&p[0..=dp], &mut dwork)?
    } else {
        // Discrete-time case: Schur-Cohn algorithm
        schur_cohn_stability(&p[0..=dp], &mut dwork)?
    };

    Ok(StabilityResult {
        stable: num_unstable == 0,
        num_unstable,
        actual_degree: dp,
        warning: iwarn,
    })
}

/// Compute stability using Routh criterion (continuous-time).
fn routh_stability(p: &[f64], dwork: &mut [f64]) -> Result<usize, PolynomialError> {
    const ZERO: f64 = 0.0;

    // Copy polynomial coefficients to workspace
    dwork[..p.len()].copy_from_slice(p);

    let mut nz = 0;
    let mut k = p.len() - 1;

    // Routh algorithm: compute coefficients and count sign changes
    while k > 0 {
        if dwork[k] == ZERO {
            return Err(PolynomialError::InconclusiveStability);
        }

        let alpha = dwork[k + 1] / dwork[k];
        if alpha < ZERO {
            nz += 1;
        }
        k -= 1;

        // Update coefficients: dwork[i] := dwork[i] - alpha * dwork[i-1]
        let mut i = k;
        while i >= 2 {
            dwork[i] -= alpha * dwork[i - 1];
            i -= 2;
        }
    }

    Ok(nz)
}

/// Compute stability using Schur-Cohn algorithm (discrete-time).
fn schur_cohn_stability(p: &[f64], dwork: &mut [f64]) -> Result<usize, PolynomialError> {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;

    let dp = p.len() - 1;

    // Copy polynomial coefficients in reverse order (reciprocal polynomial)
    for i in 0..=dp {
        dwork[i] = p[dp - i];
    }

    let mut signum = ONE;
    let mut nz = 0;
    let mut k = 1;

    // Schur-Cohn algorithm
    while k <= dp {
        // Find index of maximum absolute value in dwork[k..k+k1]
        let k1 = dp - k + 2;
        let k2 = dp + 2;

        let max_idx = idamax(&dwork[k - 1..k - 1 + k1]);
        let alpha = dwork[k - 1 + max_idx];

        if alpha == ZERO {
            return Err(PolynomialError::InconclusiveStability);
        }

        // Copy and scale: dwork[k2..k2+k1] = dwork[k..k+k1] / alpha
        for i in 0..k1 {
            dwork[k2 + i] = dwork[k + i] / alpha;
        }

        let p1 = dwork[k2];
        let pk1 = dwork[k2 + k1 - 1];

        // Compute T^k P(x): Schur transform
        for i in 1..k1 {
            dwork[k + i] = p1 * dwork[dp + 1 + i] - pk1 * dwork[k2 + k1 - i];
        }

        // Update k and check sign
        k += 1;

        if dwork[k] == ZERO {
            return Err(PolynomialError::InconclusiveStability);
        }

        signum *= dwork[k].signum();
        if signum < ZERO {
            nz += 1;
        }
    }

    Ok(nz)
}

/// Find index of element with maximum absolute value (equivalent to BLAS IDAMAX).
#[inline]
fn idamax(x: &[f64]) -> usize {
    if x.is_empty() {
        return 0;
    }

    let mut max_idx = 0;
    let mut max_val = x[0].abs();

    for (i, &val) in x.iter().enumerate().skip(1) {
        let abs_val = val.abs();
        if abs_val > max_val {
            max_val = abs_val;
            max_idx = i;
        }
    }

    max_idx
}

#[cfg(test)]
mod tests {
    use super::*;

    // ===== MC01TD Tests =====

    #[test]
    fn test_stable_continuous_first_order() {
        // P(x) = 1 + x, zero at x = -1 (left half-plane)
        let p = vec![1.0, 1.0];
        let result = mc01td('C', &p).unwrap();
        assert!(result.stable);
        assert_eq!(result.num_unstable, 0);
        assert_eq!(result.actual_degree, 1);
    }

    #[test]
    fn test_unstable_continuous_first_order() {
        // P(x) = -1 + x, zero at x = 1 (right half-plane)
        let p = vec![-1.0, 1.0];
        let result = mc01td('C', &p).unwrap();
        assert!(!result.stable);
        assert_eq!(result.num_unstable, 1);
    }

    #[test]
    fn test_stable_continuous_second_order() {
        // P(x) = 1 + 3x + 2x², zeros at x = -1, -2 (both in left half-plane)
        let p = vec![1.0, 3.0, 2.0];
        let result = mc01td('C', &p).unwrap();
        assert!(result.stable);
        assert_eq!(result.num_unstable, 0);
    }

    #[test]
    fn test_unstable_continuous_second_order() {
        // P(x) = -1 + x², zeros at x = ±1 (one in right half-plane)
        let p = vec![-1.0, 0.0, 1.0];
        let result = mc01td('C', &p).unwrap();
        assert!(!result.stable);
        assert!(result.num_unstable > 0);
    }

    #[test]
    fn test_stable_discrete_first_order() {
        // P(x) = 1 + 0.5x, zero at x = -0.5 (inside unit circle)
        let p = vec![1.0, 0.5];
        let result = mc01td('D', &p).unwrap();
        assert!(result.stable);
        assert_eq!(result.num_unstable, 0);
    }

    #[test]
    fn test_unstable_discrete_first_order() {
        // P(x) = 1 + 2x, zero at x = -2 (outside unit circle)
        let p = vec![1.0, 2.0];
        let result = mc01td('D', &p).unwrap();
        assert!(!result.stable);
        assert_eq!(result.num_unstable, 1);
    }

    #[test]
    fn test_zero_polynomial() {
        let p = vec![0.0, 0.0, 0.0];
        let result = mc01td('C', &p);
        assert!(matches!(result, Err(PolynomialError::ZeroPolynomial)));
    }

    #[test]
    fn test_invalid_dico() {
        let p = vec![1.0, 1.0];
        let result = mc01td('X', &p);
        assert!(matches!(result, Err(PolynomialError::InvalidDico('X'))));
    }

    #[test]
    fn test_trailing_zeros_trimmed() {
        // P(x) = 1 + 2x with trailing zeros
        let p = vec![1.0, 2.0, 0.0, 0.0];
        let result = mc01td('C', &p).unwrap();
        assert_eq!(result.actual_degree, 1);
        assert_eq!(result.warning, 2);
    }

    #[test]
    fn test_constant_polynomial() {
        // P(x) = 5 (degree 0, no zeros)
        let p = vec![5.0];
        let result = mc01td('C', &p).unwrap();
        assert!(result.stable);
        assert_eq!(result.actual_degree, 0);
    }

    #[test]
    fn test_case_insensitive_dico() {
        let p = vec![1.0, 1.0];

        // Lowercase 'c'
        let result1 = mc01td('c', &p).unwrap();
        assert!(result1.stable);

        // Lowercase 'd'
        let result2 = mc01td('d', &p).unwrap();
        assert!(result2.stable);
    }

    #[test]
    fn test_empty_polynomial() {
        let p: Vec<f64> = vec![];
        let result = mc01td('C', &p);
        assert!(matches!(result, Err(PolynomialError::ZeroPolynomial)));
    }

    // Test helper functions

    #[test]
    fn test_idamax_basic() {
        let x = vec![1.0, -3.0, 2.0, -5.0, 4.0];
        assert_eq!(idamax(&x), 3); // -5.0 has largest absolute value
    }

    #[test]
    fn test_idamax_empty() {
        let x: Vec<f64> = vec![];
        assert_eq!(idamax(&x), 0);
    }

    #[test]
    fn test_idamax_single() {
        let x = vec![42.0];
        assert_eq!(idamax(&x), 0);
    }

    #[test]
    fn test_idamax_first_element() {
        let x = vec![-10.0, 2.0, 3.0];
        assert_eq!(idamax(&x), 0);
    }
}
