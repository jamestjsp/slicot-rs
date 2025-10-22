//! TC - Transformation/Controllability Chapter
//!
//! This module contains polynomial matrix transformation and controllability routines
//! from SLICOT chapter TC.

use ndarray::Array3;

/// TC01OD - Dual of a left/right polynomial matrix representation
///
/// Finds the dual right (left) polynomial matrix representation of a given left (right)
/// polynomial matrix representation, where the right and left polynomial matrix
/// representations are of the form Q(s)*inv(P(s)) and inv(P(s))*Q(s) respectively.
///
/// # Arguments
///
/// * `leri` - Indicates whether a left or right matrix fraction is input:
///   - `'L'`: A left matrix fraction is input
///   - `'R'`: A right matrix fraction is input
/// * `pcoeff` - Coefficients of the denominator matrix P(s). For LERI='L', this is P×P×INDLIM.
///   For LERI='R', this is M×M×INDLIM. Modified in-place to contain P'(s).
///   PCOEFF[:,:,k-1] contains coefficient in s^(INDLIM-k) for k=1..INDLIM.
/// * `qcoeff` - Coefficients of the numerator matrix Q(s). On entry: P×M×INDLIM.
///   On exit: M×P×INDLIM containing Q'(s) of dual system.
///   QCOEFF[:,:,k-1] contains coefficient in s^(INDLIM-k) for k=1..INDLIM.
///
/// # Returns
///
/// * `Ok(())` on success
/// * `Err(String)` with error message on invalid parameters
///
/// # Method
///
/// If the given M-input/P-output left (right) polynomial matrix representation has
/// numerator matrix Q(s) and denominator matrix P(s), its dual P-input/M-output right
/// (left) polynomial matrix representation simply has numerator matrix Q'(s) and
/// denominator matrix P'(s), where ' denotes transpose.
///
/// # Example from SLICOT HTML Documentation
///
/// For M=2, P=2, INDLIM=3, LERI='L':
///
/// Input P(s) (2×2×3):
/// ```text
/// element (1,1): 2.0  3.0  1.0  (coefficients for s^2, s^1, s^0)
/// element (1,2): 5.0  7.0 -6.0
/// element (2,1): 4.0 -1.0 -1.0
/// element (2,2): 3.0  2.0  2.0
/// ```
///
/// Output P'(s) (2×2×3) - transposed:
/// ```text
/// element (1,1): 2.0  3.0  1.0
/// element (1,2): 4.0 -1.0 -1.0  (was element (2,1))
/// element (2,1): 5.0  7.0 -6.0  (was element (1,2))
/// element (2,2): 3.0  2.0  2.0
/// ```
///
/// # References
///
/// SLICOT routine TC01OD.f
pub fn tc01od(
    leri: char,
    pcoeff: &mut Array3<f64>,
    qcoeff: &mut Array3<f64>,
) -> Result<(), String> {
    // Validate LERI parameter
    let is_left = match leri {
        'L' | 'l' => true,
        'R' | 'r' => false,
        _ => {
            return Err(format!(
                "Invalid LERI parameter: '{}'. Must be 'L' or 'R'.",
                leri
            ))
        }
    };

    // Extract dimensions from the arrays
    let (p_rows, p_cols, indlim) = pcoeff.dim();
    let (q_rows, q_cols, q_indlim) = qcoeff.dim();

    // Validate that P is square
    if p_rows != p_cols {
        return Err(format!(
            "PCOEFF must be square in first two dimensions, got {}×{}",
            p_rows, p_cols
        ));
    }

    // Validate INDLIM consistency
    if indlim != q_indlim {
        return Err(format!(
            "PCOEFF and QCOEFF must have same third dimension (INDLIM), got {} vs {}",
            indlim, q_indlim
        ));
    }

    // Determine M and P from Q dimensions
    let (m, p) = if is_left {
        (q_cols, q_rows)
    } else {
        (q_rows, q_cols)
    };

    // Validate dimensions
    if indlim < 1 {
        return Err(format!("INDLIM must be >= 1, got {}", indlim));
    }

    // Validate P dimensions match LERI mode
    let expected_porm = if is_left { p } else { m };
    if p_rows != expected_porm {
        return Err(format!(
            "PCOEFF dimension mismatch: expected {}×{} for LERI='{}', got {}×{}",
            expected_porm, expected_porm, leri, p_rows, p_cols
        ));
    }

    // Quick return for zero dimensions
    if m == 0 || p == 0 {
        return Ok(());
    }

    // Transpose numerator matrix Q(s) for each polynomial degree
    // The algorithm transposes in-place using the swap pattern from SLICOT
    // For non-square matrices, we use the SLICOT approach from the Fortran code
    let (max_mp, min_mp) = if m > p { (m, p) } else { (p, m) };

    if max_mp != 1 {
        for k in 0..indlim {
            // For each degree, transpose Q(:,:,k)
            // SLICOT's approach handles rectangular matrices with DSWAP and DCOPY
            for j in 0..max_mp {
                if j < min_mp {
                    // Swap lower triangle
                    for i in (j + 1)..min_mp {
                        let temp = qcoeff[[i, j, k]];
                        qcoeff[[i, j, k]] = qcoeff[[j, i, k]];
                        qcoeff[[j, i, k]] = temp;
                    }
                } else if j >= p && j < m {
                    // Copy column to row when M > P
                    for i in 0..p {
                        qcoeff[[j, i, k]] = qcoeff[[i, j, k]];
                    }
                } else if j >= m && j < p {
                    // Copy row to column when P > M
                    for i in 0..m {
                        qcoeff[[i, j, k]] = qcoeff[[j, i, k]];
                    }
                }
            }
        }
    }

    // Transpose denominator matrix P(s) for each polynomial degree
    // P is porm×porm (square), so transpose is straightforward
    let porm = p_rows;
    if porm > 1 {
        for k in 0..indlim {
            // Transpose in-place using swap for efficiency
            for i in 0..porm {
                for j in (i + 1)..porm {
                    let temp = pcoeff[[i, j, k]];
                    pcoeff[[i, j, k]] = pcoeff[[j, i, k]];
                    pcoeff[[j, i, k]] = temp;
                }
            }
        }
    }

    Ok(())
}

#[cfg(test)]
mod tests {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::arr3;

    #[test]
    fn test_tc01od_left_example() {
        // Test case from TC01OD.html example
        // M=2, P=2, INDLIM=3, LERI='L'

        // Input PCOEFF (P×P×INDLIM = 2×2×3)
        // Data from HTML: each element has 3 coefficients for s^2, s^1, s^0
        let mut pcoeff = arr3(&[
            [
                [2.0, 3.0, 1.0], // P(1,1,:)
                [5.0, 7.0, -6.0],
            ], // P(1,2,:)
            [
                [4.0, -1.0, -1.0], // P(2,1,:)
                [3.0, 2.0, 2.0],
            ], // P(2,2,:)
        ]);

        // Input QCOEFF (P×M×INDLIM = 2×2×3)
        let mut qcoeff = arr3(&[
            [
                [6.0, -1.0, 5.0], // Q(1,1,:)
                [1.0, 1.0, 1.0],
            ], // Q(1,2,:)
            [
                [1.0, 7.0, 5.0], // Q(2,1,:)
                [4.0, 1.0, -1.0],
            ], // Q(2,2,:)
        ]);

        let result = tc01od('L', &mut pcoeff, &mut qcoeff);
        assert!(result.is_ok(), "tc01od failed: {:?}", result.err());

        // Expected PCOEFF after transpose (same as input due to specific structure)
        // P'(1,1) = P(1,1) = [2.0, 3.0, 1.0]
        // P'(1,2) = P(2,1) = [4.0, -1.0, -1.0]
        // P'(2,1) = P(1,2) = [5.0, 7.0, -6.0]
        // P'(2,2) = P(2,2) = [3.0, 2.0, 2.0]
        assert_abs_diff_eq!(pcoeff[[0, 0, 0]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pcoeff[[0, 0, 1]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pcoeff[[0, 0, 2]], 1.0, epsilon = 1e-10);

        assert_abs_diff_eq!(pcoeff[[0, 1, 0]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pcoeff[[0, 1, 1]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pcoeff[[0, 1, 2]], -1.0, epsilon = 1e-10);

        assert_abs_diff_eq!(pcoeff[[1, 0, 0]], 5.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pcoeff[[1, 0, 1]], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pcoeff[[1, 0, 2]], -6.0, epsilon = 1e-10);

        assert_abs_diff_eq!(pcoeff[[1, 1, 0]], 3.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pcoeff[[1, 1, 1]], 2.0, epsilon = 1e-10);
        assert_abs_diff_eq!(pcoeff[[1, 1, 2]], 2.0, epsilon = 1e-10);

        // Expected QCOEFF after transpose (M×P×INDLIM = 2×2×3)
        // Q'(1,1) = Q(1,1) = [6.0, -1.0, 5.0]
        // Q'(1,2) = Q(2,1) = [1.0, 7.0, 5.0]
        // Q'(2,1) = Q(1,2) = [1.0, 1.0, 1.0]
        // Q'(2,2) = Q(2,2) = [4.0, 1.0, -1.0]
        assert_abs_diff_eq!(qcoeff[[0, 0, 0]], 6.0, epsilon = 1e-10);
        assert_abs_diff_eq!(qcoeff[[0, 0, 1]], -1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(qcoeff[[0, 0, 2]], 5.0, epsilon = 1e-10);

        assert_abs_diff_eq!(qcoeff[[0, 1, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(qcoeff[[0, 1, 1]], 7.0, epsilon = 1e-10);
        assert_abs_diff_eq!(qcoeff[[0, 1, 2]], 5.0, epsilon = 1e-10);

        assert_abs_diff_eq!(qcoeff[[1, 0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(qcoeff[[1, 0, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(qcoeff[[1, 0, 2]], 1.0, epsilon = 1e-10);

        assert_abs_diff_eq!(qcoeff[[1, 1, 0]], 4.0, epsilon = 1e-10);
        assert_abs_diff_eq!(qcoeff[[1, 1, 1]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(qcoeff[[1, 1, 2]], -1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tc01od_right_mode() {
        // Test LERI='R' mode with 3×2 system
        // Note: QCOEFF must be MAX(M,P)×MAX(M,P)×INDLIM to hold both input and output
        let m = 3;
        let p = 2;
        let indlim = 2;
        let maxmp = m.max(p);

        // PCOEFF is M×M×INDLIM for right mode
        let mut pcoeff = Array3::<f64>::zeros((m, m, indlim));
        pcoeff[[0, 0, 0]] = 1.0;
        pcoeff[[1, 1, 0]] = 2.0;
        pcoeff[[2, 2, 0]] = 3.0;
        pcoeff[[0, 1, 1]] = 0.5;

        // QCOEFF must be MAX(M,P)×MAX(M,P)×INDLIM to accommodate transpose
        let mut qcoeff = Array3::<f64>::zeros((maxmp, maxmp, indlim));
        // Input data is M×P (3×2)
        qcoeff[[0, 0, 0]] = 1.0;
        qcoeff[[1, 1, 0]] = 2.0;
        qcoeff[[2, 0, 1]] = 0.3;

        let result = tc01od('R', &mut pcoeff, &mut qcoeff);
        assert!(result.is_ok(), "tc01od failed: {:?}", result.err());

        // Check P transpose worked
        assert_abs_diff_eq!(pcoeff[[1, 0, 1]], 0.5, epsilon = 1e-10); // Transposed from (0,1)

        // Check Q transpose worked (M×P becomes P×M)
        assert_abs_diff_eq!(qcoeff[[0, 0, 0]], 1.0, epsilon = 1e-10); // Q'(1,1) = Q(1,1)
        assert_abs_diff_eq!(qcoeff[[1, 1, 0]], 2.0, epsilon = 1e-10); // Q'(2,2) = Q(2,2)
        assert_abs_diff_eq!(qcoeff[[0, 2, 1]], 0.3, epsilon = 1e-10); // Q'(1,3) = Q(3,1)
    }

    #[test]
    fn test_tc01od_zero_dimensions() {
        // Test quick return for M=0
        let mut pcoeff = Array3::<f64>::zeros((2, 2, 3));
        let mut qcoeff = Array3::<f64>::zeros((2, 0, 3)); // P=2, M=0

        let result = tc01od('L', &mut pcoeff, &mut qcoeff);
        assert!(result.is_ok());
    }

    #[test]
    fn test_tc01od_scalar_system() {
        // Test 1×1 system (scalar case)
        let mut pcoeff = arr3(&[[[1.0, 2.0, 3.0]]]);
        let mut qcoeff = arr3(&[[[4.0, 5.0, 6.0]]]);

        let result = tc01od('L', &mut pcoeff, &mut qcoeff);
        assert!(result.is_ok());

        // For 1×1, transpose doesn't change anything
        assert_abs_diff_eq!(pcoeff[[0, 0, 0]], 1.0, epsilon = 1e-10);
        assert_abs_diff_eq!(qcoeff[[0, 0, 0]], 4.0, epsilon = 1e-10);
    }

    #[test]
    fn test_tc01od_invalid_leri() {
        let mut pcoeff = Array3::<f64>::zeros((2, 2, 3));
        let mut qcoeff = Array3::<f64>::zeros((2, 2, 3));

        let result = tc01od('X', &mut pcoeff, &mut qcoeff);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid LERI"));
    }

    #[test]
    fn test_tc01od_dimension_mismatch() {
        // PCOEFF not square
        let mut pcoeff = Array3::<f64>::zeros((2, 3, 3));
        let mut qcoeff = Array3::<f64>::zeros((2, 2, 3));

        let result = tc01od('L', &mut pcoeff, &mut qcoeff);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be square"));
    }

    #[test]
    fn test_tc01od_indlim_mismatch() {
        let mut pcoeff = Array3::<f64>::zeros((2, 2, 3));
        let mut qcoeff = Array3::<f64>::zeros((2, 2, 2)); // Different INDLIM

        let result = tc01od('L', &mut pcoeff, &mut qcoeff);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("same third dimension"));
    }
}
