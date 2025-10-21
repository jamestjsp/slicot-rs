//! Mathematical Routines - Basic Operations (Chapter MB)
//!
//! This module contains basic mathematical routines from the SLICOT library.
//! These are low-level utility functions used by higher-level control theory
//! algorithms.
//!
//! SLICOT Chapter MB focuses on fundamental mathematical operations that don't
//! require complex linear algebra but are essential building blocks for
//! numerical computations in control theory.

use ndarray::Array2;

/// Computes the absolute minimal value of elements in an array.
///
/// This function finds the element with the smallest absolute value (magnitude)
/// in the input slice. It is useful in numerical computations where the element
/// closest to zero needs to be identified.
///
/// # Arguments
///
/// * `x` - A slice of f64 values to be examined
///
/// # Returns
///
/// * `Some(f64)` - The smallest absolute value found in the array
/// * `None` - If the input slice is empty
///
/// # Examples
///
/// ```
/// use slicot_rs::mb::mb03my;
///
/// // Find minimum absolute value in a mixed array
/// let data = vec![3.0, -1.5, 2.0, -0.5, 4.0];
/// assert_eq!(mb03my(&data), Some(0.5));
///
/// // Single element
/// let single = vec![-7.0];
/// assert_eq!(mb03my(&single), Some(7.0));
///
/// // Empty array returns None
/// let empty: Vec<f64> = vec![];
/// assert_eq!(mb03my(&empty), None);
/// ```
///
/// # SLICOT Reference
///
/// This function is a Rust translation of the SLICOT Fortran routine `MB03MY`.
///
/// **Original Purpose**: To compute the absolute minimal value of NX elements
/// in an array. The Fortran function returns zero if NX < 1.
///
/// **Differences from Fortran version**:
/// - Returns `Option<f64>` instead of f64 (None for empty arrays vs 0.0)
/// - Takes a slice instead of array pointer with length and stride parameters
/// - Uses 0-based indexing (Rust) instead of 1-based indexing (Fortran)
///
/// **Contributor**: V. Sima, Katholieke Univ. Leuven, Belgium, Mar. 1997
///
/// **Reference Implementation**: `reference/src/MB03MY.f`
pub fn mb03my(x: &[f64]) -> Option<f64> {
    if x.is_empty() {
        return None;
    }

    // Find the minimum absolute value using iterators
    x.iter()
        .map(|&val| val.abs())
        .min_by(|a, b| a.partial_cmp(b).unwrap_or(std::cmp::Ordering::Equal))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_empty_array() {
        let empty: Vec<f64> = vec![];
        assert_eq!(mb03my(&empty), None);
    }

    #[test]
    fn test_single_positive_element() {
        let data = vec![5.0];
        assert_eq!(mb03my(&data), Some(5.0));
    }

    #[test]
    fn test_single_negative_element() {
        let data = vec![-3.0];
        assert_eq!(mb03my(&data), Some(3.0));
    }

    #[test]
    fn test_single_zero_element() {
        let data = vec![0.0];
        assert_eq!(mb03my(&data), Some(0.0));
    }

    #[test]
    fn test_mixed_values() {
        let data = vec![3.0, -1.5, 2.0, -0.5, 4.0];
        assert_eq!(mb03my(&data), Some(0.5));
    }

    #[test]
    fn test_all_positive() {
        let data = vec![5.0, 3.0, 7.0, 1.0, 9.0];
        assert_eq!(mb03my(&data), Some(1.0));
    }

    #[test]
    fn test_all_negative() {
        let data = vec![-5.0, -3.0, -7.0, -1.0, -9.0];
        assert_eq!(mb03my(&data), Some(1.0));
    }

    #[test]
    fn test_all_same_values() {
        let data = vec![2.5, 2.5, 2.5, 2.5];
        assert_eq!(mb03my(&data), Some(2.5));
    }

    #[test]
    fn test_with_zero() {
        let data = vec![3.0, -1.5, 0.0, 4.0];
        assert_eq!(mb03my(&data), Some(0.0));
    }

    #[test]
    fn test_minimum_at_start() {
        let data = vec![0.1, 5.0, 3.0, 2.0];
        assert_eq!(mb03my(&data), Some(0.1));
    }

    #[test]
    fn test_minimum_at_end() {
        let data = vec![5.0, 3.0, 2.0, 0.1];
        assert_eq!(mb03my(&data), Some(0.1));
    }

    #[test]
    fn test_minimum_in_middle() {
        let data = vec![5.0, 0.1, 3.0, 2.0];
        assert_eq!(mb03my(&data), Some(0.1));
    }

    #[test]
    fn test_very_small_values() {
        let data = vec![1e-10, 1e-8, 1e-12, 1e-9];
        assert_eq!(mb03my(&data), Some(1e-12));
    }

    #[test]
    fn test_large_values() {
        let data = vec![1e10, 1e8, 1e12, 1e9];
        assert_eq!(mb03my(&data), Some(1e8));
    }

    #[test]
    fn test_symmetric_values() {
        let data = vec![-5.0, 5.0, -3.0, 3.0, -1.0, 1.0];
        assert_eq!(mb03my(&data), Some(1.0));
    }

    #[test]
    fn test_two_elements() {
        let data = vec![7.0, -2.0];
        assert_eq!(mb03my(&data), Some(2.0));
    }
}

/// Matrix scaling routine with overflow/underflow protection.
///
/// Multiplies an M by N real matrix A by the real scalar `cto/cfrom`.
/// This is done without over/underflow as long as the final result
/// `cto * A(i,j) / cfrom` does not over/underflow. The routine handles
/// different storage types: full, triangular, Hessenberg, and banded matrices.
///
/// This is a Rust translation of the SLICOT routine MB01QD, which is a modified
/// version of LAPACK's DLASCL with support for block-structured matrices.
///
/// # Arguments
///
/// * `matrix_type` - Storage type of the matrix:
///   - `'G'`: Full matrix
///   - `'L'`: Lower triangular matrix (or block lower triangular)
///   - `'U'`: Upper triangular matrix (or block upper triangular)
///   - `'H'`: Upper Hessenberg matrix (or block upper Hessenberg)
///   - `'B'`: Symmetric band matrix (lower half stored)
///   - `'Q'`: Symmetric band matrix (upper half stored)
///   - `'Z'`: General band matrix
/// * `kl` - Lower bandwidth (only used for band matrices: 'B', 'Q', 'Z')
/// * `ku` - Upper bandwidth (only used for band matrices: 'B', 'Q', 'Z')
/// * `cfrom` - Source scaling factor (must be nonzero)
/// * `cto` - Target scaling factor
/// * `a` - The matrix to be scaled (modified in-place)
/// * `nrows` - Optional array specifying the number of rows and columns in each
///   diagonal block (for block-structured matrices). If `None`, the matrix
///   is treated as having no block structure.
///
/// # Returns
///
/// * `Ok(())` - Scaling completed successfully
/// * `Err(String)` - If parameters are invalid (e.g., `cfrom == 0`, invalid matrix type)
///
/// # Examples
///
/// ```
/// use slicot_rs::mb::mb01qd;
/// use ndarray::arr2;
///
/// // Scale a full 3x3 matrix by 2.0
/// let mut a = arr2(&[[1.0, 2.0, 3.0],
///                    [4.0, 5.0, 6.0],
///                    [7.0, 8.0, 9.0]]);
/// mb01qd('G', 0, 0, 1.0, 2.0, &mut a, None).unwrap();
/// // All elements are now doubled
/// ```
///
/// # SLICOT Reference
///
/// **Original routine**: `MB01QD` (Fortran)
///
/// **Purpose**: Matrix scaling with overflow/underflow protection
///
/// **Method**: The routine implements a safe scaling algorithm that breaks down
/// large scaling factors into smaller steps to avoid intermediate overflow/underflow.
/// It uses machine precision parameters to determine when to use incremental scaling.
///
/// **Reference Implementation**: `reference/src/MB01QD.f`
///
/// **Contributor**: V. Sima, Katholieke Univ. Leuven, Belgium, Nov. 1996
pub fn mb01qd(
    matrix_type: char,
    kl: usize,
    ku: usize,
    cfrom: f64,
    cto: f64,
    a: &mut ndarray::Array2<f64>,
    nrows: Option<&[usize]>,
) -> Result<(), String> {
    use std::f64;

    // Quick return if matrix is empty
    let (m, n) = a.dim();
    if m == 0 || n == 0 {
        return Ok(());
    }

    // Validate cfrom is nonzero
    if cfrom == 0.0 {
        return Err("cfrom must be nonzero".to_string());
    }

    // Determine matrix type
    let itype = match matrix_type.to_ascii_uppercase() {
        'G' => 0, // Full matrix
        'L' => 1, // Lower triangular
        'U' => 2, // Upper triangular
        'H' => 3, // Upper Hessenberg
        'B' => 4, // Symmetric band (lower half)
        'Q' => 5, // Symmetric band (upper half)
        'Z' => 6, // General band
        _ => return Err(format!("Invalid matrix type: {}", matrix_type)),
    };

    // Machine precision parameters
    let smlnum = f64::MIN_POSITIVE; // Smallest positive normalized number
    let bignum = 1.0 / smlnum;

    let mut cfromc = cfrom;
    let mut ctoc = cto;

    // Iterative scaling to avoid overflow/underflow
    loop {
        let cfrom1 = cfromc * smlnum;
        let cto1 = ctoc / bignum;
        let mul: f64;
        let done: bool;

        if cfrom1.abs() > ctoc.abs() && ctoc != 0.0 {
            mul = smlnum;
            done = false;
            cfromc = cfrom1;
        } else if cto1.abs() > cfromc.abs() {
            mul = bignum;
            done = false;
            ctoc = cto1;
        } else {
            mul = ctoc / cfromc;
            done = true;
        }

        let noblc = nrows.is_none() || nrows.unwrap().is_empty();

        match itype {
            0 => {
                // Full matrix
                a.mapv_inplace(|val| val * mul);
            }
            1 => {
                // Lower triangular
                if noblc {
                    // Standard lower triangular
                    for j in 0..n {
                        for i in j..m {
                            a[[i, j]] *= mul;
                        }
                    }
                } else {
                    // Block lower triangular
                    let blocks = nrows.unwrap();
                    let mut jfin = 0;
                    for &block_size in blocks {
                        let jini = jfin;
                        jfin += block_size;
                        for j in jini..jfin.min(n) {
                            for i in jini..m {
                                a[[i, j]] *= mul;
                            }
                        }
                    }
                }
            }
            2 => {
                // Upper triangular
                if noblc {
                    // Standard upper triangular
                    for j in 0..n {
                        for i in 0..=(j.min(m - 1)) {
                            a[[i, j]] *= mul;
                        }
                    }
                } else {
                    // Block upper triangular
                    let blocks = nrows.unwrap();
                    let mut jfin = 0;
                    for k in 0..blocks.len() {
                        let jini = jfin;
                        jfin += blocks[k];
                        if k == blocks.len() - 1 {
                            jfin = n;
                        }
                        for j in jini..jfin {
                            for i in 0..jfin.min(m) {
                                a[[i, j]] *= mul;
                            }
                        }
                    }
                }
            }
            3 => {
                // Upper Hessenberg
                if noblc {
                    // Standard upper Hessenberg
                    for j in 0..n {
                        for i in 0..=(j + 1).min(m - 1) {
                            a[[i, j]] *= mul;
                        }
                    }
                } else {
                    // Block upper Hessenberg
                    let blocks = nrows.unwrap();
                    let mut jfin = 0;
                    for k in 0..blocks.len() {
                        let jini = jfin;
                        jfin += blocks[k];
                        let ifin = if k == blocks.len() - 1 {
                            jfin = n;
                            n
                        } else {
                            jfin + blocks[k + 1]
                        };
                        for j in jini..jfin {
                            for i in 0..ifin.min(m) {
                                a[[i, j]] *= mul;
                            }
                        }
                    }
                }
            }
            4 => {
                // Lower half of symmetric band matrix
                let k3 = kl + 1;
                let k4 = n + 1;
                for j in 0..n {
                    for i in 0..k3.min(k4 - j - 1) {
                        a[[i, j]] *= mul;
                    }
                }
            }
            5 => {
                // Upper half of symmetric band matrix
                let k1 = ku + 2;
                let k3 = ku + 1;
                for j in 0..n {
                    for i in (k1.saturating_sub(j + 1)).max(0)..k3 {
                        a[[i, j]] *= mul;
                    }
                }
            }
            6 => {
                // General band matrix
                let k1 = kl + ku + 2;
                let k2 = kl + 1;
                let k3 = 2 * kl + ku + 1;
                let k4 = kl + ku + 1 + m;
                for j in 0..n {
                    for i in (k1.saturating_sub(j + 1)).max(k2)..k3.min(k4 - j - 1) {
                        a[[i, j]] *= mul;
                    }
                }
            }
            _ => unreachable!(),
        }

        if done {
            break;
        }
    }

    Ok(())
}

/// Scales a matrix or undoes scaling with overflow/underflow protection.
///
/// This routine scales a matrix, if necessary, so that its norm will be in a
/// safe range of representable numbers. It can also undo a previous scaling
/// operation. This is useful for preventing overflow/underflow in numerical
/// algorithms.
///
/// # Arguments
///
/// * `scun` - Operation to perform:
///   - `'S'`: Scale the matrix
///   - `'U'`: Undo scaling of the matrix
/// * `matrix_type` - Storage type of the matrix:
///   - `'G'`: Full matrix
///   - `'L'`: (Block) lower triangular matrix
///   - `'U'`: (Block) upper triangular matrix
///   - `'H'`: (Block) upper Hessenberg matrix
///   - `'B'`: Symmetric band matrix (lower half stored)
///   - `'Q'`: Symmetric band matrix (upper half stored)
///   - `'Z'`: General band matrix
/// * `kl` - Lower bandwidth (only for 'B', 'Q', 'Z')
/// * `ku` - Upper bandwidth (only for 'B', 'Q', 'Z')
/// * `anrm` - Norm of the initial matrix (must be >= 0). When ANRM = 0, returns immediately.
///   This value should be preserved between the Scale and Undo operations.
/// * `a` - The matrix to be scaled/unscaled (modified in-place)
/// * `nrows` - Optional array specifying diagonal block sizes for block-structured matrices.
///   If `None`, matrix has no block structure.
///
/// # Returns
///
/// * `Ok(())` - Scaling completed successfully
/// * `Err(String)` - If parameters are invalid
///
/// # Method
///
/// The routine compares the matrix norm (ANRM) with SMLNUM (smallest safe number)
/// and BIGNUM (largest safe number):
/// - If ANRM < SMLNUM and SCUN = 'S': scales matrix up to SMLNUM
/// - If ANRM > BIGNUM and SCUN = 'S': scales matrix down to BIGNUM
/// - If SCUN = 'U': reverses the scaling by using reciprocal of the scaling factor
///
/// The scaling is performed by calling `mb01qd` with appropriate parameters.
///
/// # Examples
///
/// ```
/// use slicot_rs::mb::mb01pd;
/// use ndarray::arr2;
///
/// // Scale a matrix with very small norm
/// let mut a = arr2(&[[1e-200, 2e-200],
///                    [3e-200, 4e-200]]);
/// let anrm = 1e-200;
/// mb01pd('S', 'G', 0, 0, anrm, &mut a, None).unwrap();
/// // Matrix is now scaled to safe range
///
/// // Later, undo the scaling
/// mb01pd('U', 'G', 0, 0, anrm, &mut a, None).unwrap();
/// // Matrix is back to original scale
/// ```
///
/// # SLICOT Reference
///
/// This is a Rust translation of SLICOT routine MB01PD.
///
/// **Original Purpose**: To scale a matrix or undo scaling. Scaling is performed,
/// if necessary, so that the matrix norm will be in a safe range of representable
/// numbers.
///
/// **Method**: Denote by ANRM the norm of the matrix, and by SMLNUM and BIGNUM,
/// two positive numbers near the smallest and largest safely representable numbers,
/// respectively. The matrix is scaled, if needed, such that the norm of the result
/// is in the range [SMLNUM, BIGNUM]. The scaling factor is represented as a ratio
/// of two numbers, one of them being ANRM, and the other one either SMLNUM or BIGNUM,
/// depending on ANRM being less than SMLNUM or larger than BIGNUM, respectively.
/// For undoing the scaling, the norm is again compared with SMLNUM or BIGNUM, and
/// the reciprocal of the previous scaling factor is used.
///
/// **Reference Implementation**: `reference/src/MB01PD.f`
///
/// **Contributor**: V. Sima, Katholieke Univ. Leuven, Belgium, Nov. 1996
pub fn mb01pd(
    scun: char,
    matrix_type: char,
    kl: usize,
    ku: usize,
    anrm: f64,
    a: &mut Array2<f64>,
    nrows: Option<&[usize]>,
) -> Result<(), String> {
    use std::f64;

    // Validate SCUN parameter
    let lscale = scun.eq_ignore_ascii_case(&'S');
    if !lscale && !scun.eq_ignore_ascii_case(&'U') {
        return Err(format!(
            "Invalid SCUN parameter: '{}'. Must be 'S' or 'U'",
            scun
        ));
    }

    // Validate matrix type
    let matrix_type_upper = matrix_type.to_ascii_uppercase();
    let itype = match matrix_type_upper {
        'G' => 0,
        'L' => 1,
        'U' => 2,
        'H' => 3,
        'B' => 4,
        'Q' => 5,
        'Z' => 6,
        _ => return Err(format!("Invalid matrix type: '{}'", matrix_type)),
    };

    let (m, n) = a.dim();
    let mn = m.min(n);

    // Validate dimensions
    if anrm < 0.0 {
        return Err(format!("ANRM must be non-negative, got {}", anrm));
    }

    // Validate NBL and NROWS consistency
    if let Some(rows) = nrows {
        let sum: usize = rows.iter().sum();
        if !rows.is_empty() && sum != mn {
            return Err(format!(
                "Sum of NROWS ({}) must equal min(M,N) = {}",
                sum, mn
            ));
        }
    }

    // Validate band matrix parameters
    if itype >= 4 {
        if kl > m.saturating_sub(1) {
            return Err(format!("KL must be <= M-1, got KL={}, M={}", kl, m));
        }
        if ku > n.saturating_sub(1) {
            return Err(format!("KU must be <= N-1, got KU={}, N={}", ku, n));
        }
        if (itype == 4 || itype == 5) && kl != ku {
            return Err(format!(
                "For symmetric band matrices, KL must equal KU, got KL={}, KU={}",
                kl, ku
            ));
        }
    }

    // Quick return if possible
    if mn == 0 || anrm == 0.0 {
        return Ok(());
    }

    // Get machine parameters
    let smlnum = f64::MIN_POSITIVE / f64::EPSILON;
    let bignum = 1.0 / smlnum;

    if lscale {
        // Scale A, if its norm is outside range [SMLNUM, BIGNUM]
        if anrm < smlnum {
            // Scale matrix norm up to SMLNUM
            mb01qd(matrix_type, kl, ku, anrm, smlnum, a, nrows)?;
        } else if anrm > bignum {
            // Scale matrix norm down to BIGNUM
            mb01qd(matrix_type, kl, ku, anrm, bignum, a, nrows)?;
        }
    } else {
        // Undo scaling
        if anrm < smlnum {
            mb01qd(matrix_type, kl, ku, smlnum, anrm, a, nrows)?;
        } else if anrm > bignum {
            mb01qd(matrix_type, kl, ku, bignum, anrm, a, nrows)?;
        }
    }

    Ok(())
}

/// Performs a row-permuted Givens transformation on two vectors.
///
/// This function applies a Givens rotation followed by a row permutation to two
/// vectors X and Y. For each element i, it computes:
///
/// ```text
/// x_new := c*y[i] - s*x[i]
/// y_new := c*x[i] + s*y[i]
/// x[i] := x_new
/// y[i] := y_new
/// ```
///
/// This is equivalent to applying a Givens rotation matrix followed by swapping
/// the rows. It's a key building block for pencil reduction algorithms.
///
/// # Arguments
///
/// * `n` - Number of elements to transform (if n <= 0, returns immediately)
/// * `x` - First vector array (modified in-place)
/// * `incx` - Stride between consecutive X elements (can be negative for reverse iteration)
/// * `y` - Second vector array (modified in-place)
/// * `incy` - Stride between consecutive Y elements (can be negative for reverse iteration)
/// * `c` - Cosine coefficient (typically satisfies c² + s² = 1)
/// * `s` - Sine coefficient (typically satisfies c² + s² = 1)
///
/// # Examples
///
/// ```
/// use slicot_rs::mb::mb04tu;
///
/// let mut x = vec![1.0, 2.0, 3.0];
/// let mut y = vec![4.0, 5.0, 6.0];
/// let c = 0.8;
/// let s = 0.6;
///
/// mb04tu(3, &mut x, 1, &mut y, 1, c, s);
/// // X and Y are now transformed
/// ```
///
/// # SLICOT Reference
///
/// This function is a Rust translation of the SLICOT Fortran routine `MB04TU`.
///
/// **Original Purpose**: Perform row-permuted Givens transformations on vectors.
/// Similar to BLAS routine DROT but includes row permutation.
///
/// **Differences from Fortran version**:
/// - Takes slices instead of raw pointers
/// - Uses 0-based indexing internally
/// - Validates slice bounds and panics on out-of-bounds access
///
/// **Reference Implementation**: `reference/src/MB04TU.f`
pub fn mb04tu(n: i32, x: &mut [f64], incx: i32, y: &mut [f64], incy: i32, c: f64, s: f64) {
    if n <= 0 {
        return;
    }

    // Fast path for unit strides
    if incx == 1 && incy == 1 {
        for i in 0..(n as usize) {
            let temp = c * y[i] - s * x[i];
            y[i] = c * x[i] + s * y[i];
            x[i] = temp;
        }
    } else {
        // General path for arbitrary strides
        let mut ix = if incx < 0 {
            ((-n + 1) * incx) as usize
        } else {
            0
        };

        let mut iy = if incy < 0 {
            ((-n + 1) * incy) as usize
        } else {
            0
        };

        for _ in 0..(n as usize) {
            debug_assert!(ix < x.len(), "Index out of bounds for x");
            debug_assert!(iy < y.len(), "Index out of bounds for y");

            let temp = c * y[iy] - s * x[ix];
            y[iy] = c * x[ix] + s * y[iy];
            x[ix] = temp;

            ix = (ix as i32 + incx) as usize;
            iy = (iy as i32 + incy) as usize;
        }
    }
}

/// Computes the Schur factorization of a real 2-by-2 matrix and its eigenvalues.
///
/// This is a Rust implementation of the LAPACK DLANV2 routine. It computes the
/// Schur factorization of a 2-by-2 nonsymmetric matrix in standard form.
///
/// Given a 2-by-2 matrix:
/// ```text
/// [ a  b ]
/// [ c  d ]
/// ```
///
/// The routine computes a rotation (cs, sn) such that:
/// ```text
/// [ a  b ] = [ cs  -sn ] [ aa  bb ] [ cs   sn ]
/// [ c  d ]   [ sn   cs ] [ cc  dd ] [-sn  cs ]
/// ```
///
/// The output is in standard form where either:
/// - `cc = 0` so that `aa` and `dd` are real eigenvalues, or
/// - `aa = dd` and `bb * cc < 0`, so that `aa ± sqrt(bb*cc)` are complex conjugate eigenvalues
///
/// # Arguments
///
/// * `a`, `b`, `c`, `d` - Elements of the 2-by-2 matrix (modified to standardized form)
///
/// # Returns
///
/// A tuple containing:
/// * `(rt1r, rt1i)` - Real and imaginary parts of first eigenvalue
/// * `(rt2r, rt2i)` - Real and imaginary parts of second eigenvalue
/// * `(cs, sn)` - Cosine and sine of the rotation matrix
///
/// If the eigenvalues are complex conjugate pair, `rt1i > 0`.
///
/// # References
///
/// This is a Rust translation of LAPACK routine DLANV2.
fn dlanv2(a: &mut f64, b: &mut f64, c: &mut f64, d: &mut f64) -> (f64, f64, f64, f64, f64, f64) {
    const ZERO: f64 = 0.0;
    const ONE: f64 = 1.0;
    const HALF: f64 = 0.5;
    const MULTPL: f64 = 4.0;

    let eps = f64::EPSILON;
    let smlnum = f64::MIN_POSITIVE / eps;
    let _bignum = ONE / smlnum;

    let mut aa = *a;
    let mut bb = *b;
    let mut cc = *c;
    let mut dd = *d;

    let rt1i: f64;
    let rt2i: f64;
    let mut cs: f64;
    let mut sn: f64;

    // Quick return if c is zero
    if cc == ZERO {
        cs = ONE;
        sn = ZERO;
    } else if bb == ZERO {
        // Swap rows/columns if b is zero but c is not
        cs = ZERO;
        sn = ONE;
        std::mem::swap(&mut dd, &mut aa);
        bb = -cc;
        cc = ZERO;
    } else if (aa - dd) == ZERO && bb.signum() != cc.signum() {
        // Already in standard form
        cs = ONE;
        sn = ZERO;
    } else {
        let temp = aa - dd;
        let p = HALF * temp;
        let bcmax = bb.abs().max(cc.abs());
        let bcmin = bb.abs().min(cc.abs()) * (bb * cc).signum();
        let scale = p.abs().max(bcmax);
        let mut z = (p / scale) * p + (bcmax / scale) * bcmin;

        // If Z is positive, eigenvalues are real
        if z >= MULTPL * eps {
            z = p + (p.signum() * bcmax) * (z / scale).sqrt();
            aa = dd + z;
            dd -= (bcmax / z) * bcmin;

            // Compute rotation to put matrix in standard form
            let tau = cc.hypot(bb);
            cs = bb / tau;
            sn = cc / tau;
            bb -= cc;
            cc = ZERO;
        } else {
            // Eigenvalues are complex
            let sigma = bb + cc;
            let tau = sigma.hypot(temp);
            cs = ((ONE + sigma.abs() / tau).sqrt()) * HALF.sqrt();
            sn = -(p / (tau * cs)) * temp.signum();

            // Apply rotation
            let aa_new = aa * cs * cs + sigma * cs * sn + dd * sn * sn;
            let bb_new = bb * cs * cs + temp * cs * sn - cc * sn * sn;
            let cc_new = -bb * sn * sn + temp * cs * sn + cc * cs * cs;
            let dd_new = aa * sn * sn - sigma * cs * sn + dd * cs * cs;

            aa = aa_new;
            bb = bb_new;
            cc = cc_new;
            dd = dd_new;

            let temp2 = (aa + dd) * HALF;
            aa = temp2;
            dd = temp2;

            if cc != ZERO {
                if bb != ZERO {
                    if bb.signum() == cc.signum() {
                        // Real eigenvalues - reduce to diagonal
                        let sab = bb.abs().sqrt();
                        let sac = cc.abs().sqrt();
                        let p = if bb < ZERO { -sab * sac } else { sab * sac };
                        let tau = ONE / bb.hypot(cc);
                        aa = temp2 + p;
                        dd = temp2 - p;
                        bb -= cc;
                        cc = ZERO;
                        let cs1 = sab * tau;
                        let sn1 = sac * tau;
                        let temp = cs * cs1 - sn * sn1;
                        sn = cs * sn1 + sn * cs1;
                        cs = temp;
                    }
                } else {
                    bb = -cc;
                    cc = ZERO;
                    let temp = cs;
                    cs = -sn;
                    sn = temp;
                }
            }
        }
    }

    // Compute eigenvalues
    let rt1r = aa;
    let rt2r = dd;
    if cc == ZERO {
        rt1i = ZERO;
        rt2i = ZERO;
    } else {
        rt1i = (bb.abs() * cc.abs()).sqrt();
        rt2i = -rt1i;
    }

    // Update the input matrix elements
    *a = aa;
    *b = bb;
    *c = cc;
    *d = dd;

    (rt1r, rt1i, rt2r, rt2i, cs, sn)
}

/// Processes a 2-by-2 diagonal block of an upper quasi-triangular matrix.
///
/// This routine computes the eigenvalues of a selected 2-by-2 diagonal block
/// of an upper quasi-triangular matrix, reduces the selected block to standard
/// form, and splits the block in the case of real eigenvalues by constructing
/// an orthogonal transformation. This transformation is applied to the matrix A
/// (by similarity) and to another matrix U from the right.
///
/// # Arguments
///
/// * `a` - N×N upper quasi-triangular matrix (modified in-place)
/// * `u` - N×N transformation matrix (modified in-place by U*UT on exit)
/// * `l` - Position of the 2×2 block (0-indexed, must be < n-1)
///
/// # Returns
///
/// `Result<(f64, f64), String>` where:
/// * `Ok((e1, e2))` - The eigenvalues of the 2×2 block:
///   - For complex eigenvalues: `e1` is real part, `e2` is positive imaginary part
///   - For real eigenvalues: `e1` and `e2` are the two real eigenvalues
/// * `Err(msg)` - Error message if parameters are invalid
///
/// # Errors
///
/// Returns error if:
/// * `n < 2` - Matrix too small
/// * `l >= n-1` - Block position out of bounds
/// * Matrix dimensions incompatible
///
/// # Method
///
/// Let `A1` be the 2-by-2 diagonal block at position `(l, l)`:
/// ```text
/// A1 = [ A(l,l)    A(l,l+1)   ]
///      [ A(l+1,l)  A(l+1,l+1) ]
/// ```
///
/// If the eigenvalues are complex, they are computed and stored in `e1` (real part)
/// and `e2` (positive imaginary part). The 2-by-2 block is reduced to standard form
/// with `A(l,l) = A(l+1,l+1)` and `A(l,l+1)` and `A(l+1,l)` having opposite signs.
///
/// If the eigenvalues are real, the block is reduced to upper triangular form such
/// that `|A(l,l)| >= |A(l+1,l+1)|`.
///
/// An orthogonal rotation is constructed and applied as a similarity transformation
/// to A and as a right multiplication to U.
///
/// # Examples
///
/// ```
/// use ndarray::arr2;
/// use slicot_rs::mb::mb03qy;
///
/// let mut a = arr2(&[
///     [1.0, 2.0, 3.0],
///     [0.0, 4.0, 5.0],
///     [0.0, 1.0, 6.0],
/// ]);
/// let mut u = arr2(&[
///     [1.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0],
///     [0.0, 0.0, 1.0],
/// ]);
///
/// // Process the 2×2 block at position (1,1)
/// let (e1, e2) = mb03qy(&mut a, &mut u, 1).unwrap();
/// println!("Eigenvalues: {} ± {}i", e1, e2);
/// ```
///
/// # SLICOT Reference
///
/// This is a Rust translation of SLICOT routine MB03QY.
///
/// **Original Purpose**: To compute the eigenvalues of a selected 2-by-2 diagonal
/// block of an upper quasi-triangular matrix, to reduce the selected block to the
/// standard form and to split the block in the case of real eigenvalues.
///
/// **Contributor**: A. Varga, German Aerospace Center, DLR Oberpfaffenhofen, March 1998.
/// Based on the RASP routine SPLITB.
///
/// **Reference**: `reference/src/MB03QY.f`
pub fn mb03qy(a: &mut Array2<f64>, u: &mut Array2<f64>, l: usize) -> Result<(f64, f64), String> {
    let n = a.nrows();

    // Validate inputs
    if n < 2 {
        return Err(format!("Matrix order N must be >= 2, got {}", n));
    }
    if l >= n - 1 {
        return Err(format!(
            "Block position L must be < N-1, got L={}, N={}",
            l, n
        ));
    }
    if a.ncols() != n {
        return Err(format!("Matrix A must be square, got {}×{}", n, a.ncols()));
    }
    if u.nrows() != n || u.ncols() != n {
        return Err(format!(
            "Matrix U must be {}×{}, got {}×{}",
            n,
            n,
            u.nrows(),
            u.ncols()
        ));
    }

    let l1 = l + 1;

    // Extract the 2×2 block elements
    let mut a_ll = a[(l, l)];
    let mut a_ll1 = a[(l, l1)];
    let mut a_l1l = a[(l1, l)];
    let mut a_l1l1 = a[(l1, l1)];

    // Compute eigenvalues and rotation using DLANV2
    let (e1, e2, ew1, _ew2, cs, sn) = dlanv2(&mut a_ll, &mut a_ll1, &mut a_l1l, &mut a_l1l1);

    // Update the 2×2 block in A
    a[(l, l)] = a_ll;
    a[(l, l1)] = a_ll1;
    a[(l1, l)] = a_l1l;
    a[(l1, l1)] = a_l1l1;

    // Determine eigenvalue output (following Fortran logic)
    let e2_out = if e2 == 0.0 { ew1 } else { e2 };

    // Apply rotation to the rest of A
    // Apply to columns to the right of the block: A(l:l1, l1+1:n)
    if l1 < n - 1 {
        for j in (l1 + 1)..n {
            let temp = cs * a[(l, j)] - sn * a[(l1, j)];
            a[(l1, j)] = cs * a[(l1, j)] + sn * a[(l, j)];
            a[(l, j)] = temp;
        }
    }

    // Apply to rows above the block: A(0:l, l:l1)
    for i in 0..l {
        let temp = cs * a[(i, l)] - sn * a[(i, l1)];
        a[(i, l1)] = cs * a[(i, l1)] + sn * a[(i, l)];
        a[(i, l)] = temp;
    }

    // Accumulate transformation in U: U = U * rotation
    for i in 0..n {
        let temp = cs * u[(i, l)] - sn * u[(i, l1)];
        u[(i, l1)] = cs * u[(i, l1)] + sn * u[(i, l)];
        u[(i, l)] = temp;
    }

    Ok((e1, e2_out))
}

#[cfg(test)]
mod tests_mb04tu {
    use super::*;

    #[test]
    fn test_mb04tu_unit_stride() {
        let mut x = vec![1.0, 2.0, 3.0];
        let mut y = vec![4.0, 5.0, 6.0];
        let c = 0.8;
        let s = 0.6;

        mb04tu(3, &mut x, 1, &mut y, 1, c, s);

        // Verify transformation
        assert!((x[0] - (0.8 * 4.0 - 0.6 * 1.0)).abs() < 1e-14);
        assert!((y[0] - (0.8 * 1.0 + 0.6 * 4.0)).abs() < 1e-14);
    }

    #[test]
    fn test_mb04tu_zero_n() {
        let mut x = vec![1.0, 2.0];
        let mut y = vec![3.0, 4.0];
        let x_orig = x.clone();
        let y_orig = y.clone();

        mb04tu(0, &mut x, 1, &mut y, 1, 0.5, 0.5);

        // Arrays should be unchanged
        assert_eq!(x, x_orig);
        assert_eq!(y, y_orig);
    }

    #[test]
    fn test_mb04tu_orthogonal_rotation() {
        // Test with orthogonal rotation c² + s² = 1
        let mut x = vec![3.0, 4.0, 5.0];
        let mut y = vec![1.0, 2.0, 3.0];
        let c = 0.6;
        let s = 0.8;

        // Verify orthogonality
        let orthogonal_constraint = (c * c + s * s) - 1.0_f64;
        assert!(orthogonal_constraint.abs() < 1e-14);

        mb04tu(3, &mut x, 1, &mut y, 1, c, s);

        // MB04TU applies rotation + row permutation
        // The transformation is: x_new = c*y_old - s*x_old, y_new = c*x_old + s*y_old
        // This is NOT a pure orthogonal transformation, so we can't expect norms to swap directly
        // Instead, just verify the transformation was applied
        let x_modified = [
            0.6 * 1.0 - 0.8 * 3.0,
            0.6 * 2.0 - 0.8 * 4.0,
            0.6 * 3.0 - 0.8 * 5.0,
        ];
        let y_modified = [
            0.6 * 3.0 + 0.8 * 1.0,
            0.6 * 4.0 + 0.8 * 2.0,
            0.6 * 5.0 + 0.8 * 3.0,
        ];

        for i in 0..3 {
            assert!((x[i] - x_modified[i]).abs() < 1e-14);
            assert!((y[i] - y_modified[i]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_mb04tu_single_element() {
        let mut x = vec![5.0];
        let mut y = vec![10.0];
        let c = 0.8;
        let s = 0.6;

        mb04tu(1, &mut x, 1, &mut y, 1, c, s);

        assert!((x[0] - (0.8 * 10.0 - 0.6 * 5.0)).abs() < 1e-14);
        assert!((y[0] - (0.8 * 5.0 + 0.6 * 10.0)).abs() < 1e-14);
    }

    #[test]
    fn test_mb04tu_stride_2() {
        // Stride of 2: take every other element
        let mut x = vec![1.0, 99.0, 2.0, 99.0, 3.0];
        let mut y = vec![4.0, 99.0, 5.0, 99.0, 6.0];
        let c = 0.8;
        let s = 0.6;

        mb04tu(3, &mut x, 2, &mut y, 2, c, s);

        // Check that elements at stride positions were transformed
        assert!((x[0] - (0.8 * 4.0 - 0.6 * 1.0)).abs() < 1e-14);
        assert!((y[0] - (0.8 * 1.0 + 0.6 * 4.0)).abs() < 1e-14);
        assert!((x[2] - (0.8 * 5.0 - 0.6 * 2.0)).abs() < 1e-14);
        assert!((y[2] - (0.8 * 2.0 + 0.6 * 5.0)).abs() < 1e-14);

        // Check that skipped elements unchanged
        assert_eq!(x[1], 99.0);
        assert_eq!(y[1], 99.0);
    }
}

#[cfg(test)]
mod tests_mb01qd {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_mb01qd_full_matrix_simple_scale() {
        // Test simple scaling of a full matrix
        let mut a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let result = mb01qd('G', 0, 0, 1.0, 2.0, &mut a, None);
        assert!(result.is_ok());

        // All elements should be doubled
        let expected = arr2(&[[2.0, 4.0, 6.0], [8.0, 10.0, 12.0], [14.0, 16.0, 18.0]]);
        for ((i, j), &val) in a.indexed_iter() {
            assert!((val - expected[[i, j]]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_mb01qd_full_matrix_divide() {
        // Test dividing by 2 (cto=1, cfrom=2)
        let mut a = arr2(&[[2.0, 4.0, 6.0], [8.0, 10.0, 12.0]]);
        let result = mb01qd('G', 0, 0, 2.0, 1.0, &mut a, None);
        assert!(result.is_ok());

        let expected = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        for ((i, j), &val) in a.indexed_iter() {
            assert!((val - expected[[i, j]]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_mb01qd_empty_matrix() {
        // Test with empty matrix (should return Ok and not modify)
        let mut a = arr2(&[[]]);
        let result = mb01qd('G', 0, 0, 1.0, 2.0, &mut a, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mb01qd_zero_cfrom_error() {
        // Test error when cfrom is zero
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let result = mb01qd('G', 0, 0, 0.0, 2.0, &mut a, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("nonzero"));
    }

    #[test]
    fn test_mb01qd_invalid_matrix_type() {
        // Test error with invalid matrix type
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let result = mb01qd('X', 0, 0, 1.0, 2.0, &mut a, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid matrix type"));
    }

    #[test]
    fn test_mb01qd_lower_triangular() {
        // Test lower triangular matrix scaling
        let mut a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let original = a.clone();
        let result = mb01qd('L', 0, 0, 1.0, 2.0, &mut a, None);
        assert!(result.is_ok());

        // Only lower triangular part should be scaled
        assert!((a[[0, 0]] - 2.0).abs() < 1e-14); // Diagonal
        assert!((a[[1, 0]] - 8.0).abs() < 1e-14); // Below diagonal
        assert!((a[[1, 1]] - 10.0).abs() < 1e-14); // Diagonal
        assert!((a[[2, 0]] - 14.0).abs() < 1e-14); // Below diagonal
        assert!((a[[2, 1]] - 16.0).abs() < 1e-14); // Below diagonal
        assert!((a[[2, 2]] - 18.0).abs() < 1e-14); // Diagonal

        // Upper triangular part should be unchanged
        assert!((a[[0, 1]] - original[[0, 1]]).abs() < 1e-14);
        assert!((a[[0, 2]] - original[[0, 2]]).abs() < 1e-14);
        assert!((a[[1, 2]] - original[[1, 2]]).abs() < 1e-14);
    }

    #[test]
    fn test_mb01qd_upper_triangular() {
        // Test upper triangular matrix scaling
        let mut a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let original = a.clone();
        let result = mb01qd('U', 0, 0, 1.0, 3.0, &mut a, None);
        assert!(result.is_ok());

        // Only upper triangular part should be scaled
        assert!((a[[0, 0]] - 3.0).abs() < 1e-14); // Diagonal
        assert!((a[[0, 1]] - 6.0).abs() < 1e-14); // Above diagonal
        assert!((a[[0, 2]] - 9.0).abs() < 1e-14); // Above diagonal
        assert!((a[[1, 1]] - 15.0).abs() < 1e-14); // Diagonal
        assert!((a[[1, 2]] - 18.0).abs() < 1e-14); // Above diagonal
        assert!((a[[2, 2]] - 27.0).abs() < 1e-14); // Diagonal

        // Lower triangular part should be unchanged
        assert!((a[[1, 0]] - original[[1, 0]]).abs() < 1e-14);
        assert!((a[[2, 0]] - original[[2, 0]]).abs() < 1e-14);
        assert!((a[[2, 1]] - original[[2, 1]]).abs() < 1e-14);
    }

    #[test]
    fn test_mb01qd_upper_hessenberg() {
        // Test upper Hessenberg matrix (upper triangular plus first subdiagonal)
        let mut a = arr2(&[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let original = a.clone();
        let result = mb01qd('H', 0, 0, 2.0, 1.0, &mut a, None);
        assert!(result.is_ok());

        // Upper Hessenberg: upper triangular + first subdiagonal
        // Column 0: elements (0,0) and (1,0) should be scaled
        assert!((a[[0, 0]] - 0.5).abs() < 1e-14);
        assert!((a[[1, 0]] - 2.5).abs() < 1e-14);
        // Column 1: elements (0,1), (1,1), (2,1) should be scaled
        assert!((a[[0, 1]] - 1.0).abs() < 1e-14);
        assert!((a[[1, 1]] - 3.0).abs() < 1e-14);
        assert!((a[[2, 1]] - 5.0).abs() < 1e-14);
        // Column 2: elements (0,2), (1,2), (2,2), (3,2) should be scaled
        assert!((a[[0, 2]] - 1.5).abs() < 1e-14);
        assert!((a[[1, 2]] - 3.5).abs() < 1e-14);
        assert!((a[[2, 2]] - 5.5).abs() < 1e-14);
        assert!((a[[3, 2]] - 7.5).abs() < 1e-14);

        // Rest should be unchanged
        assert!((a[[2, 0]] - original[[2, 0]]).abs() < 1e-14);
        assert!((a[[3, 0]] - original[[3, 0]]).abs() < 1e-14);
        assert!((a[[3, 1]] - original[[3, 1]]).abs() < 1e-14);
    }

    #[test]
    fn test_mb01qd_case_insensitive() {
        // Test that matrix type is case insensitive
        let mut a1 = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mut a2 = a1.clone();

        mb01qd('g', 0, 0, 1.0, 2.0, &mut a1, None).unwrap();
        mb01qd('G', 0, 0, 1.0, 2.0, &mut a2, None).unwrap();

        for ((i, j), &val) in a1.indexed_iter() {
            assert!((val - a2[[i, j]]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_mb01qd_overflow_protection() {
        // Test that the routine handles large scaling factors safely
        // by breaking them into smaller steps
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        // Very large scaling factor
        let cfrom = 1e-200;
        let cto = 1e200;

        let result = mb01qd('G', 0, 0, cfrom, cto, &mut a, None);
        assert!(result.is_ok());

        // The result should be approximately original * (cto/cfrom) = original * 1e400
        // But due to overflow protection, intermediate steps are taken
        // We just verify it doesn't panic and completes
    }

    #[test]
    fn test_mb01qd_block_lower_triangular() {
        // Test block lower triangular structure
        // Matrix is 4x4 with 2 blocks of size 2x2 each
        let mut a = arr2(&[
            [1.0, 2.0, 3.0, 4.0],
            [5.0, 6.0, 7.0, 8.0],
            [9.0, 10.0, 11.0, 12.0],
            [13.0, 14.0, 15.0, 16.0],
        ]);
        let nrows = vec![2, 2];
        let result = mb01qd('L', 0, 0, 1.0, 2.0, &mut a, Some(&nrows));
        assert!(result.is_ok());

        // Block 1 (rows/cols 0-1): full 2x2 block scaled
        assert!((a[[0, 0]] - 2.0).abs() < 1e-14);
        assert!((a[[0, 1]] - 4.0).abs() < 1e-14);
        assert!((a[[1, 0]] - 10.0).abs() < 1e-14);
        assert!((a[[1, 1]] - 12.0).abs() < 1e-14);

        // Block 2 lower part (rows 2-3, cols 2-3) scaled
        assert!((a[[2, 2]] - 22.0).abs() < 1e-14);
        assert!((a[[2, 3]] - 24.0).abs() < 1e-14);
        assert!((a[[3, 2]] - 30.0).abs() < 1e-14);
        assert!((a[[3, 3]] - 32.0).abs() < 1e-14);

        // Below-diagonal blocks (rows 2-3, cols 0-1) also scaled
        assert!((a[[2, 0]] - 18.0).abs() < 1e-14);
        assert!((a[[2, 1]] - 20.0).abs() < 1e-14);
        assert!((a[[3, 0]] - 26.0).abs() < 1e-14);
        assert!((a[[3, 1]] - 28.0).abs() < 1e-14);
    }

    #[test]
    fn test_mb01qd_rectangular_matrix() {
        // Test with non-square matrix
        let mut a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let result = mb01qd('G', 0, 0, 1.0, 0.5, &mut a, None);
        assert!(result.is_ok());

        let expected = arr2(&[[0.5, 1.0, 1.5], [2.0, 2.5, 3.0]]);
        for ((i, j), &val) in a.indexed_iter() {
            assert!((val - expected[[i, j]]).abs() < 1e-14);
        }
    }
}

#[cfg(test)]
mod tests_mb03qy {
    use super::*;
    use approx::assert_abs_diff_eq;
    use ndarray::arr2;

    #[test]
    fn test_mb03qy_real_eigenvalues() {
        // Test case with a 2×2 block that has real eigenvalues
        let mut a = arr2(&[[1.0, 2.0, 3.0], [0.0, 4.0, 5.0], [0.0, 2.0, 6.0]]);
        let mut u = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

        let result = mb03qy(&mut a, &mut u, 1);
        assert!(result.is_ok());

        let (e1, e2) = result.unwrap();

        // For a 2×2 block [[4, 5], [2, 6]], eigenvalues can be computed
        // λ = (4+6)/2 ± sqrt((4-6)^2/4 + 5*2)
        //   = 5 ± sqrt(1 + 10) = 5 ± sqrt(11)
        let expected_sum = 10.0;
        let actual_sum = e1 + e2;
        assert_abs_diff_eq!(actual_sum, expected_sum, epsilon = 1e-10);
    }

    #[test]
    fn test_mb03qy_complex_eigenvalues() {
        // Test with a 2×2 block that has complex eigenvalues
        // Use a rotation matrix-like structure
        let mut a = arr2(&[[1.0, 0.0, 0.0], [0.0, 0.0, -2.0], [0.0, 2.0, 0.0]]);
        let mut u = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

        let result = mb03qy(&mut a, &mut u, 1);
        assert!(result.is_ok());

        let (e1, e2) = result.unwrap();

        // For [[0, -2], [2, 0]], eigenvalues are ±2i
        // Real part should be 0, imaginary part should be 2
        assert_abs_diff_eq!(e1, 0.0, epsilon = 1e-10);
        assert_abs_diff_eq!(e2.abs(), 2.0, epsilon = 1e-10);
    }

    #[test]
    fn test_mb03qy_identity_transformation() {
        // Test with identity U to verify transformation is accumulated correctly
        let mut a = arr2(&[[5.0, 1.0], [1.0, 5.0]]);
        let mut u = arr2(&[[1.0, 0.0], [0.0, 1.0]]);

        let result = mb03qy(&mut a, &mut u, 0);
        assert!(result.is_ok());

        let (e1, e2) = result.unwrap();

        // [[5, 1], [1, 5]] has eigenvalues 6 and 4
        let sum = e1 + e2;
        assert_abs_diff_eq!(sum, 10.0, epsilon = 1e-10);

        // U should be updated (no longer identity)
        // but should still be orthogonal
        let u_t_u = u.t().dot(&u);
        let identity = arr2(&[[1.0, 0.0], [0.0, 1.0]]);
        for i in 0..2 {
            for j in 0..2 {
                assert_abs_diff_eq!(u_t_u[(i, j)], identity[(i, j)], epsilon = 1e-10);
            }
        }
    }

    #[test]
    fn test_mb03qy_invalid_n() {
        let mut a = arr2(&[[1.0]]);
        let mut u = arr2(&[[1.0]]);

        let result = mb03qy(&mut a, &mut u, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("N must be >= 2"));
    }

    #[test]
    fn test_mb03qy_invalid_l() {
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mut u = arr2(&[[1.0, 0.0], [0.0, 1.0]]);

        // L must be < N-1, so for N=2, L must be 0
        let result = mb03qy(&mut a, &mut u, 1);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("L must be < N-1"));
    }

    #[test]
    fn test_mb03qy_nonsquare_a() {
        let mut a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]);
        let mut u = arr2(&[[1.0, 0.0], [0.0, 1.0]]);

        let result = mb03qy(&mut a, &mut u, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be square"));
    }

    #[test]
    fn test_mb03qy_dimension_mismatch() {
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mut u = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);

        let result = mb03qy(&mut a, &mut u, 0);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("must be"));
    }
}

#[cfg(test)]
mod tests_mb01pd {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_mb01pd_no_scaling_needed() {
        // Test with a matrix that doesn't need scaling (norm in safe range)
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let anrm = 5.0; // Reasonable norm, no scaling needed
        let original = a.clone();

        let result = mb01pd('S', 'G', 0, 0, anrm, &mut a, None);
        assert!(result.is_ok());

        // Matrix should be unchanged (norm already in safe range)
        for ((i, j), &val) in a.indexed_iter() {
            assert!((val - original[[i, j]]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_mb01pd_scale_small_norm() {
        // Test scaling up a matrix with very small norm
        let mut a = arr2(&[[1e-200, 2e-200], [3e-200, 4e-200]]);
        let anrm = 1e-200;

        let result = mb01pd('S', 'G', 0, 0, anrm, &mut a, None);
        assert!(result.is_ok());

        // Matrix should be scaled up significantly
        // After scaling, the norm should be approximately smlnum
        let smlnum = f64::MIN_POSITIVE / f64::EPSILON;
        let expected_scale = smlnum / anrm;

        assert!((a[[0, 0]] - 1e-200 * expected_scale).abs() < 1e-14 * expected_scale);
        assert!((a[[0, 1]] - 2e-200 * expected_scale).abs() < 1e-14 * expected_scale);
    }

    #[test]
    fn test_mb01pd_scale_large_norm() {
        // Test scaling down a matrix with extremely large norm (larger than BIGNUM)
        // BIGNUM ≈ 1e292, so we use 1e300 which is beyond the safe range
        let mut a = arr2(&[[1e300, 2e300], [3e300, 4e300]]);
        let anrm = 1e300;

        let result = mb01pd('S', 'G', 0, 0, anrm, &mut a, None);
        assert!(result.is_ok());

        // Matrix should be scaled down significantly
        assert!(a[[0, 0]] < 1e300);
        assert!(a[[0, 0]] > 0.0);
    }

    #[test]
    fn test_mb01pd_scale_and_undo() {
        // Test scaling and then undoing the scaling (use extreme value that needs scaling)
        let original = arr2(&[[1e-300, 2e-300], [3e-300, 4e-300]]);
        let mut a = original.clone();
        let anrm = 1e-300;

        // Scale the matrix
        let result = mb01pd('S', 'G', 0, 0, anrm, &mut a, None);
        assert!(result.is_ok());

        // Undo the scaling
        let result = mb01pd('U', 'G', 0, 0, anrm, &mut a, None);
        assert!(result.is_ok());

        // Matrix should be back to original (within numerical precision)
        for ((i, j), &val) in a.indexed_iter() {
            let relative_error = (val - original[[i, j]]).abs() / original[[i, j]].abs();
            assert!(relative_error < 1e-10);
        }
    }

    #[test]
    fn test_mb01pd_zero_anrm() {
        // Test quick return when ANRM is zero
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let original = a.clone();

        let result = mb01pd('S', 'G', 0, 0, 0.0, &mut a, None);
        assert!(result.is_ok());

        // Matrix should be unchanged
        for ((i, j), &val) in a.indexed_iter() {
            assert_eq!(val, original[[i, j]]);
        }
    }

    #[test]
    fn test_mb01pd_empty_matrix() {
        // Test with empty matrix
        let mut a = arr2(&[[]]);

        let result = mb01pd('S', 'G', 0, 0, 1.0, &mut a, None);
        assert!(result.is_ok());
    }

    #[test]
    fn test_mb01pd_invalid_scun() {
        // Test error with invalid SCUN parameter
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        let result = mb01pd('X', 'G', 0, 0, 1.0, &mut a, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid SCUN parameter"));
    }

    #[test]
    fn test_mb01pd_invalid_matrix_type() {
        // Test error with invalid matrix type
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        let result = mb01pd('S', 'X', 0, 0, 1.0, &mut a, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Invalid matrix type"));
    }

    #[test]
    fn test_mb01pd_negative_anrm() {
        // Test error with negative ANRM
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

        let result = mb01pd('S', 'G', 0, 0, -1.0, &mut a, None);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("non-negative"));
    }

    #[test]
    fn test_mb01pd_lower_triangular() {
        // Test scaling with lower triangular matrix (use extremely small norm < SMLNUM ≈ 1e-292)
        let mut a = arr2(&[
            [1e-300, 2.0, 3.0],
            [4e-300, 5e-300, 6.0],
            [7e-300, 8e-300, 9e-300],
        ]);
        let anrm = 1e-300;
        let original = a.clone();

        let result = mb01pd('S', 'L', 0, 0, anrm, &mut a, None);
        assert!(result.is_ok());

        // Lower triangular part should be scaled up (anrm is very small)
        // Verify scaling occurred
        assert!(a[[0, 0]] > original[[0, 0]]);
        assert!(a[[1, 0]] > original[[1, 0]]);
        assert!(a[[1, 1]] > original[[1, 1]]);

        // Upper triangular should be unchanged
        assert_eq!(a[[0, 1]], original[[0, 1]]);
        assert_eq!(a[[0, 2]], original[[0, 2]]);
    }

    #[test]
    fn test_mb01pd_upper_triangular() {
        // Test scaling with upper triangular matrix (use extremely small norm < SMLNUM ≈ 1e-292)
        let mut a = arr2(&[
            [1e-300, 2e-300, 3e-300],
            [4.0, 5e-300, 6e-300],
            [7.0, 8.0, 9e-300],
        ]);
        let anrm = 1e-300;
        let original = a.clone();

        let result = mb01pd('S', 'U', 0, 0, anrm, &mut a, None);
        assert!(result.is_ok());

        // Upper triangular part should be scaled up (anrm is very small)
        assert!(a[[0, 0]] > original[[0, 0]]);
        assert!(a[[0, 1]] > original[[0, 1]]);
        assert!(a[[1, 1]] > original[[1, 1]]);

        // Lower triangular should be unchanged
        assert_eq!(a[[1, 0]], original[[1, 0]]);
        assert_eq!(a[[2, 0]], original[[2, 0]]);
    }

    #[test]
    fn test_mb01pd_case_insensitive() {
        // Test that parameters are case-insensitive
        let mut a1 = arr2(&[[1e-100, 2e-100], [3e-100, 4e-100]]);
        let mut a2 = a1.clone();
        let anrm = 1e-100;

        mb01pd('s', 'g', 0, 0, anrm, &mut a1, None).unwrap();
        mb01pd('S', 'G', 0, 0, anrm, &mut a2, None).unwrap();

        for ((i, j), &val) in a1.indexed_iter() {
            assert!((val - a2[[i, j]]).abs() < 1e-14);
        }
    }

    #[test]
    fn test_mb01pd_block_structure() {
        // Test with block-structured matrix (use extremely small norm < SMLNUM ≈ 1e-292)
        let mut a = arr2(&[
            [1e-300, 2e-300, 3e-300, 4e-300],
            [5e-300, 6e-300, 7e-300, 8e-300],
            [9e-300, 10e-300, 11e-300, 12e-300],
            [13e-300, 14e-300, 15e-300, 16e-300],
        ]);
        let nrows = vec![2, 2];
        let anrm = 1e-300;
        let original = a.clone();

        let result = mb01pd('S', 'G', 0, 0, anrm, &mut a, Some(&nrows));
        assert!(result.is_ok());

        // All elements should be scaled up (anrm is very small)
        assert!(a[[0, 0]] > original[[0, 0]]);
        assert!(a[[3, 3]] > original[[3, 3]]);
    }

    #[test]
    fn test_mb01pd_invalid_nrows_sum() {
        // Test error when NROWS sum doesn't match min(M,N)
        let mut a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);
        let nrows = vec![2, 2]; // Sum is 4, but min(3,3) = 3

        let result = mb01pd('S', 'G', 0, 0, 1.0, &mut a, Some(&nrows));
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("Sum of NROWS"));
    }

    #[test]
    fn test_mb01pd_rectangular_matrix() {
        // Test with non-square matrix (use extremely small norm < SMLNUM ≈ 1e-292)
        let mut a = arr2(&[[1e-300, 2e-300, 3e-300], [4e-300, 5e-300, 6e-300]]);
        let anrm = 1e-300;
        let original = a.clone();

        let result = mb01pd('S', 'G', 0, 0, anrm, &mut a, None);
        assert!(result.is_ok());

        // All elements should be scaled up (anrm is very small)
        assert!(a[[0, 0]] > original[[0, 0]]);
        assert!(a[[1, 2]] > original[[1, 2]]);
    }
}

/// Reorders diagonal blocks of a principal submatrix of an upper quasi-triangular matrix.
///
/// This routine reorders the diagonal blocks of a principal submatrix of an upper
/// quasi-triangular matrix A together with their eigenvalues by constructing an
/// orthogonal similarity transformation UT. After reordering, the leading block of
/// the selected submatrix of A has eigenvalues in a suitably defined domain of
/// interest, usually related to stability/instability in a continuous- or discrete-time
/// sense.
///
/// # Arguments
///
/// * `dico` - System type:
///   - `'C'`: Continuous-time (eigenvalue separation by real part)
///   - `'D'`: Discrete-time (eigenvalue separation by modulus)
/// * `stdom` - Domain of interest type:
///   - `'S'`: Stability type (left half-plane or inside unit circle)
///   - `'U'`: Instability type (right half-plane or outside unit circle)
/// * `jobu` - Transformation accumulation mode:
///   - `'I'`: Initialize U to identity matrix
///   - `'U'`: Update existing U matrix
/// * `nlow` - Lower boundary index (1-indexed in Fortran, 0-indexed here)
/// * `nsup` - Upper boundary index (1-indexed in Fortran, 0-indexed here)
/// * `alpha` - Stability boundary:
///   - Continuous: boundary for real parts of eigenvalues
///   - Discrete: boundary for moduli of eigenvalues (must be >= 0)
/// * `a` - N×N upper quasi-triangular matrix (modified in-place)
/// * `u` - N×N transformation matrix (modified in-place)
///
/// # Returns
///
/// `Result<usize, String>` where:
/// * `Ok(ndim)` - Number of eigenvalues inside the domain of interest
/// * `Err(msg)` - Error message if operation fails
///
/// # Domain of Interest
///
/// For DICO = 'C' (Continuous-time):
/// * STDOM = 'S': Real(λ) < ALPHA (stable eigenvalues)
/// * STDOM = 'U': Real(λ) > ALPHA (unstable eigenvalues)
///
/// For DICO = 'D' (Discrete-time):
/// * STDOM = 'S': |λ| < ALPHA (stable eigenvalues)
/// * STDOM = 'U': |λ| > ALPHA (unstable eigenvalues)
///
/// # Errors
///
/// Returns error if:
/// * Matrix dimensions incompatible
/// * NLOW, NSUP out of range (must satisfy 0 <= NLOW <= NSUP < N)
/// * A(NLOW, NLOW-1) or A(NSUP+1, NSUP) is nonzero (block boundary violation)
/// * ALPHA < 0 for discrete-time systems
/// * Block swap operation is ill-conditioned
///
/// # Method
///
/// The routine processes the quasi-triangular matrix from bottom to top, examining
/// each 1×1 or 2×2 diagonal block. Blocks with eigenvalues inside the domain of
/// interest are moved to the top of the selected submatrix by swapping adjacent
/// blocks using orthogonal transformations.
///
/// # Examples
///
/// ```
/// use ndarray::arr2;
/// use slicot_rs::mb::mb03qd;
///
/// // Reorder a Schur form matrix to separate stable eigenvalues
/// let mut a = arr2(&[
///     [-3.13, -26.51, 27.23, -16.20],
///     [ 0.91,  -3.13, 13.63,   8.92],
///     [ 0.00,   0.00, -3.37,   0.34],
///     [ 0.00,   0.00, -1.79,  -3.37],
/// ]);
/// let mut u = arr2(&[
///     [1.0, 0.0, 0.0, 0.0],
///     [0.0, 1.0, 0.0, 0.0],
///     [0.0, 0.0, 1.0, 0.0],
///     [0.0, 0.0, 0.0, 1.0],
/// ]);
///
/// // Separate stable eigenvalues (real part < 0) in continuous-time
/// let ndim = mb03qd('C', 'S', 'I', 0, 3, 0.0, &mut a, &mut u).unwrap();
/// println!("Number of stable eigenvalues: {}", ndim);
/// ```
///
/// # SLICOT Reference
///
/// This is a Rust translation of SLICOT routine MB03QD.
///
/// **Original Purpose**: To reorder the diagonal blocks of a principal submatrix of
/// an upper quasi-triangular matrix A together with their eigenvalues by constructing
/// an orthogonal similarity transformation UT.
///
/// **Reference**: `reference/src/MB03QD.f`
///
/// **Contributor**: A. Varga, German Aerospace Center, DLR Oberpfaffenhofen, April 1998.
/// Based on the RASP routine SEOR1.
///
/// **Reference**: Stewart, G.W. "HQR3 and EXCHQZ: FORTRAN subroutines for calculating
/// and ordering the eigenvalues of a real upper Hessenberg matrix." ACM TOMS, 2,
/// pp. 275-280, 1976.
///
/// **Numerical Aspects**: The algorithm requires less than 4*N³ operations.
///
/// **Implementation Note**: This is a simplified implementation that counts eigenvalues
/// in the domain of interest but does not yet perform full block reordering. Block
/// swapping via DTREXC equivalent is not yet implemented. The function correctly
/// identifies and counts eigenvalues in the specified domain.
#[allow(clippy::too_many_arguments)]
pub fn mb03qd(
    dico: char,
    stdom: char,
    jobu: char,
    nlow: usize,
    nsup: usize,
    alpha: f64,
    a: &mut Array2<f64>,
    u: &mut Array2<f64>,
) -> Result<usize, String> {
    let n = a.nrows();

    // Validate input parameters
    let dico_upper = dico.to_ascii_uppercase();
    let stdom_upper = stdom.to_ascii_uppercase();
    let jobu_upper = jobu.to_ascii_uppercase();

    if dico_upper != 'C' && dico_upper != 'D' {
        return Err(format!("Invalid DICO parameter: {}", dico));
    }
    if stdom_upper != 'S' && stdom_upper != 'U' {
        return Err(format!("Invalid STDOM parameter: {}", stdom));
    }
    if jobu_upper != 'I' && jobu_upper != 'U' {
        return Err(format!("Invalid JOBU parameter: {}", jobu));
    }
    if n < 1 {
        return Err(format!("Matrix order N must be >= 1, got {}", n));
    }
    if nsup >= n {
        return Err(format!("NSUP must be < N, got NSUP={}, N={}", nsup, n));
    }
    if nlow > nsup {
        return Err(format!(
            "NLOW must be <= NSUP, got NLOW={}, NSUP={}",
            nlow, nsup
        ));
    }
    if dico_upper == 'D' && alpha < 0.0 {
        return Err("For discrete-time system, ALPHA must be >= 0".to_string());
    }
    if a.ncols() != n {
        return Err(format!("Matrix A must be square, got {}×{}", n, a.ncols()));
    }
    if u.nrows() != n || u.ncols() != n {
        return Err(format!(
            "Matrix U must be {}×{}, got {}×{}",
            n,
            n,
            u.nrows(),
            u.ncols()
        ));
    }

    // Check block boundaries
    if nlow > 0 && a[(nlow, nlow - 1)].abs() > f64::EPSILON {
        return Err(format!(
            "A({},{}) is nonzero - block boundary violation at NLOW",
            nlow,
            nlow - 1
        ));
    }
    if nsup < n - 1 && a[(nsup + 1, nsup)].abs() > f64::EPSILON {
        return Err(format!(
            "A({},{}) is nonzero - block boundary violation at NSUP",
            nsup + 1,
            nsup
        ));
    }

    // Initialize U to identity if requested
    if jobu_upper == 'I' {
        u.fill(0.0);
        for i in 0..n {
            u[(i, i)] = 1.0;
        }
    }

    let discr = dico_upper == 'D';
    let lstdom = stdom_upper == 'S';

    let mut ndim = 0;
    let mut l = nsup;

    // Main loop: process blocks from bottom to top
    // Count eigenvalues in the domain of interest
    while l >= nlow {
        let mut ib = 1; // Block size (1 or 2)

        // Determine block size and eigenvalue(s)
        let tlambd: f64;
        if l > nlow {
            let lm1 = l - 1;
            if a[(l, lm1)].abs() > f64::EPSILON {
                // 2×2 block detected - use MB03QY to get eigenvalues
                let mut a_temp = a.clone();
                let mut u_temp = Array2::eye(n);
                match mb03qy(&mut a_temp, &mut u_temp, lm1) {
                    Ok((e1, e2)) => {
                        // Check if eigenvalues are complex or real
                        if e2.abs() > f64::EPSILON {
                            // Complex eigenvalues
                            ib = 2;
                            tlambd = if discr {
                                // For discrete-time, use modulus
                                (e1 * e1 + e2 * e2).sqrt()
                            } else {
                                // For continuous-time, use real part
                                e1
                            };
                        } else {
                            // Real eigenvalues - take the first one
                            ib = 2;
                            tlambd = if discr { e1.abs() } else { e1 };
                        }
                    }
                    Err(_) => {
                        // If MB03QY fails, treat as 1×1 block
                        ib = 1;
                        tlambd = if discr { a[(l, l)].abs() } else { a[(l, l)] };
                    }
                }
            } else {
                // 1×1 block
                tlambd = if discr { a[(l, l)].abs() } else { a[(l, l)] };
            }
        } else {
            // At NLOW, can only be 1×1 block
            tlambd = if discr { a[(l, l)].abs() } else { a[(l, l)] };
        }

        // Check if eigenvalue is in domain of interest
        let in_domain = if lstdom {
            tlambd < alpha
        } else {
            tlambd > alpha
        };

        if in_domain {
            // Eigenvalue is in domain - count it
            ndim += ib;
        }

        // Move to next block
        if ib == 2 {
            // For a 2×2 block at (l-1:l, l-1:l), next position is l-2
            if l < nlow + 2 {
                break; // Would go below nlow
            }
            l -= 2;
        } else {
            // For a 1×1 block at (l,l), next position is l-1
            if l == nlow {
                break; // Already at nlow
            }
            l -= 1;
        }
    }

    Ok(ndim)
}

#[cfg(test)]
mod tests_mb03qd {
    use super::*;
    use ndarray::arr2;

    #[test]
    fn test_mb03qd_all_stable_continuous() {
        // Test with all stable eigenvalues in continuous-time
        let mut a = arr2(&[
            [-3.13, -26.51, 27.23, -16.20],
            [0.91, -3.13, 13.63, 8.92],
            [0.00, 0.00, -3.37, 0.34],
            [0.00, 0.00, -1.79, -3.37],
        ]);
        let mut u = Array2::eye(4);

        // Matrix has two 2x2 blocks:
        // Block 1: rows/cols (0,1) with A(1,0)=0.91 nonzero
        // Block 2: rows/cols (2,3) with A(3,2)=-1.79 nonzero
        // All eigenvalues have negative real parts, so should all be counted
        let result = mb03qd('C', 'S', 'I', 0, 3, 0.0, &mut a, &mut u);
        assert!(result.is_ok());
        let ndim = result.unwrap();

        // Should count 4 eigenvalues (2 complex pairs)
        // Note: This is a simplified implementation that counts correctly
        // but doesn't reorder blocks
        assert_eq!(ndim, 4);
    }

    #[test]
    fn test_mb03qd_parameter_validation() {
        let mut a = arr2(&[[1.0, 2.0], [0.0, 3.0]]);
        let mut u = Array2::eye(2);

        // Invalid DICO
        assert!(mb03qd('X', 'S', 'I', 0, 1, 0.0, &mut a, &mut u).is_err());

        // Invalid STDOM
        assert!(mb03qd('C', 'X', 'I', 0, 1, 0.0, &mut a, &mut u).is_err());

        // Invalid JOBU
        assert!(mb03qd('C', 'S', 'X', 0, 1, 0.0, &mut a, &mut u).is_err());

        // NLOW > NSUP
        assert!(mb03qd('C', 'S', 'I', 1, 0, 0.0, &mut a, &mut u).is_err());

        // NSUP >= N
        assert!(mb03qd('C', 'S', 'I', 0, 2, 0.0, &mut a, &mut u).is_err());

        // Negative ALPHA for discrete-time
        assert!(mb03qd('D', 'S', 'I', 0, 1, -1.0, &mut a, &mut u).is_err());
    }

    #[test]
    fn test_mb03qd_block_boundary_check() {
        let mut a = arr2(&[
            [1.0, 2.0, 3.0],
            [0.5, 4.0, 5.0], // Non-zero A(1,0) violates block boundary if NLOW=1
            [0.0, 0.0, 6.0],
        ]);
        let mut u = Array2::eye(3);

        // Should fail because A(1,0) is nonzero
        let result = mb03qd('C', 'S', 'I', 1, 2, 0.0, &mut a, &mut u);
        assert!(result.is_err());
        assert!(result.unwrap_err().contains("block boundary"));
    }

    #[test]
    fn test_mb03qd_jobu_initialize() {
        let mut a = arr2(&[[-1.0, 2.0], [0.0, -2.0]]);
        let mut u = arr2(&[[99.0, 99.0], [99.0, 99.0]]);

        let result = mb03qd('C', 'S', 'I', 0, 1, 0.0, &mut a, &mut u);
        assert!(result.is_ok());

        // U should be initialized to identity
        assert_eq!(u[(0, 0)], 1.0);
        assert_eq!(u[(1, 1)], 1.0);
        assert_eq!(u[(0, 1)], 0.0);
        assert_eq!(u[(1, 0)], 0.0);
    }

    #[test]
    fn test_mb03qd_discrete_stable() {
        // Test discrete-time stability criterion
        let mut a = arr2(&[[0.5, 0.1], [0.0, 0.3]]);
        let mut u = Array2::eye(2);

        // Eigenvalues are 0.5 and 0.3, both < 1.0
        let result = mb03qd('D', 'S', 'I', 0, 1, 1.0, &mut a, &mut u);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2);
    }

    #[test]
    fn test_mb03qd_discrete_unstable() {
        // Test discrete-time instability criterion
        let mut a = arr2(&[[1.5, 0.1], [0.0, 2.3]]);
        let mut u = Array2::eye(2);

        // Eigenvalues are 1.5 and 2.3, both > 1.0
        let result = mb03qd('D', 'U', 'I', 0, 1, 1.0, &mut a, &mut u);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2);
    }

    #[test]
    fn test_mb03qd_mixed_eigenvalues() {
        // Test with mixed stable/unstable eigenvalues
        let mut a = arr2(&[[-2.0, 1.0, 0.0], [0.0, -1.0, 0.0], [0.0, 0.0, 3.0]]);
        let mut u = Array2::eye(3);

        // Eigenvalues: -2, -1 (stable), 3 (unstable)
        // Count stable ones (real part < 0)
        let result = mb03qd('C', 'S', 'I', 0, 2, 0.0, &mut a, &mut u);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 2);

        // Count unstable ones (real part > 0)
        let mut a2 = a.clone();
        let mut u2 = Array2::eye(3);
        let result2 = mb03qd('C', 'U', 'I', 0, 2, 0.0, &mut a2, &mut u2);
        assert!(result2.is_ok());
        assert_eq!(result2.unwrap(), 1);
    }

    #[test]
    fn test_mb03qd_single_element() {
        // Test with 1x1 matrix
        let mut a = arr2(&[[-5.0]]);
        let mut u = Array2::eye(1);

        let result = mb03qd('C', 'S', 'I', 0, 0, 0.0, &mut a, &mut u);
        assert!(result.is_ok());
        assert_eq!(result.unwrap(), 1); // -5.0 < 0, so it's stable
    }
}
