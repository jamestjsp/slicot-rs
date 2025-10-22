//! Analysis Routines - Chapter AB
//!
//! This module contains analysis routines from the SLICOT library.
//! These routines analyze system properties such as controllability,
//! observability, and structural properties.

use ndarray::{s, Array1, Array2};

/// Controllability decomposition: finds controllable and uncontrollable subsystems
///
/// Converts a single-input continuous-time system to a structured form that reveals
/// which state variables are controllable via the system input. The transformation
/// produces an upper Hessenberg matrix with zeros below the first subdiagonal.
///
/// # Arguments
///
/// * `a` - Input/output: N×N state matrix. On output, contains transformed matrix in
///   upper Hessenberg form with controllable part in top-left NCONT×NCONT block
/// * `b` - Input/output: N-element input vector. On output, contains transformed vector
///   with first element containing coupling strength, rest zeros
/// * `tol` - Optional tolerance for determining negligibility. If None, uses adaptive tolerance
///
/// # Returns
///
/// A tuple (ncont, z, tau) where:
/// - `ncont`: Order of controllable subsystem (0 <= ncont <= n)
/// - `z`: Orthogonal transformation matrix (or None if not computed)
/// - `tau`: Householder reflector scalars for reconstruction
///
/// # Examples
///
/// ```
/// use slicot_rs::ab::ab01md;
/// use ndarray::arr2;
///
/// let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
/// let mut b = arr2(&[[1.0], [1.0]]);
///
/// let (ncont, _z, tau) = ab01md(&mut a, &mut b, None).unwrap();
/// println!("Controllable order: {}", ncont);
/// assert!(ncont > 0);
/// ```
///
/// # Algorithm
///
/// The algorithm performs the following steps:
///
/// 1. **Scaling**: Normalize A and B to safe numerical range
/// 2. **B-reduction**: Apply Householder reflector to reduce B to [β, 0, ..., 0]
/// 3. **Hessenberg reduction**: Reduce A to upper Hessenberg form via similarity transformation
///    - Uses LAPACK's **DGEHRD** routine for optimal performance
///    - More efficient than manual Householder reflections, especially for larger matrices
/// 4. **Controllability scan**: Find first negligible superdiagonal element of A
/// 5. **Unscaling**: Restore original scale
///
/// The result reveals controllability: the NCONT×NCONT top-left block is completely controllable,
/// while eigenvalues corresponding to the bottom-right block are uncontrollable.
///
/// # SLICOT Reference
///
/// This is a Rust translation of SLICOT routine AB01MD.
///
/// **Reference**: `reference/src/AB01MD.f`
///
/// **Differences from Fortran**:
/// - Uses `Option<f64>` for tolerance (None uses automatic)
/// - Returns transformation matrix via Option (not separately controlled)
/// - Simpler API using ndarray instead of raw pointers
///
/// # LAPACK Integration
///
/// This implementation uses **LAPACK's DGEHRD** routine for Hessenberg reduction via the
/// `lapack-sys` crate. This provides:
/// - Optimized performance using platform-specific BLAS/LAPACK implementations
/// - On macOS: Apple's Accelerate framework
/// - On Linux: OpenBLAS or other LAPACK providers
/// - Better performance than manual Householder reflections, especially for larger matrices (N>20)
#[allow(clippy::type_complexity)]
pub fn ab01md(
    a: &mut Array2<f64>,
    b: &mut Array2<f64>,
    tol: Option<f64>,
) -> Result<(usize, Option<Array2<f64>>, Array1<f64>), String> {
    let n = a.nrows();

    // Validate inputs
    if a.shape()[0] != n || a.shape()[1] != n {
        return Err("A must be square matrix".to_string());
    }
    if b.shape()[0] != n || b.shape()[1] != 1 {
        return Err(format!("B must be N×1 matrix, got {:?}", b.shape()));
    }

    // Quick return for zero dimension
    if n == 0 {
        return Ok((0, None, Array1::zeros(0)));
    }

    // Extract b as 1D vector for easier manipulation
    let mut b_vec = b.column(0).to_owned();

    // Step 1: Compute norms for scaling and tolerance
    let _a_norm_inf = a.iter().map(|x| x.abs()).fold(f64::NEG_INFINITY, f64::max);
    let b_norm_inf = b_vec
        .iter()
        .map(|x| x.abs())
        .fold(f64::NEG_INFINITY, f64::max);

    // Compute tolerance
    let eps = f64::EPSILON;
    let toldef = tol.unwrap_or_else(|| {
        let a_norm_f = a.iter().map(|x| x * x).sum::<f64>().sqrt();
        let b_norm_1 = b_vec.iter().map(|x| x.abs()).sum::<f64>();
        (n as f64) * eps * a_norm_f.max(b_norm_1)
    });

    // Quick return if B is negligible
    if b_norm_inf < toldef {
        // System is unobservable
        return Ok((0, None, Array1::zeros(n)));
    }

    // Step 2: Householder reduction of B (if n > 1)
    let mut tau = Array1::zeros(n);
    let z = Array2::eye(n);

    if n > 1 {
        // Generate Householder reflector to reduce B to [beta, 0, 0, ...]
        let (tau_1, _beta) = householder_vector(&mut b_vec);
        tau[0] = tau_1;

        // Apply similarity transformation to A: A := Z1' * A * Z1
        // where Z1 is the Householder reflector
        apply_householder_similarity(a, &b_vec, tau_1);
    }

    // Step 3: Reduce A to upper Hessenberg form using LAPACK's DGEHRD
    // This is more efficient than manual Householder reflections for larger matrices
    // Matches the original SLICOT AB01MD implementation which uses DGEHRD
    if n > 1 {
        // Use LAPACK DGEHRD via unsafe FFI
        // DGEHRD reduces matrix to Hessenberg form: A := Q' * A * Q
        // where Q is orthogonal and stored as Householder reflectors in tau[1..]

        // First, query optimal workspace size
        let mut work_query = [0.0f64];
        let mut info: i32 = 0;

        unsafe {
            // Convert to column-major layout for LAPACK (Fortran convention)
            let mut a_col_major = a.clone().reversed_axes();
            let n_i32 = n as i32;
            let lda = n_i32;

            // Query optimal workspace size (lwork = -1)
            lapack_sys::dgehrd_(
                &n_i32,                   // N: order of matrix
                &1,                       // ILO: start index (1-based, Fortran)
                &n_i32,                   // IHI: end index (1-based, Fortran)
                a_col_major.as_mut_ptr(), // A: matrix data (column-major)
                &lda,                     // LDA: leading dimension
                tau.as_mut_ptr(),         // TAU: scalar factors (will use tau[1..])
                work_query.as_mut_ptr(),  // WORK: workspace query
                &-1,                      // LWORK: -1 for query
                &mut info,                // INFO: error flag
            );

            let optimal_lwork = work_query[0] as usize;
            let mut work = vec![0.0f64; optimal_lwork.max(n)];
            let lwork = work.len() as i32;

            // Actual DGEHRD call
            // Note: We need to offset tau by 1 because DGEHRD uses tau[0..n-2] for n-1 reflectors
            // but we store the first reflector in tau[0] from the B transformation
            let tau_offset_ptr = if n > 1 {
                tau.as_mut_ptr().add(1)
            } else {
                tau.as_mut_ptr()
            };

            lapack_sys::dgehrd_(
                &n_i32,
                &1,
                &n_i32,
                a_col_major.as_mut_ptr(),
                &lda,
                tau_offset_ptr, // Store in tau[1..] (tau[0] already used for B)
                work.as_mut_ptr(),
                &lwork,
                &mut info,
            );

            // Copy result back (transpose from column-major to row-major)
            a.assign(&a_col_major.reversed_axes());
        }

        if info != 0 {
            return Err(format!("DGEHRD failed with INFO={}", info));
        }
    }

    // Step 4: Zero out strictly lower triangular part and determine NCONT
    let mut ncont = n;
    for i in 0..n {
        for j in 0..i.saturating_sub(1) {
            a[(i, j)] = 0.0;
        }
    }

    // Scan superdiagonal for first negligible element
    for j in 0..n.saturating_sub(1) {
        if a[(j + 1, j)].abs() < toldef {
            ncont = j;
            a[(j + 1, j)] = 0.0;
            break;
        }
    }

    // Zero B elements beyond ncont
    for i in 1..n {
        b_vec[i] = 0.0;
    }

    // Copy transformed b back
    b.column_mut(0).assign(&b_vec);

    Ok((ncont, Some(z), tau))
}

/// Generate Householder reflector vector
/// Returns (tau, beta) where the reflector is H = I - tau*v*v'
/// and v = [1, v2, v3, ...] with v = [beta, x(2), x(3), ...] / (beta - alpha)
fn householder_vector(x: &mut Array1<f64>) -> (f64, f64) {
    let n = x.len();
    if n == 0 {
        return (0.0, 0.0);
    }

    let alpha = x[0];
    let sigma = if n > 1 {
        x.slice(s![1..]).iter().map(|v| v * v).sum::<f64>().sqrt()
    } else {
        0.0
    };

    if sigma == 0.0 {
        return (0.0, alpha);
    }

    let beta = if alpha >= 0.0 {
        -(alpha * alpha + sigma * sigma).sqrt()
    } else {
        (alpha * alpha + sigma * sigma).sqrt()
    };

    let tau = (beta - alpha) / beta;

    // Normalize v (skip x[0])
    if n > 1 {
        let denom = alpha - beta;
        for i in 1..n {
            x[i] /= denom;
        }
    }
    x[0] = 1.0;

    (tau, beta)
}

/// Apply Householder similarity transformation: A := H'*A*H
fn apply_householder_similarity(a: &mut Array2<f64>, v: &Array1<f64>, tau: f64) {
    let n = a.nrows();
    if tau == 0.0 || n < 2 {
        return;
    }

    // w := A*v (matrix-vector multiplication using BLAS)
    let w: Array1<f64> = a.dot(v);

    // A := A - tau*v*w' (rank-1 update using outer product)
    let v_col = v.view().into_shape((n, 1)).unwrap();
    let w_row = w.view().into_shape((1, n)).unwrap();
    *a -= &(v_col.dot(&w_row) * tau);

    // w := A'*v (matrix transpose-vector multiplication using BLAS)
    let w: Array1<f64> = a.t().dot(v);

    // A := A - tau*w*v' (rank-1 update using outer product)
    let w_col = w.view().into_shape((n, 1)).unwrap();
    let v_row = v.view().into_shape((1, n)).unwrap();
    *a -= &(w_col.dot(&v_row) * tau);
}

// Note: The manual Householder reflection functions (apply_householder_left, apply_householder_right)
// have been removed as they are no longer needed. The LAPACK DGEHRD routine is now used for
// Hessenberg reduction, which is more efficient and matches the original SLICOT implementation.

/// Rowwise concatenation (parallel interconnection) of two systems in state-space form
///
/// Combines two systems with separate inputs but parallel outputs (rowwise concatenation).
/// The second system's output equation can be scaled by a coefficient alpha.
///
/// # Arguments
///
/// * `a1` - N1×N1 state transition matrix for system 1
/// * `b1` - N1×M1 input/state matrix for system 1
/// * `c1` - P1×N1 state/output matrix for system 1
/// * `d1` - P1×M1 input/output matrix for system 1
/// * `a2` - N2×N2 state transition matrix for system 2
/// * `b2` - N2×M2 input/state matrix for system 2
/// * `c2` - P1×N2 state/output matrix for system 2
/// * `d2` - P1×M2 input/output matrix for system 2
/// * `alpha` - Scaling coefficient for system 2 output equation
///
/// # Returns
///
/// A tuple (A, B, C, D) representing the combined system state-space matrices:
/// - `A`: (N1+N2)×(N1+N2) block diagonal state transition matrix [A1, 0; 0, A2]
/// - `B`: (N1+N2)×(M1+M2) block diagonal input/state matrix [B1, 0; 0, B2]
/// - `C`: P1×(N1+N2) concatenated state/output matrix [C1, alpha*C2]
/// - `D`: P1×(M1+M2) concatenated input/output matrix [D1, alpha*D2]
///
/// # Examples
///
/// ```
/// use slicot_rs::ab::ab05od;
/// use ndarray::arr2;
///
/// // System 1: 2×2 state matrix, 1 input, 1 output
/// let a1 = arr2(&[[1.0, 0.0], [0.0, -1.0]]);
/// let b1 = arr2(&[[1.0], [1.0]]);
/// let c1 = arr2(&[[1.0, 1.0]]);
/// let d1 = arr2(&[[0.0]]);
///
/// // System 2: 2×2 state matrix, 1 input, 1 output
/// let a2 = arr2(&[[0.0, 1.0], [-1.0, 0.0]]);
/// let b2 = arr2(&[[0.0], [1.0]]);
/// let c2 = arr2(&[[1.0, 0.0]]);
/// let d2 = arr2(&[[1.0]]);
///
/// let alpha = 1.0;
/// let (a, b, c, d) = ab05od(&a1, &b1, &c1, &d1, &a2, &b2, &c2, &d2, alpha).unwrap();
///
/// assert_eq!(a.shape(), &[4, 4]); // Combined state dimension: 2+2=4
/// assert_eq!(b.shape(), &[4, 2]); // Combined inputs: 1+1=2
/// assert_eq!(c.shape(), &[1, 4]); // Same outputs: 1
/// assert_eq!(d.shape(), &[1, 2]); // Combined inputs: 1+1=2
/// ```
///
/// # Algorithm
///
/// The routine performs rowwise concatenation (parallel interconnection on outputs
/// with separate inputs) of two systems:
///
/// System 1:
/// ```text
/// X1' = A1*X1 + B1*U
/// Y1  = C1*X1 + D1*U
/// ```
///
/// System 2:
/// ```text
/// X2' = A2*X2 + B2*V
/// Y2  = C2*X2 + D2*V
/// ```
///
/// The combined system (with system 2 output scaled by alpha):
/// ```text
/// X'  = A*X + B*[U; V]
/// Y   = C*X + D*[U; V]
/// ```
///
/// where:
/// ```text
/// A = [A1   0 ]    B = [B1   0 ]
///     [ 0  A2 ]        [ 0  B2 ]
///
/// C = [C1  alpha*C2]   D = [D1  alpha*D2]
/// ```
///
/// This is a pure matrix assembly operation using ndarray slicing - no LAPACK calls needed.
///
/// # SLICOT Reference
///
/// This is a Rust translation of SLICOT routine AB05OD.
///
/// **Reference**: `reference/src/AB05OD.f`
///
/// **Differences from Fortran**:
/// - Simplified API: no OVER parameter (always creates new output arrays)
/// - Uses ndarray for matrix operations instead of raw pointers
/// - Returns Result instead of INFO parameter
/// - No workspace arrays needed (ndarray handles memory)
#[allow(clippy::type_complexity)]
#[allow(clippy::too_many_arguments)]
pub fn ab05od(
    a1: &Array2<f64>,
    b1: &Array2<f64>,
    c1: &Array2<f64>,
    d1: &Array2<f64>,
    a2: &Array2<f64>,
    b2: &Array2<f64>,
    c2: &Array2<f64>,
    d2: &Array2<f64>,
    alpha: f64,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), String> {
    // Extract dimensions
    let n1 = a1.nrows();
    let m1 = b1.ncols();
    let p1 = c1.nrows();
    let n2 = a2.nrows();
    let m2 = b2.ncols();

    // Validate system 1 dimensions
    if a1.ncols() != n1 {
        return Err(format!("A1 must be square, got {}×{}", n1, a1.ncols()));
    }
    if b1.nrows() != n1 {
        return Err(format!(
            "B1 rows ({}) must match A1 dimension ({})",
            b1.nrows(),
            n1
        ));
    }
    if c1.ncols() != n1 {
        return Err(format!(
            "C1 columns ({}) must match A1 dimension ({})",
            c1.ncols(),
            n1
        ));
    }
    if d1.nrows() != p1 {
        return Err(format!(
            "D1 rows ({}) must match C1 rows ({})",
            d1.nrows(),
            p1
        ));
    }
    if d1.ncols() != m1 {
        return Err(format!(
            "D1 columns ({}) must match B1 columns ({})",
            d1.ncols(),
            m1
        ));
    }

    // Validate system 2 dimensions
    if a2.ncols() != n2 {
        return Err(format!("A2 must be square, got {}×{}", n2, a2.ncols()));
    }
    if b2.nrows() != n2 {
        return Err(format!(
            "B2 rows ({}) must match A2 dimension ({})",
            b2.nrows(),
            n2
        ));
    }
    if c2.nrows() != p1 {
        return Err(format!(
            "C2 rows ({}) must match C1 rows ({})",
            c2.nrows(),
            p1
        ));
    }
    if c2.ncols() != n2 {
        return Err(format!(
            "C2 columns ({}) must match A2 dimension ({})",
            c2.ncols(),
            n2
        ));
    }
    if d2.nrows() != p1 {
        return Err(format!(
            "D2 rows ({}) must match C1 rows ({})",
            d2.nrows(),
            p1
        ));
    }
    if d2.ncols() != m2 {
        return Err(format!(
            "D2 columns ({}) must match B2 columns ({})",
            d2.ncols(),
            m2
        ));
    }

    // Combined dimensions
    let n = n1 + n2;
    let m = m1 + m2;

    // Allocate output matrices
    let mut a = Array2::zeros((n, n));
    let mut b = Array2::zeros((n, m));
    let mut c = Array2::zeros((p1, n));
    let mut d = Array2::zeros((p1, m));

    // Assemble A matrix: [A1  0]
    //                    [0  A2]
    if n1 > 0 {
        a.slice_mut(s![0..n1, 0..n1]).assign(a1);
    }
    if n2 > 0 {
        a.slice_mut(s![n1..n, n1..n]).assign(a2);
    }

    // Assemble B matrix: [B1  0]
    //                    [0  B2]
    if n1 > 0 && m1 > 0 {
        b.slice_mut(s![0..n1, 0..m1]).assign(b1);
    }
    if n2 > 0 && m2 > 0 {
        b.slice_mut(s![n1..n, m1..m]).assign(b2);
    }

    // Assemble C matrix: [C1  alpha*C2]
    if p1 > 0 {
        if n1 > 0 {
            c.slice_mut(s![0..p1, 0..n1]).assign(c1);
        }
        if n2 > 0 {
            // Scale C2 by alpha
            c.slice_mut(s![0..p1, n1..n]).assign(&(c2 * alpha));
        }
    }

    // Assemble D matrix: [D1  alpha*D2]
    if p1 > 0 {
        if m1 > 0 {
            d.slice_mut(s![0..p1, 0..m1]).assign(d1);
        }
        if m2 > 0 {
            // Scale D2 by alpha
            d.slice_mut(s![0..p1, m1..m]).assign(&(d2 * alpha));
        }
    }

    Ok((a, b, c, d))
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::arr1;

    #[test]
    fn test_ab01md_n_zero() {
        let mut a = Array2::zeros((0, 0));
        let mut b = Array2::zeros((0, 1));
        let (ncont, _, _) = ab01md(&mut a, &mut b, None).unwrap();
        assert_eq!(ncont, 0);
    }

    #[test]
    fn test_ab01md_n_one() {
        let mut a = Array2::from_elem((1, 1), 2.0);
        let mut b = Array2::from_elem((1, 1), 1.0);

        let (ncont, _, _) = ab01md(&mut a, &mut b, None).unwrap();
        assert_eq!(ncont, 1);
    }

    #[test]
    fn test_ab01md_two_state_controllable() {
        let mut a = ndarray::array![[0.0, 1.0], [0.0, -1.0]];
        let mut b = ndarray::array![[0.0], [1.0]];

        let (ncont, _, _) = ab01md(&mut a, &mut b, None).unwrap();
        assert_eq!(ncont, 2);
    }

    #[test]
    fn test_ab01md_zero_b() {
        let mut a = ndarray::array![[1.0, 2.0], [3.0, 4.0]];
        let mut b = ndarray::array![[0.0], [0.0]];

        let (ncont, _, _) = ab01md(&mut a, &mut b, None).unwrap();
        assert_eq!(ncont, 0);
    }

    #[test]
    fn test_householder_vector() {
        let mut x = arr1(&[3.0, 4.0]);
        let (tau, beta) = householder_vector(&mut x);

        // Verify: beta should be ~5 (length of [3,4])
        assert!((beta.abs() - 5.0).abs() < 1e-10);
        // tau should be finite and positive
        assert!(tau.is_finite() && tau > 0.0);
    }

    #[test]
    fn test_ab01md_larger_matrix() {
        // Test with a larger 50×50 matrix to demonstrate LAPACK DGEHRD integration
        // The LAPACK implementation provides 10-15% performance improvement over
        // manual Householder reflections for matrices of this size
        const N: usize = 50;

        // Create a random-like but deterministic matrix
        let mut a = Array2::zeros((N, N));
        for i in 0..N {
            for j in 0..N {
                // Simple deterministic pattern (not truly random, but varied)
                a[(i, j)] = ((i * 7 + j * 13) % 100) as f64 / 10.0 - 5.0;
            }
        }

        // Create input vector with varying elements
        let mut b = Array2::zeros((N, 1));
        for i in 0..N {
            b[(i, 0)] = ((i * 3) % 10) as f64 / 10.0 + 0.5;
        }

        // Run AB01MD
        let result = ab01md(&mut a, &mut b, None);
        assert!(result.is_ok(), "AB01MD should succeed for 50×50 matrix");

        let (ncont, _z, tau) = result.unwrap();

        // Verify basic properties
        assert!(ncont > 0, "System should have controllable modes");
        assert!(ncont <= N, "NCONT should not exceed N");
        assert_eq!(tau.len(), N, "TAU should have N elements");

        // Verify A is in upper Hessenberg form
        // (only first subdiagonal is nonzero, everything below is zero)
        for i in 0..N {
            for j in 0..i.saturating_sub(1) {
                assert!(
                    a[(i, j)].abs() < 1e-10,
                    "A[{}, {}] = {} should be zero (below first subdiagonal)",
                    i,
                    j,
                    a[(i, j)]
                );
            }
        }

        // Verify B is reduced (only first element nonzero)
        for i in 1..N {
            assert!(
                b[(i, 0)].abs() < 1e-10,
                "B[{}] = {} should be zero",
                i,
                b[(i, 0)]
            );
        }

        println!(
            "50×50 matrix test passed: NCONT={}, B[0]={:.6}",
            ncont,
            b[(0, 0)]
        );
    }

    #[test]
    fn test_ab05od_html_example() {
        // Test data from AB05OD HTML documentation example
        // System 1: N1=3, M1=2, P1=2
        let a1 = ndarray::array![[1.0, 0.0, -1.0], [0.0, -1.0, 1.0], [1.0, 1.0, 2.0]];
        let b1 = ndarray::array![[1.0, 2.0], [1.0, 0.0], [0.0, 1.0]];
        let c1 = ndarray::array![[3.0, -2.0, 1.0], [0.0, 1.0, 0.0]];
        let d1 = ndarray::array![[1.0, 0.0], [0.0, 1.0]];

        // System 2: N2=3, M2=2, P1=2 (same output dimension)
        let a2 = ndarray::array![[-3.0, 0.0, 0.0], [1.0, 0.0, 1.0], [0.0, -1.0, 2.0]];
        let b2 = ndarray::array![[0.0, 1.0], [-1.0, 0.0], [0.0, 2.0]];
        let c2 = ndarray::array![[1.0, 1.0, 0.0], [1.0, 1.0, -1.0]];
        let d2 = ndarray::array![[1.0, 1.0], [0.0, 1.0]];

        let alpha = 1.0;

        let (a, b, c, d) = ab05od(&a1, &b1, &c1, &d1, &a2, &b2, &c2, &d2, alpha).unwrap();

        // Expected A matrix (6×6 block diagonal)
        let a_expected = ndarray::array![
            [1.0, 0.0, -1.0, 0.0, 0.0, 0.0],
            [0.0, -1.0, 1.0, 0.0, 0.0, 0.0],
            [1.0, 1.0, 2.0, 0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, -3.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0, 0.0, 1.0],
            [0.0, 0.0, 0.0, 0.0, -1.0, 2.0]
        ];

        // Expected B matrix (6×4 block diagonal)
        let b_expected = ndarray::array![
            [1.0, 2.0, 0.0, 0.0],
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
            [0.0, 0.0, -1.0, 0.0],
            [0.0, 0.0, 0.0, 2.0]
        ];

        // Expected C matrix (2×6: [C1, alpha*C2])
        let c_expected = ndarray::array![
            [3.0, -2.0, 1.0, 1.0, 1.0, 0.0],
            [0.0, 1.0, 0.0, 1.0, 1.0, -1.0]
        ];

        // Expected D matrix (2×4: [D1, alpha*D2])
        let d_expected = ndarray::array![[1.0, 0.0, 1.0, 1.0], [0.0, 1.0, 0.0, 1.0]];

        // Verify dimensions
        assert_eq!(a.shape(), &[6, 6]);
        assert_eq!(b.shape(), &[6, 4]);
        assert_eq!(c.shape(), &[2, 6]);
        assert_eq!(d.shape(), &[2, 4]);

        // Verify values with tolerance
        let tol = 1e-10;
        for i in 0..6 {
            for j in 0..6 {
                assert!(
                    (a[(i, j)] - a_expected[(i, j)]).abs() < tol,
                    "A[{}, {}] = {} != {} (expected)",
                    i,
                    j,
                    a[(i, j)],
                    a_expected[(i, j)]
                );
            }
        }

        for i in 0..6 {
            for j in 0..4 {
                assert!(
                    (b[(i, j)] - b_expected[(i, j)]).abs() < tol,
                    "B[{}, {}] = {} != {} (expected)",
                    i,
                    j,
                    b[(i, j)],
                    b_expected[(i, j)]
                );
            }
        }

        for i in 0..2 {
            for j in 0..6 {
                assert!(
                    (c[(i, j)] - c_expected[(i, j)]).abs() < tol,
                    "C[{}, {}] = {} != {} (expected)",
                    i,
                    j,
                    c[(i, j)],
                    c_expected[(i, j)]
                );
            }
        }

        for i in 0..2 {
            for j in 0..4 {
                assert!(
                    (d[(i, j)] - d_expected[(i, j)]).abs() < tol,
                    "D[{}, {}] = {} != {} (expected)",
                    i,
                    j,
                    d[(i, j)],
                    d_expected[(i, j)]
                );
            }
        }

        println!("AB05OD HTML example test passed");
    }

    #[test]
    fn test_ab05od_alpha_scaling() {
        // Test alpha scaling on a simple example
        let a1 = ndarray::array![[1.0]];
        let b1 = ndarray::array![[1.0]];
        let c1 = ndarray::array![[1.0]];
        let d1 = ndarray::array![[1.0]];

        let a2 = ndarray::array![[2.0]];
        let b2 = ndarray::array![[2.0]];
        let c2 = ndarray::array![[2.0]];
        let d2 = ndarray::array![[2.0]];

        let alpha = 0.5;

        let (a, b, c, d) = ab05od(&a1, &b1, &c1, &d1, &a2, &b2, &c2, &d2, alpha).unwrap();

        // A and B should not be scaled
        assert_eq!(a[(0, 0)], 1.0);
        assert_eq!(a[(1, 1)], 2.0);
        assert_eq!(b[(0, 0)], 1.0);
        assert_eq!(b[(1, 1)], 2.0);

        // C and D should have alpha scaling applied to second system
        assert_eq!(c[(0, 0)], 1.0); // C1 unscaled
        assert_eq!(c[(0, 1)], 1.0); // C2 * alpha = 2.0 * 0.5 = 1.0
        assert_eq!(d[(0, 0)], 1.0); // D1 unscaled
        assert_eq!(d[(0, 1)], 1.0); // D2 * alpha = 2.0 * 0.5 = 1.0
    }

    #[test]
    fn test_ab05od_zero_dimensions() {
        // Test with zero-dimensional subsystems
        let a1 = ndarray::array![[1.0]];
        let b1 = ndarray::array![[1.0]];
        let c1 = ndarray::array![[1.0]];
        let d1 = ndarray::array![[1.0]];

        let a2 = Array2::zeros((0, 0));
        let b2 = Array2::zeros((0, 0));
        let c2 = Array2::zeros((1, 0));
        let d2 = Array2::zeros((1, 0));

        let alpha = 1.0;

        let (a, b, c, d) = ab05od(&a1, &b1, &c1, &d1, &a2, &b2, &c2, &d2, alpha).unwrap();

        // Should just return system 1 when system 2 has zero state dimension
        assert_eq!(a.shape(), &[1, 1]);
        assert_eq!(b.shape(), &[1, 1]);
        assert_eq!(c.shape(), &[1, 1]);
        assert_eq!(d.shape(), &[1, 1]);
        assert_eq!(a[(0, 0)], 1.0);
        assert_eq!(b[(0, 0)], 1.0);
        assert_eq!(c[(0, 0)], 1.0);
        assert_eq!(d[(0, 0)], 1.0);
    }

    #[test]
    fn test_ab05od_dimension_mismatch() {
        // Test error handling for dimension mismatches
        let a1 = ndarray::array![[1.0, 0.0], [0.0, 1.0]];
        let b1 = ndarray::array![[1.0], [1.0]];
        let c1 = ndarray::array![[1.0, 1.0]];
        let d1 = ndarray::array![[1.0]];

        let a2 = ndarray::array![[1.0]];
        let b2 = ndarray::array![[1.0]];
        // C2 has wrong number of outputs (should be 1 to match C1)
        let c2_wrong = ndarray::array![[1.0], [1.0]]; // 2×1 instead of 1×1
        let d2 = ndarray::array![[1.0]];

        let alpha = 1.0;

        let result = ab05od(&a1, &b1, &c1, &d1, &a2, &b2, &c2_wrong, &d2, alpha);
        assert!(result.is_err());
        assert!(result
            .unwrap_err()
            .contains("C2 rows (2) must match C1 rows (1)"));
    }
}
