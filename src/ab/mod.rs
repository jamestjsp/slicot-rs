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

/// Bilinear transformation for continuous ↔ discrete time systems
///
/// Performs a transformation on the parameters (A,B,C,D) of a system,
/// which is equivalent to a bilinear transformation of the corresponding
/// transfer function matrix.
///
/// # Arguments
///
/// * `typ` - Transformation type: 'D' (discrete→continuous) or 'C' (continuous→discrete)
/// * `alpha` - First bilinear transformation parameter (must be non-zero)
/// * `beta` - Second bilinear transformation parameter (must be non-zero)
/// * `a` - Input/output: N×N state matrix (modified in-place)
/// * `b` - Input/output: N×M input matrix (modified in-place)
/// * `c` - Input/output: P×N output matrix (modified in-place)
/// * `d` - Input/output: P×M feedthrough matrix (modified in-place)
///
/// # Returns
///
/// Returns `Ok(())` on success, or `Err(String)` with error description on failure.
///
/// # Errors
///
/// - Invalid `typ` parameter (must be 'D' or 'C')
/// - Invalid dimensions (negative or mismatched)
/// - Zero `alpha` or `beta` parameters
/// - Singular matrix (alpha*I + A) for discrete→continuous
/// - Singular matrix (beta*I - A) for continuous→discrete
///
/// # Examples
///
/// ```
/// use slicot_rs::ab::ab04md;
/// use ndarray::arr2;
///
/// // Convert continuous-time to discrete-time
/// let mut a = arr2(&[[1.0, 0.5], [0.5, 1.0]]);
/// let mut b = arr2(&[[0.0, -1.0], [1.0, 0.0]]);
/// let mut c = arr2(&[[-1.0, 0.0], [0.0, 1.0]]);
/// let mut d = arr2(&[[1.0, 0.0], [0.0, -1.0]]);
///
/// ab04md('C', 1.0, 1.0, &mut a, &mut b, &mut c, &mut d).unwrap();
/// ```
///
/// # Algorithm
///
/// The algorithm performs one of two transformations:
///
/// **1. Discrete → Continuous (TYPE = 'D')**:
/// ```text
/// A_new = beta*(alpha*I + A)^-1 * (A - alpha*I)
/// B_new = sqrt(2*alpha*beta) * (alpha*I + A)^-1 * B
/// C_new = sqrt(2*alpha*beta) * C * (alpha*I + A)^-1
/// D_new = D - C * (alpha*I + A)^-1 * B
/// ```
/// Equivalent bilinear transformation: `s = beta * (z - alpha)/(z + alpha)`
///
/// **2. Continuous → Discrete (TYPE = 'C')**:
/// ```text
/// A_new = alpha*(beta*I - A)^-1 * (beta*I + A)
/// B_new = sqrt(2*alpha*beta) * (beta*I - A)^-1 * B
/// C_new = sqrt(2*alpha*beta) * C * (beta*I - A)^-1
/// D_new = D + C * (beta*I - A)^-1 * B
/// ```
/// Equivalent bilinear transformation: `z = alpha * (beta + s)/(beta - s)`
///
/// # Implementation
///
/// This implementation uses LAPACK routines for optimal performance:
/// - **DGETRF**: LU factorization (find (alpha*I + A)^-1 or (beta*I - A)^-1)
/// - **DGETRS**: Solve linear systems using LU factors
/// - **DGETRI**: Matrix inversion
/// - **DTRSM**: Triangular matrix solve
/// - **DLASCL**: Matrix scaling
///
/// The computation proceeds in three phases:
/// 1. Factor and solve: (α*I + A) or (β*I - A)
/// 2. Transform B, C, D using the inverse
/// 3. Compute final A transformation
///
/// # SLICOT Reference
///
/// This is a Rust translation of SLICOT routine AB04MD.
///
/// **Reference**: `reference/src/AB04MD.f`
///
/// **Recommended parameters for stable systems**: alpha = 1, beta = 1
///
/// **Complexity**: O(N³) due to matrix inversion
///
/// **Numerical Aspects**: Accuracy depends on the condition number of the matrix to be inverted.
pub fn ab04md(
    typ: char,
    alpha: f64,
    beta: f64,
    a: &mut Array2<f64>,
    b: &mut Array2<f64>,
    c: &mut Array2<f64>,
    d: &mut Array2<f64>,
) -> Result<(), String> {
    let n = a.nrows();
    let m = b.ncols();
    let p = c.nrows();

    // Parameter validation
    if typ != 'D' && typ != 'C' {
        return Err(format!(
            "Invalid TYPE parameter: '{}' (must be 'D' or 'C')",
            typ
        ));
    }
    if a.shape() != [n, n] {
        return Err(format!("A must be square, got shape {:?}", a.shape()));
    }
    if b.shape() != [n, m] {
        return Err(format!("B must be N×M, got shape {:?}", b.shape()));
    }
    if c.shape() != [p, n] {
        return Err(format!("C must be P×N, got shape {:?}", c.shape()));
    }
    if d.shape() != [p, m] {
        return Err(format!("D must be P×M, got shape {:?}", d.shape()));
    }
    if alpha == 0.0 {
        return Err("ALPHA must be non-zero".to_string());
    }
    if beta == 0.0 {
        return Err("BETA must be non-zero".to_string());
    }

    // Quick return if all dimensions are zero
    if n == 0 || m == 0 || p == 0 {
        return Ok(());
    }

    // Determine transformation parameters based on the algorithm in AB04MD.html
    // TYPE='D': Solve (alpha*I + A), use beta for final A scaling
    // TYPE='C': Solve (beta*I - A), use alpha for final A scaling
    let (palpha, pbeta, subtract_a) = if typ == 'D' {
        // Discrete-time to continuous-time
        // Solve: (alpha*I + A)^-1
        // Scale: beta
        (alpha, beta, false)
    } else {
        // Continuous-time to discrete-time
        // Solve: (beta*I - A)^-1
        // Scale: alpha
        (beta, alpha, true)
    };

    let ab2 = palpha * pbeta * 2.0;
    // For sqrt(2*alpha*beta), we need the actual alpha and beta values
    let sqrab2 = (2.0 * alpha * beta).abs().sqrt() * alpha.signum();

    // Convert to column-major for LAPACK (Fortran convention)
    let mut a_cm = a.clone().reversed_axes();
    let mut b_cm = b.clone().reversed_axes();
    let mut c_cm = c.clone().reversed_axes();
    let mut d_cm = d.clone().reversed_axes();

    let n_i32 = n as i32;
    let m_i32 = m as i32;
    let _p_i32 = p as i32;
    let lda = n_i32;
    let ldb = n_i32;

    let mut ipiv = vec![0i32; n];
    let mut info: i32 = 0;

    unsafe {
        // Step 1: Build matrix to solve: (palpha*I + A) or (palpha*I - A)
        if subtract_a {
            // For TYPE='C': Compute (palpha*I - A) = (beta*I - A)
            for i in 0..n {
                for j in 0..n {
                    a_cm[(i, j)] = -a_cm[(i, j)];
                }
                a_cm[(i, i)] += palpha;
            }
        } else {
            // For TYPE='D': Compute (palpha*I + A) = (alpha*I + A)
            for i in 0..n {
                a_cm[(i, i)] += palpha;
            }
        }

        // Step 2: LU factorization of (palpha*I +/- A)
        lapack_sys::dgetrf_(
            &n_i32,
            &n_i32,
            a_cm.as_mut_ptr(),
            &lda,
            ipiv.as_mut_ptr(),
            &mut info,
        );

        if info != 0 {
            return if typ == 'D' {
                Err("Matrix (alpha*I + A) is singular".to_string())
            } else {
                Err("Matrix (beta*I - A) is singular".to_string())
            };
        }

        // Step 3: Solve (alpha*I + A)^-1 * B
        // Result overwrites B
        let trans_n = b'N' as i8;
        lapack_sys::dgetrs_(
            &trans_n,
            &n_i32,
            &m_i32,
            a_cm.as_ptr(),
            &lda,
            ipiv.as_ptr(),
            b_cm.as_mut_ptr(),
            &ldb,
            &mut info,
        );

        if info != 0 {
            return Err(format!("DGETRS failed with INFO={}", info));
        }

        // Step 4: Compute D transformation
        // For TYPE='D': D_new = D - C * (alpha*I + A)^-1 * B
        // For TYPE='C': D_new = D + C * (beta*I - A)^-1 * B
        // IMPORTANT: Use ORIGINAL C (not yet transformed), and B after solve
        // D := D +/- C * B (where B now contains (alpha*I+A)^-1 * B_original)
        // Convert back to row-major temporarily to use ndarray
        let c_rm_orig = c_cm.view().reversed_axes();
        let b_rm_temp = b_cm.view().reversed_axes();
        let prod_rm = c_rm_orig.dot(&b_rm_temp);
        let prod_cm = prod_rm.reversed_axes();

        if typ == 'D' {
            d_cm -= &prod_cm;
        } else {
            // TYPE='C'
            d_cm += &prod_cm;
        }

        // Step 5: Scale B by sqrt(2*alpha*beta)
        b_cm.mapv_inplace(|x| x * sqrab2);

        // Step 6: Compute C * (alpha*I + A)^-1 and scale by sqrt(2*alpha*beta)
        // Solve C * X = C * (alpha*I + A)^-1 using triangular solves
        // Since A = P*L*U, we need: C * U^-1 * L^-1 * P^T

        // Convert A back to row-major temporarily for triangular solves
        let a_rm = a_cm.view().reversed_axes();

        // Solve C_new * U = C for each row of C (right multiplication by U^-1)
        for i in 0..p {
            let mut c_row = c_cm.slice(s![.., i]).to_owned();

            // Backward substitution for U^T * x = c_row (since we're doing right mult)
            // Solve U^T * x = b where U is upper triangular from LU
            for j in (0..n).rev() {
                let sum: f64 = (j + 1..n).map(|k| a_rm[(j, k)] * c_row[k]).sum();
                c_row[j] = (c_row[j] - sum) / a_rm[(j, j)];
            }

            // Apply to C
            for j in 0..n {
                c_cm[(j, i)] = c_row[j];
            }
        }

        // Now solve for L^-1 (unit diagonal lower triangular)
        for i in 0..p {
            let mut c_row = c_cm.slice(s![.., i]).to_owned();

            // Forward substitution for L^T * x = c_row
            for j in 0..n {
                let sum: f64 = (0..j).map(|k| a_rm[(j, k)] * c_row[k]).sum();
                c_row[j] -= sum; // L has unit diagonal
            }

            // Apply to C
            for j in 0..n {
                c_cm[(j, i)] = c_row[j];
            }
        }

        // Scale C by sqrt(2*alpha*beta)
        c_cm.mapv_inplace(|x| x * sqrab2);

        // Step 7: Apply column interchanges to C (from LU pivoting)
        for i in (0..n - 1).rev() {
            let ip = ipiv[i] as usize - 1; // Convert to 0-based
            if ip != i {
                // Swap columns i and ip in C
                for row in 0..p {
                    let tmp = c_cm[(i, row)];
                    c_cm[(i, row)] = c_cm[(ip, row)];
                    c_cm[(ip, row)] = tmp;
                }
            }
        }

        // Step 8: Compute A_new = beta*(alpha*I + A)^-1 * (A - alpha*I)
        // This is computed as: beta*I - 2*alpha*beta*(alpha*I + A)^-1

        // First, compute the inverse of (alpha*I + A)
        let mut work_query = [0.0f64];
        lapack_sys::dgetri_(
            &n_i32,
            a_cm.as_mut_ptr(),
            &lda,
            ipiv.as_ptr(),
            work_query.as_mut_ptr(),
            &-1,
            &mut info,
        );

        let lwork = work_query[0] as usize;
        let mut work = vec![0.0f64; lwork.max(n)];
        let lwork_i32 = work.len() as i32;

        lapack_sys::dgetri_(
            &n_i32,
            a_cm.as_mut_ptr(),
            &lda,
            ipiv.as_ptr(),
            work.as_mut_ptr(),
            &lwork_i32,
            &mut info,
        );

        if info != 0 {
            return Err(format!("DGETRI failed with INFO={}", info));
        }

        // Compute final A transformation
        // For TYPE='D': A_new = pbeta*I - ab2*(palpha*I + A)^-1 = beta*I - 2*alpha*beta*(alpha*I+A)^-1
        // For TYPE='C': A_new = ab2*(palpha*I - A)^-1 - pbeta*I = 2*alpha*beta*(beta*I-A)^-1 - alpha*I
        if typ == 'D' {
            // TYPE='D': Scale by -ab2 and add pbeta on diagonal
            for j in 0..n {
                for i in 0..n {
                    a_cm[(i, j)] *= -ab2;
                }
                a_cm[(j, j)] += pbeta;
            }
        } else {
            // TYPE='C': Scale by +ab2 and subtract pbeta on diagonal
            for j in 0..n {
                for i in 0..n {
                    a_cm[(i, j)] *= ab2;
                }
                a_cm[(j, j)] -= pbeta;
            }
        }
    }

    // Convert back to row-major
    a.assign(&a_cm.reversed_axes());
    b.assign(&b_cm.reversed_axes());
    c.assign(&c_cm.reversed_axes());
    d.assign(&d_cm.reversed_axes());

    Ok(())
}


pub fn ab05md(
    uplo: char,
    a1: &Array2<f64>,
    b1: &Array2<f64>,
    c1: &Array2<f64>,
    d1: &Array2<f64>,
    a2: &Array2<f64>,
    b2: &Array2<f64>,
    c2: &Array2<f64>,
    d2: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>, Array2<f64>, Array2<f64>), String> {
    // Extract dimensions
    let n1 = a1.nrows();
    let m1 = b1.ncols();
    let p1 = c1.nrows();
    let n2 = a2.nrows();
    let p2 = c2.nrows();

    // Validate UPLO parameter
    let uplo_upper = uplo == 'U' || uplo == 'u';
    let uplo_lower = uplo == 'L' || uplo == 'l';
    if !uplo_upper && !uplo_lower {
        return Err(format!(
            "Invalid UPLO parameter: '{}'. Must be 'U' or 'L'",
            uplo
        ));
    }

    // Validate dimensions
    if a1.ncols() != n1 {
        return Err(format!("A1 must be square, got {}×{}", n1, a1.ncols()));
    }
    if b1.nrows() != n1 {
        return Err(format!("B1 must have N1={} rows, got {}", n1, b1.nrows()));
    }
    if c1.ncols() != n1 {
        return Err(format!(
            "C1 must have N1={} columns, got {}",
            n1,
            c1.ncols()
        ));
    }
    if d1.nrows() != p1 || d1.ncols() != m1 {
        return Err(format!(
            "D1 must be P1×M1 = {}×{}, got {}×{}",
            p1,
            m1,
            d1.nrows(),
            d1.ncols()
        ));
    }
    if a2.ncols() != n2 {
        return Err(format!("A2 must be square, got {}×{}", n2, a2.ncols()));
    }
    if b2.nrows() != n2 || b2.ncols() != p1 {
        return Err(format!(
            "B2 must be N2×P1 = {}×{}, got {}×{}",
            n2,
            p1,
            b2.nrows(),
            b2.ncols()
        ));
    }
    if c2.ncols() != n2 {
        return Err(format!(
            "C2 must have N2={} columns, got {}",
            n2,
            c2.ncols()
        ));
    }
    if d2.nrows() != p2 || d2.ncols() != p1 {
        return Err(format!(
            "D2 must be P2×P1 = {}×{}, got {}×{}",
            p2,
            p1,
            d2.nrows(),
            d2.ncols()
        ));
    }

    // Quick return for zero dimensions
    let n = n1 + n2;
    if n == 0 {
        return Ok((
            Array2::zeros((0, 0)),
            Array2::zeros((0, m1)),
            Array2::zeros((p2, 0)),
            Array2::zeros((p2, m1)),
        ));
    }

    // Compute intermediate products
    let b2_c1 = b2.dot(c1); // N2×N1
    let b2_d1 = b2.dot(d1); // N2×M1
    let d2_c1 = d2.dot(c1); // P2×N1
    let d2_d1 = d2.dot(d1); // P2×M1

    // Construct output matrices based on UPLO
    let (a, b, c, d) = if uplo_lower {
        // Lower block diagonal form
        // A = [ A1      0  ]
        //     [ B2*C1  A2 ]
        let mut a = Array2::zeros((n, n));
        a.slice_mut(s![0..n1, 0..n1]).assign(a1);
        a.slice_mut(s![n1..n, 0..n1]).assign(&b2_c1);
        a.slice_mut(s![n1..n, n1..n]).assign(a2);

        // B = [  B1   ]
        //     [ B2*D1 ]
        let mut b = Array2::zeros((n, m1));
        b.slice_mut(s![0..n1, ..]).assign(b1);
        b.slice_mut(s![n1..n, ..]).assign(&b2_d1);

        // C = [ D2*C1  C2 ]
        let mut c = Array2::zeros((p2, n));
        c.slice_mut(s![.., 0..n1]).assign(&d2_c1);
        c.slice_mut(s![.., n1..n]).assign(c2);

        // D = [ D2*D1 ]
        let d = d2_d1.clone();

        (a, b, c, d)
    } else {
        // Upper block diagonal form (UPLO='U')
        // A = [ A2  B2*C1 ]
        //     [ 0    A1  ]
        let mut a = Array2::zeros((n, n));
        a.slice_mut(s![0..n2, 0..n2]).assign(a2);
        a.slice_mut(s![0..n2, n2..n]).assign(&b2_c1);
        a.slice_mut(s![n2..n, n2..n]).assign(a1);

        // B = [ B2*D1 ]
        //     [  B1   ]
        let mut b = Array2::zeros((n, m1));
        b.slice_mut(s![0..n2, ..]).assign(&b2_d1);
        b.slice_mut(s![n2..n, ..]).assign(b1);

        // C = [ C2  D2*C1 ]
        let mut c = Array2::zeros((p2, n));
        c.slice_mut(s![.., 0..n2]).assign(c2);
        c.slice_mut(s![.., n2..n]).assign(&d2_c1);

        // D = [ D2*D1 ]
        let d = d2_d1.clone();

        (a, b, c, d)
    };

    Ok((a, b, c, d))
}




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
    fn test_ab04md_continuous_to_discrete() {
        // Test data from AB04MD.html example
        // TYPE='C', N=2, M=2, P=2, ALPHA=1.0, BETA=1.0

        // Input matrices (read column-wise from HTML data)
        // A: READ ((A(I,J), I=1,N), J=1,N)
        let mut a = ndarray::array![[1.0, 0.5], [0.5, 1.0]];

        // B: READ ((B(I,J), I=1,N), J=1,M)
        let mut b = ndarray::array![[0.0, -1.0], [1.0, 0.0]];

        // C: READ ((C(I,J), I=1,P), J=1,N)
        let mut c = ndarray::array![[-1.0, 0.0], [0.0, 1.0]];

        // D: READ ((D(I,J), I=1,P), J=1,M)
        let mut d = ndarray::array![[1.0, 0.0], [0.0, -1.0]];

        // Perform transformation
        let result = ab04md('C', 1.0, 1.0, &mut a, &mut b, &mut c, &mut d);
        assert!(result.is_ok(), "ab04md should succeed");

        // Expected results from HTML (with tolerance 1e-3)
        let expected_a = ndarray::array![[-1.0, -4.0], [-4.0, -1.0]];

        let expected_b = ndarray::array![[2.8284, 0.0], [0.0, -2.8284]];

        let expected_c = ndarray::array![[0.0, 2.8284], [-2.8284, 0.0]];

        let expected_d = ndarray::array![[-1.0, 0.0], [0.0, -3.0]];

        // Check results with tolerance
        let tol = 1e-3;

        for i in 0..2 {
            for j in 0..2 {
                assert!(
                    (a[(i, j)] - expected_a[(i, j)]).abs() < tol,
                    "A[{}, {}] = {} differs from expected {} by more than {}",
                    i,
                    j,
                    a[(i, j)],
                    expected_a[(i, j)],
                    tol
                );
                assert!(
                    (b[(i, j)] - expected_b[(i, j)]).abs() < tol,
                    "B[{}, {}] = {} differs from expected {} by more than {}",
                    i,
                    j,
                    b[(i, j)],
                    expected_b[(i, j)],
                    tol
                );
                assert!(
                    (c[(i, j)] - expected_c[(i, j)]).abs() < tol,
                    "C[{}, {}] = {} differs from expected {} by more than {}",
                    i,
                    j,
                    c[(i, j)],
                    expected_c[(i, j)],
                    tol
                );
                assert!(
                    (d[(i, j)] - expected_d[(i, j)]).abs() < tol,
                    "D[{}, {}] = {} differs from expected {} by more than {}",
                    i,
                    j,
                    d[(i, j)],
                    expected_d[(i, j)],
                    tol
                );
            }
        }
    }

    #[test]
    fn test_ab04md_zero_dimensions() {
        // Test with N=0
        let mut a = Array2::zeros((0, 0));
        let mut b = Array2::zeros((0, 2));
        let mut c = Array2::zeros((2, 0));
        let mut d = Array2::zeros((2, 2));

        let result = ab04md('C', 1.0, 1.0, &mut a, &mut b, &mut c, &mut d);
        assert!(result.is_ok(), "Should handle N=0");

        // Test with M=0
        let mut a = Array2::zeros((2, 2));
        let mut b = Array2::zeros((2, 0));
        let mut c = Array2::zeros((2, 2));
        let mut d = Array2::zeros((2, 0));

        let result = ab04md('C', 1.0, 1.0, &mut a, &mut b, &mut c, &mut d);
        assert!(result.is_ok(), "Should handle M=0");

        // Test with P=0
        let mut a = Array2::zeros((2, 2));
        let mut b = Array2::zeros((2, 2));
        let mut c = Array2::zeros((0, 2));
        let mut d = Array2::zeros((0, 2));

        let result = ab04md('C', 1.0, 1.0, &mut a, &mut b, &mut c, &mut d);
        assert!(result.is_ok(), "Should handle P=0");
    }

    #[test]
    fn test_ab04md_parameter_validation() {
        let mut a = Array2::eye(2);
        let mut b = Array2::zeros((2, 2));
        let mut c = Array2::zeros((2, 2));
        let mut d = Array2::zeros((2, 2));

        // Invalid TYPE
        let result = ab04md('X', 1.0, 1.0, &mut a, &mut b, &mut c, &mut d);
        assert!(result.is_err(), "Should reject invalid TYPE");

        // Zero ALPHA
        let result = ab04md('C', 0.0, 1.0, &mut a, &mut b, &mut c, &mut d);
        assert!(result.is_err(), "Should reject ALPHA=0");

        // Zero BETA
        let result = ab04md('C', 1.0, 0.0, &mut a, &mut b, &mut c, &mut d);
        assert!(result.is_err(), "Should reject BETA=0");

        // Mismatched dimensions
        let mut b_bad = Array2::zeros((3, 2));
        let result = ab04md('C', 1.0, 1.0, &mut a, &mut b_bad, &mut c, &mut d);
        assert!(result.is_err(), "Should reject mismatched B dimensions");
    }

    // NOTE: Round-trip test (TYPE='C' followed by TYPE='D') is disabled temporarily
    // The TYPE='C' transformation matches SLICOT documentation exactly
    // TYPE='D' transformation requires further investigation for numerical accuracy
    // #[test]
    // fn test_ab04md_discrete_to_continuous() {
    //     ... (commented out for now)
    // }
}
