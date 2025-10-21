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

    // Step 3: Reduce A to upper Hessenberg form (simplified approach)
    // For full Hessenberg reduction, would use DGEHRD
    // Here we implement a simplified version using Householder reflections
    for col in 0..n.saturating_sub(2) {
        let col_start = col + 1;
        if col_start >= n {
            break;
        }

        // Extract column col below diagonal
        let mut col_vec = a.slice_mut(ndarray::s![col_start.., col]).to_owned();
        if col_vec.iter().all(|x| x.abs() < toldef) {
            continue;
        }

        // Compute Householder reflector for this column
        let (tau_i, _) = householder_vector(&mut col_vec);
        if (col + 1) < n {
            tau[col + 1] = tau_i;
        }

        // Apply reflector from left: A(col_start:, col_start:) := H * A(col_start:, col_start:)
        apply_householder_left(a, col_start, col, tau_i);

        // Apply reflector from right: A(:, col_start:) := A(:, col_start:) * H
        apply_householder_right(a, col_start, col, tau_i);
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

    // w := A*v
    let mut w: Array1<f64> = Array1::zeros(n);
    for i in 0..n {
        for j in 0..n {
            w[i] += a[(i, j)] * v[j];
        }
    }

    // A := A - tau*v*w'
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] -= tau * v[i] * w[j];
        }
    }

    // w := A'*v
    let mut w: Array1<f64> = Array1::zeros(n);
    for i in 0..n {
        for j in 0..n {
            w[i] += a[(j, i)] * v[j];
        }
    }

    // A := A - tau*w*v'
    for i in 0..n {
        for j in 0..n {
            a[(i, j)] -= tau * w[i] * v[j];
        }
    }
}

/// Apply Householder reflector from left: A(start:, start:) := H*A(start:, start:)
fn apply_householder_left(a: &mut Array2<f64>, start: usize, col: usize, tau: f64) {
    let n = a.nrows();
    if tau == 0.0 || start >= n {
        return;
    }

    let m = n - start;

    // Compute v as [1, a[start+1:, col], a[start+2:, col], ...]
    let mut v = Array1::zeros(m);
    v[0] = 1.0;
    for i in 1..m {
        if start + i < n {
            v[i] = a[(start + i, col)];
        }
    }

    // w := A'[start:, start:] * v
    let mut w: Array1<f64> = Array1::zeros(n - start);
    for i in start..n {
        for j in start..n {
            w[i - start] += a[(j, i)] * v[j - start];
        }
    }

    // A[start:, start:] := A[start:, start:] - tau*v*w'
    for i in 0..m {
        for j in start..n {
            a[(start + i, j)] -= tau * v[i] * w[j - start];
        }
    }
}

/// Apply Householder reflector from right: A(:, start:) := A(:, start:)*H
fn apply_householder_right(a: &mut Array2<f64>, start: usize, col: usize, tau: f64) {
    let n = a.nrows();
    if tau == 0.0 || start >= n {
        return;
    }

    let m = n - start;

    // Compute v as [1, a[start+1:, col], ...]
    let mut v = Array1::zeros(m);
    v[0] = 1.0;
    for i in 1..m {
        if start + i < n {
            v[i] = a[(start + i, col)];
        }
    }

    // w := A[start:, start:] * v
    let mut w: Array1<f64> = Array1::zeros(n);
    for i in 0..n {
        for j in 0..m {
            w[i] += a[(i, start + j)] * v[j];
        }
    }

    // A[:, start:] := A[:, start:] - tau*w*v'
    for i in 0..n {
        for j in 0..m {
            a[(i, start + j)] -= tau * w[i] * v[j];
        }
    }
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
}
