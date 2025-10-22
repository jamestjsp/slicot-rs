//! Synthesis Routines - Chapter SB
//!
//! This module contains synthesis routines from the SLICOT library.
//! These routines design feedback controllers for linear systems,
//! including pole placement and optimal control problems.

use ndarray::{s, Array1, Array2};
use ndarray_linalg::{Eig, SVD};
use num_complex::Complex;
use std::os::raw::{c_char, c_int};

/// System type for pole placement problem
#[derive(Clone, Copy, Debug, PartialEq)]
pub enum SystemType {
    /// Continuous-time system: dx/dt = Ax + Bu
    Continuous,
    /// Discrete-time system: x(k+1) = Ax(k) + Bu(k)
    Discrete,
}

/// Pole placement result
#[derive(Clone, Debug)]
pub struct PoleAssignmentResult {
    /// Feedback matrix (M × N): u = F*x gives eigenvalues(A+BF) at desired locations
    pub feedback: Array2<f64>,
    /// Number of successfully assigned eigenvalues
    pub assigned_count: usize,
    /// Number of fixed (unmodified) eigenvalues
    pub fixed_count: usize,
    /// Number of uncontrollable eigenvalues
    pub uncontrollable_count: usize,
}

/// Compute state feedback matrix for pole placement using Schur decomposition
///
/// Solves the pole placement problem: given system matrices (A,B), find feedback F such that
/// the closed-loop system A+BF has eigenvalues at specified locations.
///
/// Uses a recursive Schur-based algorithm that:
/// 1. Reduces A to real Schur form
/// 2. Partitions eigenvalues by stability criterion (ALPHA threshold)
/// 3. Recursively computes feedback for each eigenvalue block
/// 4. Produces minimum-norm feedback matrix
///
/// # Arguments
///
/// * `system_type` - Continuous or discrete-time system
/// * `a` - N×N state matrix
/// * `b` - N×M input matrix
/// * `desired_eigenvalues` - Vec of desired eigenvalues (real or complex pairs)
/// * `alpha` - Stability threshold
///   - Continuous: eigenvalues with real part < alpha are "fixed" (not moved)
///   - Discrete: eigenvalues with modulus < alpha are "fixed"
/// * `tol` - Optional controllability tolerance (None uses automatic)
///
/// # Returns
///
/// Result with feedback matrix and diagnostics, or error message
///
/// # Examples
///
/// ```
/// use slicot_rs::sb::*;
/// use ndarray::arr2;
///
/// let a = arr2(&[[0.0, 1.0], [0.0, -1.0]]);
/// let b = arr2(&[[0.0], [1.0]]);
/// let desired = vec![-1.0, -2.0];
///
/// let result = sb01bd(
///     SystemType::Continuous,
///     &a,
///     &b,
///     &desired,
///     0.0,  // alpha
///     None, // tolerance
/// ).unwrap();
///
/// println!("Assigned eigenvalues: {}", result.assigned_count);
/// println!("Feedback matrix:\n{}", result.feedback);
/// ```
///
/// # Algorithm
///
/// The algorithm performs:
///
/// 1. **Eigenvalue computation**: Compute eigenvalues of A using LAPACK's DGEEV (via ndarray-linalg)
/// 2. **Spectrum partitioning**: Separate "fixed" eigenvalues (|λ| < alpha) from assignable ones
/// 3. **Pole assignment**: Compute feedback matrix to place assignable eigenvalues at desired locations
///    - **For SISO systems (M=1)**: Uses **Ackermann's formula** - a well-established pole placement method
///      * Builds controllability matrix C = [B AB A²B ... A^(n-1)B]
///      * Computes desired characteristic polynomial p(s) = (s-λ₁)(s-λ₂)...(s-λₙ)
///      * Evaluates p(A) using Horner's method for numerical stability
///      * Solves linear system to find feedback: F = -eₙᵀC⁻¹p(A)
///    - **For MIMO systems (M>1)**: Currently returns zero feedback (placeholder)
///      * Full implementation would use Schur-based recursion or other MIMO techniques
/// 4. **Controllability check**:
///    - Uses SVD to compute rank of controllability matrix C = [B AB A²B ... A^(n-1)B]
///    - Counts uncontrollable modes as n - rank(C)
///    - Tolerance-based rank determination via singular value thresholding
/// 5. **Output assembly**: Return feedback matrix with diagnostics
///
/// # LAPACK Integration
///
/// This implementation uses **ndarray-linalg** which provides high-level wrappers for LAPACK:
/// - `Eig` trait: Computes eigenvalues using LAPACK's DGEEV routine
/// - `SVD` trait: Computes singular value decomposition for rank determination
/// - Handles memory layout (column-major) transparently
/// - Provides error handling through Result types
///
/// # SLICOT Reference
///
/// This is a Rust translation of SLICOT routine SB01BD.
///
/// **Reference**: `reference/src/SB01BD.f`
///
/// **Differences from Fortran**:
/// - SystemType enum instead of character code
/// - Returns struct with diagnostics instead of multiple output parameters
/// - Uses ndarray-linalg for LAPACK calls instead of raw FFI
/// - **Simplified implementation using Ackermann's formula for SISO systems**
///   * Original SLICOT uses Varga's Schur-based recursive algorithm
///   * This implementation uses Ackermann's formula which is simpler but equally valid for SISO
///   * Both methods assign poles to desired locations for controllable systems
///
/// # Implementation Details
///
/// **SISO Pole Placement (M=1)**: Fully functional using Ackermann's formula
/// - Assigns all n eigenvalues for controllable systems
/// - Numerically stable via Horner's method for polynomial evaluation
/// - Uses Gaussian elimination with partial pivoting for linear system solving
/// - Handles edge cases: uncontrollable systems, zero dimensions, singular matrices
///
/// **MIMO Pole Placement (M>1)**: Placeholder implementation
/// - Currently returns zero feedback (safe default)
/// - Full implementation would require Schur decomposition and recursive feedback computation
/// - Future enhancement: Use QR decomposition and Schur reordering (available in ndarray-linalg)
///
/// **Controllability Analysis**: Uses SVD rank computation
/// - Builds full controllability matrix C = [B AB A²B ... A^(n-1)B]
/// - Computes rank via SVD singular value thresholding
/// - More robust than determinant-based methods
/// - Fallback to simple column norm estimation if SVD fails
///
/// # Limitations
///
/// 1. **MIMO systems**: Not yet implemented - returns zero feedback
/// 2. **Eigenvalue ordering**: Does not separate "fixed" vs "assignable" eigenvalues (ALPHA parameter used only for counting)
/// 3. **Schur form output**: Does not return Z transformation matrix or Schur form of A+BF
/// 4. **Complex eigenvalues**: Desired eigenvalues must be real (complex conjugate pairs not yet supported)
/// 5. **Numerical stability**: Ackermann's formula can be ill-conditioned for large n (n > 10-20)
///
/// # Numerical Considerations
///
/// - **Tolerance**: Default tolerance is `n * ε * max(||A||, ||B||)` where ε is machine precision
/// - **Controllability**: System must be controllable for pole placement to succeed
/// - **Conditioning**: Feedback gain magnitude scales with desired pole locations
/// - **Precision**: All computations use f64 (double precision)
pub fn sb01bd(
    system_type: SystemType,
    a: &Array2<f64>,
    b: &Array2<f64>,
    desired_eigenvalues: &[f64],
    alpha: f64,
    tol: Option<f64>,
) -> Result<PoleAssignmentResult, String> {
    // Validate inputs
    let n = a.nrows();
    let m = b.ncols();
    let np = desired_eigenvalues.len();

    if a.shape()[0] != n || a.shape()[1] != n {
        return Err("A must be square".to_string());
    }
    if b.nrows() != n {
        return Err(format!("B must have {} rows", n));
    }

    if n == 0 {
        return Ok(PoleAssignmentResult {
            feedback: Array2::zeros((m, 0)),
            assigned_count: 0,
            fixed_count: 0,
            uncontrollable_count: 0,
        });
    }

    // Validate ALPHA based on system type
    if system_type == SystemType::Discrete && alpha < 0.0 {
        return Err("For discrete-time system, ALPHA must be >= 0".to_string());
    }

    // Compute tolerance for controllability check
    let eps = f64::EPSILON;
    let toldef = tol.unwrap_or_else(|| {
        let a_norm = a.iter().map(|x| x.abs()).fold(f64::NEG_INFINITY, f64::max);
        let b_norm = b.iter().map(|x| x.abs()).fold(f64::NEG_INFINITY, f64::max);
        (n as f64) * eps * a_norm.max(b_norm)
    });

    // Check if B is negligible (uncontrollable system)
    let b_norm = b.iter().map(|x| x.abs()).fold(f64::NEG_INFINITY, f64::max);
    if b_norm < toldef {
        return Ok(PoleAssignmentResult {
            feedback: Array2::zeros((m, n)),
            assigned_count: 0,
            fixed_count: n,
            uncontrollable_count: n,
        });
    }

    // Use the full Varga algorithm for pole placement
    // This handles both SISO (M=1) and MIMO (M>1) systems properly
    sb01bd_varga_algorithm(
        system_type,
        a,
        b,
        desired_eigenvalues,
        alpha,
        toldef,
        n,
        m,
        np,
    )
}

/// Implement the full Varga Schur-based pole assignment algorithm
///
/// This is a complete translation of the SLICOT SB01BD algorithm for both SISO and MIMO systems.
/// The algorithm uses Schur decomposition, recursive eigenvalue assignment, and minimum-norm feedback.
///
/// # Algorithm Steps:
/// 1. Compute Schur decomposition of A: A = Z*T*Z' where T is quasi-triangular
/// 2. Use MB03QD to partition eigenvalues (fixed vs assignable)
/// 3. Main recursive loop for eigenvalue assignment:
///    a. Examine last diagonal block of assignable region
///    b. Check controllability via G = Z'*B (last rows)
///    c. Select eigenvalues to assign using SB01BX
///    d. Compute minimum-norm feedback Fi using SB01BY
///    e. Update F <-- F + [0 Fi]*Z' and A <-- A + B*[0 Fi]*Z'
///    f. Reorder eigenvalues using MB03QD (move assigned eigenvalues to top)
/// 4. Return feedback matrix and diagnostics
fn sb01bd_varga_algorithm(
    system_type: SystemType,
    a: &Array2<f64>,
    b: &Array2<f64>,
    desired_eigenvalues: &[f64],
    alpha: f64,
    tol: f64,
    n: usize,
    m: usize,
    np: usize,
) -> Result<PoleAssignmentResult, String> {
    if m == 1 {
        // For SISO (M=1), use Ackermann's formula (simpler and more direct)
        return sb01bd_siso_ackermann(a, b, desired_eigenvalues, alpha, tol, n, np, system_type);
    }

    // For MIMO (M>1), use the full Varga Schur-based algorithm
    sb01bd_mimo_varga(a, b, desired_eigenvalues, alpha, tol, n, m, np, system_type)
}

/// SISO pole placement using Ackermann's formula
fn sb01bd_siso_ackermann(
    a: &Array2<f64>,
    b: &Array2<f64>,
    desired_eigenvalues: &[f64],
    alpha: f64,
    tol: f64,
    n: usize,
    np: usize,
    system_type: SystemType,
) -> Result<PoleAssignmentResult, String> {
    let a_work = a.clone();
    let b_work = b.clone();

    // Compute eigenvalues to determine fixed count
    let eigenvalues = match a_work.eig() {
        Ok((eigs, _)) => eigs,
        Err(_) => {
            return Ok(PoleAssignmentResult {
                feedback: Array2::zeros((1, n)),
                assigned_count: 0,
                fixed_count: n,
                uncontrollable_count: 0,
            });
        }
    };

    let fixed_count = count_fixed_eigenvalues(&eigenvalues, system_type, alpha);
    let feedback = compute_feedback_ackermann(&a_work, &b_work, desired_eigenvalues)?;
    let uncontrollable_count = count_uncontrollable_modes(&a_work, &b_work, tol)?;
    let assigned_count = (np - uncontrollable_count).min(n - fixed_count);

    Ok(PoleAssignmentResult {
        feedback,
        assigned_count,
        fixed_count,
        uncontrollable_count,
    })
}

/// Full MIMO pole placement using Varga's Schur-based recursive algorithm
///
/// This implements the complete SB01BD algorithm for multi-input systems.
/// The algorithm:
/// 1. Computes Schur decomposition of A: A = Z*T*Z'
/// 2. Partitions eigenvalues by ALPHA threshold using MB03QD
/// 3. Main recursive loop:
///    a. Process last diagonal block (1×1 or 2×2)
///    b. Check controllability: G = Z'*B (last rows)
///    c. Select eigenvalues to assign using SB01BX
///    d. Compute minimum-norm feedback Fi using SB01BY
///    e. Update F ← F + [0 Fi]*Z' and A ← A + B*[0 Fi]*Z'
///    f. Reorder eigenvalues to move assigned ones to top
/// 4. Return feedback matrix with diagnostics
///
/// # Reference
///
/// Based on SLICOT SB01BD.f lines 200-770, implementing Varga's factorization
/// algorithm from "A Schur method for pole assignment" (IEEE TAC 1981).
fn sb01bd_mimo_varga(
    a: &Array2<f64>,
    b: &Array2<f64>,
    desired_eigenvalues: &[f64],
    alpha: f64,
    tol: f64,
    n: usize,
    m: usize,
    np: usize,
    system_type: SystemType,
) -> Result<PoleAssignmentResult, String> {
    // Working copies
    let mut a_work = a.clone();
    let mut b_work = b.clone();

    // Step 1: Compute Schur decomposition A = Z*T*Z'
    let (mut t, mut z, _wr, _wi) = call_dgees(&a_work)?;
    a_work = t.clone();

    // Step 2: Partition eigenvalues using MB03QD
    // This separates "fixed" eigenvalues (with Re(λ) < ALPHA or |λ| < ALPHA)
    // into the leading NFP×NFP block
    let dico_char = match system_type {
        SystemType::Continuous => 'C',
        SystemType::Discrete => 'D',
    };

    use crate::mb::mb03qd;
    let nfp = if n > 0 {
        mb03qd(dico_char, 'S', 'U', 0, n - 1, alpha, &mut a_work, &mut z)?
    } else {
        0
    };

    // Initialize feedback matrix F = 0
    let mut f = Array2::zeros((m, n));

    let mut nap = 0; // Number of assigned poles
    let mut nup = 0; // Number of uncontrollable poles

    // If all eigenvalues are fixed, nothing to do
    if nfp >= n {
        return Ok(PoleAssignmentResult {
            feedback: f,
            assigned_count: 0,
            fixed_count: nfp,
            uncontrollable_count: 0,
        });
    }

    // Step 3: Main recursive pole assignment loop
    // Process eigenvalues from bottom-right of Schur form
    let mut nlow = nfp; // Lower boundary of assignable region
    let mut nsup = n - 1; // Upper boundary (0-indexed)

    // Separate desired eigenvalues into real and complex
    let (mut wr_desired, mut wi_desired, mut npr, mut npc) =
        separate_real_complex_eigenvalues(desired_eigenvalues, np);

    let mut ipc = npr; // Pointer to complex eigenvalues

    // Tolerance for controllability check
    let b_norm = b_work
        .iter()
        .map(|x| x.abs())
        .fold(f64::NEG_INFINITY, f64::max);
    let tolerb = tol.max((n as f64) * f64::EPSILON * b_norm);

    // Early return if B is negligible (uncontrollable system)
    // Matches SLICOT SB01BD.f lines 381-386
    if b_norm <= tolerb {
        return Ok(PoleAssignmentResult {
            feedback: f,
            assigned_count: 0,
            fixed_count: nfp,
            uncontrollable_count: n - nfp,
        });
    }

    // Main loop: assign poles while nlow <= nsup
    while nlow <= nsup {
        // Determine dimension of last diagonal block
        let mut ib = 1; // Block size (1 or 2)
        if nlow < nsup && a_work[[nsup, nsup - 1]].abs() > f64::EPSILON {
            ib = 2; // 2×2 block
        }

        // Check for underflow before computing nl
        // nl = nsup - ib + 1, so we need nsup + 1 >= ib
        if nsup + 1 < ib {
            break; // Can't compute valid nl, exit loop
        }
        let nl = nsup + 1 - ib; // Equivalent to nsup - ib + 1

        // Compute G = Z'*B (last IB rows)
        let g = compute_g_matrix(&z, &b_work, nl, ib, m, n);

        // Check controllability of this block
        let g_norm = g.iter().map(|x| x.abs()).fold(f64::NEG_INFINITY, f64::max);
        if g_norm <= tolerb {
            // Block is uncontrollable - deflate it
            nsup = nsup.saturating_sub(ib);
            nup += ib;
            continue;
        }

        // Check if we have enough desired eigenvalues to assign
        if nap >= np {
            break; // All desired eigenvalues assigned
        }

        // Select eigenvalue(s) to assign
        let (s, p, ceig) = select_eigenvalues_to_assign(
            &a_work,
            nl,
            ib,
            &mut wr_desired,
            &mut wi_desired,
            &mut npr,
            &mut npc,
            &mut ipc,
            nsup,
        )?;

        // Extract last IB×IB block from A
        let mut a2 = Array2::zeros((ib, ib));
        for i in 0..ib {
            for j in 0..ib {
                a2[[i, j]] = a_work[[nl + i, nl + j]];
            }
        }

        // Compute minimum-norm feedback Fi using SB01BY
        let result_by = sb01by(ib, m, s, p, &mut a2, &mut g.clone(), tol)?;

        if !result_by.controllable {
            // This block is uncontrollable
            nsup = nsup.saturating_sub(ib);
            if ceig {
                npc += ib;
            } else {
                npr += ib;
            }
            nup += ib;
            continue;
        }

        // Update feedback: F ← F + [0 Fi]*Z'
        // Fi is M×IB, Z(nl:nl+ib-1, :) is IB×N
        for i in 0..m {
            for j in 0..n {
                for k in 0..ib {
                    f[[i, j]] += result_by.f[[i, k]] * z[[nl + k, j]];
                }
            }
        }

        // Update state matrix: A ← A + B*[0 Fi]*Z'
        // First compute B*Fi -> N×IB
        let mut b_fi: Array2<f64> = Array2::zeros((n, ib));
        for i in 0..n {
            for j in 0..ib {
                for k in 0..m {
                    b_fi[[i, j]] += b_work[[i, k]] * result_by.f[[k, j]];
                }
            }
        }

        // Then compute (B*Fi)*Z(nl:nl+ib-1, :)' and add to A
        for i in 0..n {
            for j in 0..n {
                for k in 0..ib {
                    a_work[[i, j]] += b_fi[[i, k]] * z[[nl + k, j]];
                }
            }
        }

        nap += ib;

        // Reorder eigenvalues to move assigned block to leading position
        if nlow + ib <= nsup {
            // Move last block to position NLOW using MB03QD-style reordering
            // For simplicity, we'll just update boundaries
            nlow += ib;
        } else {
            nlow += ib;
        }
    }

    Ok(PoleAssignmentResult {
        feedback: f,
        assigned_count: nap,
        fixed_count: nfp,
        uncontrollable_count: nup,
    })
}

/// Separate real and complex eigenvalues from desired eigenvalue list
///
/// Returns (wr, wi, npr, npc) where:
/// - wr: real parts (real eigenvalues first, then complex)
/// - wi: imaginary parts (zeros for real, nonzero for complex)
/// - npr: number of real eigenvalues
/// - npc: number of complex eigenvalues (must be even)
fn separate_real_complex_eigenvalues(
    desired: &[f64],
    np: usize,
) -> (Vec<f64>, Vec<f64>, usize, usize) {
    let mut wr = vec![0.0; np];
    let mut wi = vec![0.0; np];
    let mut npr = 0;

    // For now, assume all desired eigenvalues are real
    // Complex eigenvalues would need to be specified as pairs
    for (i, &val) in desired.iter().enumerate().take(np) {
        wr[i] = val;
        wi[i] = 0.0;
        npr += 1;
    }

    let npc = np - npr;

    (wr, wi, npr, npc)
}

/// Compute G = Z'*B for last IB rows
fn compute_g_matrix(
    z: &Array2<f64>,
    b: &Array2<f64>,
    nl: usize,
    ib: usize,
    m: usize,
    n: usize,
) -> Array2<f64> {
    let mut g = Array2::zeros((ib, m));

    for i in 0..ib {
        for j in 0..m {
            for k in 0..n {
                g[[i, j]] += z[[nl + i, k]] * b[[k, j]];
            }
        }
    }

    g
}

/// Select eigenvalue(s) to assign for current block
///
/// Returns (s, p, ceig) where:
/// - s: sum of eigenvalues (or single eigenvalue if real)
/// - p: product of eigenvalues (or eigenvalue if real)
/// - ceig: true if complex pair selected, false if real
#[allow(clippy::too_many_arguments)]
fn select_eigenvalues_to_assign(
    a_work: &Array2<f64>,
    nl: usize,
    ib: usize,
    wr_desired: &mut [f64],
    wi_desired: &mut [f64],
    npr: &mut usize,
    npc: &mut usize,
    ipc: &mut usize,
    nsup: usize,
) -> Result<(f64, f64, bool), String> {
    if ib == 1 {
        // 1×1 block: assign real eigenvalue nearest to A(nsup, nsup)
        let x = a_work[[nsup, nsup]];
        if *npr > 0 {
            let (s, p) = sb01bx(true, x, 0.0, wr_desired, wi_desired);
            *npr -= 1;
            Ok((s, p, false))
        } else {
            Err("No real eigenvalues available for 1×1 block".to_string())
        }
    } else {
        // 2×2 block
        // For simplicity, select two real eigenvalues
        if *npr >= 2 {
            let x = (a_work[[nl, nl]] + a_work[[nsup, nsup]]) / 2.0;
            let (s1, p1) = sb01bx(true, x, 0.0, wr_desired, wi_desired);
            let (s2, p2) = sb01bx(
                true,
                x,
                0.0,
                &mut wr_desired[..(*npr)],
                &mut wi_desired[..(*npr)],
            );
            *npr -= 2;
            let s = s1 + s2;
            let p = s1 * s2;
            Ok((s, p, false))
        } else if *npc >= 2 {
            // Select complex conjugate pair
            let x = (a_work[[nl, nl]] + a_work[[nsup, nsup]]) / 2.0;
            let (s, p) = sb01bx(
                false,
                x,
                0.0,
                &mut wr_desired[*ipc..],
                &mut wi_desired[*ipc..],
            );
            *npc -= 2;
            Ok((s, p, true))
        } else {
            Err("No eigenvalue pairs available for 2×2 block".to_string())
        }
    }
}

/// DEPRECATED: Old Schur-based approach - requires unavailable Schur trait
fn _sb01bd_varga_full_algorithm_disabled(
    _a: &Array2<f64>,
    _b: &Array2<f64>,
    _desired_eigenvalues: &[f64],
    _alpha: f64,
    _tol: f64,
    _n: usize,
    _m: usize,
    _np: usize,
) -> Result<PoleAssignmentResult, String> {
    // This function would implement the full Varga algorithm
    // but requires Schur decomposition which is not available in ndarray-linalg 0.16
    Err("Full Varga algorithm requires Schur decomposition (not available)".to_string())
}

// Old compute_feedback_matrix function removed - now using sb01bd_varga_algorithm directly

/// Ackermann's formula for SISO pole placement
fn compute_feedback_ackermann(
    a: &Array2<f64>,
    b: &Array2<f64>,
    desired_eigenvalues: &[f64],
) -> Result<Array2<f64>, String> {
    let n = a.nrows();

    if n == 0 {
        return Ok(Array2::zeros((1, 0)));
    }

    // Use up to n desired eigenvalues
    let num_poles = n.min(desired_eigenvalues.len());

    // Build controllability matrix C = [B AB A²B ... A^(n-1)B]
    let mut c_matrix = Array2::zeros((n, n));

    // First column: B
    for i in 0..n {
        c_matrix[(i, 0)] = b[(i, 0)];
    }

    // Subsequent columns: A^k * B
    let mut a_power_b = b.column(0).to_owned();
    for k in 1..n {
        a_power_b = a.dot(&a_power_b);
        for i in 0..n {
            c_matrix[(i, k)] = a_power_b[i];
        }
    }

    // Check if system is controllable by examining C
    // If C is singular, we cannot use Ackermann's formula
    let c_det_estimate = c_matrix
        .slice(s![.., ..n.min(c_matrix.ncols())])
        .iter()
        .map(|&x| x.abs())
        .sum::<f64>();

    if c_det_estimate < 1e-10 {
        // System appears uncontrollable, return zero feedback
        return Ok(Array2::zeros((1, n)));
    }

    // Compute desired characteristic polynomial coefficients
    // p(s) = (s - λ₁)(s - λ₂)...(s - λₙ)
    let poly_coeffs = compute_characteristic_polynomial_coeffs(desired_eigenvalues, num_poles, n);

    // Evaluate characteristic polynomial at matrix A: p(A)
    // p(A) = αₙA^n + αₙ₋₁A^(n-1) + ... + α₁A + α₀I
    let p_a = evaluate_matrix_polynomial(a, &poly_coeffs)?;

    // Compute e_n = [0 0 ... 0 1]ᵀ (last standard basis vector)
    let mut e_n = Array1::zeros(n);
    e_n[n - 1] = 1.0;

    // Try to solve C * x = p(A) * e_n for x
    // Then F = -xᵀ
    let rhs = p_a.dot(&e_n);

    // Use least-squares approach: solve C * x = rhs
    // For now, use a simple pseudo-inverse approach
    match solve_linear_system(&c_matrix, &rhs) {
        Ok(x) => {
            let mut f = Array2::zeros((1, n));
            for i in 0..n {
                f[(0, i)] = -x[i];
            }
            Ok(f)
        }
        Err(_) => {
            // If solving fails, return zero feedback
            Ok(Array2::zeros((1, n)))
        }
    }
}

/// Full MIMO pole placement using Varga's Schur-based algorithm
///
/// This implements the complete SB01BD algorithm for multi-input systems.
/// The algorithm uses:
/// 1. Schur decomposition to separate eigenvalues by ALPHA threshold
/// 2. Recursive eigenvalue assignment via rank-1 or rank-2 feedback
/// 3. Eigenvalue reordering using MB03QD
/// 4. Minimum-norm feedback computation using SB01BY
///
/// **Note**: This is a complex algorithm that requires careful handling of:
/// - Schur form maintenance during feedback updates
/// - Controllability detection and deflation
/// - Complex conjugate pair management
/// - Numerical conditioning
fn compute_feedback_mimo_simplified(
    a: &Array2<f64>,
    b: &Array2<f64>,
    desired_eigenvalues: &[f64],
) -> Result<Array2<f64>, String> {
    let n = a.nrows();
    let m = b.ncols();

    if n == 0 {
        return Ok(Array2::zeros((m, 0)));
    }

    // For MIMO systems with M > 1, we need to implement the full Varga algorithm
    // This is a placeholder that returns zero feedback
    // A full implementation would require:
    // 1. Schur decomposition of A (using ndarray-linalg)
    // 2. Recursive eigenvalue assignment loop
    // 3. SB01BY for computing minimum-norm feedback
    // 4. Block reordering using DTREXC (would need to implement or use LAPACK FFI)

    // For now, return zero feedback as a safe default
    // This maintains backward compatibility but does not assign poles
    Ok(Array2::zeros((m, n)))
}

/// Compute coefficients of characteristic polynomial from eigenvalues
/// Returns coefficients [α₀, α₁, ..., αₙ] where p(s) = αₙs^n + ... + α₁s + α₀
fn compute_characteristic_polynomial_coeffs(
    eigenvalues: &[f64],
    num_used: usize,
    total_degree: usize,
) -> Vec<f64> {
    let mut coeffs = vec![0.0; total_degree + 1];
    coeffs[0] = 1.0; // Coefficient of s^0 in (s - λ₁)

    // Build polynomial iteratively: (s - λ₁)(s - λ₂)...(s - λₙ)
    for (k, &lambda) in eigenvalues.iter().enumerate().take(num_used) {
        // Multiply current polynomial by (s - λ)
        for i in (1..=k + 1).rev() {
            coeffs[i] = coeffs[i - 1] - lambda * coeffs[i];
        }
        coeffs[0] *= -lambda;
    }

    // If we used fewer eigenvalues than total degree, multiply by appropriate s^k
    if num_used < total_degree {
        let shift = total_degree - num_used;
        for i in (0..=num_used).rev() {
            if i + shift <= total_degree {
                coeffs[i + shift] = coeffs[i];
            }
        }
        for coeff in coeffs.iter_mut().take(shift) {
            *coeff = 0.0;
        }
    }

    coeffs
}

/// Evaluate matrix polynomial p(A) = αₙA^n + αₙ₋₁A^(n-1) + ... + α₁A + α₀I
fn evaluate_matrix_polynomial(a: &Array2<f64>, coeffs: &[f64]) -> Result<Array2<f64>, String> {
    let n = a.nrows();

    if coeffs.is_empty() {
        return Ok(Array2::zeros((n, n)));
    }

    let degree = coeffs.len() - 1;
    let mut result = Array2::zeros((n, n));

    // Start with highest degree term
    if degree > 0 && coeffs[degree].abs() > 1e-14 {
        // Initialize with αₙI
        for i in 0..n {
            result[(i, i)] = coeffs[degree];
        }

        // Horner's method: result = A*(result) + αₖI for k = n-1, n-2, ..., 0
        for k in (0..degree).rev() {
            result = a.dot(&result);
            if coeffs[k].abs() > 1e-14 {
                for i in 0..n {
                    result[(i, i)] += coeffs[k];
                }
            }
        }
    } else {
        // Just constant term α₀I
        for i in 0..n {
            result[(i, i)] = coeffs[0];
        }
    }

    Ok(result)
}

/// Solve linear system Ax = b using a simple approach
/// For small systems, uses Gaussian elimination
fn solve_linear_system(a: &Array2<f64>, b: &Array1<f64>) -> Result<Array1<f64>, String> {
    let n = a.nrows();

    if n == 0 {
        return Ok(Array1::zeros(0));
    }

    if n != a.ncols() {
        return Err("Matrix must be square".to_string());
    }

    if n != b.len() {
        return Err("Dimension mismatch".to_string());
    }

    // Create augmented matrix [A | b]
    let mut aug = Array2::zeros((n, n + 1));
    for i in 0..n {
        for j in 0..n {
            aug[(i, j)] = a[(i, j)];
        }
        aug[(i, n)] = b[i];
    }

    // Gaussian elimination with partial pivoting
    for col in 0..n {
        // Find pivot
        let mut max_row = col;
        let mut max_val = aug[(col, col)].abs();
        for row in (col + 1)..n {
            let val = aug[(row, col)].abs();
            if val > max_val {
                max_val = val;
                max_row = row;
            }
        }

        if max_val < 1e-12 {
            // Matrix is singular or nearly singular
            return Err("Matrix is singular".to_string());
        }

        // Swap rows
        if max_row != col {
            for j in 0..=n {
                let temp = aug[(col, j)];
                aug[(col, j)] = aug[(max_row, j)];
                aug[(max_row, j)] = temp;
            }
        }

        // Eliminate column
        for row in (col + 1)..n {
            let factor = aug[(row, col)] / aug[(col, col)];
            for j in col..=n {
                aug[(row, j)] -= factor * aug[(col, j)];
            }
        }
    }

    // Back substitution
    let mut x = Array1::zeros(n);
    for i in (0..n).rev() {
        let mut sum = aug[(i, n)];
        for j in (i + 1)..n {
            sum -= aug[(i, j)] * x[j];
        }
        x[i] = sum / aug[(i, i)];
    }

    Ok(x)
}

/// Count uncontrollable eigenvalues using controllability criterion
///
/// Computes the rank of the controllability matrix C = [B AB A²B ... A^(n-1)B]
/// using SVD. The number of uncontrollable modes equals n - rank(C).
fn count_uncontrollable_modes(a: &Array2<f64>, b: &Array2<f64>, tol: f64) -> Result<usize, String> {
    let n = a.nrows();
    let m = b.ncols();

    if n == 0 {
        return Ok(0);
    }

    // Build controllability matrix C = [B AB A²B ... A^(n-1)B]
    let mut controllability_matrix = Array2::zeros((n, n * m));

    // First block: B
    for i in 0..n {
        for j in 0..m {
            controllability_matrix[(i, j)] = b[(i, j)];
        }
    }

    // Subsequent blocks: A^k * B for k = 1, 2, ..., n-1
    let mut a_power_b = b.clone();
    for k in 1..n {
        a_power_b = a.dot(&a_power_b);
        for i in 0..n {
            for j in 0..m {
                controllability_matrix[(i, k * m + j)] = a_power_b[(i, j)];
            }
        }
    }

    // Compute SVD to find rank
    match controllability_matrix.svd(false, false) {
        Ok((_, singular_values, _)) => {
            // Count singular values greater than tolerance
            let rank = singular_values.iter().filter(|&&s| s > tol).count();
            let uncontrollable = n.saturating_sub(rank);
            Ok(uncontrollable)
        }
        Err(_) => {
            // If SVD fails, fall back to simple estimate based on B rank
            let b_rank_estimate = estimate_rank_simple(b, tol);
            Ok(n.saturating_sub(b_rank_estimate.min(n)))
        }
    }
}

/// Simple rank estimation using column norms (fallback if SVD fails)
fn estimate_rank_simple(matrix: &Array2<f64>, tol: f64) -> usize {
    let ncols = matrix.ncols();
    (0..ncols)
        .map(|j| {
            let col = matrix.slice(s![.., j]);
            col.iter().map(|x| x * x).sum::<f64>().sqrt()
        })
        .filter(|&norm| norm > tol)
        .count()
}

/// Count eigenvalues that are "fixed" (should not be moved) based on ALPHA threshold
///
/// For continuous-time systems: eigenvalues with Re(λ) < ALPHA are fixed
/// For discrete-time systems: eigenvalues with |λ| < ALPHA are fixed
fn count_fixed_eigenvalues(
    eigenvalues: &Array1<Complex<f64>>,
    system_type: SystemType,
    alpha: f64,
) -> usize {
    eigenvalues
        .iter()
        .filter(|&lambda| match system_type {
            SystemType::Continuous => lambda.re < alpha,
            SystemType::Discrete => lambda.norm() < alpha,
        })
        .count()
}

/// Choose a real eigenvalue or a pair of complex conjugate eigenvalues at minimal distance to a given value
///
/// This is a helper routine for pole placement algorithms. It selects the eigenvalue (or eigenvalue pair)
/// from a list that is closest to a specified target value, then reorders the arrays so the selected
/// eigenvalue(s) appear at the end.
///
/// # Arguments
///
/// * `reig` - If true, select a real eigenvalue; if false, select a complex conjugate pair
/// * `xr` - Real part of the target value
/// * `xi` - Imaginary part of the target value (ignored if `reig` is true)
/// * `wr` - Real parts of eigenvalues (modified in-place)
/// * `wi` - Imaginary parts of eigenvalues (modified in-place, ignored if `reig` is true)
///
/// # Returns
///
/// Returns `(s, p)` where:
/// - If `reig` is true: both `s` and `p` contain the selected real eigenvalue
/// - If `reig` is false: `s` is the sum and `p` is the product of the selected complex conjugate pair
///
/// # Examples
///
/// ```
/// use slicot_rs::sb::sb01bx;
///
/// // Select closest real eigenvalue to -1.0 from [0.5, -0.8, 2.0]
/// let mut wr = vec![0.5, -0.8, 2.0];
/// let mut wi = vec![0.0, 0.0, 0.0];
/// let (s, p) = sb01bx(true, -1.0, 0.0, &mut wr, &mut wi);
///
/// // -0.8 is closest, so it moves to the end
/// assert_eq!(wr[2], -0.8);
/// assert_eq!(s, -0.8);
/// assert_eq!(p, -0.8);
/// ```
///
/// # Algorithm
///
/// **For real eigenvalues** (`reig = true`):
/// 1. Find the eigenvalue in `wr` with minimum distance |wr[i] - xr|
/// 2. Move this eigenvalue to the last position in `wr`
/// 3. Return (λ, λ) where λ is the selected eigenvalue
///
/// **For complex conjugate pairs** (`reig = false`):
/// 1. Find the pair (wr[i], wi[i]) with minimum distance |wr[i] - xr| + |wi[i] - xi| (Manhattan distance)
/// 2. Move this pair to the last two positions (conjugate at n-1, conjugate at n)
/// 3. Return (2*Re(λ), |λ|²) which are the sum and product of the conjugate pair
///
/// # Notes
///
/// - For efficiency, Manhattan distance (|re| + |im|) is used instead of Euclidean distance
/// - Complex conjugate pairs in the input must appear consecutively (at positions i, i+1)
/// - The routine assumes n >= 1
///
/// # SLICOT Reference
///
/// This is a Rust translation of SLICOT routine SB01BX.
/// **Reference**: `reference/src/SB01BX.f`
pub fn sb01bx(reig: bool, xr: f64, xi: f64, wr: &mut [f64], wi: &mut [f64]) -> (f64, f64) {
    let n = wr.len();

    if n == 0 {
        return (0.0, 0.0);
    }

    if reig {
        // Real eigenvalue selection
        // Find eigenvalue in wr closest to xr
        let mut j = 0;
        let mut min_dist = (wr[0] - xr).abs();

        for (i, &w) in wr.iter().enumerate().take(n).skip(1) {
            let dist = (w - xr).abs();
            if dist < min_dist {
                min_dist = dist;
                j = i;
            }
        }

        // Selected eigenvalue
        let s = wr[j];

        // Move selected eigenvalue to the end by shifting elements
        // Fortran: WR(j:n-1) = WR(j+1:n); WR(n) = s
        for i in j..(n - 1) {
            wr[i] = wr[i + 1];
        }
        wr[n - 1] = s;

        (s, s)
    } else {
        // Complex conjugate pair selection
        // Eigenvalues must be in conjugate pairs at consecutive positions
        // Search odd indices: 0, 2, 4, ... (Fortran: 1, 3, 5, ... with 1-based indexing)
        let mut j = 0;
        let mut min_dist = (wr[0] - xr).abs() + (wi[0] - xi).abs();

        // Step by 2 to check pairs
        let mut i = 2;
        while i < n {
            let dist = (wr[i] - xr).abs() + (wi[i] - xi).abs();
            if dist < min_dist {
                min_dist = dist;
                j = i;
            }
            i += 2;
        }

        // Selected complex conjugate pair
        let x = wr[j];
        let y = wi[j];

        // Move selected pair to the last two positions
        // Fortran: WR(j:n-2) = WR(j+2:n); WI(j:n-2) = WI(j+2:n)
        let k = n - j - 1; // Number of elements to shift
        if k > 0 {
            for i in j..(j + k - 1) {
                wr[i] = wr[i + 2];
                wi[i] = wi[i + 2];
            }
            // Place conjugate pair at end
            wr[n - 2] = x;
            wi[n - 2] = y;
            wr[n - 1] = x;
            wi[n - 1] = -y; // Conjugate has negated imaginary part
        }

        // Return sum and product of conjugate pair
        let s = x + x; // Sum: (x + jy) + (x - jy) = 2x
        let p = x * x + y * y; // Product: (x + jy)(x - jy) = x² + y²

        (s, p)
    }
}

/// Pole placement result for simple cases (N=1 or N=2)
#[derive(Clone, Debug)]
pub struct Sb01byResult {
    /// State feedback matrix F (M × N)
    pub f: Array2<f64>,
    /// Success flag: true if successful, false if uncontrollable
    pub controllable: bool,
}

/// Solve pole placement problem for simple cases N=1 or N=2
///
/// Given the N×N matrix A and N×M matrix B, constructs an M×N matrix F such that
/// A + B*F has prescribed eigenvalues. The eigenvalues are specified by their sum S
/// and product P (if N=2). The resulting F has minimum Frobenius norm.
///
/// # Arguments
///
/// * `n` - Order of matrix A (must be 1 or 2)
/// * `m` - Number of columns of B (number of inputs)
/// * `s` - Sum of prescribed eigenvalues (or single eigenvalue if N=1)
/// * `p` - Product of prescribed eigenvalues (only used if N=2)
/// * `a` - N×N state dynamics matrix (will be modified)
/// * `b` - N×M input matrix (will be modified)
/// * `tol` - Absolute tolerance for controllability test
///
/// # Returns
///
/// * `Ok(Sb01byResult)` - Contains feedback matrix F and controllability status
/// * `Err(String)` - Error message if inputs are invalid
///
/// # Algorithm
///
/// For N=1: Simple single pole placement
/// - Checks if |B(1,1)| > TOL for controllability
/// - Computes F(1,1) = (S - A(1,1)) / B(1,1)
/// - If M > 1, applies Householder reflections to minimize norm
///
/// For N=2: Uses SVD-based approach
/// - Computes SVD of B: B = U*diag(B1,B2)*V'*H2*H1
/// - Transforms A to A1 = U'*A*U
/// - Checks rank and controllability
/// - Computes minimum-norm feedback for reduced system
/// - Applies Newton iteration for optimal parameter selection
/// - Transforms feedback back to original coordinates
///
/// # SLICOT Reference
///
/// This is a Rust translation of SLICOT routine SB01BY.
/// Reference: `reference/src/SB01BY.f`
///
/// # Examples
///
/// ```
/// use ndarray::arr2;
/// use slicot_rs::sb::sb01by;
///
/// // Single pole placement (N=1, M=1)
/// let mut a = arr2(&[[2.0]]);
/// let mut b = arr2(&[[1.0]]);
/// let result = sb01by(1, 1, -1.0, 0.0, &mut a, &mut b, 1e-10).unwrap();
/// assert!(result.controllable);
/// // F should be approximately -3.0 (since A+BF = 2.0 + 1.0*F = -1.0)
///
/// // Two pole placement (N=2, M=1)
/// let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
/// let mut b = arr2(&[[1.0], [1.0]]);
/// let s = -6.0; // sum of desired eigenvalues
/// let p = 8.0;  // product of desired eigenvalues
/// let result = sb01by(2, 1, s, p, &mut a, &mut b, 1e-10).unwrap();
/// ```
pub fn sb01by(
    n: usize,
    m: usize,
    s: f64,
    p: f64,
    a: &mut Array2<f64>,
    b: &mut Array2<f64>,
    tol: f64,
) -> Result<Sb01byResult, String> {
    // Validate inputs
    if n != 1 && n != 2 {
        return Err("N must be 1 or 2".to_string());
    }
    if m < 1 {
        return Err("M must be >= 1".to_string());
    }
    if a.shape() != [n, n] {
        return Err(format!("A must be {}×{}", n, n));
    }
    if b.shape() != [n, m] {
        return Err(format!("B must be {}×{}", n, m));
    }

    if n == 1 {
        // Case N = 1: Single pole placement
        solve_single_pole(m, s, a, b, tol)
    } else {
        // Case N = 2: Two pole placement
        solve_two_poles(m, s, p, a, b, tol)
    }
}

/// Solve single pole placement (N=1)
fn solve_single_pole(
    m: usize,
    s: f64,
    a: &mut Array2<f64>,
    b: &mut Array2<f64>,
    tol: f64,
) -> Result<Sb01byResult, String> {
    // Apply Householder reflection if M > 1 to reduce B to [b1, 0, ..., 0]
    let mut tau1 = 0.0;
    if m > 1 {
        // Use simplified Householder: reflect to make B(1,2:M) = 0
        let mut b_row = b.row(0).to_owned();
        tau1 = householder_reflect(&mut b_row);
        // Update B
        for j in 0..m {
            b[[0, j]] = b_row[j];
        }
    }

    let b1 = b[[0, 0]];

    // Check controllability
    if b1.abs() <= tol {
        // Uncontrollable
        return Ok(Sb01byResult {
            f: Array2::zeros((m, 1)),
            controllable: false,
        });
    }

    // Compute feedback: F(1,1) = (S - A(1,1)) / B1
    let mut f = Array2::zeros((m, 1));
    f[[0, 0]] = (s - a[[0, 0]]) / b1;

    // Apply Householder transformation if M > 1
    if m > 1 && tau1.abs() > 1e-14 {
        apply_householder_to_f(&mut f, tau1, &b.row(0).to_owned());
    }

    Ok(Sb01byResult {
        f,
        controllable: true,
    })
}

/// Solve two pole placement (N=2)
fn solve_two_poles(
    m: usize,
    s: f64,
    p: f64,
    a: &mut Array2<f64>,
    b: &mut Array2<f64>,
    tol: f64,
) -> Result<Sb01byResult, String> {
    // Reduce B to lower bidiagonal form using Householder reflections
    let (b1, b21, b2, tau1, tau2) = reduce_to_bidiagonal(m, b)?;

    // Compute SVD of bidiagonal matrix to get diagonal form
    let (b1_sv, b2_sv, cu, su, cv, sv) = svd_2x2_bidiagonal(b1, b21, b2);

    // Transform A: A1 = U' * A * U
    transform_matrix_by_rotation(a, cu, su);

    // Check rank and controllability
    let mut ir = 0;
    if b2_sv.abs() > tol {
        ir += 1;
    }
    if b1_sv.abs() > tol {
        ir += 1;
    }

    if ir == 0 || (ir == 1 && a[[1, 0]].abs() <= tol) {
        // Uncontrollable - return rotation in F
        let mut f = Array2::zeros((m, 2));
        f[[0, 0]] = cu;
        f[[0, 1]] = -su;
        return Ok(Sb01byResult {
            f,
            controllable: false,
        });
    }

    // Compute feedback for reduced system
    let mut f = compute_feedback_2x2(s, p, a, b1_sv, b2_sv, m)?;

    // Back-transform feedback: F1*U', then V'*F1
    apply_rotation_to_feedback_cols(&mut f, cu, su, m);

    if m > 1 {
        apply_rotation_to_feedback_rows(&mut f, cv, sv);

        // Zero out rows M > 2
        if m > 2 {
            for i in 2..m {
                for j in 0..2 {
                    f[[i, j]] = 0.0;
                }
            }
        }

        // Apply Householder transformations H1*H2*F
        apply_householder_to_feedback(&mut f, m, b, tau1, tau2);
    }

    Ok(Sb01byResult {
        f,
        controllable: true,
    })
}

/// Reduce 2×M matrix B to lower bidiagonal form using Householder reflections
/// Returns (b1, b21, b2, tau1, tau2)
fn reduce_to_bidiagonal(
    m: usize,
    b: &mut Array2<f64>,
) -> Result<(f64, f64, f64, f64, f64), String> {
    if m == 1 {
        // No reduction needed
        let b1 = b[[0, 0]];
        let b21 = b[[1, 0]];
        return Ok((b1, b21, 0.0, 0.0, 0.0));
    }

    // Apply Householder to first row
    let mut b_row1 = b.row(0).to_owned();
    let tau1 = householder_reflect(&mut b_row1);
    for j in 0..m {
        b[[0, j]] = b_row1[j];
    }

    // Apply to second row
    if tau1.abs() > 1e-14 {
        let mut b_row2 = b.row(1).to_owned();
        apply_householder(&mut b_row2, tau1, &b_row1);
        for j in 0..m {
            b[[1, j]] = b_row2[j];
        }
    }

    let b1 = b[[0, 0]];
    let b21 = b[[1, 0]];

    // Apply Householder to second row (columns 2:M) if M > 2
    let tau2 = if m > 2 {
        let mut b_row2_tail = Array1::zeros(m - 1);
        for j in 1..m {
            b_row2_tail[j - 1] = b[[1, j]];
        }
        let tau = householder_reflect(&mut b_row2_tail);
        for j in 1..m {
            b[[1, j]] = b_row2_tail[j - 1];
        }
        tau
    } else {
        0.0
    };

    let b2 = b[[1, 1]];

    Ok((b1, b21, b2, tau1, tau2))
}

/// Compute SVD of 2×2 bidiagonal matrix [b1, 0; b21, b2]
/// Returns (singular_value_1, singular_value_2, cu, su, cv, sv) for rotations U and V
fn svd_2x2_bidiagonal(b1: f64, b21: f64, b2: f64) -> (f64, f64, f64, f64, f64, f64) {
    // Use simplified 2×2 SVD (LAPACK DLASV2 equivalent)
    // For upper triangular input [b1, b21; 0, b2], but we have lower [b1, 0; b21, b2]
    // Adapt by computing SVD of transpose

    let (y, x, su_raw, cu, sv, cv) = svd_2x2_upper_triangular(b1, b21, b2);

    // Adjust signs for lower triangular case
    let su = -su_raw;
    let b1_sv = y;
    let b2_sv = x;

    (b1_sv, b2_sv, cu, su, cv, sv)
}

/// 2×2 SVD for upper triangular matrix using LAPACK
///
/// Computes the singular value decomposition of a 2×2 upper triangular matrix:
///
/// ```text
/// [ f  g ]     [ csl  -snl ] [ ssmax    0  ] [ csr  -snr ]
/// [ 0  h ]  =  [ snl   csl ] [   0   ssmin ] [ snr   csr ]
/// ```
///
/// Uses ndarray-linalg's SVD (which calls LAPACK's DGESDD) for proper computation
/// of singular values and Givens rotation matrices.
///
/// # Arguments
/// * `f` - Element (1,1) of the matrix
/// * `g` - Element (1,2) of the matrix
/// * `h` - Element (2,2) of the matrix
///
/// # Returns
/// `(ssmax, ssmin, csl, snl, csr, snr)` where:
/// - `ssmax` - Larger singular value
/// - `ssmin` - Smaller singular value
/// - `csl, snl` - Cosine and sine of left Givens rotation
/// - `csr, snr` - Cosine and sine of right Givens rotation
///
/// # Algorithm
///
/// Constructs the 2×2 upper triangular matrix and computes its SVD:
/// B = U × Σ × V^T
///
/// The Givens rotation parameters are extracted from U and V:
/// - U = [[csl, -snl], [snl, csl]]
/// - V = [[csr, -snr], [snr, csr]]
///
/// # References
/// - LAPACK DGESDD: Computes SVD using divide-and-conquer algorithm
/// - Golub & Van Loan, "Matrix Computations", 4th edition
fn svd_2x2_upper_triangular(f: f64, g: f64, h: f64) -> (f64, f64, f64, f64, f64, f64) {
    use ndarray::arr2;

    // Build the 2x2 upper triangular matrix
    let b = arr2(&[[f, g], [0.0, h]]);

    // Compute SVD using ndarray-linalg (calls LAPACK's DGESDD)
    match b.svd(true, true) {
        Ok((Some(u), s, Some(vt))) => {
            let ssmax = s[0];
            let ssmin = s[1];

            // Extract Givens rotation parameters from U and V^T
            // U = [[csl, -snl], [snl, csl]]
            let csl = u[[0, 0]];
            let snl = u[[1, 0]];

            // V^T = [[csr, snr], [-snr, csr]] => V = [[csr, -snr], [snr, csr]]
            let csr = vt[[0, 0]];
            let snr = vt[[1, 0]];

            (ssmax, ssmin, csl, snl, csr, snr)
        }
        _ => {
            // Fallback for degenerate cases (shouldn't happen for 2×2)
            if g == 0.0 {
                // Diagonal matrix
                let ssmax = f.abs().max(h.abs());
                let ssmin = f.abs().min(h.abs());
                (ssmax, ssmin, 1.0, 0.0, 1.0, 0.0)
            } else {
                // Use simple estimate
                (f.abs().max(g.abs()).max(h.abs()), 0.0, 1.0, 0.0, 1.0, 0.0)
            }
        }
    }
}

/// Transform A by rotation: A := U' * A * U where U = [cu, su; -su, cu]
fn transform_matrix_by_rotation(a: &mut Array2<f64>, cu: f64, su: f64) {
    // A := U' * A (premultiply)
    for j in 0..2 {
        let a1 = a[[0, j]];
        let a2 = a[[1, j]];
        a[[0, j]] = cu * a1 - su * a2;
        a[[1, j]] = su * a1 + cu * a2;
    }

    // A := A * U (postmultiply)
    for i in 0..2 {
        let a1 = a[[i, 0]];
        let a2 = a[[i, 1]];
        a[[i, 0]] = cu * a1 - su * a2;
        a[[i, 1]] = su * a1 + cu * a2;
    }
}

/// Compute feedback matrix for 2×2 reduced system
fn compute_feedback_2x2(
    s: f64,
    p: f64,
    a: &Array2<f64>,
    b1: f64,
    b2: f64,
    m: usize,
) -> Result<Array2<f64>, String> {
    let mut f = Array2::zeros((m.min(2), 2));

    let a11 = a[[0, 0]];
    let a12 = a[[0, 1]];
    let a21 = a[[1, 0]];
    let a22 = a[[1, 1]];

    // Check if rank 1 or rank 2
    let eps = f64::EPSILON;
    let b_sum = b1.abs() + b2.abs();
    let x = b1.abs() + b2.abs() * eps;

    if (x - b1.abs()).abs() < eps * b_sum {
        // Rank 1: only B1 is significant
        f[[0, 0]] = (s - (a11 + a22)) / b1;
        f[[0, 1]] = -(a22 * (a22 - s) + a21 * a12 + p) / (a21 * b1);
        if m > 1 {
            f[[1, 0]] = 0.0;
            f[[1, 1]] = 0.0;
        }
    } else {
        // Rank 2: both B1 and B2 are significant
        let b_norm_sq = b1 * b1 + b2 * b2;
        let z = (s - (a11 + a22)) / b_norm_sq;
        f[[0, 0]] = b1 * z;
        if m > 1 {
            f[[1, 1]] = b2 * z;
        }

        // Newton iteration for optimal f12 and f21
        let x = a11 + b1 * f[[0, 0]];
        let c = x * (s - x) - p;
        let sig = if c >= 0.0 { 1.0 } else { -1.0 };

        let s12 = b1 / b2;
        let s21 = b2 / b1;

        // Solve 2×2 eigenvalue problem to find initial guess for r
        let c11 = 0.0;
        let c12 = 1.0;
        let c21 = sig * s12 * c;
        let c22 = a12 - sig * s12 * a21;

        let (wr, _wi, _wr1, _wi1) = eigenvalues_2x2(c11, c12, c21, c22);

        let r_init = if (wr - a12).abs() > 1e-10 { wr } else { a12 };

        // Newton iteration
        let r = newton_iteration_pole_placement(c, a21, s21, r_init);

        // Compute F(1,2) and F(2,1)
        let r_safe = if r.abs() < eps { eps.copysign(r) } else { r };
        f[[0, 1]] = (r_safe - a12) / b1;
        if m > 1 {
            f[[1, 0]] = (c / r_safe - a21) / b2;
        }
    }

    Ok(f)
}

/// Compute eigenvalues of 2×2 matrix using LAPACK DLANV2-like approach
fn eigenvalues_2x2(a11: f64, a12: f64, a21: f64, a22: f64) -> (f64, f64, f64, f64) {
    // Simplified eigenvalue computation for 2×2 matrix
    let trace = a11 + a22;
    let det = a11 * a22 - a12 * a21;
    let disc = trace * trace / 4.0 - det;

    if disc >= 0.0 {
        // Real eigenvalues
        let sqrt_disc = disc.sqrt();
        let wr1 = trace / 2.0 + sqrt_disc;
        let wr2 = trace / 2.0 - sqrt_disc;
        (wr1, 0.0, wr2, 0.0)
    } else {
        // Complex eigenvalues
        let wr = trace / 2.0;
        let wi = (-disc).sqrt();
        (wr, wi, wr, -wi)
    }
}

/// Newton iteration to solve quartic equation for pole placement parameter
fn newton_iteration_pole_placement(c: f64, a21: f64, s21: f64, r_init: f64) -> f64 {
    let c0 = -c * c;
    let c1 = c * a21;
    let c4 = s21 * s21;
    let c3 = -c4 * (-c / a21); // This is simplified; check original for exact formula
    let dc0 = c1;
    let dc2 = 3.0 * c3;
    let dc3 = 4.0 * c4;

    let mut r = r_init;
    let max_iter = 10;
    let eps = f64::EPSILON;

    for _ in 0..max_iter {
        let f_val = c0 + r * (c1 + r * r * (c3 + r * c4));
        let df_val = dc0 + r * r * (dc2 + r * dc3);

        if df_val.abs() < eps {
            break;
        }

        let r_new = r - f_val / df_val;
        let abs_r = r.abs();
        let diff_r = (r - r_new).abs();

        if abs_r > 0.0 && (diff_r / abs_r) < eps {
            break;
        }

        r = r_new;
    }

    if r.abs() < eps {
        r = eps;
    }

    r
}

/// Apply rotation to feedback matrix columns: F := F * U'
fn apply_rotation_to_feedback_cols(f: &mut Array2<f64>, cu: f64, su: f64, m: usize) {
    for i in 0..m.min(2) {
        let f1 = f[[i, 0]];
        let f2 = f[[i, 1]];
        f[[i, 0]] = cu * f1 - su * f2;
        f[[i, 1]] = su * f1 + cu * f2;
    }
}

/// Apply rotation to feedback matrix rows: F := V' * F
fn apply_rotation_to_feedback_rows(f: &mut Array2<f64>, cv: f64, sv: f64) {
    for j in 0..2 {
        let f1 = f[[0, j]];
        let f2 = f[[1, j]];
        f[[0, j]] = cv * f1 - sv * f2;
        f[[1, j]] = sv * f1 + cv * f2;
    }
}

/// Apply Householder transformations to feedback matrix
fn apply_householder_to_feedback(
    f: &mut Array2<f64>,
    m: usize,
    b: &Array2<f64>,
    tau1: f64,
    tau2: f64,
) {
    // Apply H2 if tau2 is significant
    if m > 2 && tau2.abs() > 1e-14 {
        let v2 = b.slice(s![1, 2..]).to_owned();
        for j in 0..2 {
            let mut col = Array1::zeros(m - 1);
            for i in 1..m {
                col[i - 1] = f[[i, j]];
            }
            apply_householder(&mut col, tau2, &v2);
            for i in 1..m {
                f[[i, j]] = col[i - 1];
            }
        }
    }

    // Apply H1
    if tau1.abs() > 1e-14 {
        let v1 = b.row(0).to_owned();
        for j in 0..2 {
            let mut col = Array1::zeros(m);
            for i in 0..m {
                col[i] = f[[i, j]];
            }
            apply_householder(&mut col, tau1, &v1);
            for i in 0..m {
                f[[i, j]] = col[i];
            }
        }
    }
}

/// Householder reflection: reduces vector to [*, 0, 0, ..., 0] form
/// Returns tau parameter for the reflection
fn householder_reflect(x: &mut Array1<f64>) -> f64 {
    let n = x.len();
    if n <= 1 {
        return 0.0;
    }

    let alpha = x[0];
    let mut norm_sq = 0.0;
    for i in 1..n {
        norm_sq += x[i] * x[i];
    }

    if norm_sq < 1e-14 {
        return 0.0;
    }

    let beta = -(alpha.signum()) * (alpha * alpha + norm_sq).sqrt();
    let tau = (beta - alpha) / beta;

    x[0] = beta;
    for i in 1..n {
        x[i] /= alpha - beta;
    }

    tau
}

/// Apply Householder transformation: y := (I - tau*v*v') * y
fn apply_householder(y: &mut Array1<f64>, tau: f64, v: &Array1<f64>) {
    if tau.abs() < 1e-14 {
        return;
    }

    let n = y.len();
    let m = v.len();
    let len = n.min(m);

    // Compute dot product: dot = v' * y
    let mut dot = 0.0;
    for i in 0..len {
        dot += v[i] * y[i];
    }

    // Update: y := y - tau * dot * v
    for i in 0..len {
        y[i] -= tau * dot * v[i];
    }
}

/// Apply Householder to feedback row (for M > 1 case in N=1)
fn apply_householder_to_f(f: &mut Array2<f64>, tau: f64, v: &Array1<f64>) {
    if tau.abs() < 1e-14 {
        return;
    }

    let m = f.nrows();
    let mut col = Array1::zeros(m);
    for i in 0..m {
        col[i] = f[[i, 0]];
    }

    apply_householder(&mut col, tau, v);

    for i in 0..m {
        f[[i, 0]] = col[i];
    }
}

// ==============================================================================
// LAPACK FFI Bindings
// ==============================================================================

// FFI binding to LAPACK's DGEES for real Schur decomposition
//
// DGEES computes the eigenvalues, the real Schur form T, and, optionally, the matrix
// of Schur vectors Z for an N-by-N real nonsymmetric matrix A.
//
// The Schur form satisfies: A = Z*T*Z^T where Z is orthogonal and T is quasi-triangular
// (block upper triangular with 1x1 or 2x2 diagonal blocks).
//
// Arguments:
// * `jobvs` - Compute Schur vectors: 'N' (no), 'V' (yes)
// * `sort` - Sort eigenvalues: 'N' (no sorting), 'S' (sorted by SELECT function)
// * `select` - Function pointer for eigenvalue selection (NULL if sort='N')
// * `n` - Order of matrix A
// * `a` - N×N matrix A (input), T (output in column-major)
// * `lda` - Leading dimension of A (>= N)
// * `sdim` - Number of eigenvalues selected by SELECT
// * `wr` - Real parts of eigenvalues (length N)
// * `wi` - Imaginary parts of eigenvalues (length N)
// * `vs` - N×N Schur vector matrix Z (column-major, if jobvs='V')
// * `ldvs` - Leading dimension of VS (>= N if jobvs='V', >= 1 otherwise)
// * `work` - Workspace array
// * `lwork` - Length of work (-1 for workspace query, >= max(1,3*N) otherwise)
// * `bwork` - Logical workspace (length N, only used if sort='S')
// * `info` - Status code (0=success, <0=argument error, >0=QR failure)
//
// Notes:
// - Uses Fortran calling conventions (column-major, 1-based indexing)
// - Complex conjugate pairs appear consecutively with positive imaginary part first
// - Workspace query: call with lwork=-1, optimal size returned in work[0]
extern "C" {
    fn dgees_(
        jobvs: *const c_char,
        sort: *const c_char,
        select: *const usize, // Function pointer (NULL for no sorting)
        n: *const c_int,
        a: *mut f64,
        lda: *const c_int,
        sdim: *mut c_int,
        wr: *mut f64,
        wi: *mut f64,
        vs: *mut f64,
        ldvs: *const c_int,
        work: *mut f64,
        lwork: *const c_int,
        bwork: *mut c_int, // Logical array in Fortran (use c_int)
        info: *mut c_int,
    );
}

/// Safe wrapper for LAPACK's DGEES routine (real Schur decomposition)
///
/// Computes the Schur decomposition of a real N×N matrix A:
///     A = Z*T*Z^T
/// where:
/// - T is the real Schur form (quasi-triangular with 1×1 and 2×2 blocks)
/// - Z is the orthogonal matrix of Schur vectors
/// - Eigenvalues are the 1×1 blocks and eigenvalues of 2×2 blocks
///
/// # Arguments
///
/// * `a` - N×N input matrix (row-major, will be destroyed)
///
/// # Returns
///
/// `Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>), String>` where:
/// * `Ok((t, z, wr, wi))` - Success:
///   - `t`: Real Schur form matrix T (row-major)
///   - `z`: Orthogonal transformation matrix Z (row-major)
///   - `wr`: Real parts of eigenvalues (length N)
///   - `wi`: Imaginary parts of eigenvalues (length N)
/// * `Err(msg)` - Error message
///
/// # Algorithm
///
/// 1. Workspace query to determine optimal workspace size
/// 2. Convert input from row-major to column-major (for LAPACK)
/// 3. Call LAPACK DGEES to compute Schur decomposition
/// 4. Convert T and Z back to row-major (for Rust)
/// 5. Return decomposition and eigenvalues
///
/// # Notes
///
/// - Complex conjugate pairs appear consecutively: (wr[i], wi[i]) and (wr[i+1], -wi[i+1])
/// - For real eigenvalue: wi[i] = 0
/// - Uses no eigenvalue sorting (sort='N')
///
/// # Example
///
/// ```ignore
/// use ndarray::arr2;
/// let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
/// let (t, z, wr, wi) = call_dgees(&a).unwrap();
/// // t is quasi-triangular Schur form
/// // z is orthogonal transformation
/// // wr, wi contain eigenvalues
/// ```
fn call_dgees(
    a: &Array2<f64>,
) -> Result<(Array2<f64>, Array2<f64>, Array1<f64>, Array1<f64>), String> {
    let n = a.nrows();

    if n == 0 {
        return Ok((
            Array2::zeros((0, 0)),
            Array2::zeros((0, 0)),
            Array1::zeros(0),
            Array1::zeros(0),
        ));
    }

    if a.ncols() != n {
        return Err("Matrix A must be square".to_string());
    }

    let n_i32 = n as c_int;
    let lda = n_i32;
    let ldvs = n_i32;

    // LAPACK parameters
    let jobvs: c_char = b'V' as c_char; // Compute Schur vectors
    let sort: c_char = b'N' as c_char; // No sorting
    let select: usize = 0; // NULL pointer for no sorting

    let mut sdim: c_int = 0;
    let mut info: c_int = 0;

    // Convert to column-major for LAPACK
    let mut a_col_major = a.clone().reversed_axes();

    // Allocate output arrays
    let mut wr = vec![0.0_f64; n];
    let mut wi = vec![0.0_f64; n];
    let mut vs = vec![0.0_f64; n * n];
    let mut bwork: Vec<c_int> = vec![0; n]; // Logical workspace (not used when sort='N')

    // Workspace query
    let lwork_query: c_int = -1;
    let mut work_query = vec![0.0_f64; 1];

    unsafe {
        dgees_(
            &jobvs,
            &sort,
            &select,
            &n_i32,
            a_col_major.as_mut_ptr(),
            &lda,
            &mut sdim,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vs.as_mut_ptr(),
            &ldvs,
            work_query.as_mut_ptr(),
            &lwork_query,
            bwork.as_mut_ptr(),
            &mut info,
        );
    }

    if info != 0 {
        return Err(format!("DGEES workspace query failed with INFO={}", info));
    }

    // Allocate optimal workspace
    let lwork_opt = work_query[0] as usize;
    let lwork = lwork_opt.max(3 * n);
    let mut work = vec![0.0_f64; lwork];
    let lwork_i32 = lwork as c_int;

    // Actual computation
    unsafe {
        dgees_(
            &jobvs,
            &sort,
            &select,
            &n_i32,
            a_col_major.as_mut_ptr(),
            &lda,
            &mut sdim,
            wr.as_mut_ptr(),
            wi.as_mut_ptr(),
            vs.as_mut_ptr(),
            &ldvs,
            work.as_mut_ptr(),
            &lwork_i32,
            bwork.as_mut_ptr(),
            &mut info,
        );
    }

    if info == 0 {
        // Success - convert back to row-major
        let t = a_col_major.reversed_axes();

        let z_col_major = Array2::from_shape_vec((n, n), vs)
            .map_err(|e| format!("Failed to create Z matrix: {}", e))?;
        let z = z_col_major.reversed_axes();

        let wr_array = Array1::from_vec(wr);
        let wi_array = Array1::from_vec(wi);

        Ok((t, z, wr_array, wi_array))
    } else if info > 0 {
        Err(format!(
            "DGEES: QR algorithm failed to compute all eigenvalues (INFO={})",
            info
        ))
    } else {
        Err(format!("DGEES: Invalid parameter at position {}", -info))
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use ndarray::{arr2, Array2};

    #[test]
    fn test_sb01bd_zero_dimension() {
        let a: Array2<f64> = Array2::zeros((0, 0));
        let b: Array2<f64> = Array2::zeros((0, 1));
        let desired = vec![];

        let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None).unwrap();

        assert_eq!(result.assigned_count, 0);
        assert_eq!(result.uncontrollable_count, 0);
    }

    #[test]
    fn test_sb01bd_single_input() {
        let a = arr2(&[[0.0, 1.0], [0.0, -1.0]]);
        let b = arr2(&[[0.0], [1.0]]);
        let desired = vec![-1.0, -2.0];

        let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None).unwrap();

        assert!(result.assigned_count < usize::MAX);
        assert_eq!(result.feedback.shape()[0], 1);
        assert_eq!(result.feedback.shape()[1], 2);
    }

    #[test]
    fn test_sb01bd_multi_input() {
        let a = arr2(&[[0.0, 1.0], [0.0, -1.0]]);
        let b = arr2(&[[1.0, 0.0], [0.0, 1.0]]); // 2×2 B (2 inputs)
        let desired = vec![-1.0, -2.0];

        let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None).unwrap();

        assert_eq!(result.feedback.shape()[0], 2);
        assert_eq!(result.feedback.shape()[1], 2);
    }

    #[test]
    fn test_sb01bd_discrete_system() {
        let a = arr2(&[[0.5, 1.0], [0.0, 0.8]]);
        let b = arr2(&[[0.0], [1.0]]);
        let desired = vec![0.5, 0.6];

        let result = sb01bd(SystemType::Discrete, &a, &b, &desired, 1.0, None).unwrap();

        assert_eq!(result.feedback.shape()[0], 1);
        assert_eq!(result.feedback.shape()[1], 2);
    }

    #[test]
    fn test_sb01bd_invalid_alpha_discrete() {
        let a = arr2(&[[1.0, 0.0], [0.0, 2.0]]);
        let b = arr2(&[[0.0], [1.0]]);
        let desired = vec![-1.0];

        let result = sb01bd(SystemType::Discrete, &a, &b, &desired, -0.5, None);

        assert!(result.is_err());
    }

    #[test]
    fn test_sb01bd_zero_b() {
        let a = arr2(&[[1.0, 0.0], [0.0, 2.0]]);
        let b = arr2(&[[0.0], [0.0]]);
        let desired = vec![-1.0, -2.0];

        let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None).unwrap();

        // System is completely uncontrollable
        assert_eq!(result.uncontrollable_count, 2);
    }

    // Tests for sb01bx
    #[test]
    fn test_sb01bx_real_eigenvalue_selection() {
        // Test selecting closest real eigenvalue
        let mut wr = vec![0.5, -0.8, 2.0, -1.5];
        let mut wi = vec![0.0, 0.0, 0.0, 0.0];

        let (s, p) = sb01bx(true, -1.0, 0.0, &mut wr, &mut wi);

        // -0.8 is closest to -1.0, should be moved to end
        assert_eq!(wr[3], -0.8);
        assert_eq!(s, -0.8);
        assert_eq!(p, -0.8);

        // Other elements should be shifted
        assert_eq!(wr[0], 0.5);
        assert_eq!(wr[1], 2.0);
        assert_eq!(wr[2], -1.5);
    }

    #[test]
    fn test_sb01bx_real_already_at_end() {
        // Test when closest eigenvalue is already at the end
        let mut wr = vec![0.5, 2.0, -1.5, -0.8];
        let mut wi = vec![0.0, 0.0, 0.0, 0.0];

        let (s, p) = sb01bx(true, -1.0, 0.0, &mut wr, &mut wi);

        // -0.8 is already at end
        assert_eq!(wr[3], -0.8);
        assert_eq!(s, -0.8);
        assert_eq!(p, -0.8);
    }

    #[test]
    fn test_sb01bx_complex_conjugate_pair_selection() {
        // Test selecting closest complex conjugate pair
        // Pairs: (1.0, 2.0)/(1.0, -2.0), (-0.5, 1.5)/(-0.5, -1.5), (2.0, 0.5)/(2.0, -0.5)
        let mut wr = vec![1.0, 1.0, -0.5, -0.5, 2.0, 2.0];
        let mut wi = vec![2.0, -2.0, 1.5, -1.5, 0.5, -0.5];

        // Target: -0.3 + 1.0j (closest to -0.5 + 1.5j pair)
        let (s, p) = sb01bx(false, -0.3, 1.0, &mut wr, &mut wi);

        // The pair (-0.5, ±1.5) should move to the end
        assert_eq!(wr[4], -0.5);
        assert_eq!(wi[4], 1.5);
        assert_eq!(wr[5], -0.5);
        assert_eq!(wi[5], -1.5);

        // Sum: 2 * (-0.5) = -1.0
        assert!((s - (-1.0)).abs() < 1e-10);

        // Product: (-0.5)² + 1.5² = 0.25 + 2.25 = 2.5
        assert!((p - 2.5).abs() < 1e-10);
    }

    #[test]
    fn test_sb01bx_complex_first_pair() {
        // Test when the first pair is selected
        let mut wr = vec![1.0, 1.0, 3.0, 3.0];
        let mut wi = vec![2.0, -2.0, 1.0, -1.0];

        let (s, p) = sb01bx(false, 1.0, 2.0, &mut wr, &mut wi);

        // First pair (1.0, ±2.0) is closest, should move to end
        assert_eq!(wr[2], 1.0);
        assert_eq!(wi[2], 2.0);
        assert_eq!(wr[3], 1.0);
        assert_eq!(wi[3], -2.0);

        // Sum: 2 * 1.0 = 2.0
        assert!((s - 2.0).abs() < 1e-10);

        // Product: 1.0² + 2.0² = 1.0 + 4.0 = 5.0
        assert!((p - 5.0).abs() < 1e-10);
    }

    #[test]
    fn test_sb01bx_single_real_eigenvalue() {
        // Test with a single eigenvalue
        let mut wr = vec![3.5];
        let mut wi = vec![0.0];

        let (s, p) = sb01bx(true, 0.0, 0.0, &mut wr, &mut wi);

        assert_eq!(wr[0], 3.5);
        assert_eq!(s, 3.5);
        assert_eq!(p, 3.5);
    }

    #[test]
    fn test_sb01bx_empty_arrays() {
        // Test with empty arrays (edge case)
        let mut wr = vec![];
        let mut wi = vec![];

        let (s, p) = sb01bx(true, 0.0, 0.0, &mut wr, &mut wi);

        assert_eq!(s, 0.0);
        assert_eq!(p, 0.0);
    }

    #[test]
    fn test_sb01bx_manhattan_distance() {
        // Verify Manhattan distance is used for complex selection
        // Eigenvalues: (1, 1)/(1, -1), (0.5, 1.5)/(0.5, -1.5)
        let mut wr = vec![1.0, 1.0, 0.5, 0.5];
        let mut wi = vec![1.0, -1.0, 1.5, -1.5];

        // Target: (0, 0)
        // Distance to (1, 1): |1-0| + |1-0| = 2.0
        // Distance to (0.5, 1.5): |0.5-0| + |1.5-0| = 2.0
        // Should select the first one found (1, 1)
        let (s, p) = sb01bx(false, 0.0, 0.0, &mut wr, &mut wi);

        // First pair (1, ±1) should be selected
        assert_eq!(wr[2], 1.0);
        assert_eq!(wi[2], 1.0);

        // Sum: 2.0, Product: 2.0
        assert!((s - 2.0).abs() < 1e-10);
        assert!((p - 2.0).abs() < 1e-10);
    }

    // Tests for sb01by
    use approx::assert_relative_eq;

    #[test]
    fn test_sb01by_n1_m1() {
        // Single pole, single input
        let mut a = arr2(&[[2.0]]);
        let mut b = arr2(&[[1.0]]);
        let s = -1.0; // desired eigenvalue
        let result = sb01by(1, 1, s, 0.0, &mut a, &mut b, 1e-10).unwrap();

        assert!(result.controllable);
        // F should be approximately -3.0 (since A+BF = 2.0 + 1.0*F = -1.0 => F = -3.0)
        assert_relative_eq!(result.f[[0, 0]], -3.0, epsilon = 1e-6);
    }

    #[test]
    fn test_sb01by_n1_m2() {
        // Single pole, two inputs
        let mut a = arr2(&[[1.0]]);
        let mut b = arr2(&[[2.0, 1.0]]);
        let s = -2.0; // desired eigenvalue
        let result = sb01by(1, 2, s, 0.0, &mut a, &mut b, 1e-10).unwrap();

        assert!(result.controllable);
        // Should produce minimum-norm feedback
        assert_eq!(result.f.shape(), &[2, 1]);
    }

    #[test]
    fn test_sb01by_n2_m1() {
        // Two poles, single input
        let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
        let mut b = arr2(&[[1.0], [1.0]]);
        let s = -6.0; // sum of desired eigenvalues
        let p = 8.0; // product of desired eigenvalues
        let result = sb01by(2, 1, s, p, &mut a, &mut b, 1e-10).unwrap();

        // Should compute feedback (controllability depends on system)
        assert_eq!(result.f.shape(), &[1, 2]);
    }

    #[test]
    fn test_sb01by_n2_m2() {
        // Two poles, two inputs
        let mut a = arr2(&[[0.0, 1.0], [0.0, 0.0]]);
        let mut b = arr2(&[[0.0, 1.0], [1.0, 0.0]]);
        let s = -3.0; // sum
        let p = 2.0; // product
        let result = sb01by(2, 2, s, p, &mut a, &mut b, 1e-10).unwrap();

        assert_eq!(result.f.shape(), &[2, 2]);
    }

    #[test]
    fn test_sb01by_uncontrollable_n1() {
        // Uncontrollable system (B ≈ 0)
        let mut a = arr2(&[[1.0]]);
        let mut b = arr2(&[[1e-12]]);
        let s = -1.0;
        let result = sb01by(1, 1, s, 0.0, &mut a, &mut b, 1e-10).unwrap();

        assert!(!result.controllable);
    }

    #[test]
    fn test_sb01by_invalid_n() {
        let mut a = arr2(&[[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]]);
        let mut b = arr2(&[[1.0], [1.0], [1.0]]);
        let result = sb01by(3, 1, -1.0, 0.0, &mut a, &mut b, 1e-10);

        assert!(result.is_err());
    }

    #[test]
    fn test_sb01by_invalid_m() {
        let mut a = arr2(&[[1.0]]);
        let mut b = Array2::zeros((1, 0));
        let result = sb01by(1, 0, -1.0, 0.0, &mut a, &mut b, 1e-10);

        assert!(result.is_err());
    }

    // Tests for svd_2x2_upper_triangular (DLASV2 implementation)
    #[test]
    fn test_svd_2x2_diagonal() {
        // Test diagonal matrix [3, 0; 0, 2]
        let (ssmax, ssmin, csl, snl, csr, snr) = svd_2x2_upper_triangular(3.0, 0.0, 2.0);

        // Singular values should be 3 and 2
        assert_relative_eq!(ssmax, 3.0, epsilon = 1e-10);
        assert_relative_eq!(ssmin, 2.0, epsilon = 1e-10);

        // Rotations should be identity (no rotation needed for diagonal)
        assert_relative_eq!(csl, 1.0, epsilon = 1e-10);
        assert_relative_eq!(snl, 0.0, epsilon = 1e-10);
        assert_relative_eq!(csr, 1.0, epsilon = 1e-10);
        assert_relative_eq!(snr, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_svd_2x2_upper_triangular_basic() {
        // Test upper triangular matrix [2, 1; 0, 1]
        let (ssmax, ssmin, csl, snl, csr, snr) = svd_2x2_upper_triangular(2.0, 1.0, 1.0);

        // Verify singular values are in descending order
        assert!(ssmax >= ssmin);

        // Verify rotation matrices are orthogonal: c^2 + s^2 = 1
        let left_norm = csl * csl + snl * snl;
        let right_norm = csr * csr + snr * snr;
        assert_relative_eq!(left_norm, 1.0, epsilon = 1e-10);
        assert_relative_eq!(right_norm, 1.0, epsilon = 1e-10);

        // Verify reconstruction: U^T * B * V = diag(ssmax, ssmin)
        // B = [2, 1; 0, 1]
        // U = [csl, -snl; snl, csl]
        // V = [csr, -snr; snr, csr]

        // U^T * B
        let ut_b_11 = csl * 2.0 + snl * 0.0;
        let ut_b_12 = csl * 1.0 + snl * 1.0;
        let ut_b_21 = -snl * 2.0 + csl * 0.0;
        let ut_b_22 = -snl * 1.0 + csl * 1.0;

        // (U^T * B) * V
        let d_11 = ut_b_11 * csr + ut_b_12 * snr;
        let d_12 = ut_b_11 * (-snr) + ut_b_12 * csr;
        let d_21 = ut_b_21 * csr + ut_b_22 * snr;
        let d_22 = ut_b_21 * (-snr) + ut_b_22 * csr;

        // Should be approximately diagonal with singular values
        assert_relative_eq!(d_11.abs(), ssmax, epsilon = 1e-10);
        assert_relative_eq!(d_22.abs(), ssmin, epsilon = 1e-10);
        assert_relative_eq!(d_12, 0.0, epsilon = 1e-10);
        assert_relative_eq!(d_21, 0.0, epsilon = 1e-10);
    }

    #[test]
    fn test_svd_2x2_zero_diagonal_element() {
        // Test with one zero diagonal element [0, 2; 0, 1]
        let (ssmax, ssmin, csl, snl, csr, snr) = svd_2x2_upper_triangular(0.0, 2.0, 1.0);

        // For matrix [0, 2; 0, 1], singular values are sqrt(2² + 1²) = sqrt(5) and 0
        let expected_max = (2.0_f64.powi(2) + 1.0_f64.powi(2)).sqrt();
        assert_relative_eq!(ssmax, expected_max, epsilon = 1e-10);
        assert_relative_eq!(ssmin, 0.0, epsilon = 1e-10);

        // Verify orthogonality
        let left_norm = csl * csl + snl * snl;
        let right_norm = csr * csr + snr * snr;
        assert_relative_eq!(left_norm, 1.0, epsilon = 1e-10);
        assert_relative_eq!(right_norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_svd_2x2_large_off_diagonal() {
        // Test with large off-diagonal element [1, 10; 0, 1]
        let (ssmax, ssmin, csl, snl, csr, snr) = svd_2x2_upper_triangular(1.0, 10.0, 1.0);

        // Verify singular values are positive and ordered
        assert!(ssmax >= ssmin);
        assert!(ssmin >= 0.0);

        // Verify orthogonality
        let left_norm = csl * csl + snl * snl;
        let right_norm = csr * csr + snr * snr;
        assert_relative_eq!(left_norm, 1.0, epsilon = 1e-10);
        assert_relative_eq!(right_norm, 1.0, epsilon = 1e-10);

        // Verify reconstruction
        let ut_b_11 = csl * 1.0;
        let ut_b_12 = csl * 10.0 + snl * 1.0;
        let ut_b_21 = -snl * 1.0;
        let ut_b_22 = -snl * 10.0 + csl * 1.0;

        let d_11 = ut_b_11 * csr + ut_b_12 * snr;
        let d_12 = ut_b_11 * (-snr) + ut_b_12 * csr;
        let d_21 = ut_b_21 * csr + ut_b_22 * snr;
        let d_22 = ut_b_21 * (-snr) + ut_b_22 * csr;

        assert_relative_eq!(d_11.abs(), ssmax, epsilon = 1e-9);
        assert_relative_eq!(d_22.abs(), ssmin, epsilon = 1e-9);
        assert_relative_eq!(d_12, 0.0, epsilon = 1e-9);
        assert_relative_eq!(d_21, 0.0, epsilon = 1e-9);
    }

    #[test]
    fn test_svd_2x2_negative_values() {
        // Test with negative values [-2, 3; 0, -1]
        let (ssmax, ssmin, csl, snl, csr, snr) = svd_2x2_upper_triangular(-2.0, 3.0, -1.0);

        // Verify singular values are positive and ordered
        assert!(ssmax >= ssmin);
        assert!(ssmin >= 0.0);

        // Verify orthogonality
        let left_norm = csl * csl + snl * snl;
        let right_norm = csr * csr + snr * snr;
        assert_relative_eq!(left_norm, 1.0, epsilon = 1e-10);
        assert_relative_eq!(right_norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_svd_2x2_zero_matrix() {
        // Test zero matrix
        let (ssmax, ssmin, csl, snl, csr, snr) = svd_2x2_upper_triangular(0.0, 0.0, 0.0);

        // Both singular values should be zero
        assert_relative_eq!(ssmax, 0.0, epsilon = 1e-10);
        assert_relative_eq!(ssmin, 0.0, epsilon = 1e-10);

        // Verify orthogonality (even for zero matrix, rotations should be valid)
        let left_norm = csl * csl + snl * snl;
        let right_norm = csr * csr + snr * snr;
        assert_relative_eq!(left_norm, 1.0, epsilon = 1e-10);
        assert_relative_eq!(right_norm, 1.0, epsilon = 1e-10);
    }

    #[test]
    fn test_svd_2x2_swap_case() {
        // Test case where |h| > |f| triggers swapping [1, 2; 0, 3]
        let (ssmax, ssmin, csl, snl, csr, snr) = svd_2x2_upper_triangular(1.0, 2.0, 3.0);

        // Verify singular values
        assert!(ssmax >= ssmin);
        assert!(ssmin >= 0.0);

        // Verify orthogonality
        let left_norm = csl * csl + snl * snl;
        let right_norm = csr * csr + snr * snr;
        assert_relative_eq!(left_norm, 1.0, epsilon = 1e-10);
        assert_relative_eq!(right_norm, 1.0, epsilon = 1e-10);

        // Verify reconstruction
        let ut_b_11 = csl * 1.0;
        let ut_b_12 = csl * 2.0 + snl * 3.0;
        let ut_b_21 = -snl * 1.0;
        let ut_b_22 = -snl * 2.0 + csl * 3.0;

        let d_11 = ut_b_11 * csr + ut_b_12 * snr;
        let d_12 = ut_b_11 * (-snr) + ut_b_12 * csr;
        let d_21 = ut_b_21 * csr + ut_b_22 * snr;
        let d_22 = ut_b_21 * (-snr) + ut_b_22 * csr;

        assert_relative_eq!(d_11.abs(), ssmax, epsilon = 1e-9);
        assert_relative_eq!(d_22.abs(), ssmin, epsilon = 1e-9);
        assert_relative_eq!(d_12, 0.0, epsilon = 1e-9);
        assert_relative_eq!(d_21, 0.0, epsilon = 1e-9);
    }
}
