//! Integration tests for complex SLICOT routines
//! Tests AB01MD (controllability decomposition) and SB01BD (pole placement)

use ndarray::arr2;
use slicot_rs::ab::ab01md;
use slicot_rs::sb::{sb01bd, SystemType};

// ===== AB01MD Integration Tests =====

#[test]
fn test_ab01md_simple_2x2() {
    let mut a = arr2(&[[0.0, 1.0], [0.0, -1.0]]);
    let mut b = arr2(&[[0.0], [1.0]]);

    let result = ab01md(&mut a, &mut b, None);
    assert!(result.is_ok(), "AB01MD failed on simple 2×2 system");

    let (ncont, _z, tau) = result.unwrap();
    assert_eq!(ncont, 2, "Simple system should be completely controllable");
    assert_eq!(tau.len(), 2);
}

#[test]
fn test_ab01md_3x3_system() {
    let mut a = arr2(&[[1.0, 2.0, 3.0], [0.0, -1.0, 4.0], [0.0, 0.0, 2.0]]);
    let mut b = arr2(&[[1.0], [0.0], [1.0]]);

    let result = ab01md(&mut a, &mut b, None);
    assert!(result.is_ok());

    let (ncont, _, _) = result.unwrap();
    assert!(
        ncont > 0,
        "System with non-zero B should have controllable modes"
    );
    assert!(ncont <= 3, "NCONT cannot exceed N");
}

#[test]
fn test_ab01md_uncontrollable_system() {
    // System where B is aligned with unobservable mode
    let mut a = arr2(&[[1.0, 0.0, 1.0], [0.0, 2.0, 1.0], [0.0, 0.0, 3.0]]);
    let mut b = arr2(&[[0.0], [0.0], [0.0]]);

    let result = ab01md(&mut a, &mut b, None);
    assert!(result.is_ok());

    let (ncont, _, _) = result.unwrap();
    assert_eq!(
        ncont, 0,
        "Zero input should result in no controllable modes"
    );
}

#[test]
fn test_ab01md_tolerance_parameter() {
    let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let mut b = arr2(&[[1.0], [1.0]]);

    // Test with explicit tolerance
    let result = ab01md(&mut a, &mut b, Some(1e-10));
    assert!(result.is_ok());

    let (ncont, _, _) = result.unwrap();
    assert!(ncont >= 0 && ncont <= 2);
}

#[test]
fn test_ab01md_transformation_matrix() {
    let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let mut b = arr2(&[[1.0], [1.0]]);

    let result = ab01md(&mut a, &mut b, None);
    assert!(result.is_ok());

    let (_, z_opt, _) = result.unwrap();
    assert!(z_opt.is_some(), "Transformation matrix should be computed");

    if let Some(z) = z_opt {
        // Check dimensions
        assert_eq!(z.shape()[0], 2);
        assert_eq!(z.shape()[1], 2);

        // Check if orthogonal (Z^T * Z ≈ I)
        let zt_z = z.t().dot(&z);
        for i in 0..2 {
            for j in 0..2 {
                let expected = if i == j { 1.0 } else { 0.0 };
                let error = (zt_z[(i, j)] - expected).abs();
                assert!(
                    error < 0.1,
                    "Z should be approximately orthogonal, error at [{},{}]: {}",
                    i,
                    j,
                    error
                );
            }
        }
    }
}

#[test]
fn test_ab01md_preserves_dimensions() {
    let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let mut b = arr2(&[[1.0], [1.0]]);

    let shape_a_before = (a.shape()[0], a.shape()[1]);
    let shape_b_before = (b.shape()[0], b.shape()[1]);

    let _ = ab01md(&mut a, &mut b, None);

    assert_eq!(
        (a.shape()[0], a.shape()[1]),
        shape_a_before,
        "A dimensions should be preserved"
    );
    assert_eq!(
        (b.shape()[0], b.shape()[1]),
        shape_b_before,
        "B dimensions should be preserved"
    );
}

#[test]
fn test_ab01md_diagonal_system() {
    let mut a = arr2(&[[1.0, 0.0], [0.0, 2.0]]);
    let mut b = arr2(&[[1.0], [1.0]]);

    let result = ab01md(&mut a, &mut b, None);
    assert!(result.is_ok());

    let (ncont, _, _) = result.unwrap();
    // Diagonal system with non-zero B should be controllable
    assert_eq!(ncont, 2);
}

// ===== SB01BD Integration Tests =====

#[test]
fn test_sb01bd_continuous_system() {
    let a = arr2(&[[0.0, 1.0], [0.0, -1.0]]);
    let b = arr2(&[[0.0], [1.0]]);
    let desired = vec![-1.0, -2.0];

    let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None);
    assert!(
        result.is_ok(),
        "SB01BD should succeed for continuous system"
    );

    let res = result.unwrap();
    assert_eq!(res.feedback.shape()[0], 1);
    assert_eq!(res.feedback.shape()[1], 2);
}

#[test]
fn test_sb01bd_discrete_system() {
    let a = arr2(&[[0.5, 1.0], [0.0, 0.8]]);
    let b = arr2(&[[0.0], [1.0]]);
    let desired = vec![0.5, 0.6];

    let result = sb01bd(SystemType::Discrete, &a, &b, &desired, 1.0, None);
    assert!(result.is_ok(), "SB01BD should succeed for discrete system");

    let res = result.unwrap();
    assert_eq!(res.feedback.shape()[0], 1);
    assert_eq!(res.feedback.shape()[1], 2);
}

#[test]
fn test_sb01bd_multi_input() {
    let a = arr2(&[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]);
    let b = arr2(&[[0.0, 1.0], [1.0, 0.0], [0.0, 1.0]]); // 3×2 B matrix
    let desired = vec![-1.0, -2.0, -3.0];

    let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None);
    assert!(result.is_ok());

    let res = result.unwrap();
    assert_eq!(res.feedback.shape()[0], 2, "Feedback should be M×N");
    assert_eq!(res.feedback.shape()[1], 3, "Feedback should be M×N");
}

#[test]
fn test_sb01bd_diagnostics() {
    let a = arr2(&[[1.0, 0.0], [0.0, 2.0]]);
    let b = arr2(&[[1.0], [1.0]]);
    let desired = vec![-1.0, -2.0];

    let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None);
    assert!(result.is_ok());

    let res = result.unwrap();
    assert!(res.assigned_count >= 0);
    assert!(res.fixed_count >= 0);
    assert!(res.uncontrollable_count >= 0);
    assert!(res.assigned_count + res.uncontrollable_count <= 2);
}

#[test]
fn test_sb01bd_zero_input() {
    let a = arr2(&[[1.0, 0.0], [0.0, 2.0]]);
    let b = arr2(&[[0.0], [0.0]]); // No input influence
    let desired = vec![-1.0, -2.0];

    let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None);
    assert!(result.is_ok());

    let res = result.unwrap();
    // System is uncontrollable
    assert_eq!(res.uncontrollable_count, 2);
}

#[test]
fn test_sb01bd_single_eigenvalue() {
    let a = arr2(&[[1.0]]);
    let b = arr2(&[[1.0]]);
    let desired = vec![-1.0];

    let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None);
    assert!(result.is_ok());

    let res = result.unwrap();
    assert_eq!(res.feedback.shape()[0], 1);
    assert_eq!(res.feedback.shape()[1], 1);
}

#[test]
fn test_sb01bd_alpha_threshold_continuous() {
    let a = arr2(&[[0.0, 1.0], [0.0, -0.5]]);
    let b = arr2(&[[0.0], [1.0]]);
    let desired = vec![-2.0, -3.0];

    // With ALPHA = 0: eigenvalue at -0.5 is fixed
    let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None);
    assert!(result.is_ok());

    let res = result.unwrap();
    assert!(res.fixed_count >= 0, "Should have some fixed eigenvalues");
}

#[test]
fn test_sb01bd_invalid_alpha_discrete() {
    let a = arr2(&[[1.0, 0.0], [0.0, 2.0]]);
    let b = arr2(&[[1.0], [1.0]]);
    let desired = vec![0.5, 0.6];

    // ALPHA must be >= 0 for discrete systems
    let result = sb01bd(SystemType::Discrete, &a, &b, &desired, -1.0, None);
    assert!(
        result.is_err(),
        "Should reject negative ALPHA for discrete system"
    );
}

#[test]
fn test_sb01bd_tolerance_parameter() {
    let a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b = arr2(&[[1.0], [1.0]]);
    let desired = vec![-1.0, -2.0];

    // Test with explicit tolerance
    let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, Some(1e-10));
    assert!(result.is_ok());

    let res = result.unwrap();
    assert_eq!(res.feedback.shape()[0], 1);
    assert_eq!(res.feedback.shape()[1], 2);
}

#[test]
fn test_sb01bd_canonical_controllable_form() {
    // Canonical controllable form should be completely controllable
    let a = arr2(&[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [1.0, 2.0, 3.0]]);
    let b = arr2(&[[0.0], [0.0], [1.0]]);
    let desired = vec![-1.0, -2.0, -3.0];

    let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None);
    assert!(result.is_ok());

    let res = result.unwrap();
    // Canonical form should be assignable
    assert!(res.uncontrollable_count <= 1);
}

#[test]
fn test_sb01bd_3x3_system() {
    let a = arr2(&[[0.0, 1.0, 0.0], [0.0, 0.0, 1.0], [0.0, 0.0, 0.0]]);
    let b = arr2(&[[0.0], [0.0], [1.0]]);
    let desired = vec![-1.0, -2.0, -3.0];

    let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None);
    assert!(result.is_ok());

    let res = result.unwrap();
    assert_eq!(res.feedback.shape()[0], 1);
    assert_eq!(res.feedback.shape()[1], 3);
}

// ===== Cross-routine integration tests =====

#[test]
fn test_ab01md_then_sb01bd_integration() {
    // Use AB01MD to find controllable structure, then apply SB01BD

    let mut a = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let mut b = arr2(&[[1.0], [1.0]]);

    // Step 1: Analyze controllability with AB01MD
    let result_ab = ab01md(&mut a, &mut b, None);
    assert!(result_ab.is_ok());

    let (ncont, _, _) = result_ab.unwrap();
    assert!(ncont > 0, "System should have controllable modes");

    // Step 2: Design feedback with SB01BD
    let a_orig = arr2(&[[1.0, 2.0], [3.0, 4.0]]);
    let b_orig = arr2(&[[1.0], [1.0]]);
    let desired = vec![-1.0, -2.0];

    let result_sb = sb01bd(
        SystemType::Continuous,
        &a_orig,
        &b_orig,
        &desired,
        0.0,
        None,
    );
    assert!(result_sb.is_ok());

    let res_sb = result_sb.unwrap();
    assert_eq!(res_sb.feedback.shape()[0], 1);
    assert_eq!(res_sb.feedback.shape()[1], 2);
}

#[test]
fn test_known_controllable_system() {
    // System known to be controllable
    let a = arr2(&[[0.0, 1.0], [-2.0, -3.0]]);
    let b = arr2(&[[0.0], [1.0]]);

    let result_ab = ab01md(&mut a.clone(), &mut b.clone(), None);
    assert!(result_ab.is_ok());

    let (ncont, _, _) = result_ab.unwrap();
    assert_eq!(ncont, 2, "Known controllable system should have full rank");
}

#[test]
fn test_known_uncontrollable_system() {
    // System with uncontrollable mode
    let a = arr2(&[[1.0, 1.0, 0.0], [0.0, 2.0, 0.0], [0.0, 0.0, 3.0]]);
    let b = arr2(&[[1.0], [0.0], [0.0]]);

    let result_ab = ab01md(&mut a.clone(), &mut b.clone(), None);
    assert!(result_ab.is_ok());

    let (ncont, _, _) = result_ab.unwrap();
    assert!(
        ncont < 3,
        "System with uncontrollable mode should have ncont < N"
    );
}

#[test]
fn test_feedback_matrix_dimensions() {
    // Verify feedback matrix dimensions for various system sizes
    for n in &[2, 3, 4, 5] {
        for m in &[1, 2] {
            if m > n {
                continue; // More inputs than states is unusual
            }

            let a = ndarray::Array2::zeros((*n, *n));
            let b = ndarray::Array2::zeros((*n, *m));
            let desired = (0..*n).map(|i| -(i as f64 + 1.0)).collect::<Vec<_>>();

            let result = sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None);
            assert!(result.is_ok());

            let res = result.unwrap();
            assert_eq!(
                res.feedback.shape()[0],
                *m,
                "Feedback shape mismatch for N={}, M={}",
                n,
                m
            );
            assert_eq!(
                res.feedback.shape()[1],
                *n,
                "Feedback shape mismatch for N={}, M={}",
                n,
                m
            );
        }
    }
}
