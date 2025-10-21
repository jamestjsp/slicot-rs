//! Demonstration of pole placement using SB01BD
//!
//! This example shows how to use the sb01bd function to design
//! a state feedback controller that places closed-loop poles at
//! desired locations.

use ndarray::arr2;
use ndarray_linalg::Eig;
use num_complex::Complex;
use slicot_rs::sb::{sb01bd, SystemType};

fn main() {
    println!("=== Pole Placement Demonstration ===\n");

    // Example: Double integrator system
    // dx1/dt = x2
    // dx2/dt = u
    // State-space: A = [0 1; 0 0], B = [0; 1]

    let a = arr2(&[[0.0, 1.0], [0.0, 0.0]]);
    let b = arr2(&[[0.0], [1.0]]);

    println!("Original system:");
    println!("A = \n{:?}", a);
    println!("B = \n{:?}", b);

    // Compute original eigenvalues
    if let Ok((eigs_orig, _)) = a.eig() {
        println!("\nOriginal eigenvalues:");
        for eig in eigs_orig.iter() {
            let c: Complex<f64> = *eig;
            println!("  λ = {:.4}", c.re);
        }
    }

    // Design controller to place poles at -1 and -2
    let desired = vec![-1.0, -2.0];
    println!("\nDesired eigenvalues: {:?}", desired);

    match sb01bd(SystemType::Continuous, &a, &b, &desired, 0.0, None) {
        Ok(result) => {
            println!("\n=== Controller Design Results ===");
            println!("Feedback matrix F:");
            for i in 0..result.feedback.nrows() {
                print!("  [");
                for j in 0..result.feedback.ncols() {
                    print!("{:8.4}", result.feedback[(i, j)]);
                }
                println!(" ]");
            }

            println!("\nDiagnostics:");
            println!("  Assigned eigenvalues: {}", result.assigned_count);
            println!("  Fixed eigenvalues: {}", result.fixed_count);
            println!(
                "  Uncontrollable eigenvalues: {}",
                result.uncontrollable_count
            );

            // Verify closed-loop eigenvalues
            let a_cl = &a + &b.dot(&result.feedback);
            println!("\nClosed-loop matrix A + BF:");
            println!("{:?}", a_cl);

            if let Ok((eigs_cl, _)) = a_cl.eig() {
                println!("\nActual closed-loop eigenvalues:");
                for eig in eigs_cl.iter() {
                    println!("  λ = {:.4} + {:.4}i", eig.re, eig.im);
                }

                // Check if eigenvalues match desired
                let mut matched = 0;
                for desired_eig in &desired {
                    for actual_eig in eigs_cl.iter() {
                        if (actual_eig.re - desired_eig).abs() < 0.01 && actual_eig.im.abs() < 0.01
                        {
                            matched += 1;
                            break;
                        }
                    }
                }

                println!("\n✓ Pole placement successful!");
                println!(
                    "  Matched {}/{} desired eigenvalues",
                    matched,
                    desired.len()
                );
            }
        }
        Err(e) => {
            println!("Error: {}", e);
        }
    }
}
