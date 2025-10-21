//! Demonstration of SB01BX - eigenvalue selection for pole placement
//!
//! This example shows how to use sb01bx to select the closest eigenvalue
//! or eigenvalue pair from a set, which is a common operation in pole
//! placement algorithms.

use slicot_rs::sb::sb01bx;

fn main() {
    println!("SB01BX Demo - Eigenvalue Selection for Pole Placement\n");

    // Example 1: Select closest real eigenvalue
    println!("Example 1: Real eigenvalue selection");
    println!("-------------------------------------");
    let mut wr = vec![0.5, -0.8, 2.0, -1.5];
    let mut wi = vec![0.0, 0.0, 0.0, 0.0];
    let target = -1.0;

    println!("Available eigenvalues: {:?}", wr);
    println!("Target eigenvalue: {}", target);

    let (s, _p) = sb01bx(true, target, 0.0, &mut wr, &mut wi);

    println!("Selected eigenvalue: {}", s);
    println!("Eigenvalues after reordering: {:?}", wr);
    println!("(Selected eigenvalue moved to end)\n");

    // Example 2: Select closest complex conjugate pair
    println!("Example 2: Complex conjugate pair selection");
    println!("-------------------------------------------");
    let mut wr = vec![1.0, 1.0, -0.5, -0.5, 2.0, 2.0];
    let mut wi = vec![2.0, -2.0, 1.5, -1.5, 0.5, -0.5];
    let target_re = -0.3;
    let target_im = 1.0;

    println!("Available pairs:");
    for i in (0..wr.len()).step_by(2) {
        println!("  {} ± {}j", wr[i], wi[i]);
    }
    println!("Target: {} + {}j", target_re, target_im);

    let (s, p) = sb01bx(false, target_re, target_im, &mut wr, &mut wi);

    println!("\nSelected pair:");
    println!("  Sum (2*Re): {}", s);
    println!("  Product (|λ|²): {}", p);
    println!("\nPairs after reordering:");
    for i in (0..wr.len()).step_by(2) {
        println!("  {} ± {}j", wr[i], wi[i]);
    }
    println!("(Selected pair moved to end)\n");

    // Example 3: Use in pole placement context
    println!("Example 3: Pole placement application");
    println!("--------------------------------------");
    println!("In pole placement, you might have:");
    println!("  - Current system eigenvalues: [-0.5, 2.0, -1.5]");
    println!("  - Desired eigenvalues: [-2.0, -3.0, -4.0]");
    println!("  - Need to match them in optimal order\n");

    let mut current = vec![-0.5, 2.0, -1.5];
    let mut wi_current = vec![0.0, 0.0, 0.0];
    let desired = [-2.0, -3.0, -4.0];

    println!("Selecting eigenvalue closest to {}:", desired[0]);
    let (selected, _) = sb01bx(true, desired[0], 0.0, &mut current, &mut wi_current);
    println!("  Selected: {} (from position moved to end)", selected);
    println!("  Remaining eigenvalues: {:?}\n", &current[..2]);
}
