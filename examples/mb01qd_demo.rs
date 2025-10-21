//! Demonstration of the MB01QD matrix scaling routine

use ndarray::arr2;
use slicot_rs::mb::mb01qd;

fn main() {
    println!("=== MB01QD Matrix Scaling Demonstration ===\n");

    // Test 1: Full matrix scaling
    println!("Test 1: Scaling a full 3x3 matrix by 2.0");
    let mut a = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

    println!("Original matrix:");
    println!("{}\n", a);

    mb01qd('G', 0, 0, 1.0, 2.0, &mut a, None).unwrap();

    println!("After scaling by 2.0:");
    println!("{}\n", a);

    // Test 2: Upper triangular scaling
    println!("Test 2: Scaling only the upper triangular part by 10.0");
    let mut b = arr2(&[[1.0, 2.0, 3.0], [4.0, 5.0, 6.0], [7.0, 8.0, 9.0]]);

    println!("Original matrix:");
    println!("{}\n", b);

    mb01qd('U', 0, 0, 1.0, 10.0, &mut b, None).unwrap();

    println!("After scaling upper triangular part by 10.0:");
    println!("{}\n", b);

    // Test 3: Lower triangular scaling
    println!("Test 3: Scaling only the lower triangular part by 0.5");
    let mut c = arr2(&[[10.0, 20.0, 30.0], [40.0, 50.0, 60.0], [70.0, 80.0, 90.0]]);

    println!("Original matrix:");
    println!("{}\n", c);

    mb01qd('L', 0, 0, 1.0, 0.5, &mut c, None).unwrap();

    println!("After scaling lower triangular part by 0.5:");
    println!("{}\n", c);

    println!("All tests completed successfully!");
}
