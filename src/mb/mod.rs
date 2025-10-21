//! Mathematical Routines - Basic Operations (Chapter MB)
//!
//! This module contains basic mathematical routines from the SLICOT library.
//! These are low-level utility functions used by higher-level control theory
//! algorithms.
//!
//! SLICOT Chapter MB focuses on fundamental mathematical operations that don't
//! require complex linear algebra but are essential building blocks for
//! numerical computations in control theory.

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
        let x_modified = vec![
            0.6 * 1.0 - 0.8 * 3.0,
            0.6 * 2.0 - 0.8 * 4.0,
            0.6 * 3.0 - 0.8 * 5.0,
        ];
        let y_modified = vec![
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
