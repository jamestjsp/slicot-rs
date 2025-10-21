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
