//! slicot-rs: Rust translation of SLICOT library
//!
//! SLICOT (Subroutine Library In COntrol Theory) is a comprehensive library
//! for control theoretical computations. This crate provides idiomatic Rust
//! implementations of SLICOT routines, organized by the original library's
//! chapter structure.
//!
//! # Organization
//!
//! The library is organized into modules corresponding to SLICOT chapters:
//! - `mb`: Mathematical routines - Basic operations (MB03MY, MB04TU, etc.)
//! - `ma`: Mathematical routines - Advanced operations (MA02FD, MA01AD, etc.)
//! - `ab`: Analysis routines (AB01MD - controllability decomposition, etc.)
//! - `sb`: Synthesis routines (SB01BD - pole placement, etc.)
//! - Additional chapters will be added as routines are translated
//!
//! # Example
//!
//! ```
//! use slicot_rs::mb::mb03my;
//!
//! let data = vec![3.0, -1.5, 2.0, -0.5, 4.0];
//! let min_abs = mb03my(&data);
//! assert_eq!(min_abs, Some(0.5));
//! ```

pub mod ab;
pub mod ma;
pub mod mb;
pub mod sb;
