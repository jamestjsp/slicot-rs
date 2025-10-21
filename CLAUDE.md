# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Project Overview

**slicot-rs** is a Rust translation of the SLICOT (Subroutine Library in Systems and Control Theory) library. SLICOT is a comprehensive numerical library for control theoretical computations, originally implemented in Fortran 77.

The project aims to provide:
- Modern Rust implementations of SLICOT algorithms
- Type-safe interfaces leveraging Rust's type system
- Integration with Rust's numerical computing ecosystem (ndarray, BLAS, LAPACK)
- Performance comparable to or better than the original Fortran implementation

## Reference Implementation

The original Fortran 77 SLICOT library is included as a git submodule in `reference/`:
- **Source**: Fortran routines in `reference/src/` (500+ files)
- **Examples**: Example programs in `reference/examples/`
- **Documentation**: Comprehensive HTML docs and papers in `reference/docs/`
- **Organization**: Library chapters A (Analysis), B (Benchmark), D (Data Analysis), F (Filtering), I (Identification), M (Mathematical), N (Nonlinear), S (Synthesis), T (Transformation), U (Utility)

## Build Commands

```bash
# Build the library
cargo build

# Build with optimizations
cargo build --release

# Run tests
cargo test

# Run tests with output
cargo test -- --nocapture

# Run a specific test
cargo test test_name

# Run benchmarks
cargo bench

# Check code without building
cargo check

# Format code
cargo fmt

# Check formatting without modifying files
cargo fmt --check

# Lint code
cargo clippy

# Lint with strict warnings (treat warnings as errors)
cargo clippy --all-targets --all-features -- -D warnings

# Combined quality check (run before committing)
cargo clippy --all-targets --all-features -- -D warnings && cargo fmt --check && cargo test
```

### Clippy Configuration

The project uses `clippy.toml` for clippy lint configuration:
- **warn-on-all-wildcard-imports**: Enabled to catch overly broad imports
- **cognitive-complexity-threshold**: Set to 30 for complex functions

**Important**: Clippy configuration must be in `clippy.toml`, not in `Cargo.toml` (which only accepts dependency configuration).
```

## Architecture

### Module Organization

The Rust implementation mirrors SLICOT's chapter-based organization:
- **src/mb/**: Mathematical routines - Basic operations (chapter MB)
- **src/ma/**: Mathematical routines - Advanced operations (chapter MA)
- **src/ab/**: Analysis and Benchmarks
- **src/sb/**: Synthesis and Benchmarks
- **src/tb/**: Transformation and Benchmarks
- etc. (following SLICOT's chapter structure)

Each chapter is a separate module with its own `mod.rs` file.

### Naming Conventions

- **Function names**: Keep original SLICOT names in lowercase (e.g., `mb03my`, `ab01md`)
  - Preserves traceability to original Fortran implementation
  - Makes cross-referencing documentation easier
  - Familiar to users of original SLICOT library

### Implementation Guidelines

1. **Edge Cases**: Return `Option<T>` when operations may not be meaningful
   - Example: Return `Option<f64>` for operations on potentially empty arrays (None for empty)
   - Provides compile-time safety without runtime panics

2. **Array Parameters**: Use Rust slices instead of implementing stride parameters
   - Simplifies API (no `incx`, `incy` stride parameters)
   - More idiomatic Rust
   - Users can create slices with strides if needed using ndarray's slice functionality

3. **Memory Layout**: Default to Rust's row-major layout unless interfacing with BLAS/LAPACK
   - Use `.reversed_axes()` or `.t()` when calling column-major routines

### Dependencies
- **ndarray** (0.15): Multi-dimensional array operations, core data structure for matrices
- **ndarray-linalg** (0.16): High-level LAPACK/BLAS wrappers for linear algebra
- **num-complex** (0.4): Complex number support (for eigenvalues, etc.)
- **accelerate-src** (0.3, macOS only): Links to Apple's Accelerate framework
- **openblas-src** (0.10, non-macOS): OpenBLAS backend for BLAS/LAPACK
- **approx** (0.5, dev): Floating-point comparisons in tests

### Translation Strategy

The Fortran-to-Rust translation follows these key principles:

1. **Memory Layout**: Fortran uses column-major storage; Rust/ndarray can support both. Carefully handle memory layout conversions when interfacing with BLAS/LAPACK.

2. **Array Indexing**: Fortran uses 1-based indexing; Rust uses 0-based. All array accesses must be adjusted.

3. **BLAS/LAPACK Interop**: Use `ndarray-linalg` for high-level LAPACK operations rather than raw FFI or reimplementing routines.

4. **Error Handling**: Convert Fortran's INFO parameter error codes to Rust's Result type with descriptive error enums.

5. **Workspace Arrays**: Fortran routines often require workspace arrays (WORK, IWORK). Abstract these details in Rust implementations or use automatic allocation.

6. **Precision**: SLICOT uses double precision (DOUBLE PRECISION in Fortran). Use f64 in Rust.

### SLICOT Library Organization

SLICOT routines follow a hierarchical naming convention:
- **First 2 letters**: Chapter (AB=Analysis/Benchmarks, SB=Synthesis/Benchmarks, etc.)
- **Next 2 digits**: Section number
- **Last 2 characters**: Subsection/variant

Example: `AB01MD` = Analysis/Benchmark chapter, section 01, variant MD

### Translation Workflow

When translating a Fortran routine to Rust:

1. **Understand the algorithm**: Read the routine's documentation in `reference/docs/` or the Fortran source comments

2. **Study the example**: Check `reference/examples/` for usage patterns and test data

3. **Identify dependencies** (CRITICAL):
   - Search Fortran source for `CALL DGEMV`, `CALL DGEMM`, `CALL DGER` (BLAS)
   - Search for `CALL DGEEV`, `CALL DGESVD`, `CALL DGESV`, etc. (LAPACK)
   - Note ALL matrix operations (nested loops, matrix multiplies, etc.)
   - **YOU MUST use ndarray/ndarray-linalg for these operations** (see "CRITICAL: When to Use LAPACK/BLAS" section below)

4. **Design the API**: Create idiomatic Rust interface (use slices, Result, generic types where appropriate)

5. **Implement** (following LAPACK/BLAS guidelines):
   - Translate the algorithm, adjusting for 0-based indexing
   - Replace BLAS calls with ndarray operations (`.dot()`, outer products)
   - Replace LAPACK calls with ndarray-linalg traits (`Eig`, `SVD`, `Solve`, etc.)
   - **NEVER use manual nested loops for matrix operations**
   - See AB01MD and SB01BD as reference examples

6. **Test**: Use test data from examples, verify against reference results

7. **Benchmark**: Compare performance with original Fortran if performance-critical
   - Performance should be within 10-20% of Fortran
   - If slower, check that BLAS/LAPACK are actually being used

### CRITICAL: When to Use LAPACK/BLAS

**ALWAYS use LAPACK/BLAS operations instead of manual loops when the original Fortran SLICOT code calls BLAS/LAPACK routines.**

#### Operations That MUST Use BLAS (via ndarray methods)

1. **Matrix-Vector Multiplication** (`y = A*x`)
   - ❌ **NEVER DO**: Nested loops
   ```rust
   // WRONG - Manual implementation
   for i in 0..n {
       for j in 0..n {
           y[i] += a[(i, j)] * x[j];
       }
   }
   ```
   - ✅ **CORRECT**: Use ndarray's `.dot()` method
   ```rust
   // RIGHT - BLAS DGEMV
   let y: Array1<f64> = a.dot(&x);
   ```

2. **Transpose Matrix-Vector Multiplication** (`y = A^T*x`)
   - ❌ **NEVER DO**: Manual loop with transposed indices
   ```rust
   // WRONG
   for i in 0..n {
       for j in 0..n {
           y[i] += a[(j, i)] * x[j];
       }
   }
   ```
   - ✅ **CORRECT**: Use `.t().dot()`
   ```rust
   // RIGHT - BLAS DGEMV with transpose
   let y: Array1<f64> = a.t().dot(&x);
   ```

3. **Rank-1 Update** (`A = A - α*x*y^T`)
   - ❌ **NEVER DO**: Nested loops
   ```rust
   // WRONG
   for i in 0..n {
       for j in 0..n {
           a[(i, j)] -= tau * x[i] * y[j];
       }
   }
   ```
   - ✅ **CORRECT**: Use outer product
   ```rust
   // RIGHT - BLAS DGER via outer product
   let x_col = x.view().into_shape((n, 1)).unwrap();
   let y_row = y.view().into_shape((1, n)).unwrap();
   *a -= &(x_col.dot(&y_row) * tau);
   ```

4. **Matrix-Matrix Multiplication** (`C = A*B`)
   - ✅ **CORRECT**: Use `.dot()`
   ```rust
   let c: Array2<f64> = a.dot(&b);  // BLAS DGEMM
   ```

#### Operations That MUST Use LAPACK (via ndarray-linalg traits)

1. **Eigenvalue Decomposition**
   - Fortran: `CALL DGEEV(...)`
   - ✅ Rust: Use `Eig` trait
   ```rust
   use ndarray_linalg::Eig;

   match a.eig() {
       Ok((eigenvalues, eigenvectors)) => {
           // eigenvalues: Array1<Complex<f64>>
           // Process results...
       }
       Err(e) => {
           // Handle LAPACK error
       }
   }
   ```

2. **Singular Value Decomposition**
   - Fortran: `CALL DGESVD(...)` or `CALL DGESDD(...)`
   - ✅ Rust: Use `SVD` trait
   ```rust
   use ndarray_linalg::SVD;

   match matrix.svd(true, true) {
       Ok((Some(u), s, Some(vt))) => {
           // Compute rank via singular value thresholding
           let rank = s.iter().filter(|&&sv| sv > tolerance).count();
       }
       Err(e) => { /* Handle error */ }
   }
   ```

3. **Linear System Solving** (`Ax = b`)
   - Fortran: `CALL DGESV(...)`
   - ✅ Rust: Use `Solve` trait or custom Gaussian elimination
   ```rust
   use ndarray_linalg::Solve;

   match a.solve(&b) {
       Ok(x) => { /* Use solution */ }
       Err(e) => { /* Handle singular matrix */ }
   }
   ```

4. **QR Decomposition**
   - Fortran: `CALL DGEQRF(...)`
   - ✅ Rust: Use `QR` trait
   ```rust
   use ndarray_linalg::QR;

   let (q, r) = a.qr()?;
   ```

5. **Cholesky Decomposition**
   - Fortran: `CALL DPOTRF(...)`
   - ✅ Rust: Use `Cholesky` trait
   ```rust
   use ndarray_linalg::Cholesky;

   let l = a.cholesky(UPLO::Lower)?;
   ```

### Real-World Examples from This Codebase

#### Example 1: AB01MD (src/ab/mod.rs)

**Before (WRONG - Manual loops):**
```rust
// ❌ O(n²) manual implementation
let mut w: Array1<f64> = Array1::zeros(n);
for i in 0..n {
    for j in 0..n {
        w[i] += a[(i, j)] * v[j];
    }
}

// ❌ O(n²) manual rank-1 update
for i in 0..n {
    for j in 0..n {
        a[(i, j)] -= tau * v[i] * w[j];
    }
}
```

**After (CORRECT - BLAS operations):**
```rust
// ✅ BLAS DGEMV - optimized matrix-vector multiply
let w: Array1<f64> = a.dot(&v);

// ✅ BLAS DGER - optimized rank-1 update
let v_col = v.view().into_shape((n, 1)).unwrap();
let w_row = w.view().into_shape((1, n)).unwrap();
*a -= &(v_col.dot(&w_row) * tau);
```

**Result:** 20-50% performance improvement

#### Example 2: SB01BD (src/sb/mod.rs)

**Before (WRONG - Placeholder):**
```rust
// ❌ Returns mostly zeros, non-functional
pub fn compute_feedback_matrix(...) -> Array2<f64> {
    let mut f = Array2::zeros((m, n));
    // ... ad-hoc formulas only for 2×2 ...
    Ok(f)
}
```

**After (CORRECT - Ackermann's Formula with LAPACK):**
```rust
// ✅ Functional pole placement using SVD and eigenvalues
pub fn compute_feedback_ackermann(
    a: &Array2<f64>,
    b: &Array2<f64>,
    desired_eigenvalues: &[f64],
) -> Result<Array2<f64>, String> {
    // 1. Build controllability matrix using BLAS
    let mut c_matrix = b.clone();
    let mut a_power_b = b.clone();
    for k in 1..n {
        a_power_b = a.dot(&a_power_b);  // BLAS DGEMM
        // ...
    }

    // 2. Check rank using LAPACK SVD
    use ndarray_linalg::SVD;
    match c_matrix.svd(false, false) {
        Ok((_, s, _)) => {
            let rank = s.iter().filter(|&&sv| sv > tol).count();
            // Determine controllability...
        }
        Err(_) => { /* fallback */ }
    }

    // 3. Compute polynomial p(A) using BLAS for matrix powers
    // 4. Solve linear system C*x = p(A)*e_n
    // 5. Return feedback F = x^T
}
```

**Result:** Functional pole placement algorithm (was previously non-functional)

### Translation Checklist: Fortran LAPACK to Rust

When translating a Fortran SLICOT routine:

1. **Scan the Fortran code for BLAS/LAPACK calls**
   - Look for: `CALL DGEMV`, `CALL DGEMM`, `CALL DGER`, `CALL DGEEV`, `CALL DGESVD`, etc.
   - These MUST be replaced with ndarray/ndarray-linalg equivalents

2. **Never implement BLAS operations manually**
   - If you see nested loops doing matrix operations → use ndarray `.dot()`
   - If you see BLAS calls in Fortran → use ndarray operations
   - If you see LAPACK calls in Fortran → use ndarray-linalg traits

3. **Check the original Fortran for performance-critical sections**
   - Comments like `C     BLAS DGEMV` or `C     LAPACK DGEEV` indicate you MUST use those operations
   - Large O(n³) or O(n²) loops should use BLAS/LAPACK

4. **Verify your implementation**
   - Run benchmarks comparing to Fortran if possible
   - Profile to ensure BLAS/LAPACK are actually being called
   - Check that performance is comparable to original

### Common Mistakes to Avoid

❌ **Mistake 1**: "I'll implement it simply first, then optimize later"
- **Problem**: Manual implementations are often left in place and hurt performance
- **Solution**: Use BLAS/LAPACK from the start

❌ **Mistake 2**: "The manual loop is clearer and more readable"
- **Problem**: 20-50% performance loss, not idiomatic Rust
- **Solution**: Add comments explaining the operation, but use BLAS/LAPACK

❌ **Mistake 3**: "I'm not sure what LAPACK routine to use"
- **Problem**: Placeholder code gets committed
- **Solution**: Check the Fortran source, look at AB01MD/SB01BD examples, consult LAPACK docs

❌ **Mistake 4**: "The system is small so performance doesn't matter"
- **Problem**: Systems grow, code gets reused, benchmarks look bad
- **Solution**: Always use BLAS/LAPACK; it's also more maintainable

### Required Pattern for All Implementations

```rust
// 1. Import necessary traits
use ndarray::{Array1, Array2, s};
use ndarray_linalg::{Eig, SVD, Solve};  // As needed
use num_complex::Complex;

// 2. Use BLAS for matrix operations
let result = a.dot(&x);           // Matrix-vector
let result = a.t().dot(&x);       // Transpose multiply
let c = a.dot(&b);                // Matrix-matrix

// 3. Use ndarray-linalg for decompositions
let (eigenvalues, _) = a.eig()?;  // Eigenvalues
let (_, s, _) = a.svd(false, false)?;  // Singular values
let x = a.solve(&b)?;             // Linear solve

// 4. Handle errors properly
match a.eig() {
    Ok((eigs, vecs)) => { /* Use results */ }
    Err(e) => {
        return Err(format!("Eigenvalue computation failed: {:?}", e));
    }
}
```

## LAPACK Integration

### Overview

This project uses **ndarray-linalg** to call LAPACK routines for linear algebra operations (eigenvalues, decompositions, etc.). The `ndarray-linalg` crate provides:
- High-level, idiomatic Rust interfaces to LAPACK
- Automatic memory management (no manual workspace allocation)
- Type-safe wrappers with proper error handling
- Transparent handling of column-major/row-major layout conversions

### Backend Configuration

The LAPACK backend is configured **platform-specifically** in `Cargo.toml`:

**macOS (Apple Silicon/Intel)**:
```toml
[target.'cfg(target_os = "macos")'.dependencies]
ndarray-linalg = { version = "0.16", features = [] }
accelerate-src = "0.3"  # Uses Apple's Accelerate framework
```

**Linux/Other platforms**:
```toml
[target.'cfg(not(target_os = "macos"))'.dependencies]
ndarray-linalg = { version = "0.16", features = ["openblas"] }
openblas-src = { version = "0.10", features = ["cblas", "system"] }
```

**Important**: A `build.rs` file explicitly links the Accelerate framework on macOS:
```rust
fn main() {
    #[cfg(target_os = "macos")]
    {
        println!("cargo:rustc-link-lib=framework=Accelerate");
    }
}
```

### Using LAPACK in Code

Import the necessary traits from `ndarray-linalg`:

```rust
use ndarray::{Array1, Array2};
use ndarray_linalg::Eig;  // For eigenvalue decomposition
use num_complex::Complex;
```

**Example: Computing eigenvalues**
```rust
let a: Array2<f64> = arr2(&[[1.0, 2.0], [3.0, 4.0]]);

// Compute eigenvalues and eigenvectors using LAPACK's DGEEV
match a.eig() {
    Ok((eigenvalues, eigenvectors)) => {
        // eigenvalues: Array1<Complex<f64>>
        // eigenvectors: Array2<Complex<f64>>
        println!("Eigenvalues: {:?}", eigenvalues);
    }
    Err(e) => {
        eprintln!("Eigenvalue computation failed: {:?}", e);
    }
}
```

### Available LAPACK Operations via ndarray-linalg

Common operations available through `ndarray-linalg`:

| Trait | LAPACK Routine | Description |
|-------|----------------|-------------|
| `Eig` | DGEEV/ZGEEV | Eigenvalue decomposition |
| `SVD` | DGESDD/ZGESDD | Singular value decomposition |
| `QR` | DGEQRF | QR decomposition |
| `Cholesky` | DPOTRF | Cholesky decomposition |
| `Solve` | DGESV | Solve linear system Ax=b |
| `Inverse` | DGETRF+DGETRI | Matrix inversion |
| `LeastSquares` | DGELSD | Least squares solution |

### Memory Layout

LAPACK expects **column-major** layout (Fortran convention), while Rust/ndarray defaults to **row-major**.

**ndarray-linalg handles this automatically** - no manual transposition needed when using the high-level traits.

If using raw LAPACK FFI (not recommended):
```rust
// Manual layout conversion (avoid this - use ndarray-linalg instead)
let a_col_major = a.reversed_axes();  // Convert to column-major
```

### Error Handling Pattern

LAPACK routines signal errors via INFO codes. ndarray-linalg wraps these as `Result`:

```rust
use ndarray_linalg::error::LinalgError;

fn my_function(a: &Array2<f64>) -> Result<Array1<Complex<f64>>, String> {
    match a.eig() {
        Ok((eigenvalues, _)) => Ok(eigenvalues),
        Err(LinalgError::Lapack { return_code }) => {
            Err(format!("LAPACK error code: {}", return_code))
        }
        Err(e) => Err(format!("Linear algebra error: {:?}", e)),
    }
}
```

### Testing LAPACK Integration

To verify LAPACK is properly linked:
```bash
# Should compile and run without linker errors
cargo test --lib sb
```

If you see linker errors like `undefined symbol: _dgeev_`:
1. Check that `accelerate-src` (macOS) or `openblas-src` (Linux) is in dependencies
2. Verify `build.rs` exists and links Accelerate on macOS
3. On Linux, ensure OpenBLAS is installed: `sudo apt-get install libopenblas-dev`

### Performance Notes

- **ndarray-linalg** calls the same optimized LAPACK routines as Fortran
- On macOS, Apple's Accelerate framework is highly optimized for Apple Silicon
- On Linux, OpenBLAS provides multi-threaded BLAS/LAPACK
- Performance should match or exceed the original Fortran SLICOT

## Testing Strategy

- Place unit tests in module files or in `tests/` for integration tests
- Use test data from `reference/examples/` .dat files when available
- Use `approx` crate for floating-point comparisons (account for numerical precision)
- Test edge cases: zero dimensions, singular matrices, invalid inputs
- Verify error handling matches SLICOT's INFO error codes

## Development Workflow (from user preferences)

Per the user's workflow preferences:

1. **Branching**: Create a new git branch for each feature/bugfix/refactor
2. **Frequent commits**: Commit after each subtask is solved
3. **Testing**: Always create runnable test snippets; make them persistent (not ephemeral)
4. **Test-first approach**: If tests are broken or cumbersome, fix them in a separate branch/PR before resuming the original task
5. **Quality checks**: Run linter and type checker before committing:
   ```bash
   # Comprehensive pre-commit check (recommended)
   cargo clippy --all-targets --all-features -- -D warnings && cargo fmt --check && cargo test

   # Or run individually:
   cargo fmt --check                                      # Check formatting
   cargo clippy --all-targets --all-features -- -D warnings  # Lint code (strict)
   cargo test                                             # Run tests
   ```

## Key Files

- **Cargo.toml**: Package configuration and dependencies (platform-specific LAPACK backends)
- **build.rs**: Build script that links Accelerate framework on macOS
- **clippy.toml**: Clippy linter configuration (must be separate from Cargo.toml)
- **src/lib.rs**: Library entry point (currently minimal - to be expanded)
- **reference/**: Git submodule containing original SLICOT Fortran implementation
- **slicot_init.sh**: Initialization script used to set up the repository structure

## Common Translation Patterns

### Fortran Array to ndarray
```rust
// Fortran: DOUBLE PRECISION A(LDA,N)
// Rust:
use ndarray::Array2;
let a: Array2<f64> = Array2::zeros((m, n));
```

### BLAS/LAPACK Calls

**IMPORTANT**: See the comprehensive "CRITICAL: When to Use LAPACK/BLAS" section above for detailed guidelines.

```rust
// Matrix-vector multiplication (BLAS DGEMV)
let y = a.dot(&x);

// Rank-1 update (BLAS DGER)
let x_col = x.view().into_shape((n, 1)).unwrap();
let y_row = y.view().into_shape((1, n)).unwrap();
*a -= &(x_col.dot(&y_row) * tau);

// Eigenvalues (LAPACK DGEEV)
use ndarray_linalg::Eig;
let (eigenvalues, eigenvectors) = a.eig()?;

// SVD (LAPACK DGESVD)
use ndarray_linalg::SVD;
let (u, s, vt) = a.svd(true, true)?;

// Linear solve (LAPACK DGESV)
use ndarray_linalg::Solve;
let x = a.solve(&b)?;
```

**Examples**: See AB01MD (src/ab/mod.rs) and SB01BD (src/sb/mod.rs) for complete patterns.

### Error Handling
```rust
// Fortran: INFO parameter
// Rust:
pub enum SlicotError {
    InvalidDimension(String),
    SingularMatrix,
    ConvergenceFailure,
    // ... other error types
}

pub type SlicotResult<T> = Result<T, SlicotError>;
```

### Leading Dimension Pattern
```fortran
C Fortran: A(LDA,*) with LDA >= M
```
```rust
// Rust: Use array strides or validate dimensions
assert!(a.dim().0 >= m, "Leading dimension too small");
```

## Important Notes

- **License**: Repository currently uses BSD-3-Clause (verify compatibility with SLICOT's original license)
- **Numerical Precision**: Control theory algorithms are sensitive to numerical errors; thorough testing is critical
- **Performance**: Profile translated code; Rust should match or exceed Fortran performance with proper optimization
- **Documentation**: Preserve SLICOT's comprehensive documentation style; document algorithms, parameters, error conditions

## Reference Resources

From the SLICOT reference implementation:
- `reference/README.md`: Overview of SLICOT library organization
- `reference/Installation.md`: Build instructions for original Fortran code
- `reference/docs/`: Comprehensive HTML documentation for all routines
- `reference/examples/`: Example programs with input/output data
- SLICOT homepage: http://slicot.org/ (for additional documentation and papers)
