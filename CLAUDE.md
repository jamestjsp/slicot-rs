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
3. **Identify dependencies**: Note which BLAS/LAPACK routines are called
4. **Design the API**: Create idiomatic Rust interface (use slices, Result, generic types where appropriate)
5. **Implement**: Translate the algorithm, adjusting for 0-based indexing and Rust idioms
6. **Test**: Use test data from examples, verify against reference results
7. **Benchmark**: Compare performance with original Fortran if performance-critical

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
```rust
// Use ndarray-linalg or direct BLAS/LAPACK bindings
use ndarray_linalg::*;
```

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
