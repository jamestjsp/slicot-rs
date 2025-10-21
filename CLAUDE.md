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

# Lint code
cargo clippy
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
- **blas-src** (0.8): BLAS backend for linear algebra operations
- **lapack-src** (0.8): LAPACK backend for advanced linear algebra
- **approx** (0.5, dev): Floating-point comparisons in tests

### Translation Strategy

The Fortran-to-Rust translation follows these key principles:

1. **Memory Layout**: Fortran uses column-major storage; Rust/ndarray can support both. Carefully handle memory layout conversions when interfacing with BLAS/LAPACK.

2. **Array Indexing**: Fortran uses 1-based indexing; Rust uses 0-based. All array accesses must be adjusted.

3. **BLAS/LAPACK Interop**: Use existing Rust BLAS/LAPACK bindings rather than reimplementing low-level routines.

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
   cargo fmt --check   # Check formatting
   cargo clippy        # Lint code
   cargo test          # Run tests
   ```

## Key Files

- **Cargo.toml**: Package configuration and dependencies
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
