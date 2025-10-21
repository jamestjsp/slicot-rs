---
name: slicot-knowledge
description: This skill should be used when working with SLICOT (Subroutine Library In Control Theory) routines, translating Fortran 77 to Rust, parsing SLICOT HTML documentation, creating test cases from SLICOT examples, or understanding SLICOT data formats and library organization.
---

# SLICOT Knowledge Skill

Comprehensive knowledge base for working with the SLICOT library, including Fortran-to-Rust translation, documentation parsing, and test data extraction.

## When to Use This Skill

Activate this skill when:

1. **Translating SLICOT routines** from Fortran 77 to Rust
2. **Parsing SLICOT HTML documentation** to understand routine specifications
3. **Creating test cases** from SLICOT example programs and data
4. **Understanding SLICOT data formats** (matrix storage, polynomials, etc.)
5. **Navigating SLICOT library organization** (chapters, naming conventions)
6. **Interpreting Fortran READ statements** for test data
7. **Handling column-major to row-major conversions**
8. **Planning translation order** using the dependency tree (translate leaves first)

## SLICOT Overview

SLICOT is a numerical library for control theoretical computations implemented in Fortran 77. It provides:

- State-space system analysis and synthesis
- Model reduction and realization
- Controller design (pole placement, LQR, H₂, H∞)
- Filtering and identification
- Numerical linear algebra for control theory

### Library Organization

SLICOT routines are organized into functional chapters:

- **A** - Analysis Routines
- **B** - Benchmark and Test Problems
- **D** - Data Analysis
- **F** - Filtering
- **I** - Identification
- **M** - Mathematical Routines
- **N** - Nonlinear Systems
- **S** - Synthesis Routines
- **T** - Transformation Routines
- **U** - Utility Routines

### Naming Convention

Routine names follow the pattern: `XXYYZZ`

- `XX` - Chapter (e.g., AB, SB, MA)
- `YY` - Section number (01, 02, etc.)
- `ZZ` - Variant/subsection (MD, ND, etc.)

**Example**: `AB01MD` = Analysis/Benchmark chapter, section 01, variant MD

## Working with SLICOT HTML Documentation

SLICOT routines have comprehensive HTML documentation in `reference/doc/*.html`.

### Standard Documentation Structure

Each HTML file contains these sections in order:

1. **Header** - Routine name and brief description
2. **Purpose** - Detailed problem description and mathematical formulation
3. **Specification** - Fortran subroutine signature with parameter types
4. **Arguments** - Detailed parameter descriptions by category:
   - Mode Parameters (JOBZ, UPLO, TRANS, etc.)
   - Input/Output Parameters
   - Tolerances
   - Workspace (DWORK, IWORK, LDWORK, LIWORK)
   - Error Indicator (INFO)
5. **Method** - Algorithm description and numerical techniques
6. **References** - Academic papers and textbooks
7. **Numerical Aspects** - Complexity and stability analysis
8. **Example** - Complete program with:
   - **Program Text** - Fortran example code
   - **Program Data** - Test inputs
   - **Program Results** - Expected outputs

### Parsing Workflow

#### For Implementation

1. Read **Purpose** to understand the mathematical problem
2. Read **Method** to understand the algorithm
3. Read **Specification** to design the function signature
4. Read **Arguments** to implement parameter handling and validation
5. Read **Numerical Aspects** to set performance expectations

#### For Testing

1. Read **Example Program Text** to find READ statements showing data format
2. Read **Program Data** to extract test inputs
3. Read **Program Results** to extract expected outputs
4. Parse data according to Fortran column-major convention
5. Write test with appropriate numerical tolerances (~1e-3 to 5e-3)

### Key Information Locations

| Need | Look Here |
|------|-----------|
| What does this routine do? | **Purpose** section |
| What algorithm is used? | **Method** section |
| Function signature | **Specification** section |
| Parameter details | **Arguments** section |
| Test inputs | **Example → Program Data** |
| Expected results | **Example → Program Results** |
| Data input format | **Example → Program Text** (READ statements) |

## Parsing Test Data (CRITICAL)

SLICOT uses Fortran column-major storage, but HTML examples may present data differently.

### The Golden Rule

**ALWAYS examine the Fortran READ statements in "Program Text" to determine the true data format.**

### Common Fortran READ Patterns

#### Pattern 1: Column-wise Matrix Read

```fortran
READ ( NIN, FMT = * ) ( ( A(I,J), I = 1,M ), J = 1,N )
```

Reads M×N matrix **column-by-column**:
- First M values → Column 1
- Next M values → Column 2
- Continue for N columns

#### Pattern 2: Row-wise Matrix Read

```fortran
READ ( NIN, FMT = * ) ( ( A(I,J), J = 1,N ), I = 1,M )
```

Reads M×N matrix **row-by-row**:
- First N values → Row 1
- Next N values → Row 2
- Continue for M rows

#### Pattern 3: Vector Read

```fortran
READ ( NIN, FMT = * ) ( B(I), I = 1,N )
```

Reads vector sequentially.

### Implied DO Loop Interpretation

For nested loops: `( ( ARRAY(I,J), <inner> ), <outer> )`

- **Inner loop** varies fastest (reads contiguously)
- **Outer loop** varies slowest (advances to next block)

Example: `((A(I,J), I=1,3), J=1,2)` reads:
```
A(1,1), A(2,1), A(3,1),  ← Column 1
A(1,2), A(2,2), A(3,2)   ← Column 2
```

### Practical Example

**HTML Data**:
```
  0.0000  2.0000  1.0000
 -1.0000 -0.1000  1.0000
```

**Fortran Code**:
```fortran
READ ( NIN, FMT = * ) ( ( B(I,J), I = 1,N ), J = 1,M )
```

With N=3, M=2, this reads **column-wise**:

Sequence: `0.0, -1.0, 0.0, 2.0, -0.1, 1.0` (reading HTML left-to-right, top-to-bottom)

**WAIT - the HTML shows 3 numbers per line for N=3, suggesting the lines might represent something else. Let me re-check the actual sequence.**

Actually, with N=3 rows, M=2 columns:
- HTML line 1 has 3 numbers
- HTML line 2 has 3 numbers
- Total: 6 numbers for a 3×2 matrix ✓

Reading HTML sequentially: `0.0, 2.0, 1.0, -1.0, -0.1, 1.0`

With READ `((B(I,J), I=1,3), J=1,2)`:
- Column 1: B(1,1)=0.0, B(2,1)=2.0, B(3,1)=1.0
- Column 2: B(1,2)=-1.0, B(2,2)=-0.1, B(3,2)=1.0

**Row-major representation** (mathematical notation):
```
B = [  0.0  -1.0 ]
    [  2.0  -0.1 ]
    [  1.0   1.0 ]
```

### Common Pitfalls

1. **Assuming HTML presentation = data order**
   - Solution: Check Fortran READ loops

2. **Mismatched dimensions**
   - Solution: Count total numbers (should = M×N), verify with dimensions

3. **Ignoring workspace arrays**
   - Solution: Only parse problem data (A, B, C, D, etc.), skip DWORK/IWORK

4. **Wrong tolerance in tests**
   - HTML shows ~4-5 decimal places
   - Use tolerance 1e-3 to 5e-3 for comparisons
   - Adjust if algorithm has natural variations

## SLICOT Data Formats

### Matrix Storage Schemes

#### 1. Conventional (Full) Storage
- **Column-major** ordering (Fortran standard)
- Element A(i,j) at index `i-1 + (j-1)*LDA` (C/Rust 0-based)
- Leading dimension `LDA` ≥ number of rows

#### 2. Packed Storage
- For symmetric, Hermitian, or triangular matrices
- Only relevant triangle stored in 1D array
- Array names end with 'P'
- **Upper triangle** (UPLO='U'): A(i,j) → AP[i + j(j-1)/2] for i ≤ j
- **Lower triangle** (UPLO='L'): A(i,j) → AP[i + (2n-j)(j-1)/2] for i ≥ j

#### 3. Band Storage
- For banded matrices (kl subdiagonals, ku superdiagonals)
- Stored in (kl+ku+1) × n array
- Array names end with 'B'

#### 4. Special Formats
- **Tridiagonal**: 3 separate 1D arrays (diagonal, super-diagonal, sub-diagonal)
- **Bidiagonal**: 2 separate 1D arrays
- **Householder reflectors**: Factored form for orthogonal transformations

### Polynomial Storage

#### Scalar Polynomials
- 1D array P(M) where P(i+1) = coefficient of z^i
- Degree n requires M ≥ n+1

#### Vector Polynomials
- 2D array P(K,M): K components, M coefficients
- Optional degree array DEGP(K)

#### Matrix Polynomials
- 3D array P(K,L,M): K×L matrices, M coefficients
- Optional degree array DEGP(K,L)

## Fortran to Rust Type Mapping

| Fortran | Rust |
|---------|------|
| `CHARACTER*1` | `char` or custom `enum` |
| `INTEGER` | `i32` or `usize` |
| `DOUBLE PRECISION` | `f64` |
| `DOUBLE PRECISION array(N)` | `&[f64]` or `&mut [f64]` or `Array1<f64>` |
| `DOUBLE PRECISION array(LDA,*)` | `&[f64]` or `&mut [f64]` or `Array2<f64>` |
| `INFO` parameter | `Result<T, String>` or custom error enum |

### Common Mode Parameters

```rust
// JOBZ parameter example
pub enum JobZ {
    None,      // 'N': Do not compute
    Factored,  // 'F': Factored form
    Init,      // 'I': Initialize and compute
}

// UPLO parameter
pub enum Uplo {
    Upper,     // 'U'
    Lower,     // 'L'
}
```

## BLAS/LAPACK Dependencies

SLICOT routines extensively use BLAS and LAPACK. When translating to Rust:

### Use ndarray for BLAS Operations

```rust
// Matrix-vector multiply (DGEMV)
let y: Array1<f64> = a.dot(&x);

// Transpose multiply
let y: Array1<f64> = a.t().dot(&x);

// Matrix-matrix multiply (DGEMM)
let c: Array2<f64> = a.dot(&b);

// Rank-1 update (DGER): A = A - α*x*y^T
let x_col = x.view().into_shape((n, 1)).unwrap();
let y_row = y.view().into_shape((1, n)).unwrap();
*a -= &(x_col.dot(&y_row) * alpha);
```

### Use ndarray-linalg for LAPACK Operations

```rust
use ndarray_linalg::{Eig, SVD, Solve, QR, Cholesky};

// Eigenvalue decomposition (DGEEV)
let (eigenvalues, eigenvectors) = a.eig()?;

// SVD (DGESVD, DGESDD)
let (u, s, vt) = a.svd(true, true)?;

// Linear solve (DGESV)
let x = a.solve(&b)?;

// QR decomposition (DGEQRF)
let (q, r) = a.qr()?;

// Cholesky (DPOTRF)
let l = a.cholesky(UPLO::Lower)?;
```

## Common SLICOT Patterns

### Pattern 1: Workspace Allocation

Fortran routines require workspace arrays (DWORK, IWORK).

**In Rust**: Abstract workspace or allocate internally.

```rust
// Option 1: Internal allocation
let mut dwork = Array1::<f64>::zeros(ldwork);

// Option 2: Workspace query (if supported)
// Call with ldwork=-1, read optimal from dwork[0]
```

### Pattern 2: INFO Error Codes

```fortran
INFO = 0:  Success
INFO < 0:  Parameter -INFO is invalid
INFO > 0:  Algorithm-specific error
```

**In Rust**: Use `Result` type

```rust
pub fn routine(...) -> Result<Output, SlicotError> {
    // Validate parameters
    if n == 0 {
        return Ok(default_output);  // Quick return
    }

    // Compute

    // Check convergence
    if !converged {
        return Err(SlicotError::ConvergenceFailure);
    }

    Ok(output)
}
```

### Pattern 3: Input/Output Arrays

Many SLICOT routines modify arrays in-place.

```rust
pub fn routine(
    a: &mut Array2<f64>,  // Modified in-place
    b: &Array1<f64>,      // Read-only input
) -> Result<usize, String>
```

### Pattern 4: Zero-Dimension Edge Cases

Many routines accept N=0 or M=0 as valid (quick return).

```rust
// Check Fortran source for quick return conditions
if n == 0 {
    return Ok(default_output);
}
```

Don't assume zero dimensions are errors unless Fortran code validates them.

## Reference Files

For detailed information, consult the bundled reference documentation:

### slicot-library-overview.md
- Complete SLICOT library organization
- Data storage format specifications
- BLAS/LAPACK dependencies
- Numerical precision and tolerances
- Performance considerations

**Use when**: Understanding SLICOT architecture, data formats, or library conventions.

### test-data-parsing.md
- Detailed guide to parsing HTML test data
- Fortran READ statement interpretation
- Column-major to row-major conversion
- Worked examples (AB01MD, AB01ND, TF01MD, AB05MD)
- Common pitfalls and solutions

**Use when**: Creating test cases from SLICOT HTML examples.

### html-documentation-structure.md
- Section-by-section HTML documentation guide
- Navigation strategies
- Information extraction workflows
- Common patterns (mode parameters, workspace, error codes)
- Quick reference tables

**Use when**: Parsing SLICOT HTML documentation for implementation or testing.

### routine-dependencies.md
- Comprehensive dependency tree analysis (Level 0/leaves → Level 1 → Level 2+)
- Currently implemented routines and their dependency status
- Missing dependencies blocking existing implementations
- Translation order recommendations (bottom-up approach)
- Quick win opportunities and critical path analysis
- Detailed dependency mappings for specific routines
- Parallel translation opportunities

**Use when**: Planning which SLICOT routines to translate next, understanding routine dependencies, or organizing translation work.

## Workflow for Translating a SLICOT Routine

### Step 1: Understand the Routine

1. Read HTML documentation **Purpose** section
2. Read **Method** section for algorithm overview
3. Identify required BLAS/LAPACK operations

### Step 2: Design the API

1. Examine **Specification** for parameters
2. Map Fortran types to Rust types
3. Decide on function signature:
   - Use `&Array2<f64>` for matrices
   - Use `&mut` for in-place modifications
   - Return `Result<T, E>` for error handling

### Step 3: Implement the Algorithm

1. Translate **Method** description to Rust
2. Replace BLAS calls with `ndarray` operations
3. Replace LAPACK calls with `ndarray-linalg` traits
4. Handle workspace internally
5. Implement parameter validation
6. Add zero-dimension edge cases

### Step 4: Create Tests

1. Read **Example Program Text** for READ statements
2. Parse **Program Data** according to Fortran format
3. Extract **Program Results** as expected values
4. Write test with appropriate tolerance (1e-3 to 5e-3)
5. Test edge cases (zero dimensions, singular matrices)

### Step 5: Verify

1. Run tests and compare with expected results
2. Check numerical stability
3. Benchmark if performance-critical
4. Cross-reference with Fortran source in `reference/src/`

## Common Translation Challenges

### Challenge 1: Column-Major vs Row-Major

**Problem**: Fortran uses column-major, Rust/ndarray default is row-major.

**Solution**:
- `ndarray` supports both layouts
- Use `.reversed_axes()` or `.t()` for transpose
- `ndarray-linalg` handles layout conversions automatically

### Challenge 2: 1-Based vs 0-Based Indexing

**Problem**: Fortran uses 1-based indexing, Rust uses 0-based.

**Solution**: Adjust all array indices by -1.

```fortran
DO I = 1, N
   A(I,J) = ...
```

```rust
for i in 0..n {
    a[(i, j)] = ...;
}
```

### Challenge 3: Workspace Management

**Problem**: Fortran requires explicit workspace arrays.

**Solution**: Allocate workspace internally in Rust wrapper.

```rust
// Calculate required workspace
let ldwork = 3 * n + m;
let mut dwork = Array1::<f64>::zeros(ldwork);
```

### Challenge 4: Error Handling

**Problem**: Fortran uses INFO parameter for error codes.

**Solution**: Map to Rust Result type.

```rust
pub enum SlicotError {
    InvalidDimension(String),
    SingularMatrix,
    ConvergenceFailure,
    LapackError(i32),
}

pub type SlicotResult<T> = Result<T, SlicotError>;
```

## Quick Reference

### Finding Information

| Question | Location |
|----------|----------|
| What does routine XYZ do? | HTML: **Purpose** section |
| How does the algorithm work? | HTML: **Method** section |
| What are the parameters? | HTML: **Arguments** section |
| What's the function signature? | HTML: **Specification** section |
| How to parse test data? | HTML: **Example → Program Text** (READ statements) |
| What are test inputs? | HTML: **Example → Program Data** |
| What are expected results? | HTML: **Example → Program Results** |
| What's the complexity? | HTML: **Numerical Aspects** section |
| Where's the Fortran source? | `reference/src/XYZ.f` |
| Where are Fortran examples? | `reference/examples/TXYZ.f` |

### Essential Reminders

1. **Always check Fortran READ statements** for data format
2. **Use appropriate tolerances** (1e-3 to 5e-3) in tests
3. **Check for zero-dimension edge cases** in Fortran source
4. **Use ndarray-linalg** for LAPACK operations (never reimplement)
5. **Use ndarray .dot()** for BLAS operations (never manual loops)
6. **Validate parameters** but don't be stricter than Fortran
7. **Trust Fortran source** over HTML if they conflict

## Getting Help

If stuck:

1. **Read the reference files** in this skill
2. **Examine Fortran source** in `reference/src/`
3. **Look at Fortran examples** in `reference/examples/`
4. **Check SLICOT papers** referenced in HTML documentation
5. **Cross-reference with existing Rust implementations** in `src/`
6. **Review LAPACK/BLAS documentation** for specific operations
