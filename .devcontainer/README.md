# Development Container Setup

This directory contains the configuration for the VS Code Dev Container used for slicot-rs development.

## Requirements

The development container automatically installs the following dependencies:

### BLAS/LAPACK Libraries

The project requires BLAS and LAPACK libraries for linear algebra operations:

- `libopenblas-dev` - OpenBLAS library (includes optimized BLAS and LAPACK)
- `liblapack-dev` - LAPACK reference implementation
- `libblas-dev` - BLAS reference implementation
- `gfortran` - GNU Fortran compiler (required by some LAPACK packages)
- `pkg-config` - Helper tool for library discovery

These packages are automatically installed via the `postCreateCommand` when the container is first created.

## Usage

### First Time Setup

1. Open the repository in VS Code
2. When prompted, click "Reopen in Container" or use the Command Palette:
   - Press `F1` or `Ctrl+Shift+P`
   - Select "Dev Containers: Reopen in Container"
3. Wait for the container to build and the `postCreateCommand` to complete
4. Verify the setup by running:
   ```bash
   cargo build
   cargo test
   ```

### Manual Package Installation

If the `postCreateCommand` fails or you need to reinstall packages:

```bash
sudo apt-get update
sudo apt-get install -y libopenblas-dev liblapack-dev libblas-dev gfortran pkg-config
```

### Verifying LAPACK/BLAS Installation

Check if the libraries are properly installed:

```bash
# Check for OpenBLAS
pkg-config --modversion openblas

# Check for LAPACK
dpkg -l | grep -i lapack

# List installed BLAS/LAPACK related packages
dpkg -l | grep -E 'openblas|lapack|blas'
```

## Base Image

The container uses the official Microsoft Rust devcontainer:
- Image: `mcr.microsoft.com/devcontainers/rust:1-1-bullseye`
- Debian version: Bullseye (Debian 11)
- Rust: Latest stable version

## Alternative: Using a Custom Dockerfile

For more reproducible builds or if you need additional dependencies, a Dockerfile-based approach is available. This bakes the packages into the container image, making setup faster and more reliable.

### Using the Optional Dockerfile

1. Rename `Dockerfile.optional` to `Dockerfile`:
   ```bash
   cd .devcontainer
   mv Dockerfile.optional Dockerfile
   ```

2. Update `devcontainer.json` to use the Dockerfile:
   ```json
   {
     "name": "Rust",
     "build": {
       "dockerfile": "Dockerfile",
       "context": ".."
     },
     // ... rest of the configuration
   }
   ```

3. Rebuild the container:
   - Press `F1` or `Ctrl+Shift+P`
   - Select "Dev Containers: Rebuild Container"

See the `reference/.devcontainer/` folder for a more comprehensive Dockerfile-based setup used by the SLICOT C wrapper project.

## Troubleshooting

### Cargo build fails with "cannot find -lopenblas"

This means the OpenBLAS library is not installed or not found:

```bash
# Install the package
sudo apt-get update && sudo apt-get install -y libopenblas-dev

# Verify installation
ldconfig -p | grep openblas
```

### Network issues during package installation

If `apt-get update` fails due to network issues:

1. Check your internet connection
2. Rebuild the container: "Dev Containers: Rebuild Container"
3. If problems persist, consider using a Dockerfile approach with cached layers

### Rust toolchain issues

Verify your Rust installation:

```bash
rustc --version
cargo --version
rustup show
```

Update Rust if needed:

```bash
rustup update
```

## Platform-Specific Notes

### macOS (Host System)

When running on macOS, the container runs Linux (Debian), so it uses OpenBLAS. On native macOS, the project uses Apple's Accelerate framework instead (configured in `Cargo.toml`).

### Linux (Host System)

The container setup is the same whether the host is Linux, macOS, or Windows.

## References

- [VS Code Dev Containers](https://code.visualstudio.com/docs/devcontainers/containers)
- [OpenBLAS Documentation](https://www.openblas.net/)
- [ndarray-linalg Crate](https://docs.rs/ndarray-linalg/)
- [SLICOT Library](http://slicot.org/)
