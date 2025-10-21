# 1. Create and initialize your new repository
# mkdir slicot-rs
# cd slicot-rs
git init

# 2. Add the SLICOT-Reference as a submodule
git submodule add git@github.com:jamestjsp/SLICOT_C.git reference

# 3. Create initial Rust project structure
cargo init --lib

# 4. Create a basic directory structure
mkdir -p tests examples benches

# 5. Create initial .gitignore (cargo init creates one, but let's ensure it's complete)
cat >> .gitignore << 'EOF'
/target/
**/*.rs.bk
Cargo.lock
EOF

# 6. Create a README
cat > README.md << 'EOF'
# slicot-rs

A Rust translation of the SLICOT (Subroutine Library in Systems and Control Theory) library.

## Reference Implementation

The original Fortran 77 reference implementation is included as a git submodule in `reference/`.

## Building
```bash
cargo build
```

## Testing
```bash
cargo test
```

## License

TODO: Check SLICOT license compatibility
EOF

# 7. Initial commit
git add .
git commit -m "Initial commit: Rust translation of SLICOT library"

# 8. If you have a remote repository (e.g., on GitHub)
# git remote add origin https://github.com/yourusername/slicot-rs.git
# git push -u origin main