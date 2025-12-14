# ProteinMPNN Tests

This directory contains tests for the ProteinMPNN package.

## Running Tests

### Install test dependencies

```bash
pip install -e ".[test]"
```

### Run all tests

```bash
pytest tests/
```

### Run specific test files

```bash
# Test installation and weights
pytest tests/test_installation.py -v

# Test inference
pytest tests/test_inference.py -v
```

### Run with coverage

```bash
pytest tests/ --cov=protein_mpnn_utils --cov=protein_mpnn_run
```

## Test Description

- **test_installation.py**: Verifies that the package is correctly installed via pip and that model weights are included in the installation.
- **test_inference.py**: Tests that ProteinMPNN can successfully run inference on test PDB files.
