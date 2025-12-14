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

### Run only installation tests (no GPU required)

```bash
pytest tests/ -m "not inference"
```

This is what runs in CI - it verifies the package installs correctly and weights are downloaded, but doesn't run actual inference.

### Run only inference tests (requires GPU/model execution)

```bash
pytest tests/ -m inference
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

- **test_installation.py**: Verifies that the package is correctly installed via pip and that model weights are downloaded from GitHub during installation.
- **test_inference.py**: Tests that ProteinMPNN can successfully run inference on test PDB files (marked with `@pytest.mark.inference` and `@pytest.mark.slow`).

## Test Markers

- `inference`: Tests that run actual model inference (skipped in CI)
- `slow`: Tests that take significant time to run
