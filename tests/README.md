# ProteinMPNN Tests

This directory contains tests for the ProteinMPNN package.

## Running Tests

### Install test dependencies

```bash
uv pip install -e ".[test]"
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
- `performance`: Performance benchmarking tests (see below)

---

# Performance Testing

This directory also contains performance benchmarking and profiling tools.

## Quick Start

### Run Performance Tests
```bash
# From repo root
pytest test_performance.py -v -s

# Skip performance tests in regular testing
pytest -v -m "not performance"
```

### Run Benchmarks
```bash
# Quick benchmark (3 configs, ~10 seconds)
python tests/benchmark_runner.py --quick

# Standard benchmark (7 configs, ~1 minute)
python tests/benchmark_runner.py

# Comprehensive benchmark (10 configs, ~2-3 minutes)
python tests/benchmark_runner.py --comprehensive

# Save as baseline for future comparisons
python tests/benchmark_runner.py --baseline

# Compare with baseline
python tests/benchmark_runner.py --compare baseline_results.json
```

## Performance Test Structure

Tests use **6MRR.pdb** (68 residue protein) located at `inputs/PDB_monomers/pdbs/6MRR.pdb`.

### Test Classes:

1. **TestSingleStructurePerformance** - Baseline measurements
2. **TestMultipleStructuresPerformance** - Sequential processing (baseline for future batching)
3. **TestMultipleTemperaturesPerformance** - Multiple sampling temperatures
4. **TestMemoryProfiler** - GPU memory tracking
5. **TestDetailedProfiling** - cProfile integration

## Workflow for Optimization

### 1. Establish Baseline
```bash
python tests/benchmark_runner.py --baseline
# Creates baseline_results.json
```

### 2. Make Optimizations
Edit code, implement batching, etc.

### 3. Compare Performance
```bash
python tests/benchmark_runner.py --compare baseline_results.json
```

## Advanced Profiling

See [../PERFORMANCE_TESTING.md](../PERFORMANCE_TESTING.md) for detailed profiling with:
- PyTorch Profiler (CUDA analysis)
- py-spy (live profiling)
- line_profiler (line-by-line)
- Chrome tracing visualization
