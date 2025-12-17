# Performance Testing Guide for ProteinMPNN

This guide explains how to profile and benchmark the ProteinMPNN wrapper for performance optimization.

## Quick Start

### Run all performance tests:
```bash
pytest test_performance.py -v -s
```

### Run specific test categories:
```bash
# Single structure tests only
pytest test_performance.py -v -s -k "single_structure"

# Multiple structures tests only
pytest test_performance.py -v -s -k "multiple_structures"

# Memory profiling tests
pytest test_performance.py -v -s -k "memory"
```

### Skip performance tests in regular CI:
```bash
pytest -v -m "not performance"
```

## Profiling Methods

### 1. Basic Timing (Built into tests)
The performance tests automatically print timing information:

```bash
pytest test_performance.py::TestSingleStructurePerformance::test_single_structure_10_sequences -v -s
```

Output:
```
[Perf] 1 structure, 10 sequences: 2.456s (0.246s per seq)
```

### 2. Python cProfile (Detailed Function-Level)
Use the detailed profiling test to see top functions by cumulative time:

```bash
pytest test_performance.py::TestDetailedProfiling::test_profile_single_structure_10_sequences -v -s
```

This prints the top 20 functions consuming the most time.

### 3. PyTorch Profiler (GPU/CUDA Analysis)
Use the helper function in your own scripts:

```python
from test_performance import profile_workload
from protein_mpnn import ProteinMPNN

model = ProteinMPNN(device='cuda')

profile_workload(
    "batch_generation",
    model.sample,
    pdb_path_or_str="path/to/file.pdb",
    num_seq_per_target=10,
    batch_size=1
)
```

This generates:
- Console output with CUDA timing breakdown
- `trace_batch_generation.json` - Chrome trace format (view at chrome://tracing)

### 4. Line Profiler (Line-by-Line Analysis)
For detailed line-by-line profiling:

```bash
# Install line_profiler
pip install line_profiler

# Profile specific functions
kernprof -l -v test_performance.py
```

Add `@profile` decorator to functions you want to profile in detail.

### 5. py-spy (Live Profiling)
For production-like profiling without modifying code:

```bash
# Install py-spy
pip install py-spy

# Top-like live view
py-spy top -- python test_performance.py

# Generate flamegraph
py-spy record -o profile.svg -- python test_performance.py

# Record and generate speedscope format
py-spy record --format speedscope -o profile.speedscope.json -- python test_performance.py
```

View speedscope files at: https://www.speedscope.app/

## Performance Test Structure

### Test Classes

1. **TestSingleStructurePerformance**
   - Baseline measurements for single structure
   - Varying sequence counts (1, 10)
   - Different batch sizes
   - Use these to establish baseline performance

2. **TestMultipleStructuresPerformance**
   - Currently measures sequential processing
   - Use as baseline before implementing batched generation
   - Compare "before" vs "after" optimization

3. **TestMultipleTemperaturesPerformance**
   - Measures cost of multiple sampling temperatures
   - Important for understanding temperature loop overhead

4. **TestMemoryProfiler**
   - GPU memory usage tracking
   - Helps identify memory bottlenecks
   - Requires CUDA

5. **TestDetailedProfiling**
   - Uses cProfile for detailed breakdown
   - Good for identifying hotspots

## Benchmarking Workflow

### Before Optimization:
```bash
# 1. Run baseline tests and save results
pytest test_performance.py -v -s > baseline_results.txt

# 2. Profile with py-spy
py-spy record -o baseline_profile.svg -- pytest test_performance.py -v -s -k "10_sequences"

# 3. Generate PyTorch profiler trace
python -c "
from test_performance import profile_workload
from protein_mpnn import ProteinMPNN
model = ProteinMPNN(device='cuda', suppress_print=True)
profile_workload('baseline', model.sample, pdb_path_or_str=open('examples/1BC8.pdb').read(), num_seq_per_target=10)
"
```

### After Optimization:
```bash
# 1. Run same tests
pytest test_performance.py -v -s > optimized_results.txt

# 2. Compare timing
diff baseline_results.txt optimized_results.txt

# 3. Profile again
py-spy record -o optimized_profile.svg -- pytest test_performance.py -v -s -k "10_sequences"
```

### Calculate Speedup:
```python
# Extract times from test output
baseline_time = 2.456  # seconds from baseline
optimized_time = 1.234  # seconds from optimized

speedup = baseline_time / optimized_time
print(f"Speedup: {speedup:.2f}x")
print(f"Time saved: {baseline_time - optimized_time:.3f}s ({(1 - optimized_time/baseline_time)*100:.1f}%)")
```

## Integration with CI/CD

### GitHub Actions Example:
```yaml
# .github/workflows/performance.yml
name: Performance Tests

on:
  pull_request:
    branches: [main]
  workflow_dispatch:  # Manual trigger

jobs:
  performance:
    runs-on: [self-hosted, gpu]  # Requires GPU runner
    steps:
      - uses: actions/checkout@v3
      - name: Set up Python
        uses: actions/setup-python@v4
        with:
          python-version: '3.9'
      - name: Install dependencies
        run: |
          pip install -e .
          pip install pytest pytest-benchmark
      - name: Run performance tests
        run: |
          pytest test_performance.py -v -s --benchmark-only
      - name: Compare with baseline
        run: |
          # Store results and compare with main branch
          pytest test_performance.py --benchmark-compare
```

## Best Practices

### 1. **Consistent Environment**
- Use same GPU model for all tests
- Close other GPU processes
- Run multiple times and average results
- Clear CUDA cache between runs: `torch.cuda.empty_cache()`

### 2. **Controlled Variables**
- Fix random seeds for reproducibility
- Use same input structures
- Same model weights
- Same batch sizes for comparison

### 3. **Meaningful Comparisons**
- Measure end-to-end time (sample() method)
- Measure throughput (sequences per second)
- Measure per-sequence time
- Track GPU memory usage

### 4. **Profiling Hygiene**
- Profile representative workloads (not toy examples)
- Profile full pipeline, not isolated functions
- Profile with realistic batch sizes
- Don't optimize prematurely - profile first!

## Common Performance Patterns

### Expected Results (on A100/H100):
```
Single structure, 1 sequence:        ~0.2-0.5s
Single structure, 10 sequences:      ~2-5s (0.2-0.5s per seq)
Batch size impact:                   Minimal (current implementation)
Multiple temperatures:               Linear scaling with # temps
GPU memory per structure (~100 AA):  ~500-1000 MB peak
```

### Red Flags:
- Sublinear scaling with batch_size (should be ~constant or better)
- Super-linear scaling with num_seq_per_target (should be ~linear)
- High GPU memory but low utilization (memory bound)
- Low GPU utilization (<50%) (CPU bound or I/O bound)

## Troubleshooting

### "Tests are slower than expected"
1. Check GPU utilization: `nvidia-smi dmon`
2. Check for CPU bottlenecks: `py-spy top`
3. Verify CUDA is being used: Check test output for "cuda:0"
4. Check for other processes: `nvidia-smi`

### "Memory errors during tests"
1. Reduce batch_size
2. Reduce num_seq_per_target
3. Use smaller structures
4. Clear cache: `torch.cuda.empty_cache()`

### "Inconsistent results"
1. Check for other GPU processes
2. Fix random seeds explicitly
3. Run warmup iterations (first run may be slower due to CUDA init)
4. Average multiple runs

## Additional Tools

### Memory Profilers:
```bash
# PyTorch memory profiler
python -m torch.utils.bottleneck test_performance.py

# General memory profiler
pip install memory_profiler
mprof run test_performance.py
mprof plot
```

### Visualization Tools:
- **Chrome Tracing**: chrome://tracing (for PyTorch profiler traces)
- **Speedscope**: https://www.speedscope.app (for py-spy recordings)
- **TensorBoard**: For PyTorch profiler integration
- **Snakeviz**: For cProfile visualization (`pip install snakeviz`)

## References

- [PyTorch Profiler Tutorial](https://pytorch.org/tutorials/recipes/recipes/profiler_recipe.html)
- [py-spy Documentation](https://github.com/benfred/py-spy)
- [pytest-benchmark](https://pytest-benchmark.readthedocs.io/)
- [CUDA Best Practices](https://docs.nvidia.com/cuda/cuda-c-best-practices-guide/)
