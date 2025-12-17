"""
Performance tests for ProteinMPNN wrapper.

Run with:
    pytest test_performance.py -v
    pytest test_performance.py -v --profile  # With cProfile
    pytest test_performance.py -v -k "single_structure"  # Run specific test

Mark tests to skip in regular CI:
    pytest -v -m "not performance"
"""

import pytest
import time
import torch
import numpy as np
from pathlib import Path
import tempfile
from protein_mpnn import ProteinMPNN


# Mark all tests in this file as performance tests
pytestmark = pytest.mark.performance


@pytest.fixture(scope="module")
def test_pdb_path():
    """Path to test PDB file (6MRR - 68 residue protein)."""
    pdb_path = Path(__file__).parent / "inputs" / "PDB_monomers" / "pdbs" / "6MRR.pdb"
    if not pdb_path.exists():
        # Try relative to repo root
        pdb_path = Path(__file__).parent.parent / "inputs" / "PDB_monomers" / "pdbs" / "6MRR.pdb"

    assert pdb_path.exists(), f"Test PDB file not found at {pdb_path}"
    return str(pdb_path)


@pytest.fixture(scope="module")
def model():
    """Load model once for all tests."""
    print("\n[Setup] Loading ProteinMPNN model...")
    start = time.time()
    model = ProteinMPNN(
        model_name='v_48_020',
        device='cuda' if torch.cuda.is_available() else 'cpu',
        suppress_print=True
    )
    elapsed = time.time() - start
    print(f"[Setup] Model loaded in {elapsed:.2f}s")
    return model


class TestSingleStructurePerformance:
    """Test performance for single structure with varying sequence counts."""

    def test_single_structure_single_sequence(self, model, test_pdb_path, benchmark=None):
        """Baseline: 1 structure, 1 sequence."""
        def run():
            return model.sample(
                pdb_path_or_str=test_pdb_path,
                num_seq_per_target=1,
                batch_size=1,
                sampling_temp="0.1",
                seed=42
            )

        if benchmark:
            result = benchmark(run)
        else:
            start = time.time()
            result = run()
            elapsed = time.time() - start
            print(f"\n[Perf] 1 structure, 1 sequence: {elapsed:.3f}s")

        assert len(result) == 1

    def test_single_structure_10_sequences(self, model, test_pdb_path):
        """1 structure, 10 sequences (sequential generation)."""
        start = time.time()
        result = model.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=10,
            batch_size=1,
            sampling_temp="0.1",
            seed=42
        )
        elapsed = time.time() - start

        protein_name = list(result.keys())[0]
        assert len(result[protein_name]['sequences']) == 10
        print(f"\n[Perf] 1 structure, 10 sequences: {elapsed:.3f}s ({elapsed/10:.3f}s per seq)")

    def test_single_structure_10_sequences_batch5(self, model, test_pdb_path):
        """1 structure, 10 sequences with batch_size=5."""
        start = time.time()
        result = model.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=10,
            batch_size=5,
            sampling_temp="0.1",
            seed=42
        )
        elapsed = time.time() - start

        protein_name = list(result.keys())[0]
        assert len(result[protein_name]['sequences']) == 10
        print(f"\n[Perf] 1 structure, 10 sequences (batch=5): {elapsed:.3f}s ({elapsed/10:.3f}s per seq)")


class TestMultipleStructuresPerformance:
    """Test performance for multiple structures (currently sequential)."""

    def test_5_structures_sequential(self, model, test_pdb_path):
        """5 structures, 1 sequence each (sequential calls)."""
        start = time.time()
        results = []
        for i in range(5):
            result = model.sample(
                pdb_path_or_str=test_pdb_path,
                num_seq_per_target=1,
                batch_size=1,
                sampling_temp="0.1",
                seed=42 + i
            )
            results.append(result)
        elapsed = time.time() - start

        assert len(results) == 5
        print(f"\n[Perf] 5 structures (sequential), 1 seq each: {elapsed:.3f}s ({elapsed/5:.3f}s per structure)")

    def test_5_structures_10_sequences_sequential(self, model, test_pdb_path):
        """5 structures, 10 sequences each (sequential calls)."""
        start = time.time()
        results = []
        for i in range(5):
            result = model.sample(
                pdb_path_or_str=test_pdb_path,
                num_seq_per_target=10,
                batch_size=1,
                sampling_temp="0.1",
                seed=42 + i
            )
            results.append(result)
        elapsed = time.time() - start

        assert len(results) == 5
        total_sequences = sum(len(list(r.values())[0]['sequences']) for r in results)
        assert total_sequences == 50
        print(f"\n[Perf] 5 structures (sequential), 10 seq each: {elapsed:.3f}s ({elapsed/50:.3f}s per seq)")


class TestMultipleTemperaturesPerformance:
    """Test performance with multiple sampling temperatures."""

    def test_single_structure_3_temperatures(self, model, test_pdb_path):
        """1 structure with 3 different temperatures (1 seq per temp)."""
        start = time.time()
        result = model.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=1,  # Generate 1 seq per temp
            batch_size=1,
            sampling_temp="0.1 0.2 0.5",  # 3 temperatures
            seed=42
        )
        elapsed = time.time() - start

        protein_name = list(result.keys())[0]
        assert len(result[protein_name]['sequences']) == 3
        assert len(result[protein_name]['sampling_temperatures']) == 3
        print(f"\n[Perf] 1 structure, 3 temperatures: {elapsed:.3f}s ({elapsed/3:.3f}s per temp)")


class TestMemoryProfiler:
    """Memory usage tests."""

    @pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA not available")
    def test_gpu_memory_usage(self, model, test_pdb_path):
        """Monitor GPU memory usage during generation."""
        torch.cuda.empty_cache()
        torch.cuda.reset_peak_memory_stats()

        initial_memory = torch.cuda.memory_allocated() / 1024**2  # MB

        result = model.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=10,
            batch_size=1,
            sampling_temp="0.1",
            seed=42
        )

        peak_memory = torch.cuda.max_memory_allocated() / 1024**2  # MB
        final_memory = torch.cuda.memory_allocated() / 1024**2  # MB

        print(f"\n[Memory] Initial: {initial_memory:.1f}MB, Peak: {peak_memory:.1f}MB, Final: {final_memory:.1f}MB")
        print(f"[Memory] Peak increase: {peak_memory - initial_memory:.1f}MB")


@pytest.fixture
def profiler_context():
    """Context manager for cProfile profiling."""
    import cProfile
    import pstats
    from io import StringIO

    profiler = cProfile.Profile()
    yield profiler

    # Print stats after test
    s = StringIO()
    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')
    ps.print_stats(20)  # Top 20 functions
    print("\n" + "="*80)
    print("PROFILER OUTPUT (Top 20 functions by cumulative time)")
    print("="*80)
    print(s.getvalue())


class TestDetailedProfiling:
    """Tests with detailed profiling output."""

    def test_profile_single_structure_10_sequences(self, model, test_pdb_path, profiler_context):
        """Profile a representative workload."""
        profiler_context.enable()

        result = model.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=10,
            batch_size=1,
            sampling_temp="0.1",
            seed=42
        )

        profiler_context.disable()

        protein_name = list(result.keys())[0]
        assert len(result[protein_name]['sequences']) == 10


# Helper function for manual profiling
def profile_workload(workload_name: str, func, *args, **kwargs):
    """
    Helper function to profile a specific workload with torch profiler.

    Example usage:
        from test_performance import profile_workload
        profile_workload(
            "my_test",
            model.sample,
            pdb_path_or_str=content,
            num_seq_per_target=10
        )
    """
    import torch.profiler as profiler

    with profiler.profile(
        activities=[
            profiler.ProfilerActivity.CPU,
            profiler.ProfilerActivity.CUDA,
        ],
        record_shapes=True,
        profile_memory=True,
        with_stack=True,
    ) as prof:
        result = func(*args, **kwargs)

    # Save detailed trace
    prof.export_chrome_trace(f"trace_{workload_name}.json")

    # Print summary
    print(f"\n{'='*80}")
    print(f"PyTorch Profiler Summary: {workload_name}")
    print('='*80)
    print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=20))

    return result


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
