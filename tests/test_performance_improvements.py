"""
Tests for performance improvements: auto-batching, caching, and compilation.

Run with:
    pytest test_performance_improvements.py -v -s
"""

import pytest
import time
import torch
from pathlib import Path
from protein_mpnn import ProteinMPNN


@pytest.fixture(scope="module")
def test_pdb_path():
    """Path to test PDB file (6MRR - 68 residue protein)."""
    pdb_path = Path(__file__).parent / "inputs" / "PDB_monomers" / "pdbs" / "6MRR.pdb"
    if not pdb_path.exists():
        # Try relative to repo root
        pdb_path = Path(__file__).parent.parent / "inputs" / "PDB_monomers" / "pdbs" / "6MRR.pdb"

    assert pdb_path.exists(), f"Test PDB file not found at {pdb_path}"
    return str(pdb_path)


class TestAutoBatching:
    """Test automatic batch size selection."""

    def test_auto_batching_enabled_on_gpu(self, test_pdb_path):
        """Auto-batching should activate when batch_size not specified and num_seq > 1."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = ProteinMPNN(
            model_name='v_48_020',
            device='cuda',
            suppress_print=False  # Enable to see auto-batch message
        )

        # Don't specify batch_size - should auto-select
        result = model.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=10,
            # batch_size parameter omitted - defaults to 1
            sampling_temp="0.1",
            seed=42
        )

        protein_name = list(result.keys())[0]
        assert len(result[protein_name]['sequences']) == 10

    def test_manual_batch_size_override(self, test_pdb_path):
        """Manual batch_size should override auto-selection."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = ProteinMPNN(
            model_name='v_48_020',
            device='cuda',
            suppress_print=True
        )

        # Explicitly set batch_size - should not auto-select
        result = model.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=10,
            batch_size=2,  # Explicit override
            sampling_temp="0.1",
            seed=42
        )

        protein_name = list(result.keys())[0]
        assert len(result[protein_name]['sequences']) == 10

    def test_auto_batching_performance_gain(self, test_pdb_path):
        """Auto-batching should provide speedup for larger workloads."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        model = ProteinMPNN(
            model_name='v_48_020',
            device='cuda',
            suppress_print=True
        )

        # Time with explicit batch_size=1
        start = time.time()
        result1 = model.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=16,
            batch_size=1,
            sampling_temp="0.1",
            seed=42
        )
        time_no_batching = time.time() - start

        model = ProteinMPNN(
            model_name='v_48_020',
            device='cuda',
            suppress_print=True
        )

        # Time with larger batch_size (divisible)
        start = time.time()
        result2 = model.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=16,
            batch_size=8,
            sampling_temp="0.1",
            seed=43
        )
        time_with_batching = time.time() - start

        # Should be faster with batching
        speedup = time_no_batching / time_with_batching
        print(f"\nBatching speedup: {speedup:.2f}x ({time_no_batching:.2f}s -> {time_with_batching:.2f}s)")

        # Verify correct number of sequences
        assert len(result1[list(result1.keys())[0]]['sequences']) == 16
        assert len(result2[list(result2.keys())[0]]['sequences']) == 16

        # Should see at least 1.5x speedup with batch_size=8 vs 1
        assert speedup > 1.5, f"Expected >1.5x speedup, got {speedup:.2f}x"


class TestTorchCompile:
    """Test torch.compile performance improvements."""

    def test_compile_speedup(self, test_pdb_path):
        """torch.compile should provide speedup for inference."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if not hasattr(torch, 'compile'):
            pytest.skip("torch.compile not available (requires PyTorch 2.0+)")

        # Time without compilation
        model_no_compile = ProteinMPNN(
            model_name='v_48_020',
            device='cuda',
            compile_model=False,
            suppress_print=True
        )

        # Warmup run
        _ = model_no_compile.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=2,
            batch_size=2,
            sampling_temp="0.1",
            seed=42
        )

        # Timed run without compilation
        start = time.time()
        result_no_compile = model_no_compile.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=16,
            batch_size=8,
            sampling_temp="0.1",
            seed=42
        )
        time_no_compile = time.time() - start

        # Time with compilation
        model_with_compile = ProteinMPNN(
            model_name='v_48_020',
            device='cuda',
            compile_model=True,
            suppress_print=True
        )

        # Warmup run (compilation happens here - will be slower)
        print("\n[Compilation warmup - this will be slow]")
        _ = model_with_compile.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=2,
            batch_size=2,
            sampling_temp="0.1",
            seed=42
        )

        # Timed run with compilation (should be fast now)
        start = time.time()
        result_with_compile = model_with_compile.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=16,
            batch_size=8,
            sampling_temp="0.1",
            seed=42
        )
        time_with_compile = time.time() - start

        # Calculate speedup
        speedup = time_no_compile / time_with_compile
        print(f"\ntorch.compile speedup: {speedup:.2f}x ({time_no_compile:.2f}s -> {time_with_compile:.2f}s)")

        # Verify correct number of sequences
        protein_name_1 = list(result_no_compile.keys())[0]
        protein_name_2 = list(result_with_compile.keys())[0]
        assert len(result_no_compile[protein_name_1]['sequences']) == 16
        assert len(result_with_compile[protein_name_2]['sequences']) == 16

        # torch.compile should provide at least 1.2x speedup
        assert speedup > 1.2, f"Expected >1.2x speedup, got {speedup:.2f}x"

    def test_compile_correctness(self, test_pdb_path):
        """torch.compile should produce identical results to non-compiled model."""
        if not torch.cuda.is_available():
            pytest.skip("CUDA not available")

        if not hasattr(torch, 'compile'):
            pytest.skip("torch.compile not available (requires PyTorch 2.0+)")

        # Non-compiled model
        model_no_compile = ProteinMPNN(
            model_name='v_48_020',
            device='cuda',
            compile_model=False,
            suppress_print=True
        )

        result_no_compile = model_no_compile.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=4,
            batch_size=4,
            sampling_temp="0.1",
            seed=42
        )

        # Compiled model
        model_with_compile = ProteinMPNN(
            model_name='v_48_020',
            device='cuda',
            compile_model=True,
            suppress_print=True
        )

        result_with_compile = model_with_compile.sample(
            pdb_path_or_str=test_pdb_path,
            num_seq_per_target=4,
            batch_size=4,
            sampling_temp="0.1",
            seed=42
        )

        # Results should be identical with same seed
        protein_name_1 = list(result_no_compile.keys())[0]
        protein_name_2 = list(result_with_compile.keys())[0]

        sequences_1 = result_no_compile[protein_name_1]['sequences']
        sequences_2 = result_with_compile[protein_name_2]['sequences']

        assert len(sequences_1) == len(sequences_2)
        assert sequences_1 == sequences_2, "torch.compile should produce identical sequences with same seed"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
