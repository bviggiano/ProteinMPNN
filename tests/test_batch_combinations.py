"""
Test different combinations of num_seq_per_target and batch_size to identify bugs.

This test file checks that the ProteinMPNN wrapper correctly generates the expected
number of sequences for various parameter combinations.
"""

import pytest
from protein_mpnn import ProteinMPNN


class TestBatchCombinations:
    """Test various combinations of num_seq_per_target and batch_size."""

    @pytest.fixture
    def model(self):
        """Initialize ProteinMPNN model once for all tests."""
        return ProteinMPNN(model_name='v_48_020', suppress_print=True)

    @pytest.fixture
    def test_pdb(self):
        """Simple test PDB structure."""
        pdb_content = """ATOM      1  N   ALA A   1      -8.901   4.127  -0.555  1.00  0.00           N
ATOM      2  CA  ALA A   1      -8.608   3.135  -1.618  1.00  0.00           C
ATOM      3  C   ALA A   1      -7.117   2.964  -1.897  1.00  0.00           C
ATOM      4  O   ALA A   1      -6.634   1.849  -1.758  1.00  0.00           O
ATOM      5  CB  ALA A   1      -9.437   3.396  -2.889  1.00  0.00           C
ATOM      6  N   GLY A   2      -6.346   4.019  -2.228  1.00  0.00           N
ATOM      7  CA  GLY A   2      -4.923   3.984  -2.537  1.00  0.00           C
ATOM      8  C   GLY A   2      -4.594   4.375  -3.970  1.00  0.00           C
ATOM      9  O   GLY A   2      -5.482   4.654  -4.780  1.00  0.00           O
ATOM     10  N   VAL A   3      -3.306   4.407  -4.321  1.00  0.00           N
ATOM     11  CA  VAL A   3      -2.850   4.762  -5.661  1.00  0.00           C
ATOM     12  C   VAL A   3      -1.337   4.663  -5.798  1.00  0.00           C
ATOM     13  O   VAL A   3      -0.605   4.362  -4.855  1.00  0.00           O
ATOM     14  CB  VAL A   3      -3.274   6.223  -6.054  1.00  0.00           C
ATOM     15  CG1 VAL A   3      -2.775   6.556  -7.464  1.00  0.00           C
ATOM     16  CG2 VAL A   3      -4.799   6.405  -5.989  1.00  0.00           C
"""
        return pdb_content

    def test_num_seq_1_batch_1(self, model, test_pdb):
        """Test: num_seq_per_target=1, batch_size=1 (expected: 1 sequence)."""
        results = model.sample(
            pdb_path_or_str=test_pdb,
            num_seq_per_target=1,
            batch_size=1,
            sampling_temp='0.1',
            seed=42
        )

        protein_name = list(results.keys())[0]
        num_sequences = len(results[protein_name]['sequences'])

        assert num_sequences == 1, f"Expected 1 sequence, got {num_sequences}"

    def test_num_seq_1_batch_2(self, model, test_pdb):
        """Test: num_seq_per_target=1, batch_size=2 (expected: 1 sequence) - LIKELY BUG."""
        results = model.sample(
            pdb_path_or_str=test_pdb,
            num_seq_per_target=1,
            batch_size=2,
            sampling_temp='0.1',
            seed=42
        )

        protein_name = list(results.keys())[0]
        num_sequences = len(results[protein_name]['sequences'])

        assert num_sequences == 1, f"Expected 1 sequence, got {num_sequences} (BUG: batch_size should not affect output count)"

    def test_num_seq_1_batch_5(self, model, test_pdb):
        """Test: num_seq_per_target=1, batch_size=5 (expected: 1 sequence) - LIKELY BUG."""
        results = model.sample(
            pdb_path_or_str=test_pdb,
            num_seq_per_target=1,
            batch_size=5,
            sampling_temp='0.1',
            seed=42
        )

        protein_name = list(results.keys())[0]
        num_sequences = len(results[protein_name]['sequences'])

        assert num_sequences == 1, f"Expected 1 sequence, got {num_sequences} (BUG: batch_size should not affect output count)"

    def test_num_seq_4_batch_1(self, model, test_pdb):
        """Test: num_seq_per_target=4, batch_size=1 (expected: 4 sequences)."""
        results = model.sample(
            pdb_path_or_str=test_pdb,
            num_seq_per_target=4,
            batch_size=1,
            sampling_temp='0.1',
            seed=42
        )

        protein_name = list(results.keys())[0]
        num_sequences = len(results[protein_name]['sequences'])

        assert num_sequences == 4, f"Expected 4 sequences, got {num_sequences}"

    def test_num_seq_4_batch_2(self, model, test_pdb):
        """Test: num_seq_per_target=4, batch_size=2 (expected: 4 sequences)."""
        results = model.sample(
            pdb_path_or_str=test_pdb,
            num_seq_per_target=4,
            batch_size=2,
            sampling_temp='0.1',
            seed=42
        )

        protein_name = list(results.keys())[0]
        num_sequences = len(results[protein_name]['sequences'])

        assert num_sequences == 4, f"Expected 4 sequences, got {num_sequences}"

    def test_num_seq_10_batch_5(self, model, test_pdb):
        """Test: num_seq_per_target=10, batch_size=5 (expected: 10 sequences)."""
        results = model.sample(
            pdb_path_or_str=test_pdb,
            num_seq_per_target=10,
            batch_size=5,
            sampling_temp='0.1',
            seed=42
        )

        protein_name = list(results.keys())[0]
        num_sequences = len(results[protein_name]['sequences'])

        assert num_sequences == 10, f"Expected 10 sequences, got {num_sequences}"

    def test_num_seq_8_batch_4(self, model, test_pdb):
        """Test: num_seq_per_target=8, batch_size=4 (expected: 8 sequences)."""
        results = model.sample(
            pdb_path_or_str=test_pdb,
            num_seq_per_target=8,
            batch_size=4,
            sampling_temp='0.1',
            seed=42
        )

        protein_name = list(results.keys())[0]
        num_sequences = len(results[protein_name]['sequences'])

        assert num_sequences == 8, f"Expected 8 sequences, got {num_sequences}"

    def test_num_seq_7_batch_3(self, model, test_pdb):
        """Test: num_seq_per_target=7, batch_size=3 (expected: 9 sequences - rounds up to next multiple)."""
        results = model.sample(
            pdb_path_or_str=test_pdb,
            num_seq_per_target=7,
            batch_size=3,
            sampling_temp='0.1',
            seed=42
        )

        protein_name = list(results.keys())[0]
        num_sequences = len(results[protein_name]['sequences'])

        # 7 sequences with batch_size=3 requires 3 batches (ceil(7/3) = 3), generating 9 sequences
        assert num_sequences == 9, f"Expected 9 sequences (3 batches × 3), got {num_sequences}"

    def test_num_seq_5_batch_3(self, model, test_pdb):
        """Test: num_seq_per_target=5, batch_size=3 (expected: 6 sequences - rounds up to next multiple)."""
        results = model.sample(
            pdb_path_or_str=test_pdb,
            num_seq_per_target=5,
            batch_size=3,
            sampling_temp='0.1',
            seed=42
        )

        protein_name = list(results.keys())[0]
        num_sequences = len(results[protein_name]['sequences'])

        # 5 sequences with batch_size=3 requires 2 batches (ceil(5/3) = 2), generating 6 sequences
        assert num_sequences == 6, f"Expected 6 sequences (2 batches × 3), got {num_sequences}"

    def test_all_output_fields_present(self, model, test_pdb):
        """Test that all expected output fields are present."""
        results = model.sample(
            pdb_path_or_str=test_pdb,
            num_seq_per_target=3,
            batch_size=1,
            sampling_temp='0.1',
            seed=42
        )

        protein_name = list(results.keys())[0]
        result = results[protein_name]

        # Check all expected fields are present
        expected_fields = [
            'native_sequence', 'sequences', 'scores', 'global_scores',
            'seq_recovery_rates', 'sampling_temperatures', 'native_score',
            'global_native_score', 'seed', 'designed_chains', 'fixed_chains'
        ]

        for field in expected_fields:
            assert field in result, f"Missing expected field: {field}"

        # Check array lengths match
        num_seqs = len(result['sequences'])
        assert len(result['scores']) == num_seqs, "scores length mismatch"
        assert len(result['global_scores']) == num_seqs, "global_scores length mismatch"
        assert len(result['seq_recovery_rates']) == num_seqs, "seq_recovery_rates length mismatch"
        assert len(result['sampling_temperatures']) == num_seqs, "sampling_temperatures length mismatch"

    def test_multiple_temperatures(self, model, test_pdb):
        """Test: Multiple temperatures should generate num_seq_per_target sequences PER temperature."""
        results = model.sample(
            pdb_path_or_str=test_pdb,
            num_seq_per_target=4,
            batch_size=2,
            sampling_temp='0.1 0.2 0.5',  # 3 temperatures
            seed=42
        )

        protein_name = list(results.keys())[0]
        num_sequences = len(results[protein_name]['sequences'])
        temps = results[protein_name]['sampling_temperatures']

        # Should get 4 sequences per temperature = 12 total
        assert num_sequences == 12, f"Expected 12 sequences (4 per temp × 3 temps), got {num_sequences}"

        # Check temperature distribution
        from collections import Counter
        temp_counts = Counter(temps)
        for temp in [0.1, 0.2, 0.5]:
            assert temp_counts[temp] == 4, f"Expected 4 sequences at temp {temp}, got {temp_counts[temp]}"


if __name__ == "__main__":
    # Run tests with verbose output
    pytest.main([__file__, "-v", "-s"])
