#!/usr/bin/env python
"""
Test suite to verify consistency between CLI implementation and model wrapper class.

This ensures that the new ProteinMPNN wrapper class produces identical results
to the command-line interface for the same seeds and inputs.
"""

import tempfile
from pathlib import Path
import pytest
import numpy as np
import argparse
import os


@pytest.mark.inference
def test_sample_consistency():
    """Test that the sample() method produces identical results to CLI"""
    import protein_mpnn_run
    from protein_mpnn import ProteinMPNN

    test_pdb = 'inputs/PDB_monomers/pdbs/6MRR.pdb'
    seed = 42
    num_seq_per_target = 4
    batch_size = 2
    sampling_temp = "0.1 0.2"

    # Run CLI version
    with tempfile.TemporaryDirectory() as temp_dir:
        args = argparse.Namespace(
            suppress_print=1,
            ca_only=False,
            path_to_model_weights='',
            model_name='v_48_020',
            use_soluble_model=False,
            seed=seed,
            save_score=1,
            save_probs=0,
            score_only=0,
            path_to_fasta='',
            conditional_probs_only=0,
            conditional_probs_only_backbone=0,
            unconditional_probs_only=0,
            backbone_noise=0.0,
            num_seq_per_target=num_seq_per_target,
            batch_size=batch_size,
            max_length=200000,
            sampling_temp=sampling_temp,
            out_folder=temp_dir,
            pdb_path=test_pdb,
            pdb_path_chains='',
            jsonl_path='',
            chain_id_jsonl='',
            fixed_positions_jsonl='',
            omit_AAs='X',
            bias_AA_jsonl='',
            bias_by_res_jsonl='',
            omit_AA_jsonl='',
            pssm_jsonl='',
            pssm_multi=0.0,
            pssm_threshold=0.0,
            pssm_log_odds_flag=0,
            pssm_bias_flag=0,
            tied_positions_jsonl='',
        )

        protein_mpnn_run.main(args)

        # Read CLI output
        output_file = Path(temp_dir) / 'seqs' / '6MRR.fa'
        with open(output_file, 'r') as f:
            cli_content = f.read()

        # Parse FASTA file
        cli_sequences = []
        cli_scores = []
        cli_global_scores = []
        cli_native_seq = None

        lines = cli_content.strip().split('\n')
        for i in range(0, len(lines), 2):
            header = lines[i]
            seq = lines[i+1] if i+1 < len(lines) else ""

            if i == 0:
                # Native sequence
                cli_native_seq = seq
                # Extract native score from header
                score_idx = header.find('score=')
                global_score_idx = header.find('global_score=')
                if score_idx != -1:
                    score_end = header.find(',', score_idx)
                    native_score_str = header[score_idx+6:score_end]
                if global_score_idx != -1:
                    score_start = global_score_idx + 13
                    score_end = header.find(',', score_start)
                    native_global_score_str = header[score_start:score_end]
            else:
                cli_sequences.append(seq)
                # Extract score from header
                # Format: >T=0.1, sample=1, score=2.1234, global_score=3.4567, seq_recovery=0.8765
                parts = header.split(', ')
                for part in parts:
                    if part.startswith('score='):
                        cli_scores.append(float(part.split('=')[1]))
                    elif part.startswith('global_score='):
                        cli_global_scores.append(float(part.split('=')[1]))

        # Read score file
        score_file = Path(temp_dir) / 'scores' / '6MRR.npz'
        cli_score_data = np.load(score_file)
        cli_scores_array = cli_score_data['score']
        cli_global_scores_array = cli_score_data['global_score']

    # Run wrapper version
    model = ProteinMPNN(
        model_name='v_48_020',
        suppress_print=True
    )

    wrapper_results = model.sample(
        pdb_path=test_pdb,
        num_seq_per_target=num_seq_per_target,
        batch_size=batch_size,
        sampling_temp=sampling_temp,
        seed=seed
    )

    # Get results
    protein_name = list(wrapper_results.keys())[0]
    wrapper_data = wrapper_results[protein_name]

    # Compare native sequences
    assert wrapper_data['native_sequence'] == cli_native_seq, \
        f"Native sequences don't match:\nWrapper: {wrapper_data['native_sequence']}\nCLI: {cli_native_seq}"

    # Compare generated sequences
    assert len(wrapper_data['sequences']) == len(cli_sequences), \
        f"Number of sequences don't match: {len(wrapper_data['sequences'])} vs {len(cli_sequences)}"

    for i, (wrapper_seq, cli_seq) in enumerate(zip(wrapper_data['sequences'], cli_sequences)):
        assert wrapper_seq == cli_seq, \
            f"Sequence {i} doesn't match:\nWrapper: {wrapper_seq}\nCLI: {cli_seq}"

    # Compare scores (allow small floating point differences)
    np.testing.assert_allclose(
        wrapper_data['scores'],
        cli_scores_array,
        rtol=1e-5,
        err_msg="Scores don't match between wrapper and CLI"
    )

    np.testing.assert_allclose(
        wrapper_data['global_scores'],
        cli_global_scores_array,
        rtol=1e-5,
        err_msg="Global scores don't match between wrapper and CLI"
    )

    print(f"\n✓ Sample consistency test passed")
    print(f"  - Generated {len(wrapper_data['sequences'])} sequences")
    print(f"  - All sequences match CLI output")
    print(f"  - All scores match CLI output")


@pytest.mark.inference
def test_score_consistency():
    """Test that the score() method produces identical results to CLI"""
    import protein_mpnn_run
    from protein_mpnn import ProteinMPNN

    test_pdb = 'inputs/PDB_monomers/pdbs/6MRR.pdb'
    seed = 42
    num_batches = 3

    # Run CLI version
    with tempfile.TemporaryDirectory() as temp_dir:
        args = argparse.Namespace(
            suppress_print=1,
            ca_only=False,
            path_to_model_weights='',
            model_name='v_48_020',
            use_soluble_model=False,
            seed=seed,
            save_score=0,
            save_probs=0,
            score_only=1,
            path_to_fasta='',
            conditional_probs_only=0,
            conditional_probs_only_backbone=0,
            unconditional_probs_only=0,
            backbone_noise=0.0,
            num_seq_per_target=num_batches,
            batch_size=1,
            max_length=200000,
            sampling_temp='0.1',
            out_folder=temp_dir,
            pdb_path=test_pdb,
            pdb_path_chains='',
            jsonl_path='',
            chain_id_jsonl='',
            fixed_positions_jsonl='',
            omit_AAs='X',
            bias_AA_jsonl='',
            bias_by_res_jsonl='',
            omit_AA_jsonl='',
            pssm_jsonl='',
            pssm_multi=0.0,
            pssm_threshold=0.0,
            pssm_log_odds_flag=0,
            pssm_bias_flag=0,
            tied_positions_jsonl='',
        )

        protein_mpnn_run.main(args)

        # Read CLI output
        score_file = Path(temp_dir) / 'score_only' / '6MRR_pdb.npz'
        cli_data = np.load(score_file)
        cli_scores = cli_data['score']
        cli_global_scores = cli_data['global_score']

    # Run wrapper version
    model = ProteinMPNN(
        model_name='v_48_020',
        suppress_print=True
    )

    wrapper_results = model.score(
        pdb_path=test_pdb,
        num_batches=num_batches,
        seed=seed
    )

    # Get results
    protein_name = list(wrapper_results.keys())[0]
    wrapper_data = wrapper_results[protein_name]['pdb_scores']

    # Compare scores
    np.testing.assert_allclose(
        wrapper_data['scores'],
        cli_scores,
        rtol=1e-5,
        err_msg="Scores don't match between wrapper and CLI"
    )

    np.testing.assert_allclose(
        wrapper_data['global_scores'],
        cli_global_scores,
        rtol=1e-5,
        err_msg="Global scores don't match between wrapper and CLI"
    )

    # Compare statistics
    assert abs(wrapper_data['mean_score'] - cli_scores.mean()) < 1e-5, \
        "Mean scores don't match"
    assert abs(wrapper_data['std_score'] - cli_scores.std()) < 1e-5, \
        "Std scores don't match"

    print(f"\n✓ Score consistency test passed")
    print(f"  - Mean score: {wrapper_data['mean_score']:.4f}")
    print(f"  - Std score: {wrapper_data['std_score']:.4f}")
    print(f"  - Sample size: {wrapper_data['sample_size']}")


@pytest.mark.inference
def test_conditional_probs_consistency():
    """Test that conditional_probs() produces identical results to CLI"""
    import protein_mpnn_run
    from protein_mpnn import ProteinMPNN

    test_pdb = 'inputs/PDB_monomers/pdbs/6MRR.pdb'
    seed = 42
    num_batches = 2

    # Run CLI version
    with tempfile.TemporaryDirectory() as temp_dir:
        args = argparse.Namespace(
            suppress_print=1,
            ca_only=False,
            path_to_model_weights='',
            model_name='v_48_020',
            use_soluble_model=False,
            seed=seed,
            save_score=0,
            save_probs=0,
            score_only=0,
            path_to_fasta='',
            conditional_probs_only=1,
            conditional_probs_only_backbone=0,
            unconditional_probs_only=0,
            backbone_noise=0.0,
            num_seq_per_target=num_batches,
            batch_size=1,
            max_length=200000,
            sampling_temp='0.1',
            out_folder=temp_dir,
            pdb_path=test_pdb,
            pdb_path_chains='',
            jsonl_path='',
            chain_id_jsonl='',
            fixed_positions_jsonl='',
            omit_AAs='X',
            bias_AA_jsonl='',
            bias_by_res_jsonl='',
            omit_AA_jsonl='',
            pssm_jsonl='',
            pssm_multi=0.0,
            pssm_threshold=0.0,
            pssm_log_odds_flag=0,
            pssm_bias_flag=0,
            tied_positions_jsonl='',
        )

        protein_mpnn_run.main(args)

        # Read CLI output
        probs_file = Path(temp_dir) / 'conditional_probs_only' / '6MRR.npz'
        cli_data = np.load(probs_file)
        cli_log_probs = cli_data['log_p']
        cli_sequence = cli_data['S']
        cli_mask = cli_data['mask']
        cli_design_mask = cli_data['design_mask']

    # Run wrapper version
    model = ProteinMPNN(
        model_name='v_48_020',
        suppress_print=True
    )

    wrapper_results = model.conditional_probs(
        pdb_path=test_pdb,
        num_batches=num_batches,
        backbone_only=False,
        seed=seed
    )

    # Get results
    protein_name = list(wrapper_results.keys())[0]
    wrapper_data = wrapper_results[protein_name]

    # Compare log probabilities
    np.testing.assert_allclose(
        wrapper_data['log_probs'],
        cli_log_probs,
        rtol=1e-5,
        err_msg="Log probabilities don't match between wrapper and CLI"
    )

    # Compare sequence
    np.testing.assert_array_equal(
        wrapper_data['sequence'],
        cli_sequence,
        err_msg="Sequences don't match"
    )

    # Compare masks
    np.testing.assert_array_equal(
        wrapper_data['mask'],
        cli_mask,
        err_msg="Masks don't match"
    )

    np.testing.assert_array_equal(
        wrapper_data['design_mask'],
        cli_design_mask,
        err_msg="Design masks don't match"
    )

    print(f"\n✓ Conditional probs consistency test passed")
    print(f"  - Log probs shape: {wrapper_data['log_probs'].shape}")
    print(f"  - Sequence length: {len(wrapper_data['sequence'])}")


@pytest.mark.inference
def test_unconditional_probs_consistency():
    """Test that unconditional_probs() produces identical results to CLI"""
    import protein_mpnn_run
    from protein_mpnn import ProteinMPNN

    test_pdb = 'inputs/PDB_monomers/pdbs/6MRR.pdb'
    seed = 42
    num_batches = 2

    # Run CLI version
    with tempfile.TemporaryDirectory() as temp_dir:
        args = argparse.Namespace(
            suppress_print=1,
            ca_only=False,
            path_to_model_weights='',
            model_name='v_48_020',
            use_soluble_model=False,
            seed=seed,
            save_score=0,
            save_probs=0,
            score_only=0,
            path_to_fasta='',
            conditional_probs_only=0,
            conditional_probs_only_backbone=0,
            unconditional_probs_only=1,
            backbone_noise=0.0,
            num_seq_per_target=num_batches,
            batch_size=1,
            max_length=200000,
            sampling_temp='0.1',
            out_folder=temp_dir,
            pdb_path=test_pdb,
            pdb_path_chains='',
            jsonl_path='',
            chain_id_jsonl='',
            fixed_positions_jsonl='',
            omit_AAs='X',
            bias_AA_jsonl='',
            bias_by_res_jsonl='',
            omit_AA_jsonl='',
            pssm_jsonl='',
            pssm_multi=0.0,
            pssm_threshold=0.0,
            pssm_log_odds_flag=0,
            pssm_bias_flag=0,
            tied_positions_jsonl='',
        )

        protein_mpnn_run.main(args)

        # Read CLI output
        probs_file = Path(temp_dir) / 'unconditional_probs_only' / '6MRR.npz'
        cli_data = np.load(probs_file)
        cli_log_probs = cli_data['log_p']
        cli_sequence = cli_data['S']
        cli_mask = cli_data['mask']
        cli_design_mask = cli_data['design_mask']

    # Run wrapper version
    model = ProteinMPNN(
        model_name='v_48_020',
        suppress_print=True
    )

    wrapper_results = model.unconditional_probs(
        pdb_path=test_pdb,
        num_batches=num_batches,
        seed=seed
    )

    # Get results
    protein_name = list(wrapper_results.keys())[0]
    wrapper_data = wrapper_results[protein_name]

    # Compare log probabilities
    np.testing.assert_allclose(
        wrapper_data['log_probs'],
        cli_log_probs,
        rtol=1e-5,
        err_msg="Log probabilities don't match between wrapper and CLI"
    )

    # Compare sequence
    np.testing.assert_array_equal(
        wrapper_data['sequence'],
        cli_sequence,
        err_msg="Sequences don't match"
    )

    # Compare masks
    np.testing.assert_array_equal(
        wrapper_data['mask'],
        cli_mask,
        err_msg="Masks don't match"
    )

    np.testing.assert_array_equal(
        wrapper_data['design_mask'],
        cli_design_mask,
        err_msg="Design masks don't match"
    )

    print(f"\n✓ Unconditional probs consistency test passed")
    print(f"  - Log probs shape: {wrapper_data['log_probs'].shape}")
    print(f"  - Sequence length: {len(wrapper_data['sequence'])}")


@pytest.mark.inference
def test_model_reusability():
    """Test that the model can be reused for multiple inferences"""
    from protein_mpnn import ProteinMPNN

    test_pdb = 'inputs/PDB_monomers/pdbs/6MRR.pdb'

    # Create model once
    model = ProteinMPNN(
        model_name='v_48_020',
        suppress_print=True
    )

    # Run multiple times with different seeds
    results1 = model.sample(pdb_path=test_pdb, num_seq_per_target=2, seed=42)
    results2 = model.sample(pdb_path=test_pdb, num_seq_per_target=2, seed=43)
    results3 = model.score(pdb_path=test_pdb, num_batches=2, seed=42)

    # Verify that different seeds produce different results
    protein_name = list(results1.keys())[0]
    assert results1[protein_name]['sequences'][0] != results2[protein_name]['sequences'][0], \
        "Different seeds should produce different sequences"

    # Verify that same seed produces same results
    results4 = model.sample(pdb_path=test_pdb, num_seq_per_target=2, seed=42)
    assert results1[protein_name]['sequences'] == results4[protein_name]['sequences'], \
        "Same seed should produce same sequences"

    print(f"\n✓ Model reusability test passed")
    print(f"  - Model can be used multiple times")
    print(f"  - Different seeds produce different results")
    print(f"  - Same seeds produce identical results")


if __name__ == '__main__':
    # Run tests
    test_sample_consistency()
    test_score_consistency()
    test_conditional_probs_consistency()
    test_unconditional_probs_consistency()
    test_model_reusability()
    print("\n" + "="*50)
    print("All consistency tests passed!")
    print("="*50)
