#!/usr/bin/env python
"""
Test suite for ProteinMPNN inference functionality.
"""

import tempfile
from pathlib import Path
import pytest


@pytest.mark.inference
@pytest.mark.slow
def test_simple_inference():
    """Test that ProteinMPNN can run inference on a test PDB file"""
    import protein_mpnn_run
    import argparse

    # Check if test input exists
    test_pdb = 'inputs/PDB_monomers/pdbs/6MRR.pdb'

    # Create temporary output directory
    with tempfile.TemporaryDirectory() as temp_dir:
        # Set up arguments
        args = argparse.Namespace(
            suppress_print=1,
            ca_only=False,
            path_to_model_weights='',
            model_name='v_48_020',
            use_soluble_model=False,
            seed=42,
            save_score=0,
            save_probs=0,
            score_only=0,
            path_to_fasta='',
            conditional_probs_only=0,
            conditional_probs_only_backbone=0,
            unconditional_probs_only=0,
            backbone_noise=0.0,
            num_seq_per_target=2,
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

        # Run inference
        print(f"\nRunning inference on {test_pdb}")
        protein_mpnn_run.main(args)

        # Check that output was generated
        output_dir = Path(temp_dir) / 'seqs'
        assert output_dir.exists(), f"Output directory not created: {output_dir}"

        output_file = output_dir / '6MRR.fa'
        assert output_file.exists(), f"Output file not created: {output_file}"

        # Verify output file has content
        with open(output_file, 'r') as f:
            content = f.read()
            assert len(content) > 0, "Output file is empty"
            assert '>' in content, "Output file doesn't appear to be in FASTA format"

        print(f"✓ Inference test passed")
        print(f"✓ Output generated at: {output_file}")
