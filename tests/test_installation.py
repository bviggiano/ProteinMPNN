#!/usr/bin/env python
"""
Test suite to verify that ProteinMPNN is correctly installed via pip
and that model weights are included in the installation.
"""

import os
import torch


def test_module_imports():
    """Test that core modules can be imported"""
    import protein_mpnn_utils
    import protein_mpnn_run
    import helper_scripts

    assert protein_mpnn_utils is not None
    assert protein_mpnn_run is not None
    assert helper_scripts is not None


def test_weights_are_from_pip_installation():
    """Verify that weights are loaded from pip installation, not from repo"""
    import protein_mpnn_run

    # Get the path where protein_mpnn_run is installed
    module_path = os.path.dirname(os.path.abspath(protein_mpnn_run.__file__))
    print(f"\nProteinMPNN module installed at: {module_path}")

    # Check that we're not in the source directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    # Expected weight paths relative to the installed module
    weight_dirs = {
        "vanilla": {
            "path": os.path.join(module_path, "vanilla_model_weights"),
            "files": ["v_48_002.pt", "v_48_010.pt", "v_48_020.pt", "v_48_030.pt"],
        },
        "ca": {
            "path": os.path.join(module_path, "ca_model_weights"),
            "files": ["v_48_002.pt", "v_48_010.pt", "v_48_020.pt"],
        },
        "soluble": {
            "path": os.path.join(module_path, "soluble_model_weights"),
            "files": [
                "v_48_002.pt",
                "v_48_010.pt",
                "v_48_020.pt",
                "v_48_030.pt",
            ],
        },
    }

    # Check that model weights exist in the installed package
    model_files_found = []
    for weight_type, weight_dict in weight_dirs.items():
        print(f"\nChecking {weight_type} weights at: {weight_dict['path']}")

        assert os.path.exists(
            weight_dict["path"]
        ), f"Weight directory not found: {weight_dict['path']}"

        # List weight files
        weight_files = [f for f in os.listdir(weight_dict["path"]) if f.endswith(".pt")]

        assert len(weight_files) > 0, f"No .pt files found in {weight_dict['path']}"

        assert set(weight_files) == set(
            weight_dict["files"]
        ), f"Weight files do not match: {weight_files} != {weight_dict['files']}"

        print(f"✓ Found {len(weight_files)} weight file(s): {', '.join(weight_files)}")
        model_files_found.extend(weight_files)

    assert len(model_files_found) > 0, "No model weight files found"
    print(f"\n✓ Total weight files found: {len(model_files_found)}")


def test_model_loading():
    """Test that a model can be successfully loaded from the pip installation"""
    import protein_mpnn_run

    # Get the path where protein_mpnn_run is installed
    module_path = os.path.dirname(os.path.abspath(protein_mpnn_run.__file__))

    # Try loading a model to verify it works
    test_model_path = os.path.join(module_path, "vanilla_model_weights", "v_48_020.pt")

    assert os.path.exists(test_model_path), f"Test model not found: {test_model_path}"

    # Load the model
    checkpoint = torch.load(test_model_path, map_location="cpu")

    assert checkpoint is not None, "Failed to load checkpoint"
    assert isinstance(checkpoint, dict), "Checkpoint should be a dictionary"

    print(f"\n✓ Successfully loaded model from: {test_model_path}")
    print(f"  Model keys: {list(checkpoint.keys())[:5]}...")
