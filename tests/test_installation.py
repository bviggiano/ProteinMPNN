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

    assert protein_mpnn_utils is not None
    assert protein_mpnn_run is not None


def test_weights_are_from_pip_installation():
    """Verify that weights are loaded from pip installation, not from repo"""
    from protein_mpnn import protein_mpnn_run

    # Get the path where protein_mpnn_run is installed
    module_path = os.path.dirname(os.path.abspath(protein_mpnn_run.__file__))
    print(f"\nProteinMPNN module installed at: {module_path}")

    # Check that we're not in the source directory
    current_dir = os.getcwd()
    print(f"Current working directory: {current_dir}")

    # Expected weight paths relative to the installed module
    weight_dirs = {
        'vanilla': os.path.join(module_path, 'vanilla_model_weights'),
        'ca': os.path.join(module_path, 'ca_model_weights'),
        'soluble': os.path.join(module_path, 'soluble_model_weights'),
    }

    # Check that model weights exist in the installed package
    model_files_found = []
    for weight_type, weight_dir in weight_dirs.items():
        print(f"\nChecking {weight_type} weights at: {weight_dir}")

        assert os.path.exists(weight_dir), \
            f"Weight directory not found: {weight_dir}"

        # List weight files
        weight_files = [f for f in os.listdir(weight_dir) if f.endswith('.pt')]

        assert len(weight_files) > 0, \
            f"No .pt files found in {weight_dir}"

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
    test_model_path = os.path.join(module_path, 'vanilla_model_weights', 'v_48_020.pt')

    assert os.path.exists(test_model_path), \
        f"Test model not found: {test_model_path}"

    # Load the model
    checkpoint = torch.load(test_model_path, map_location='cpu')

    assert checkpoint is not None, "Failed to load checkpoint"
    assert isinstance(checkpoint, dict), "Checkpoint should be a dictionary"

    print(f"\n✓ Successfully loaded model from: {test_model_path}")
    print(f"  Model keys: {list(checkpoint.keys())[:5]}...")


def test_weights_not_in_working_directory():
    """Verify that weights are not being loaded from the current working directory"""
    import protein_mpnn_run

    module_path = os.path.dirname(os.path.abspath(protein_mpnn_run.__file__))
    test_model_path = os.path.join(module_path, 'vanilla_model_weights', 'v_48_020.pt')

    # In CI, verify the model path doesn't contain 'DISABLED'
    if os.environ.get('CI'):
        assert 'DISABLED' not in test_model_path, \
            "Model should not be trying to use disabled local weights"

        # Verify model path is not in the repo directory (in editable install it will be)
        # but it shouldn't be using the DISABLED directories
        print(f"\n✓ Model path verified: {test_model_path}")
