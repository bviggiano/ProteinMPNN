#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProteinMPNN: Robust deep learning-based protein sequence design
"""

from setuptools import setup, find_packages
from setuptools.command.install import install
from setuptools.command.develop import develop
import os
import sys
import urllib.request
import shutil

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

# GitHub raw URL base
GITHUB_RAW_BASE = "https://raw.githubusercontent.com/dauparas/ProteinMPNN/main"

# Model weights to download
WEIGHT_FILES = {
    'vanilla_model_weights': [
        'v_48_002.pt',
        'v_48_010.pt',
        'v_48_020.pt',
        'v_48_030.pt',
    ],
    'ca_model_weights': [
        'v_48_002.pt',
        'v_48_010.pt',
        'v_48_020.pt',
    ],
    'soluble_model_weights': [
        'v_48_002.pt',
        'v_48_010.pt',
        'v_48_020.pt',
        'v_48_030.pt',
        'excluded_PDBs.csv',
    ],
}

def download_weights(install_lib):
    """Download model weights from GitHub to the installation directory"""
    print("\n" + "="*60)
    print("Downloading ProteinMPNN model weights from GitHub...")
    print("="*60 + "\n")

    for weight_dir, files in WEIGHT_FILES.items():
        # Create weight directory in installation location
        target_dir = os.path.join(install_lib, weight_dir)
        os.makedirs(target_dir, exist_ok=True)

        for filename in files:
            target_path = os.path.join(target_dir, filename)

            # Skip if already exists
            if os.path.exists(target_path):
                print(f"✓ {weight_dir}/{filename} already exists, skipping...")
                continue

            # Download file
            url = f"{GITHUB_RAW_BASE}/{weight_dir}/{filename}"
            print(f"Downloading {weight_dir}/{filename}...", end=' ', flush=True)

            try:
                urllib.request.urlretrieve(url, target_path)
                file_size = os.path.getsize(target_path) / (1024 * 1024)  # MB
                print(f"✓ ({file_size:.1f} MB)")
            except Exception as e:
                print(f"✗ Failed: {e}")
                print(f"You can manually download from: {url}")

    print("\n" + "="*60)
    print("Model weights download complete!")
    print("="*60 + "\n")

class PostInstallCommand(install):
    """Post-installation for installation mode."""
    def run(self):
        install.run(self)
        download_weights(self.install_lib)

class PostDevelopCommand(develop):
    """Post-installation for development mode."""
    def run(self):
        develop.run(self)
        # For editable installs, download to source directory
        download_weights(this_directory)

setup(
    name='protein-mpnn',
    version='1.0.0',
    description='Robust deep learning-based protein sequence design using ProteinMPNN',
    long_description=long_description,
    long_description_content_type='text/markdown',
    author='Justas Dauparas',
    author_email='',
    url='https://github.com/dauparas/ProteinMPNN',
    license='MIT',

    # Python version requirement
    python_requires='>=3.6',

    # Package configuration
    py_modules=[
        'protein_mpnn_run',
        'protein_mpnn_utils',
    ],

    # Include helper scripts as a package
    packages=['helper_scripts'],

    # Custom commands to download weights
    cmdclass={
        'install': PostInstallCommand,
        'develop': PostDevelopCommand,
    },

    # Dependencies
    install_requires=[
        'numpy',
        'torch>=1.8.0',
    ],

    # Test dependencies
    extras_require={
        'test': [
            'pytest>=6.0',
            'pytest-cov',
        ],
    },

    # Classifiers
    classifiers=[
        'Development Status :: 4 - Beta',
        'Intended Audience :: Science/Research',
        'License :: OSI Approved :: MIT License',
        'Programming Language :: Python :: 3',
        'Programming Language :: Python :: 3.6',
        'Programming Language :: Python :: 3.7',
        'Programming Language :: Python :: 3.8',
        'Programming Language :: Python :: 3.9',
        'Programming Language :: Python :: 3.10',
        'Topic :: Scientific/Engineering :: Bio-Informatics',
        'Topic :: Scientific/Engineering :: Artificial Intelligence',
    ],

    # Keywords
    keywords='protein design deep-learning structure biology',

    # Project URLs
    project_urls={
        'Paper': 'https://www.science.org/doi/10.1126/science.add2187',
        'Source': 'https://github.com/dauparas/ProteinMPNN',
    },
)
