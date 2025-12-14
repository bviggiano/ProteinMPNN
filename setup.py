#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ProteinMPNN: Robust deep learning-based protein sequence design
"""

from setuptools import setup, find_packages
import os

# Read the contents of README file
this_directory = os.path.abspath(os.path.dirname(__file__))
with open(os.path.join(this_directory, 'README.md'), encoding='utf-8') as f:
    long_description = f.read()

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

    # Package data - include model weights
    include_package_data=True,
    package_data={
        '': [
            'vanilla_model_weights/*.pt',
            'ca_model_weights/*.pt',
            'soluble_model_weights/*.pt',
        ],
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
