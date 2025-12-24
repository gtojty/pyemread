#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Setup script for pyemread package.

Installation:
    pip install -e .
"""

from setuptools import setup, find_packages

with open("README.md", "r", encoding="utf-8") as fh:
    long_description = fh.read()

setup(
    name="pyemread",
    version="2.1.1",
    author="Tao Gong, David Braze",
    author_email="gtojty@gmail.com",
    description="A Python package for multi-line text reading eye movement experiments",
    long_description=long_description,
    long_description_content_type="text/markdown",
    url="https://github.com/gtojty/pyemread",
    packages=find_packages(),
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Human Machine Interfaces",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    python_requires=">=3.8",
    install_requires=[
        "numpy>=1.20.0",
        "pandas>=1.3.0",
        "Pillow>=8.0.0",
        "matplotlib>=3.4.0",
    ],
    extras_require={
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "flake8>=3.9.0",
        ],
    },
    include_package_data=True,
    package_data={
        "pyemread": ["*.py"],
    },
    keywords=[
        "eye-tracking",
        "eye-movements", 
        "reading",
        "psycholinguistics",
        "cognitive science",
        "eyelink",
    ],
    project_urls={
        "Bug Tracker": "https://github.com/gtojty/pyemread/issues",
        "Documentation": "https://github.com/gtojty/pyemread#readme",
        "Source Code": "https://github.com/gtojty/pyemread",
    },
)
