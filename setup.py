"""
Setup file for nativeIMS package.

This allows the package to be installed with:
    pip install .
    pip install -e .  (for development mode)
    pip install .[web]  (with web interface dependencies)
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8")

setup(
    name="nativeims",
    version="0.1.0",
    
    # Package info
    description="A Python library for native ion mobility mass spectrometry data processing",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author info
    author="Your Name",  # TODO: Update this
    author_email="your.email@example.com",  # TODO: Update this
    
    # URLs
    url="https://github.com/yourusername/nativeIMS-processing-tools",  # TODO: Update this
    
    # License
    license="MIT",  # TODO: Update if different
    
    # Find all packages automatically
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    
    # Python version requirement
    python_requires=">=3.7",
    
    # Core dependencies (required for the library to work)
    install_requires=[
        "pandas>=1.0.0",
        "numpy>=1.18.0",
        "scipy>=1.4.0",
        "matplotlib>=3.1.0",
    ],
    
    # Optional dependencies
    # Install with: pip install .[web]
    extras_require={
        "web": [
            "streamlit>=1.0.0",
        ],
        "dev": [
            "pytest>=6.0.0",
            "pytest-cov>=2.0.0",
            "black>=20.0.0",
            "flake8>=3.8.0",
        ],
    },
    
    # Include data files
    package_data={
        "": ["*.csv", "*.txt", "*.md"],
    },
    
    # Classifiers help users find your project
    classifiers=[
        "Development Status :: 3 - Alpha",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.7",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
    ],
    
    # Keywords for searching
    keywords="mass-spectrometry ion-mobility proteomics calibration",
)
