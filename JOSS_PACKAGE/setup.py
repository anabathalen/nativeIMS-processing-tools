"""
Setup file for IMSpartacus package.

IMSpartacus: Ion Mobility Spectrometry Processing And Robust Toolkit for Analysis of Collision Cross Sections
"""

from setuptools import setup, find_packages
from pathlib import Path

# Read the README file
this_directory = Path(__file__).parent
long_description = (this_directory / "README.md").read_text(encoding="utf-8") if (this_directory / "README.md").exists() else ""

setup(
    name="imspartacus",
    version="1.0.0",
    
    # Package info
    description="Ion Mobility Spectrometry Processing And Robust Toolkit for Analysis of Collision Cross Sections",
    long_description=long_description,
    long_description_content_type="text/markdown",
    
    # Author info
    author="Your Name",  # TODO: Update this
    author_email="your.email@example.com",  # TODO: Update this
    
    # URLs
    url="https://github.com/yourusername/imspartacus",  # TODO: Update this
    project_urls={
        "Bug Tracker": "https://github.com/yourusername/imspartacus/issues",
        "Documentation": "https://github.com/yourusername/imspartacus#readme",
        "Source Code": "https://github.com/yourusername/imspartacus",
    },
    
    # License
    license="MIT",  # TODO: Update if different
    
    # Find all packages automatically
    packages=find_packages(exclude=["tests", "docs", "examples"]),
    
    # Python version requirement
    python_requires=">=3.8",
    
    # Core dependencies (required for the library to work)
    install_requires=[
        "pandas>=1.3.0",
        "numpy>=1.20.0",
        "scipy>=1.7.0",
        "matplotlib>=3.4.0",
        "plotly>=5.0.0",
        "streamlit>=1.20.0",
        "scikit-learn>=0.24.0",
    ],
    
    # Optional dependencies
    extras_require={
        "dev": [
            "pytest>=7.0.0",
            "pytest-cov>=3.0.0",
            "black>=22.0.0",
            "flake8>=4.0.0",
            "sphinx>=4.0.0",
        ],
    },
    
    # Include data files
    package_data={
        "": ["*.csv", "*.txt", "*.md"],
        "static": ["*.css"],
    },
    include_package_data=True,
    
    # Entry points (optional - for command-line tools)
    entry_points={
        "console_scripts": [
            "imspartacus=app:main",
        ],
    },
    
    # Classifiers help users find your project
    classifiers=[
        "Development Status :: 4 - Beta",
        "Intended Audience :: Science/Research",
        "Topic :: Scientific/Engineering :: Chemistry",
        "Topic :: Scientific/Engineering :: Bio-Informatics",
        "License :: OSI Approved :: MIT License",
        "Programming Language :: Python :: 3",
        "Programming Language :: Python :: 3.8",
        "Programming Language :: Python :: 3.9",
        "Programming Language :: Python :: 3.10",
        "Programming Language :: Python :: 3.11",
        "Programming Language :: Python :: 3.12",
        "Operating System :: OS Independent",
    ],
    
    # Keywords for searching
    keywords="mass-spectrometry ion-mobility proteomics calibration CCS TWIMS collision-cross-section",
)
