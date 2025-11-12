"""
Setup script to copy and rename nativeims package to imspartacus for JOSS submission.
This script will:
1. Copy all nativeims module files
2. Replace all 'nativeims' references with 'imspartacus'
3. Copy and rename the refactored pages (1-6, 8-10, 12)
4. Update all imports in the copied files
"""

import os
import shutil
from pathlib import Path
import re

# Define source and destination paths
SOURCE_ROOT = Path(r"c:\Users\h87023ab\Documents\GITHUB\PROCESSING_TOOLS\nativeIMS-processing-tools")
DEST_ROOT = SOURCE_ROOT / "JOSS_PACKAGE"

# Modules to copy from nativeims package
NATIVEIMS_MODULES = [
    "calibration",
    "extraction",
    "fitting",
    "io",
    "processing",
    "visualization"
]

# Pages to include (refactored versions)
PAGES_TO_INCLUDE = {
    "1_calibrate_refactored.py": "1_calibrate.py",
    "2_get_input_files_refactored.py": "2_generate_input_files.py",
    "3_process_output_files_refactored.py": "3_process_output_files.py",
    "4_get_calibrated_scaled_data_refactored.py": "4_get_calibrated_data.py",
    "5_plot_CCSDs_refactored.py": "5_plot_ccsds.py",
    "6_fit_data_refactored.py": "6_fit_data.py",
    "8_plot_pretty_MS_refactored.py": "7_plot_mass_spectra.py",
    "9_generate_range_files_refactored.py": "8_generate_range_files.py",
    "10_ESIProt_refactored.py": "9_esiprot.py",
    "12_origami_refactored.py": "10_origami_ciu.py",
}

def replace_package_name(content):
    """Replace all occurrences of 'nativeims' with 'imspartacus'."""
    # Replace imports
    content = re.sub(r'from nativeims\.', 'from imspartacus.', content)
    content = re.sub(r'import nativeims\.', 'import imspartacus.', content)
    content = re.sub(r'import nativeims\b', 'import imspartacus', content)
    
    # Replace in comments and docstrings
    content = content.replace('nativeims', 'imspartacus')
    content = content.replace('nativeIMS', 'IMSpartacus')
    content = content.replace('Native IMS', 'IMSpartacus')
    
    return content

def copy_module_files(module_name):
    """Copy all files from a nativeims submodule to imspartacus."""
    source_dir = SOURCE_ROOT / "nativeims" / module_name
    dest_dir = DEST_ROOT / "imspartacus" / module_name
    
    if not source_dir.exists():
        print(f"Warning: {source_dir} does not exist")
        return
    
    # Copy all Python files
    for py_file in source_dir.glob("*.py"):
        if py_file.name == "__pycache__":
            continue
            
        content = py_file.read_text(encoding="utf-8")
        content = replace_package_name(content)
        
        dest_file = dest_dir / py_file.name
        dest_file.write_text(content, encoding="utf-8")
        print(f"Copied and updated: {py_file.name} -> {module_name}/")

def copy_page_file(source_name, dest_name):
    """Copy a page file and update imports."""
    source_file = SOURCE_ROOT / "pages" / source_name
    dest_file = DEST_ROOT / "pages" / dest_name
    
    if not source_file.exists():
        print(f"Warning: {source_file} does not exist")
        return
    
    content = source_file.read_text(encoding="utf-8")
    content = replace_package_name(content)
    
    dest_file.write_text(content, encoding="utf-8")
    print(f"Copied page: {source_name} -> {dest_name}")

def create_init_files():
    """Create __init__.py files for all modules."""
    # Main package __init__.py
    main_init_source = SOURCE_ROOT / "nativeims" / "__init__.py"
    main_init_dest = DEST_ROOT / "imspartacus" / "__init__.py"
    
    if main_init_source.exists():
        content = main_init_source.read_text(encoding="utf-8")
        content = replace_package_name(content)
        main_init_dest.write_text(content, encoding="utf-8")
    else:
        # Create a basic __init__.py
        init_content = '''"""
IMSpartacus - Ion Mobility Spectrometry Processing And Robust Toolkit for Analysis of Collision Cross Sections

A comprehensive toolkit for processing TWIMS data.
"""

__version__ = "1.0.0"
__author__ = "Your Name"

# Import main modules
from . import calibration
from . import extraction
from . import fitting
from . import io
from . import processing
from . import visualization
'''
        main_init_dest.write_text(init_content, encoding="utf-8")
    
    # Create __init__.py for each submodule
    for module_name in NATIVEIMS_MODULES:
        source_init = SOURCE_ROOT / "nativeims" / module_name / "__init__.py"
        dest_init = DEST_ROOT / "imspartacus" / module_name / "__init__.py"
        
        if source_init.exists():
            content = source_init.read_text(encoding="utf-8")
            content = replace_package_name(content)
            dest_init.write_text(content, encoding="utf-8")
        else:
            dest_init.write_text(f'"""{module_name.capitalize()} module for IMSpartacus."""\n', encoding="utf-8")
        
        print(f"Created __init__.py for {module_name}")

def main():
    """Main execution function."""
    print("=" * 60)
    print("IMSpartacus Package Setup")
    print("=" * 60)
    
    # Copy module files
    print("\n1. Copying and updating module files...")
    for module_name in NATIVEIMS_MODULES:
        print(f"\n  Processing module: {module_name}")
        copy_module_files(module_name)
    
    # Create __init__ files
    print("\n2. Creating __init__.py files...")
    create_init_files()
    
    # Copy page files
    print("\n3. Copying and renumbering page files...")
    for source_name, dest_name in PAGES_TO_INCLUDE.items():
        copy_page_file(source_name, dest_name)
    
    print("\n" + "=" * 60)
    print("Setup complete!")
    print("=" * 60)
    print(f"\nFiles created in: {DEST_ROOT}")
    print("\nNext steps:")
    print("1. Review the generated files")
    print("2. Update setup.py with your information")
    print("3. Create README.md for JOSS")
    print("4. Run 'pip install -e .' in the JOSS_PACKAGE directory")

if __name__ == "__main__":
    main()
