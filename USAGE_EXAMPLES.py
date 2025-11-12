"""
Example: How to use the nativeIMS calibration library
======================================================

This script shows how to process calibrant data using the core library
(without Streamlit).
"""

from pathlib import Path
from nativeims.calibration import (
    load_bush_database,
    CalibrantDatabase,
    CalibrantProcessor,
    InstrumentParams,
    adjust_dataframe_drift_times
)
from nativeims.io.writers import write_imscal_dat, write_calibration_results_csv


def example_single_file():
    """Example: Process a single calibrant file."""
    print("=" * 60)
    print("Example 1: Processing a single file")
    print("=" * 60)
    
    # Step 1: Load the Bush database
    bush_df = load_bush_database(Path("data/bush.csv"))
    db = CalibrantDatabase(bush_df)
    
    # Step 2: Create a processor
    processor = CalibrantProcessor(db, min_r2=0.9)
    
    # Step 3: Process a single file
    result = processor.process_file(
        file_path=Path("calibrants/myoglobin/24.txt"),
        protein_name="myoglobin",
        gas_type="helium"
    )
    
    # Step 4: Check result
    if result:
        print(f"✓ Successfully processed {result.filename}")
        print(f"  Protein: {result.protein}")
        print(f"  Charge state: {result.charge_state}")
        print(f"  Drift time: {result.drift_time:.3f} ms")
        print(f"  R²: {result.r_squared:.3f}")
        print(f"  Literature CCS: {result.ccs_literature:.2f} nm²")
    else:
        print("✗ Processing failed")
    
    print()


def example_folder():
    """Example: Process all files in a folder."""
    print("=" * 60)
    print("Example 2: Processing an entire folder")
    print("=" * 60)
    
    # Load database
    bush_df = load_bush_database(Path("data/bush.csv"))
    db = CalibrantDatabase(bush_df)
    
    # Create processor
    processor = CalibrantProcessor(db, min_r2=0.9)
    
    # Process all files in a folder
    measurements, skipped = processor.process_folder(
        folder_path=Path("calibrants/myoglobin"),
        protein_name="myoglobin",
        gas_type="helium"
    )
    
    # Show results
    print(f"✓ Successfully processed {len(measurements)} files")
    print(f"✗ Skipped {len(skipped)} files")
    
    print("\nSuccessful measurements:")
    for m in measurements:
        print(f"  Charge {m.charge_state}: "
              f"drift time = {m.drift_time:.3f} ms, "
              f"R² = {m.r_squared:.3f}")
    
    if skipped:
        print("\nSkipped files:")
        for s in skipped:
            print(f"  {s}")
    
    print()


def example_full_dataset():
    """Example: Process multiple proteins and create output files."""
    print("=" * 60)
    print("Example 3: Processing complete dataset")
    print("=" * 60)
    
    # Load database
    bush_df = load_bush_database(Path("data/bush.csv"))
    db = CalibrantDatabase(bush_df)
    
    # Create processor
    processor = CalibrantProcessor(db, min_r2=0.9)
    
    # Process entire calibrant set
    # Expects folder structure:
    # calibrants/
    #   myoglobin/
    #     24.txt
    #     25.txt
    #   cytochromec/
    #     18.txt
    #     19.txt
    
    results_df = processor.process_calibrant_set(
        base_path=Path("calibrants"),
        gas_type="helium"
    )
    
    print(f"Processed {len(results_df)} total measurements")
    print("\nResults preview:")
    print(results_df.head())
    
    # Define instrument parameters
    params = InstrumentParams(
        wave_velocity=281.0,  # m/s (multiply by 0.75 for Cyclic)
        wave_height=20.0,      # V
        pressure=1.63,         # mbar
        drift_length=0.98,     # m (0.98 for Cyclic, 0.25 for Synapt)
        instrument_type='cyclic',
        inject_time=0.3        # ms (only for Cyclic)
    )
    
    # Adjust drift times for Cyclic IMS
    adjusted_df = adjust_dataframe_drift_times(results_df, params)
    
    # Write output files
    write_calibration_results_csv(
        adjusted_df,
        Path("output/calibration_results.csv")
    )
    print("\n✓ Wrote CSV file: output/calibration_results.csv")
    
    write_imscal_dat(
        adjusted_df,
        velocity=params.wave_velocity,
        voltage=params.wave_height,
        pressure=params.pressure,
        length=params.drift_length,
        output_path=Path("output/calibration.dat")
    )
    print("✓ Wrote .dat file: output/calibration.dat")
    
    print()


def example_database_queries():
    """Example: Query the Bush database."""
    print("=" * 60)
    print("Example 4: Querying the database")
    print("=" * 60)
    
    # Load database
    bush_df = load_bush_database(Path("data/bush.csv"))
    db = CalibrantDatabase(bush_df)
    
    # Get available proteins
    proteins = db.get_available_proteins()
    print(f"Available proteins: {proteins}")
    
    # Get charge states for myoglobin
    charges = db.get_available_charge_states('myoglobin')
    print(f"\nMyoglobin charge states: {charges}")
    
    # Look up specific calibrant
    result = db.lookup_calibrant('myoglobin', 24, 'helium')
    if result:
        print(f"\nMyoglobin 24+ in helium:")
        print(f"  CCS: {result['ccs']:.2f} nm²")
        print(f"  Mass: {result['mass']:.1f} Da")
    
    print()


if __name__ == "__main__":
    print("\n")
    print("╔" + "=" * 58 + "╗")
    print("║  nativeIMS Calibration Library - Usage Examples         ║")
    print("╚" + "=" * 58 + "╝")
    print()
    
    # Run all examples
    # Note: These will fail if you don't have the data files,
    # but they show you how to use the library!
    
    try:
        example_database_queries()
    except Exception as e:
        print(f"Example 4 failed: {e}\n")
    
    try:
        example_single_file()
    except Exception as e:
        print(f"Example 1 failed: {e}\n")
    
    try:
        example_folder()
    except Exception as e:
        print(f"Example 2 failed: {e}\n")
    
    try:
        example_full_dataset()
    except Exception as e:
        print(f"Example 3 failed: {e}\n")
    
    print("=" * 60)
    print("Examples complete!")
    print("=" * 60)
