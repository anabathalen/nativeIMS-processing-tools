from myutils import data_tools
from myutils import import_tools
from myutils import styling
import pandas as pd
import numpy as np
import streamlit as st
import os
import io
import matplotlib.pyplot as plt

### DEFINING NEW FUNCTIONS SPECIFIC TO THIS PAGE -------------- ###

def process_manual_data(folder_name, base_path, bush_df, calibrant_type):
    folder_path = os.path.join(base_path, folder_name)
    results = []
    plots = []
    skipped_entries = []

    # pick column from bush
    calibrant_column = 'CCS_he' if calibrant_type == 'Helium' else 'CCS_n2'

    # iterate through each file in the folder
    for filename in os.listdir(folder_path):
        if filename.endswith('.txt') and filename[0].isdigit(): # only process text files that start with a number
            file_path = os.path.join(folder_path,filename)

            data = np.loadtxt(file_path) # load the text files and save the first column as a list of drift times and the second column as a list of intensities
            drift_time = data [:, 0]
            intensity = data[:, 1]

            # gaussian fit
            params, r2, fitted_values = data_tools.fit_gaussian_with_retries(drift_time, intensity)
            if params is not None:
                amp, apex, stddev = params
                charge_state = filename.split('.')[0]

                # look up the corresponding calibrant row in the bush database using the protein name and charge state
                calibrant_row = bush_df[(bush_df['protein'] == folder_name) & (bush_df['charge'] == int(charge_state))]

                # if there is something in the row of interest, proceed
                if not calibrant_row.empty:
                    calibrant_value = calibrant_row[calibrant_column].values[0]
                    mass = calibrant_row['mass'].values[0]

                    # only add to results if the calibrant value is not NaN
                    if pd.notna(calibrant_value) and calibrant_value is not None:
                        results.append([folder_name, mass, charge_state, apex, r2, calibrant_value])
                        plots.append((drift_time, intensity, fitted_values, filename, apex, r2))

                    else:
                        skipped_entries.append(f"{folder_name} charge {charge_state} - no {calibrant_type.lower()} CCS value available. If you would like to add it, please alter ..data/bush.csv and open a pull request.")
                
                else:
                    skipped_entries.append(f"{folder_name} charge {charge_state} - no {calibrant_type.lower()} CCS value in database. If you would like to add it, please alter ..data/bush.csv and open a pull request.")
            
            else:
                skipped_entries.append(f"{folder_name} charge {filename.split('.')[0]} - Gaussian fit failed")
    
    results_df = pd.DataFrame(results, columns = ['protein', 'mass', 'charge state', 'drift time', 'r2', 'calibrant_value'])

    return results_df, plots, skipped_entries

def display_results(results_df, plots, skipped_entries):
    if not results_df.empty:
        st.markdown('<h3 class="section-header">Gaussian Fit Results</h3>', unsafe_allow_html=True)
        st.dataframe(results_df)
        st.markdown('</div>', unsafe_allow_html=True)

        # Plot all the fits
        n_plots = len(plots)
        if n_plots > 0:
            n_cols = 3
            n_rows = (n_plots + n_cols - 1) // n_cols

            plt.figure(figsize=(12, 4 * n_rows))
            for i, (drift_time, intensity, fitted_values, filename, apex, r2) in enumerate(plots):
                plt.subplot(n_rows, n_cols, i + 1)
                plt.plot(drift_time, intensity, 'b.', label='Raw Data', markersize=3)
                plt.plot(drift_time, fitted_values, 'r-', label='Gaussian Fit', linewidth=1)
                plt.title(f'{filename}\nApex: {apex:.2f}, R²: {r2:.3f}')
                plt.xlabel('Drift Time')
                plt.ylabel('Intensity')
                plt.legend()
                plt.grid()

            plt.tight_layout()
            st.pyplot(plt)
    else:
        st.markdown('<div class="warning-card">No valid calibrant data found that matches the database.</div>', unsafe_allow_html=True)

    # Show skipped entries if any
    if skipped_entries:
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">⚠️ Skipped Entries</h3>', unsafe_allow_html=True)
        st.markdown('<div class="warning-card">', unsafe_allow_html=True)
        st.write("The following entries were skipped:")
        for entry in skipped_entries:
            st.write(f"• {entry}")
        st.markdown('</div>', unsafe_allow_html=True)
        st.markdown('</div>', unsafe_allow_html=True)

def generate_dat_file(results_df, velocity, voltage, pressure, length):
    if results_df.empty:
        return None
        
    dat_content = f"# length {length}\n# velocity {velocity}\n# voltage {voltage}\n# pressure {pressure}\n"

    # Create .dat content
    for _, row in results_df.iterrows():
        protein = row['protein']
        charge_state = row['charge state']
        mass = row['mass']
        calibrant_value = row['calibrant_value'] * 100  # Convert to Ų
        drift_time = row['drift time']
        dat_content += f"{protein}_{charge_state} {mass} {charge_state} {calibrant_value} {drift_time}\n"
    
    return dat_content

### ----------------------------------------------------------- ###

styling.load_custom_css()

st.markdown('<div class = "main-header"><h1>Prcess Calibrant Data</h1><p>Fit ATDs of calibrants and generate reference files for IMSCal</p></div>', unsafe_allow_html = True)

st.markdown("""
<div class = "info-card">
    <p>Use this page to fit the ATDs of your calibrants from text files or from TWIMExtract and generate a reference file for IMSCal and/or a csv file of calibrant measured and literature arrival times. This is desiged for use with denatured calibrants, so the fitting only allows for a single peak in each ATD - consider another tool if your ATDs are not gaussian.</p>
    <p>To start, make a folder for each calibrant you used. You should name these folders according to the table below (or they won't match the bush database file). From here you can either proceed with TWIMExtract or manually. For TWIMExtract, generate a separate csv file for the ATD of each charge state and rename them 'X.csv' where X is the charge state, and save them under their respective protein folder. To extract the data manually, make a text file for each charge state (called 'X.txt' where X is the charge state) and paste the corresponding ATD from MassLynx into each file. If extracting manually, remember to set the x-axis to ms not bins! Zip these protein folders together and upload it below.</p>
    </div>
    """, unsafe_allow_html = True)

data = {
    'Protein': [
        'Denatured Myoglobin',
        'Denatured Cytochrome C',
        'Polyalanine Peptide of Length X',
        'Denatured Ubiquitin'
    ],
    'Folder Name': [
        'myoglobin',
        'cytochromec',
        'polyalanineX',
        'ubiquitin'
    ]
}

df = pd.DataFrame(data)

st.markdown('<h3 class = "section-header">Calibrant Folder Naming</h3>', unsafe_allow_html = True)
st.table(df)
st.markdown('</div>', unsafe_allow_html = True)

### FINAL PAGE FUNCTION --------------------------------------------- ###

def calibrate_page():
    # Step 1: Upload ZIP file
    st.markdown('<h3 class="section-header">📁 Upload Calibrant Data</h3>', unsafe_allow_html=True)
    uploaded_zip_file = st.file_uploader("Upload a ZIP file containing your calibrant folders", type="zip")
    st.markdown('</div>', unsafe_allow_html=True)
    
    if uploaded_zip_file is not None:
        # Extract the folders from the ZIP file
        folders, temp_dir = import_tools.handle_zip_upload(uploaded_zip_file)

        # Step 2: Read bush.csv for calibrant data
        bush_df = import_tools.read_bush()

        if bush_df.empty:
            st.markdown('<div class="error-card">Cannot proceed without the Bush calibrant database.</div>', unsafe_allow_html=True)
            return

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">⚗️ Calibration Parameters</h3>', unsafe_allow_html=True)
        st.markdown('Most of the time you should calibrate with calibrant values obtained for the same drift gas as you used in your experiment, but sometimes you might not so the option is here.')
        
        # Step 3: Dropdown for selecting calibrant type (He or N2)
        calibrant_type = st.selectbox("Which values from the Bush database would you like to calibrate with?", options=["Helium", "Nitrogen"])

        col1, col2 = st.columns(2)
        with col1:
            # Step 4: Get user inputs for parameters
            velocity = st.number_input("Enter wave velocity (m/s), multiplied by 0.75 if this is Cyclic data", min_value=0.0, value=281.0)
            voltage = st.number_input("Enter wave height (V)", min_value=0.0, value=20.0)
        with col2:
            pressure = st.number_input("Enter IMS pressure", min_value=0.0, value=1.63)
            length = st.number_input("Enter drift cell length (0.25m for Synapt, 0.98m for Cyclic)", min_value=0.0, value=0.980)

        # Step 4.5: Ask for data type
        data_type = st.radio("Is this Cyclic or Synapt data?", options=["Cyclic", "Synapt"])
        inject_time = 0.0
        if data_type.lower() == "cyclic":
            inject_time = st.number_input("Enter inject time (ms)", min_value=0.0, value=0.0)
        
        st.markdown('</div>', unsafe_allow_html=True)

        # Step 5: Process all folders and files
        all_results_df = pd.DataFrame(columns=['protein', 'mass', 'charge state', 'drift time', 'r2', 'calibrant_value'])
        all_plots = []
        all_skipped = []

        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🔬 Processing Results</h3>', unsafe_allow_html=True)
        
        for folder in folders:
            st.write(f"Processing folder: **{folder}**")
            results_df, plots, skipped_entries = process_manual_data(folder, temp_dir, bush_df, calibrant_type)
            all_results_df = pd.concat([all_results_df, results_df], ignore_index=True)
            all_plots.extend(plots)
            all_skipped.extend(skipped_entries)

        st.markdown('</div>', unsafe_allow_html=True)

        # Step 6: Display results
        display_results(all_results_df, all_plots, all_skipped)

        # Only show download options if we have valid results
        if not all_results_df.empty:
            st.markdown('<div class="section-card">', unsafe_allow_html=True)
            st.markdown('<h3 class="section-header">📥 Download Results</h3>', unsafe_allow_html=True)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # Step 7: CSV download
                csv_buffer = io.StringIO()
                all_results_df.to_csv(csv_buffer, index=False)
                st.download_button(
                    label="📊 Download Results (CSV)",
                    data=csv_buffer.getvalue(),
                    file_name="combined_gaussian_fit_results.csv",
                    mime="text/csv"
                )

            with col2:
                # Step 8: Prepare adjusted drift times for .dat file if cyclic
                if data_type.lower() == "cyclic":
                    adjusted_df = all_results_df.copy()
                    adjusted_df['drift time'] = adjusted_df['drift time'] - inject_time
                else:
                    adjusted_df = all_results_df

                # Step 9: .dat file download
                dat_file_content = generate_dat_file(adjusted_df, velocity, voltage, pressure, length)
                if dat_file_content:
                    st.download_button(
                        label="📋 Download .dat File",
                        data=dat_file_content,
                        file_name="calibration_data.dat",
                        mime="text/plain"
                    )
            
            st.markdown('</div>', unsafe_allow_html=True)
        else:
            st.markdown('<div class="error-card">No valid results to download. Please check your data and database matching.</div>', unsafe_allow_html=True)

### ----------------------------------------------------------------- ###

calibrate_page()