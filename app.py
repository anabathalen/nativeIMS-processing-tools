import streamlit as st
from myutils import styling

def main():
    """Main home page function."""
    styling.load_custom_css()

    # Main header (light pink background, straight borders)
    st.markdown(
        '<div class="main-header">'
        '<h1>Native TWIMS Processing Toolkit</h1>'
        '</div>',
        unsafe_allow_html=True
    )

    # Info card (blue background, blue borders)
    st.markdown("""
    <div class="info-card">
        <h3>🧰 About this Toolkit</h3>
        <p>This toolkit provides a set of tools for processing TWIMS data in combination with previously published tools (TWIMExtract(ref) and IMSCal(ref)).</p>
        <p>To start, you will need TWIMS data for your calibrants and samples. Use the panel on the left to navigate to 'calibrate' and start processing! If you would like to practice, download the sample data below. </p>
    </div>
    """, unsafe_allow_html=True)

    # Navigation info
    st.markdown("""
    <div class="info-card">
        <h3>📋 Available Tools</h3>
        <p>Use the sidebar to navigate between different processing tools:</p>
        <ul>
            <li><strong>Calibrate:</strong> Process calibrant data and generate reference files for IMSCal</li>
            <li><strong>Get Input Files:</strong> Generate input files for IMSCal from your data.</li>
            <li><strong>Process Output Files:</strong> Use output files from IMSCal to convert arrival times to collision cross sections.</li>
            <li><strong>Get Calibrated and Scaled Data:</strong> Uses the conversions from the 'process output files' step and the mass spectum of each sample to generate full calibrated and scaled datasets. </li>
            <li><strong>Plot CCSDs:</strong> Tool for plotting CCSDs - these can be either unsmoothed/fitted from 'Get Calibrated and Scaled Data', or fitted using 'Fit Data'.</li>
            <li><strong>Fit Data:</strong> Emulates Origin software fitting procedures to fit calibrated data.</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # References section
    st.markdown("""
    <div class="info-card">
        <h3>📚 References</h3>
        <p><sup>1</sup> I. Sergent, A. I. Adjieufack, A. Gaudel-Siri and L. Charles, <em> International Journal of Mass Spectrometry,</em>,2023, 492, 117112.</p>
        <p><sup>2</sup> S. E. Haynes, D. A. Polasky, S. M. Dixit, J. D. Majmudar, K. Neeson, B. T. Ruotolo and B. R. Martin, <em>Analytical Chemistry</em>, 2017, 89, 5669–5672.</p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()