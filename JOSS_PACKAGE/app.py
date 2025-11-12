"""
IMSpartacus - Ion Mobility Spectrometry Processing And Robust Toolkit for Analysis of Collision Cross Sections

A comprehensive toolkit for processing traveling wave ion mobility spectrometry (TWIMS) data,
including calibration, CCS conversion, peak fitting, and visualization.
"""

import streamlit as st
from myutils import styling

# Page configuration
st.set_page_config(
    page_title="IMSpartacus",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded"
)

def main():
    """Main home page function."""
    styling.load_custom_css()

    # Main header
    st.markdown(
        '<div class="main-header">'
        '<h1>⚡ IMSpartacus</h1>'
        '<p>Ion Mobility Spectrometry Processing And Robust Toolkit for Analysis of Collision Cross Sections</p>'
        '</div>',
        unsafe_allow_html=True
    )

    # About section
    st.markdown("""
    <div class="info-card">
        <h3>🧰 About IMSpartacus</h3>
        <p>IMSpartacus is a comprehensive toolkit for processing traveling wave ion mobility spectrometry (TWIMS) data. 
        It integrates with established tools like TWIMExtract<sup>1</sup> and IMSCal<sup>2</sup> to provide end-to-end 
        analysis workflows from raw data to publication-ready figures.</p>
        <p><strong>Key Features:</strong></p>
        <ul>
            <li>Drift time to CCS calibration using protein standards</li>
            <li>Automated peak detection and fitting</li>
            <li>ORIGAMI-style collision-induced unfolding (CIU) analysis</li>
            <li>ESIProt charge state deconvolution</li>
            <li>Publication-quality visualization tools</li>
            <li>Streamlit-based web interface for accessibility</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Navigation info
    st.markdown("""
    <div class="info-card">
        <h3>📋 Available Tools</h3>
        <p>Use the sidebar to navigate between different processing tools:</p>
        
        <strong>📊 Core Workflow:</strong>
        <ol>
            <li><strong>Calibrate:</strong> Process calibrant data and generate CCS calibration curves</li>
            <li><strong>Generate Input Files:</strong> Create input files for IMSCal from your data</li>
            <li><strong>Process Output Files:</strong> Convert arrival times to collision cross sections using IMSCal output</li>
            <li><strong>Get Calibrated Data:</strong> Generate full calibrated and scaled datasets with mass spectrum integration</li>
            <li><strong>Plot CCSDs:</strong> Visualize collision cross section distributions</li>
        </ol>
        
        <strong>🔬 Advanced Analysis:</strong>
        <ul>
            <li><strong>Fit Data:</strong> Peak detection and fitting with multiple peak functions (Gaussian, pseudo-Voigt, etc.)</li>
            <li><strong>Mass Spectrum Plotting:</strong> Publication-ready MS plots with advanced customization</li>
            <li><strong>Range File Generator:</strong> Create TWIMExtract range files for automated extraction</li>
            <li><strong>ESIProt:</strong> Protein charge state deconvolution</li>
            <li><strong>ORIGAMI Analysis:</strong> CCS fingerprint heatmaps for CIU experiments</li>
        </ul>
    </div>
    """, unsafe_allow_html=True)

    # Getting Started
    st.markdown("""
    <div class="info-card">
        <h3>🚀 Getting Started</h3>
        <p><strong>Prerequisites:</strong></p>
        <ul>
            <li>TWIMS data for calibrants and samples</li>
            <li>TWIMExtract for initial data extraction</li>
            <li>IMSCal for CCS calibration (steps 1-3)</li>
        </ul>
        <p><strong>Typical Workflow:</strong></p>
        <ol>
            <li>Extract drift time data using TWIMExtract</li>
            <li>Generate calibration using <em>Calibrate</em> tool</li>
            <li>Create IMSCal input files using <em>Generate Input Files</em></li>
            <li>Run IMSCal externally</li>
            <li>Process IMSCal output and visualize results</li>
            <li>Perform advanced analysis (fitting, CIU, etc.)</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)

    # References section
    st.markdown("""
    <div class="info-card">
        <h3>📚 References</h3>
        <p><sup>1</sup> I. Sergent, A. I. Adjieufack, A. Gaudel-Siri and L. Charles, 
        <em>International Journal of Mass Spectrometry</em>, 2023, <strong>492</strong>, 117112.</p>
        <p><sup>2</sup> S. E. Haynes, D. A. Polasky, S. M. Dixit, J. D. Majmudar, K. Neeson, B. T. Ruotolo and B. R. Martin, 
        <em>Analytical Chemistry</em>, 2017, <strong>89</strong>, 5669–5672.</p>
    </div>
    """, unsafe_allow_html=True)

    # Citation
    st.markdown("""
    <div class="info-card">
        <h3>📖 Citation</h3>
        <p>If you use IMSpartacus in your research, please cite:</p>
        <p><em>[Citation will be added upon JOSS publication]</em></p>
    </div>
    """, unsafe_allow_html=True)

if __name__ == "__main__":
    main()
