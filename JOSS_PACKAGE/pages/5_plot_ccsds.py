"""Refactored CCSD plotting page for Streamlit app.

This page provides an interface for plotting calibrated & scaled IMS data
with various display options, using the core imspartacus library for all
scientific computations.
"""

import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
from myutils import styling

# Import from imspartacus library
from imspartacus.visualization import (
    CCSDData,
    GaussianFitData,
    PlotSettings,
    CCSDPlotter
)


class DataLoader:
    """Handles file uploads and data loading."""
    
    @staticmethod
    def load_calibrated_csv(file) -> CCSDData | None:
        """Load calibrated CSV file into CCSDData container.
        
        Args:
            file: Uploaded file object
            
        Returns:
            CCSDData object or None if loading fails
        """
        try:
            df = pd.read_csv(file)
            return CCSDData(df, filter_threshold=0.5)
        except ValueError as e:
            st.error(f"Error loading calibrated CSV: {e}")
            return None
        except Exception as e:
            st.error(f"Unexpected error loading CSV: {e}")
            return None
    
    @staticmethod
    def load_gaussian_fits(file) -> GaussianFitData | None:
        """Load Gaussian fit parameters from CSV.
        
        Args:
            file: Uploaded file object
            
        Returns:
            GaussianFitData object or None if loading fails
        """
        try:
            df = pd.read_csv(file)
            return GaussianFitData(df)
        except ValueError as e:
            st.error(f"Error loading Gaussian fits: {e}")
            return None
        except Exception as e:
            st.error(f"Unexpected error loading Gaussian fits: {e}")
            return None


class PlotOptionsUI:
    """UI components for plot customization."""
    
    @staticmethod
    def show_basic_options() -> dict:
        """Display basic plot options and return selections."""
        st.markdown("""
        <div class="section-card">
            <div class="section-header">🎨 Plot Options</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            trace_palette_choice = st.selectbox(
                "Trace color palette (raw data)",
                list(sns.palettes.SEABORN_PALETTES.keys()) + ["Black"]
            )
            
            plot_mode = st.radio("Display Mode", ["Summed", "Stacked"])
            
            use_scaled = st.radio(
                "Intensity Type",
                ["Scaled", "Normalized"]
            ) == "Scaled"
            
            font_family = st.selectbox(
                "Font family",
                ["DejaVu Sans", "Arial", "Times New Roman", "Calibri", "Verdana"]
            )
            
            bg_option = st.radio("Background", ["White", "Transparent"])
        
        with col2:
            fig_width = st.slider("Figure width", min_value=2, max_value=20, value=6)
            fig_height = st.slider("Figure height", min_value=2, max_value=20, value=4)
            fig_dpi = st.slider("Figure DPI", min_value=100, max_value=1000, value=300)
            font_size = st.slider("Font size", min_value=5, max_value=24, value=12)
            line_thickness = st.slider(
                "Line thickness",
                min_value=0.1,
                max_value=5.0,
                value=1.0,
                step=0.1
            )
        
        return {
            'trace_palette_choice': trace_palette_choice,
            'plot_mode': plot_mode,
            'use_scaled': use_scaled,
            'font_family': font_family,
            'bg_option': bg_option,
            'fig_width': fig_width,
            'fig_height': fig_height,
            'fig_dpi': fig_dpi,
            'font_size': font_size,
            'line_thickness': line_thickness
        }
    
    @staticmethod
    def show_styling_options() -> dict:
        """Display styling options and return selections."""
        st.markdown("""
        <div class="section-card">
            <div class="section-header">✨ Styling Options</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            show_dashed_lines = st.checkbox("Show dashed CCS lines", value=True)
            show_ccs_labels = st.checkbox("Show CCS value labels inside plot", value=True)
            shade_under = st.checkbox("Shade under curves", value=True)
            black_lines = st.checkbox("Use black lines for traces", value=False)
        
        with col2:
            label_vertical_pos = st.slider(
                "Label vertical position (0 = bottom, 1 = top)",
                min_value=0.0,
                max_value=1.0,
                value=0.95
            )
            label_orientation = st.radio("Label orientation", ["Vertical", "Horizontal"])
        
        file_name = st.text_input("PNG file name", value="ccs_plot.png")
        
        return {
            'show_dashed_lines': show_dashed_lines,
            'show_ccs_labels': show_ccs_labels,
            'shade_under': shade_under,
            'black_lines': black_lines,
            'label_vertical_pos': label_vertical_pos,
            'label_orientation': label_orientation,
            'file_name': file_name
        }
    
    @staticmethod
    def show_ccs_range_options(ccs_min_default: float, ccs_max_default: float) -> dict:
        """Display CCS range and label options."""
        st.markdown("""
        <div class="section-card">
            <div class="section-header">📏 CCS Range & Labels</div>
        </div>
        """, unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        
        with col1:
            ccs_min = st.number_input("CCS x-axis min", value=ccs_min_default)
        
        with col2:
            ccs_max = st.number_input("CCS x-axis max", value=ccs_max_default)
        
        ccs_label_str = st.text_input(
            "Optional CCS label positions (comma-separated, e.g. 1200,1350)",
            value=""
        )
        
        ccs_label_values = []
        if ccs_label_str.strip():
            try:
                ccs_label_values = [
                    float(x.strip())
                    for x in ccs_label_str.split(",")
                    if x.strip()
                ]
            except Exception:
                st.warning("Could not parse CCS label positions. Please enter comma-separated numbers.")
        
        return {
            'ccs_min': ccs_min,
            'ccs_max': ccs_max,
            'ccs_label_values': ccs_label_values
        }
    
    @staticmethod
    def show_gaussian_options(
        selected_charges: list[int]
    ) -> tuple[GaussianFitData | None, bool, bool]:
        """Display Gaussian fit overlay options.
        
        Returns:
            Tuple of (gaussian_data, show_fits, shade_gaussians)
        """
        st.markdown("""
        <div class="section-card">
            <div class="section-header">📊 Optional Gaussian Fits</div>
        </div>
        """, unsafe_allow_html=True)
        
        fits_file = st.file_uploader(
            "Upload Gaussian fits CSV (Amplitude/Center_CCS/Sigma)",
            type="csv",
            key="fits_csv"
        )
        
        show_gaussian_fits = st.checkbox("Overlay Gaussian fits", value=False)
        
        gaussian_data = None
        shade_gaussians = False
        
        if fits_file and show_gaussian_fits:
            gaussian_data = DataLoader.load_gaussian_fits(fits_file)
            
            if gaussian_data is not None:
                # Filter to selected charges
                gaussian_data.df = gaussian_data.filter_charges(selected_charges)
                
                if not gaussian_data.df.empty:
                    shade_gaussians = st.checkbox(
                        "Shade under Gaussian components",
                        value=False,
                        key="shade_gaussian_components"
                    )
        
        return gaussian_data, show_gaussian_fits, shade_gaussians


class ResultsDisplay:
    """Handles display of results and downloads."""
    
    @staticmethod
    def show_maxima_info(maxima_info: dict):
        """Display local maxima information.
        
        Args:
            maxima_info: Dictionary mapping trace labels to maxima lists
        """
        st.markdown("#### 📍 Local Maxima (CCS, Intensity)")
        
        for label, maxima in maxima_info.items():
            if maxima:
                maxima_str = ", ".join([
                    f"({ccs:.1f}, {val:.2f})"
                    for ccs, val in maxima
                ])
                st.write(f"**{label}:** {maxima_str}")
            else:
                st.write(f"**{label}:** None found")
    
    @staticmethod
    def show_download_button(fig_buffer, file_name: str):
        """Display download button for plot.
        
        Args:
            fig_buffer: BytesIO buffer containing PNG data
            file_name: Name for downloaded file
        """
        if not file_name.endswith('.png'):
            file_name += '.png'
        
        st.download_button(
            "📥 Download CCS Plot as PNG",
            data=fig_buffer,
            file_name=file_name,
            mime="image/png",
            key="ccs_download"
        )


def main():
    """Main application flow."""
    # Load custom CSS
    styling.load_custom_css()
    
    # Header
    st.markdown("""
    <div class="main-header">
        <h1>📈 Plot Calibrated & Scaled IMS Data</h1>
        <p>Upload CSV file(s) generated from the previous step. This page allows you to plot the scaled CCSDs for selected charge states, either stacked or summed. Upload multiple files to display them as subplots.</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Step 1: Upload calibrated data
    st.markdown("""
    <div class="section-card">
        <div class="section-header">📁 Step 1: Upload Calibrated & Scaled CSV File(s)</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Multiple file upload
    use_multiple = st.checkbox("Upload multiple files for subplot comparison", value=False)
    
    if use_multiple:
        cal_files = st.file_uploader(
            "Upload calibrated & scaled CSV files",
            type="csv",
            accept_multiple_files=True
        )
        
        if not cal_files or len(cal_files) == 0:
            st.info("Please upload one or more calibrated & scaled CSV files to continue.")
            return
        
        # Layout selection for subplots
        layout = st.radio(
            "Subplot layout",
            ["Vertical", "Horizontal", "Grid"],
            horizontal=True
        ).lower()
    else:
        cal_file = st.file_uploader(
            "Upload calibrated & scaled CSV file",
            type="csv"
        )
        
        if not cal_file:
            st.info("Please upload a calibrated & scaled CSV file to continue.")
            return
        
        cal_files = [cal_file]
        layout = "vertical"
    
    # Load all datasets
    datasets = []
    for cal_file in cal_files:
        ccsd_data = DataLoader.load_calibrated_csv(cal_file)
        
        if ccsd_data is None:
            continue
        
        if ccsd_data.df.empty:
            st.warning(f"No data remaining after filtering in {cal_file.name}. Adjust filter threshold or check your data.")
            continue
        
        # Use filename as dataset name (remove extension)
        dataset_name = cal_file.name.rsplit('.', 1)[0]
        datasets.append((dataset_name, ccsd_data, cal_file))
    
    if not datasets:
        st.error("No valid datasets loaded.")
        return
    
    # Step 1.5: Customize titles (only for multiple files)
    if use_multiple and len(datasets) > 1:
        st.markdown("""
        <div class="section-card">
            <div class="section-header">✏️ Customize Subplot Titles</div>
        </div>
        """, unsafe_allow_html=True)
        
        show_titles = st.checkbox("Show subplot titles", value=True)
        
        if show_titles:
            col1, col2 = st.columns(2)
            
            with col1:
                title_fontsize = st.slider("Title font size", min_value=8, max_value=24, value=14)
                title_fontweight = st.selectbox("Title font weight", ["normal", "bold"], index=1)
            
            with col2:
                title_style = st.radio("Title style", ["Use filenames", "Custom labels"])
            
            # Custom title mapping
            title_map = {}
            if title_style == "Custom labels":
                st.markdown("**Enter custom titles for each dataset:**")
                for i, (name, _, _) in enumerate(datasets):
                    custom_title = st.text_input(
                        f"Title for {name}",
                        value=name,
                        key=f"title_{i}"
                    )
                    title_map[name] = custom_title
            else:
                # Use original names
                for name, _, _ in datasets:
                    title_map[name] = name
        else:
            show_titles = False
            title_map = {name: None for name, _, _ in datasets}
            title_fontsize = 14
            title_fontweight = "bold"
    
    # Step 2: Select charge states (apply to all datasets)
    st.markdown("""
    <div class="section-card">
        <div class="section-header">⚡ Step 2: Select Charge States</div>
    </div>
    """, unsafe_allow_html=True)
    
    # Get all unique charge states across all datasets
    all_charges = set()
    for _, ccsd_data, _ in datasets:
        all_charges.update(ccsd_data.get_charge_states())
    all_charges = sorted(all_charges)
    
    selected_charges = st.multiselect(
        "Select charge states to include (applies to all datasets)",
        all_charges,
        default=all_charges
    )
    
    if not selected_charges:
        st.warning("Please select at least one charge state.")
        return
    
    # Show data preview
    with st.expander("📊 View Data"):
        for name, ccsd_data, _ in datasets:
            st.markdown(f"**{name}**")
            filtered_df = ccsd_data.filter_charges(selected_charges)
            st.dataframe(filtered_df)
    
    # Step 3: Gaussian fits (optional) - can upload multiple matching files
    st.markdown("""
    <div class="section-card">
        <div class="section-header">📊 Optional Gaussian Fits</div>
    </div>
    """, unsafe_allow_html=True)
    
    gaussian_datasets = []
    show_gaussian_fits = st.checkbox("Overlay Gaussian fits", value=False)
    shade_gaussians = False
    
    if show_gaussian_fits:
        if use_multiple:
            fits_files = st.file_uploader(
                "Upload Gaussian fits CSV files (Amplitude/Center_CCS/Sigma) - name should match data files",
                type="csv",
                key="fits_csv",
                accept_multiple_files=True
            )
            
            if fits_files:
                for fits_file in fits_files:
                    gaussian_data = DataLoader.load_gaussian_fits(fits_file)
                    if gaussian_data is not None:
                        # Filter to selected charges
                        gaussian_data.df = gaussian_data.filter_charges(selected_charges)
                        if not gaussian_data.df.empty:
                            # Match by filename
                            fits_name = fits_file.name.rsplit('.', 1)[0]
                            gaussian_datasets.append((fits_name, gaussian_data))
        else:
            fits_file = st.file_uploader(
                "Upload Gaussian fits CSV (Amplitude/Center_CCS/Sigma)",
                type="csv",
                key="fits_csv"
            )
            
            if fits_file:
                gaussian_data = DataLoader.load_gaussian_fits(fits_file)
                if gaussian_data is not None:
                    gaussian_data.df = gaussian_data.filter_charges(selected_charges)
                    if not gaussian_data.df.empty:
                        dataset_name = datasets[0][0]
                        gaussian_datasets.append((dataset_name, gaussian_data))
        
        if gaussian_datasets:
            shade_gaussians = st.checkbox(
                "Shade under Gaussian components",
                value=False,
                key="shade_gaussian_components"
            )
    
    # Step 4: Plot options
    basic_opts = PlotOptionsUI.show_basic_options()
    styling_opts = PlotOptionsUI.show_styling_options()
    
    # Get CCS range from first dataset
    ccs_min_default, ccs_max_default = datasets[0][1].get_ccs_range()
    range_opts = PlotOptionsUI.show_ccs_range_options(ccs_min_default, ccs_max_default)
    
    # Build color palettes
    max_charges = len(selected_charges)
    if basic_opts['trace_palette_choice'] == "Black":
        trace_palette = ["black"] * max(1, max_charges)
    else:
        trace_palette = sns.color_palette(
            basic_opts['trace_palette_choice'],
            n_colors=max_charges
        )
    
    # Create plot settings
    # Include title settings for multiple files
    if use_multiple and len(datasets) > 1:
        settings = PlotSettings(
            fig_width=basic_opts['fig_width'],
            fig_height=basic_opts['fig_height'],
            fig_dpi=basic_opts['fig_dpi'],
            font_size=basic_opts['font_size'],
            font_family=basic_opts['font_family'],
            line_thickness=basic_opts['line_thickness'],
            plot_mode=basic_opts['plot_mode'],
            use_scaled=basic_opts['use_scaled'],
            ccs_min=range_opts['ccs_min'],
            ccs_max=range_opts['ccs_max'],
            ccs_label_values=range_opts['ccs_label_values'],
            show_dashed_lines=styling_opts['show_dashed_lines'],
            show_ccs_labels=styling_opts['show_ccs_labels'],
            label_vertical_pos=styling_opts['label_vertical_pos'],
            label_orientation=styling_opts['label_orientation'],
            shade_under=styling_opts['shade_under'],
            black_lines=styling_opts['black_lines'],
            bg_transparent=(basic_opts['bg_option'] == "Transparent"),
            trace_colors=trace_palette,
            shade_gaussians=shade_gaussians,
            title_fontsize=title_fontsize,
            title_fontweight=title_fontweight
        )
    else:
        settings = PlotSettings(
            fig_width=basic_opts['fig_width'],
            fig_height=basic_opts['fig_height'],
            fig_dpi=basic_opts['fig_dpi'],
            font_size=basic_opts['font_size'],
            font_family=basic_opts['font_family'],
            line_thickness=basic_opts['line_thickness'],
            plot_mode=basic_opts['plot_mode'],
            use_scaled=basic_opts['use_scaled'],
            ccs_min=range_opts['ccs_min'],
            ccs_max=range_opts['ccs_max'],
            ccs_label_values=range_opts['ccs_label_values'],
            show_dashed_lines=styling_opts['show_dashed_lines'],
            show_ccs_labels=styling_opts['show_ccs_labels'],
            label_vertical_pos=styling_opts['label_vertical_pos'],
            label_orientation=styling_opts['label_orientation'],
            shade_under=styling_opts['shade_under'],
            black_lines=styling_opts['black_lines'],
            bg_transparent=(basic_opts['bg_option'] == "Transparent"),
            trace_colors=trace_palette,
            shade_gaussians=shade_gaussians
        )
    
    # Step 5: Generate plot(s)
    st.markdown("""
    <div class="section-card">
        <div class="section-header">📊 Intensity vs CCS</div>
    </div>
    """, unsafe_allow_html=True)
    
    try:
        if use_multiple and len(datasets) > 1:
            # Multiple datasets - use subplots
            # Apply title map to dataset names
            plot_datasets = [
                (title_map[name], ccsd_data, selected_charges)
                for name, ccsd_data, _ in datasets
            ]
            
            fig, all_maxima_info = CCSDPlotter.plot_multiple_ccs_traces(
                datasets=plot_datasets,
                settings=settings,
                gaussian_datasets=gaussian_datasets if gaussian_datasets else None,
                show_gaussian_fits=show_gaussian_fits,
                layout=layout
            )
            
            # Display plot
            st.pyplot(fig)
            
            # Show maxima information for all datasets
            for dataset_name, maxima_info in all_maxima_info.items():
                st.markdown(f"#### 📍 {dataset_name} - Local Maxima")
                for label, maxima in maxima_info.items():
                    if maxima:
                        maxima_str = ", ".join([
                            f"({ccs:.1f}, {val:.2f})"
                            for ccs, val in maxima
                        ])
                        st.write(f"**{label}:** {maxima_str}")
                    else:
                        st.write(f"**{label}:** None found")
        else:
            # Single dataset - use original method
            name, ccsd_data, _ = datasets[0]
            
            # Get corresponding Gaussian data if available
            gaussian_data = None
            if gaussian_datasets:
                for g_name, g_data in gaussian_datasets:
                    if g_name == name:
                        gaussian_data = g_data
                        break
            
            # Generate plot
            fig, maxima_info = CCSDPlotter.plot_ccs_traces(
                ccsd_data=ccsd_data,
                selected_charges=selected_charges,
                settings=settings,
                gaussian_data=gaussian_data,
                show_gaussian_fits=show_gaussian_fits
            )
            
            # Display plot
            st.pyplot(fig)
            
            # Show maxima information
            ResultsDisplay.show_maxima_info(maxima_info)
        
        # Save and download
        fig_buffer = CCSDPlotter.save_figure_to_buffer(
            fig,
            dpi=settings.fig_dpi,
            transparent=settings.bg_transparent
        )
        
        ResultsDisplay.show_download_button(
            fig_buffer,
            styling_opts['file_name']
        )
        
    except Exception as e:
        st.error(f"Error generating plot: {e}")
        import traceback
        st.code(traceback.format_exc())


if __name__ == "__main__":
    main()
