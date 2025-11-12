"""
Advanced Mass Spectrum Plotting - Refactored
Create publication-ready mass spectrum plots with extensive customization.
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import myutils
sys.path.insert(0, str(Path(__file__).parent.parent))

import streamlit as st
import io
import numpy as np
from typing import List, Dict, Any, Optional

# Import from imspartacus package
from imspartacus.visualization import (
    SpectrumData,
    SpectrumReader,
    SpectrumProcessor,
    MassSpectrumPlotter
)

# Import Streamlit UI styling
from myutils import styling

# Apply custom styling
styling.load_custom_css()


class MassSpectrumInterface:
    """Streamlit interface for mass spectrum plotting."""
    
    @staticmethod
    def show_header():
        """Display page header."""
        st.markdown(
            '<div class="main-header">'
            '<h1>📊 Advanced Mass Spectrum Plotting</h1>'
            '<p>Create publication-ready mass spectrum plots with extensive customization and data processing</p>'
            '</div>',
            unsafe_allow_html=True
        )
        
        st.markdown("""
        <div class="info-card">
            <p>Professional mass spectrometry plotting tool with advanced features for publication-quality figures.</p>
            <p><strong>Plot Types:</strong></p>
            <ul>
                <li><strong>Single Spectrum:</strong> Detailed single spectrum with advanced annotations</li>
                <li><strong>Stacked Comparison:</strong> Multiple spectra with normalized intensities</li>
                <li><strong>Hybrid Stacked:</strong> Individual spectrum annotations and processing</li>
                <li><strong>Mirror Plot:</strong> Two spectra mirrored for comparison</li>
                <li><strong>Overlay Plot:</strong> Multiple spectra overlaid with transparency</li>
            </ul>
            <p><strong>Advanced Features:</strong> Data smoothing, baseline correction, peak detection, noise filtering, and more.</p>
        </div>
        """, unsafe_allow_html=True)
    
    @staticmethod
    def show_file_upload(plot_type: str) -> List:
        """Show file upload widget based on plot type.
        
        Args:
            plot_type: Selected plot type
            
        Returns:
            List of uploaded files
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📁 Upload Files</h3>', unsafe_allow_html=True)
        
        if plot_type == "Single Spectrum":
            uploaded_file = st.file_uploader("Upload spectrum file", type=['txt', 'csv', 'tsv'])
            uploaded_files = [uploaded_file] if uploaded_file else []
        elif plot_type == "Mirror Plot":
            uploaded_files = st.file_uploader(
                "Upload exactly 2 spectrum files for mirror plot",
                type=['txt', 'csv', 'tsv'], accept_multiple_files=True
            )
            if uploaded_files and len(uploaded_files) != 2:
                st.warning("⚠️ Please upload exactly 2 files for mirror plot")
        else:
            uploaded_files = st.file_uploader(
                "Upload spectrum files",
                type=['txt', 'csv', 'tsv'], accept_multiple_files=True
            )
        
        st.markdown('</div>', unsafe_allow_html=True)
        return uploaded_files if uploaded_files else []
    
    @staticmethod
    def show_plot_configuration() -> tuple:
        """Show plot configuration options.
        
        Returns:
            Tuple of (plot_type, output_format)
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">🎯 Plot Configuration</h3>', unsafe_allow_html=True)
        
        col1, col2 = st.columns(2)
        with col1:
            plot_type = st.selectbox(
                "Plot type:",
                ["Single Spectrum", "Stacked Comparison", "Overlay Plot", "Mirror Plot"]
            )
        with col2:
            output_format = st.selectbox("Output format:", ["PNG", "PDF", "SVG"])
        
        st.markdown('</div>', unsafe_allow_html=True)
        return plot_type, output_format
    
    @staticmethod
    def show_figure_settings() -> Dict[str, Any]:
        """Show figure dimension and styling settings.
        
        Returns:
            Dictionary of figure settings
        """
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📐 Figure Settings</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            figure_width = st.number_input("Width (inches):", 1.0, 30.0, 10.0, 0.5)
            figure_height = st.number_input("Height (inches):", 1.0, 20.0, 6.0, 0.5)
        
        with col2:
            dpi = st.selectbox("DPI:", [150, 200, 300, 400, 600, 800], index=2)
            font_size = st.number_input("Font size:", 6, 24, 12)
        
        with col3:
            font_family = st.selectbox("Font family:", ["Arial", "Times New Roman", "Helvetica", "Calibri"])
            font_weight = st.selectbox("Font weight:", ["normal", "bold"], index=1)
            background = st.selectbox("Background:", ["white", "lightgray", "black", "transparent"])
        
        with col4:
            line_color = st.selectbox("Line color:", ["black", "blue", "red", "green", "purple", "orange", "brown"])
            line_width = st.number_input("Line width:", 0.1, 5.0, 1.5, 0.1)
            line_style = st.selectbox("Line style:", ["-", "--", "-.", ":"])
            alpha = st.slider("Line transparency:", 0.1, 1.0, 1.0)
        
        st.markdown('</div>', unsafe_allow_html=True)
        
        return {
            'width': figure_width,
            'height': figure_height,
            'dpi': dpi,
            'font_size': font_size,
            'font_family': font_family,
            'font_weight': font_weight,
            'background': background,
            'line_color': line_color,
            'line_width': line_width,
            'line_style': line_style,
            'alpha': alpha
        }
    
    @staticmethod
    def show_advanced_styling(plot_type: str) -> Dict[str, Any]:
        """Show advanced styling options.
        
        Args:
            plot_type: Selected plot type
            
        Returns:
            Dictionary of advanced styling settings
        """
        settings = {}
        
        with st.expander("🎨 Advanced Styling", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                fill_under = st.checkbox("Fill under curve")
                if fill_under:
                    fill_alpha = st.slider("Fill transparency:", 0.1, 1.0, 0.3)
                else:
                    fill_alpha = 0.3
            
            with col2:
                if plot_type in ["Stacked Comparison", "Overlay Plot"]:
                    palette = st.selectbox(
                        "Color palette:",
                        ["Set2", "husl", "tab10", "Set1", "Paired", "Dark2", "Pastel1"],
                        index=0
                    )
                else:
                    palette = "Set2"
                
                if plot_type == "Mirror Plot":
                    line_color_2 = st.selectbox(
                        "Second spectrum color:",
                        ["red", "blue", "green", "purple", "orange", "black"],
                        index=0
                    )
                else:
                    line_color_2 = "red"
            
            with col3:
                if plot_type == "Stacked Comparison":
                    stack_offset = st.slider("Stack offset:", 0.5, 3.0, 1.2)
                else:
                    stack_offset = 1.2
                
                show_legend = st.checkbox("Show legend", value=True)
                if show_legend:
                    legend_pos = st.selectbox("Legend position:", ["best", "upper right", "upper left", "lower right", "lower left"])
                    legend_frame = st.checkbox("Legend frame", value=True)
                else:
                    legend_pos = "best"
                    legend_frame = True
            
            settings.update({
                'fill_under': fill_under,
                'fill_alpha': fill_alpha,
                'palette': palette,
                'line_color_2': line_color_2,
                'stack_offset': stack_offset,
                'show_legend': show_legend,
                'legend_pos': legend_pos,
                'legend_frame': legend_frame
            })
        
        return settings
    
    @staticmethod
    def show_axis_settings() -> Dict[str, Any]:
        """Show axis configuration options.
        
        Returns:
            Dictionary of axis settings
        """
        settings = {}
        
        st.markdown('<div class="section-card">', unsafe_allow_html=True)
        st.markdown('<h3 class="section-header">📊 Axis Settings</h3>', unsafe_allow_html=True)
        
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            x_min = st.number_input("X-axis min:", 0.0, value=50.0, step=50.0)
            x_max = st.number_input("X-axis max:", 100.0, value=5000.0, step=100.0)
        
        with col2:
            custom_y_range = st.checkbox("Custom Y range")
            if custom_y_range:
                y_min = st.number_input("Y-axis min:", value=0.0)
                y_max = st.number_input("Y-axis max:", value=1000.0)
            else:
                y_min = None
                y_max = None
        
        with col3:
            show_grid = st.checkbox("Show grid")
            if show_grid:
                grid_alpha = st.slider("Grid transparency:", 0.0, 1.0, 0.3)
                grid_style = st.selectbox("Grid style:", ["--", "-", "-.", ":"])
                grid_color = st.selectbox("Grid color:", ["gray", "lightgray", "black"])
            else:
                grid_alpha = 0.3
                grid_style = "--"
                grid_color = "gray"
        
        with col4:
            zoom = st.slider("Y-axis zoom:", 0.5, 10.0, 1.4)
            preserve_baseline = st.checkbox("Preserve baseline at zero", True)
        
        settings.update({
            'x_min': x_min,
            'x_max': x_max,
            'y_min': y_min,
            'y_max': y_max,
            'show_grid': show_grid,
            'grid_alpha': grid_alpha,
            'grid_style': grid_style,
            'grid_color': grid_color,
            'zoom': zoom,
            'preserve_baseline': preserve_baseline
        })
        
        # Advanced axis customization
        with st.expander("⚙️ Advanced Axis Customization", expanded=False):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                show_x_label = st.checkbox("Show X label", True)
                if show_x_label:
                    x_label_text = st.text_input("X label:", "m/z")
                else:
                    x_label_text = "m/z"
            
            with col2:
                show_y_label = st.checkbox("Show Y label", True)
                if show_y_label:
                    y_label_text = st.text_input("Y label:", "Intensity")
                else:
                    y_label_text = "Intensity"
            
            with col3:
                axis_label_size = st.number_input("Axis label size:", 8, 24, 14)
                axis_label_weight = st.selectbox("Axis label weight:", ["normal", "bold"], index=1)
                tick_label_size = st.number_input("Tick label size:", 6, 20, 12)
            
            with col4:
                show_bottom_axis = st.checkbox("Bottom axis", True)
                show_top_axis = st.checkbox("Top axis", False)
                show_left_axis = st.checkbox("Left axis", True)
                show_right_axis = st.checkbox("Right axis", False)
            
            hide_y_ticks = st.checkbox("Hide Y ticks", True)
            show_y_tick_labels = st.checkbox("Show Y tick labels", False)
            show_x_tick_labels = st.checkbox("Show X tick labels", True)
            
            custom_x_ticks = st.checkbox("Custom X tick spacing")
            if custom_x_ticks:
                x_tick_spacing = st.number_input("X tick spacing:", 1.0, 500.0, 100.0, 10.0)
            else:
                x_tick_spacing = 100.0
            
            y_tick_count = st.number_input("Y tick count:", 0, 20, 5) if not hide_y_ticks else None
            
            spine_width = st.slider("Spine width:", 0.5, 5.0, 1.5, 0.1)
            spine_color = st.selectbox("Spine color:", ["black", "gray", "white"])
            
            settings.update({
                'show_x_label': show_x_label,
                'x_label_text': x_label_text,
                'show_y_label': show_y_label,
                'y_label_text': y_label_text,
                'axis_label_size': axis_label_size,
                'axis_label_weight': axis_label_weight,
                'tick_label_size': tick_label_size,
                'show_bottom_axis': show_bottom_axis,
                'show_top_axis': show_top_axis,
                'show_left_axis': show_left_axis,
                'show_right_axis': show_right_axis,
                'hide_y_ticks': hide_y_ticks,
                'show_y_tick_labels': show_y_tick_labels,
                'show_x_tick_labels': show_x_tick_labels,
                'custom_x_ticks': custom_x_ticks,
                'x_tick_spacing': x_tick_spacing,
                'y_tick_count': y_tick_count,
                'spine_width': spine_width,
                'spine_color': spine_color
            })
        
        st.markdown('</div>', unsafe_allow_html=True)
        return settings
    
    @staticmethod
    def show_processing_settings() -> Dict[str, Any]:
        """Show data processing options.
        
        Returns:
            Dictionary of processing settings
        """
        settings = {}
        
        with st.expander("🔬 Data Processing", expanded=False):
            col1, col2, col3 = st.columns(3)
            
            with col1:
                smoothing = st.checkbox("Apply smoothing")
                if smoothing:
                    smooth_window = st.slider("Smoothing window:", 3, 21, 5, 2)
                    smooth_order = st.slider("Polynomial order:", 1, 5, 2)
                else:
                    smooth_window = 5
                    smooth_order = 2
            
            with col2:
                baseline_correction = st.checkbox("Baseline correction")
                if baseline_correction:
                    baseline_percentile = st.slider("Baseline percentile:", 1.0, 20.0, 5.0, 1.0)
                else:
                    baseline_percentile = 5.0
            
            with col3:
                normalize = st.checkbox("Normalize")
                if normalize:
                    normalize_type = st.selectbox("Normalization:", ["max", "sum"])
                else:
                    normalize_type = "max"
            
            settings.update({
                'smoothing': smoothing,
                'smooth_window': smooth_window,
                'smooth_order': smooth_order,
                'baseline_correction': baseline_correction,
                'baseline_percentile': baseline_percentile,
                'normalize': normalize,
                'normalize_type': normalize_type
            })
        
        return settings
    
    @staticmethod
    def show_title_settings() -> Dict[str, Any]:
        """Show plot title settings.
        
        Returns:
            Dictionary of title settings
        """
        settings = {}
        
        with st.expander("📝 Plot Title", expanded=False):
            plot_title = st.text_input("Title text:", "")
            if plot_title:
                col1, col2 = st.columns(2)
                with col1:
                    title_font_size = st.number_input("Title font size:", 8, 36, 16)
                with col2:
                    title_weight = st.selectbox("Title weight:", ["normal", "bold"], index=1)
            else:
                title_font_size = 16
                title_weight = "bold"
            
            settings.update({
                'title': plot_title,
                'title_font_size': title_font_size,
                'title_weight': title_weight
            })
        
        return settings


def main():
    """Main application function."""
    # Show header
    MassSpectrumInterface.show_header()
    
    # Step 1: Plot configuration
    plot_type, output_format = MassSpectrumInterface.show_plot_configuration()
    
    # Step 2: File upload
    uploaded_files = MassSpectrumInterface.show_file_upload(plot_type)
    
    if not uploaded_files or not all(f is not None for f in uploaded_files):
        st.info("👆 Please upload spectrum files to begin plotting.")
        return
    
    # Step 3: Figure settings
    figure_settings = MassSpectrumInterface.show_figure_settings()
    
    # Step 4: Advanced styling
    advanced_settings = MassSpectrumInterface.show_advanced_styling(plot_type)
    figure_settings.update(advanced_settings)
    
    # Step 5: Axis settings
    axis_settings = MassSpectrumInterface.show_axis_settings()
    figure_settings.update(axis_settings)
    
    # Step 6: Processing settings
    processing_settings = MassSpectrumInterface.show_processing_settings()
    
    # Step 7: Title settings
    title_settings = MassSpectrumInterface.show_title_settings()
    figure_settings.update(title_settings)
    
    # Step 8: Generate plot
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.markdown('<h3 class="section-header">📈 Plot Preview</h3>', unsafe_allow_html=True)
    
    try:
        # Load spectra
        spectra = []
        for uploaded_file in uploaded_files:
            spectrum = SpectrumReader.read_file(uploaded_file)
            if len(spectrum.intensity) == 0:
                st.warning(f"⚠️ Could not read data from {uploaded_file.name}")
                continue
            
            # Apply processing
            processed_spectrum = SpectrumProcessor.process(spectrum, processing_settings)
            spectra.append(processed_spectrum)
        
        if not spectra:
            st.error("No valid spectra loaded")
            return
        
        # Map plot type to internal format
        plot_type_map = {
            "Single Spectrum": "single",
            "Stacked Comparison": "stacked",
            "Overlay Plot": "overlay",
            "Mirror Plot": "mirror"
        }
        
        # Create plot
        fig = MassSpectrumPlotter.create_plot(
            spectra,
            figure_settings,
            annotations=None,
            plot_type=plot_type_map[plot_type],
            spectrum_labels=[s.name for s in spectra],
            vertical_lines=None,
            spectrum_colors=None
        )
        
        if fig:
            # Display plot
            st.pyplot(fig)
            
            # Download options
            st.markdown("### 📥 Download Options")
            col1, col2, col3 = st.columns(3)
            
            facecolor = 'none' if figure_settings['background'] == 'transparent' else figure_settings['background']
            is_transparent = figure_settings['background'] == 'transparent'
            
            with col1:
                # PNG download
                buf_png = io.BytesIO()
                fig.savefig(buf_png, format='png', dpi=figure_settings['dpi'], 
                           bbox_inches='tight', facecolor=facecolor, transparent=is_transparent)
                buf_png.seek(0)
                
                file_size_mb = len(buf_png.getvalue()) / (1024 * 1024)
                st.download_button(
                    f"📊 Download PNG ({figure_settings['dpi']} DPI, {file_size_mb:.1f}MB)",
                    buf_png.getvalue(),
                    f"{plot_type.lower().replace(' ', '_')}_plot.png",
                    "image/png"
                )
            
            with col2:
                # PDF download
                buf_pdf = io.BytesIO()
                fig.savefig(buf_pdf, format='pdf', bbox_inches='tight',
                           facecolor=facecolor, transparent=is_transparent)
                buf_pdf.seek(0)
                
                st.download_button(
                    "📄 Download PDF",
                    buf_pdf.getvalue(),
                    f"{plot_type.lower().replace(' ', '_')}_plot.pdf",
                    "application/pdf"
                )
            
            with col3:
                # SVG download
                buf_svg = io.BytesIO()
                fig.savefig(buf_svg, format='svg', bbox_inches='tight',
                           facecolor=facecolor, transparent=is_transparent)
                buf_svg.seek(0)
                
                st.download_button(
                    "🎨 Download SVG",
                    buf_svg.getvalue(),
                    f"{plot_type.lower().replace(' ', '_')}_plot.svg",
                    "image/svg+xml"
                )
        
    except Exception as e:
        st.error(f"Error: {str(e)}")
        import traceback
        with st.expander("Show error details"):
            st.code(traceback.format_exc())
    
    st.markdown('</div>', unsafe_allow_html=True)


if __name__ == "__main__":
    main()
