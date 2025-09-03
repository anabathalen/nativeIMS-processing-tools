import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from myutils import styling

# Apply custom styling
styling.load_custom_css()

# Main header
st.markdown(
    '<div class="main-header">'
    '<h1>📊 Mass Spectrum Plotting</h1>'
    '<p>Create beautiful mass spectrum plots with charge state annotations</p>'
    '</div>',
    unsafe_allow_html=True
)

# Info card
st.markdown("""
<div class="info-card">
    <p>Create publication-ready mass spectrum plots with extensive customization options.</p>
    <p><strong>Plot Types:</strong></p>
    <ul>
        <li><strong>Single Spectrum:</strong> One spectrum with detailed charge state annotations</li>
        <li><strong>Stacked Comparison:</strong> Multiple spectra stacked with shared annotations</li>
        <li><strong>Hybrid Stacked:</strong> Multiple spectra with individual annotations and colors</li>
    </ul>
    <p><strong>File Format:</strong> Tab-separated files with m/z and intensity columns (no headers).</p>
</div>
""", unsafe_allow_html=True)

# Plotting functions
def read_mass_spectrum_file(file_path):
    """Read mass spectrum file and return dataframe"""
    try:
        df = pd.read_csv(file_path, sep='\t', header=None, on_bad_lines='skip')
        df.columns = ['m/z', 'intensity']
        df['m/z'] = pd.to_numeric(df['m/z'], errors='coerce')
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
        df.dropna(inplace=True)
        return df
    except Exception as e:
        st.error(f"Error reading file {file_path}: {e}")
        return pd.DataFrame(columns=['m/z', 'intensity'])

def calculate_mz_values_specific(mass, charge_states):
    """Calculate m/z values for specific charge states"""
    return [((mass + charge) / charge, charge) for charge in charge_states]

def get_color_palette(palette_name, n_colors):
    """Get color palette based on selection"""
    palettes = {
        "Scientific": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        "Nature": ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9', '#F0E442'],
        "Colorblind Safe": sns.color_palette("colorblind", n_colors),
        "Viridis": plt.cm.viridis(np.linspace(0, 1, n_colors)),
        "Plasma": plt.cm.plasma(np.linspace(0, 1, n_colors)),
    }
    return palettes.get(palette_name, sns.color_palette("husl", n_colors))[:n_colors]

def plot_single_spectrum(file_path, mass_configs, plot_settings):
    """Plot single spectrum with annotations"""
    fig, ax = plt.subplots(figsize=(plot_settings['width'], plot_settings['height']), dpi=300)
    
    # Set styling
    if plot_settings['background'] != "white":
        fig.patch.set_facecolor(plot_settings['background'])
        ax.set_facecolor(plot_settings['background'])
    plt.rcParams.update({'font.size': plot_settings['font_size']})
    
    df = read_mass_spectrum_file(file_path)
    if df.empty:
        return None
    
    df = df[(df['m/z'] >= plot_settings['x_min']) & (df['m/z'] <= plot_settings['x_max'])]
    max_intensity = df['intensity'].max()
    
    # Plot spectrum
    ax.plot(df['m/z'], df['intensity'], color=plot_settings['line_color'], 
            linewidth=plot_settings['line_width'], linestyle=plot_settings['line_style'])
    
    ax.set_xlim(plot_settings['x_min'], plot_settings['x_max'])
    ax.set_ylim(-0.05 * max_intensity, plot_settings['zoom'] * max_intensity)
    
    # Add annotations
    legend_entries = []
    for i, config in enumerate(mass_configs):
        color = config['color']
        shape = config['shape']
        mass = config['mass']
        charge_states = config['charge_states']
        
        legend_entries.append((shape, color, mass))
        
        for charge in charge_states:
            mz = (mass + charge) / charge
            if plot_settings['x_min'] <= mz <= plot_settings['x_max']:
                # Find intensity at this m/z
                width = 100 / charge
                intensity_in_range = df[(df['m/z'] >= mz - width/2) & (df['m/z'] <= mz + width/2)]
                
                if not intensity_in_range.empty:
                    peak_intensity = intensity_in_range['intensity'].max()
                    if peak_intensity >= plot_settings['threshold'] * max_intensity:
                        y_marker = peak_intensity + plot_settings['offset'] * max_intensity
                        y_label = y_marker + plot_settings['offset'] * max_intensity
                        
                        if y_label <= plot_settings['zoom'] * max_intensity:
                            ax.scatter([mz], [y_marker], color=color, marker=shape, s=plot_settings['marker_size'])
                            ax.text(mz, y_label, f'{charge}+', ha='center', 
                                   fontsize=plot_settings['font_size'], color=color)
    
    # Legend
    if plot_settings['show_legend'] and legend_entries:
        legend_elements = [plt.Line2D([0], [0], marker=symbol, color='w', markerfacecolor=color, 
                                     markersize=8, label=f'{mass} Da') 
                          for symbol, color, mass in legend_entries]
        ax.legend(handles=legend_elements, loc=plot_settings['legend_pos'], frameon=False)
    
    # Formatting
    if plot_settings['show_grid']:
        ax.grid(True, linestyle=plot_settings['grid_style'], alpha=plot_settings['grid_alpha'])
    ax.set_xlabel('m/z', fontsize=plot_settings['font_size'])
    ax.set_title(plot_settings['title'], fontsize=plot_settings['font_size'])
    ax.set_yticks([])
    
    plt.tight_layout()
    return fig

def plot_stacked_spectra(file_paths, file_names, mass_configs, plot_settings, is_hybrid=False):
    """Plot stacked spectra - regular or hybrid mode"""
    fig, ax = plt.subplots(figsize=(plot_settings['width'], plot_settings['height']), dpi=300)
    
    # Set styling
    if plot_settings['background'] != "white":
        fig.patch.set_facecolor(plot_settings['background'])
        ax.set_facecolor(plot_settings['background'])
    plt.rcParams.update({'font.size': plot_settings['font_size']})
    
    data_frames = []
    max_intensity_overall = 0
    
    # Load and plot spectra
    for i, (file_path, file_name) in enumerate(zip(file_paths, file_names)):
        df = read_mass_spectrum_file(file_path)
        if df.empty:
            continue
            
        df = df[(df['m/z'] >= plot_settings['x_min']) & (df['m/z'] <= plot_settings['x_max'])]
        data_frames.append(df)
        
        max_intensity = df['intensity'].max()
        max_intensity_overall = max(max_intensity_overall, max_intensity)
        
        # Normalize and stack
        df['normalized'] = df['intensity'] / max_intensity * plot_settings['zoom_factors'][i]
        staggered = df['normalized'] + i * 1.2
        
        ax.plot(df['m/z'], staggered, color=plot_settings['line_color'], 
                linewidth=plot_settings['line_width'], linestyle=plot_settings['line_style'])
        
        # Spectrum labels
        x_range = plot_settings['x_max'] - plot_settings['x_min']
        label_x = plot_settings['x_max'] - (0.05 * x_range)
        ax.text(label_x, staggered.iloc[0] + 0.1, plot_settings['titles'][i], 
                fontsize=plot_settings['font_size'], ha='right')
    
    ax.set_xlim(plot_settings['x_min'], plot_settings['x_max'])
    ax.set_ylim(-0.1, 0.3 + len(file_paths) * 1.2)
    
    # Add annotations
    if is_hybrid:
        # Hybrid mode - each spectrum gets its own annotations
        legend_entries = []
        for spectrum_idx, config in enumerate(mass_configs):
            if spectrum_idx >= len(data_frames):
                continue
                
            df = data_frames[spectrum_idx]
            color = config['color']
            shape = config['shape']
            mass = config['mass']
            charge_states = config['charge_states']
            
            legend_entries.append((shape, color, mass, spectrum_idx))
            
            for charge in charge_states:
                mz = (mass + charge) / charge
                if plot_settings['x_min'] <= mz <= plot_settings['x_max']:
                    # Find intensity at this m/z in this spectrum
                    width = 100 / charge
                    intensity_in_range = df[(df['m/z'] >= mz - width/2) & (df['m/z'] <= mz + width/2)]
                    
                    if not intensity_in_range.empty:
                        peak_intensity = intensity_in_range['intensity'].max()
                        if peak_intensity >= plot_settings['threshold'] * max_intensity_overall:
                            # Position relative to this spectrum's baseline
                            spectrum_baseline = spectrum_idx * 1.2
                            normalized_peak = peak_intensity / df['intensity'].max() * plot_settings['zoom_factors'][spectrum_idx]
                            y_marker = spectrum_baseline + normalized_peak + plot_settings['offset']
                            y_label = y_marker + plot_settings['offset']
                            
                            ax.scatter([mz], [y_marker], color=color, marker=shape, s=plot_settings['marker_size'])
                            ax.text(mz, y_label, f'{charge}+', ha='center', 
                                   fontsize=plot_settings['font_size'], color=color)
        
        # Hybrid legend
        if plot_settings['show_legend'] and legend_entries:
            legend_elements = [plt.Line2D([0], [0], marker=symbol, color='w', markerfacecolor=color, 
                                         markersize=8, label=f'{mass} Da (Spectrum {spec_idx+1})') 
                              for symbol, color, mass, spec_idx in legend_entries]
            ax.legend(handles=legend_elements, loc=plot_settings['legend_pos'], frameon=False)
    
    else:
        # Regular stacked mode - shared annotations across all spectra
        for config in mass_configs:
            color = config['color']
            mass = config['mass']
            charge_states = config['charge_states']
            
            for charge in charge_states:
                mz = (mass + charge) / charge
                if plot_settings['x_min'] <= mz <= plot_settings['x_max']:
                    # Highlight across all spectra
                    thickness = 100 / charge
                    ax.fill_betweenx([-0.1, 0.3 + len(file_paths)*1.2], 
                                    mz - thickness/2, mz + thickness/2, 
                                    color=color, alpha=0.1)
                    
                    # Check if significant in any spectrum
                    max_in_range = 0
                    for df in data_frames:
                        if df.empty:
                            continue
                        intensities = df[(df['m/z'] >= mz - thickness/2) & (df['m/z'] <= mz + thickness/2)]
                        if not intensities.empty:
                            max_in_range = max(max_in_range, intensities['intensity'].max())
                    
                    if max_in_range > plot_settings['threshold'] * max_intensity_overall:
                        y_pos = 0.3 + len(file_paths)*1.2 - 0.1 * (mass_configs.index(config) + 1)
                        ax.text(mz, y_pos, f'{charge}+', ha='center', 
                               fontsize=plot_settings['font_size'], color=color)
    
    # Formatting
    if plot_settings['show_grid']:
        ax.grid(True, linestyle=plot_settings['grid_style'], alpha=plot_settings['grid_alpha'])
    ax.set_xlabel('m/z', fontsize=plot_settings['font_size'])
    ax.set_title(plot_settings['title'], fontsize=plot_settings['font_size'])
    ax.set_yticks([])
    
    plt.tight_layout()
    return fig

# UI
st.markdown('<h3 class="section-header">🎯 Plot Type Selection</h3>', unsafe_allow_html=True)
plot_type = st.selectbox("Select plot type:", 
                        ["Single Spectrum", "Stacked Comparison", "Hybrid Stacked"])

# Basic settings
st.markdown('<h3 class="section-header">⚙️ Basic Settings</h3>', unsafe_allow_html=True)
col1, col2, col3 = st.columns(3)

with col1:
    figure_width = st.number_input("Width:", 2.0, 20.0, 10.0, 0.5)
    figure_height = st.number_input("Height:", 2.0, 15.0, 6.0, 0.5)

with col2:
    font_size = st.number_input("Font size:", 6, 24, 12)
    x_min = st.number_input("X-axis min:", 0.0, value=50.0, step=50.0)

with col3:
    x_max = st.number_input("X-axis max:", 100.0, value=5000.0, step=100.0)
    line_width = st.number_input("Line width:", 0.1, 5.0, 1.0, 0.1)

# Style settings
col1, col2, col3, col4 = st.columns(4)
with col1:
    line_color = st.selectbox("Line color:", ["black", "blue", "red", "green", "purple"])
with col2:
    background = st.selectbox("Background:", ["white", "lightgray", "transparent"])
with col3:
    show_grid = st.checkbox("Show grid")
with col4:
    grid_alpha = st.slider("Grid transparency:", 0.0, 1.0, 0.3)

# File upload
st.markdown('<h3 class="section-header">📁 File Upload</h3>', unsafe_allow_html=True)

if plot_type == "Single Spectrum":
    uploaded_file = st.file_uploader("Upload spectrum file", type=['txt', 'csv', 'tsv'])
    uploaded_files = [uploaded_file] if uploaded_file else []
    file_names = [uploaded_file.name] if uploaded_file else []
else:
    uploaded_files = st.file_uploader("Upload spectrum files", type=['txt', 'csv', 'tsv'], 
                                     accept_multiple_files=True)
    file_names = [f.name for f in uploaded_files]

if uploaded_files and all(f is not None for f in uploaded_files):
    # Mass and annotation configuration
    st.markdown('<h3 class="section-header">🔬 Mass Configuration</h3>', unsafe_allow_html=True)
    
    # Color palette
    palette = st.selectbox("Color palette:", ["Scientific", "Nature", "Colorblind Safe", "Viridis", "Plasma"])
    
    # Shape options
    available_shapes = ['o', 's', 'v', '^', '*', 'D', 'h', '<', '>', 'p']
    shape_names = ['Circle', 'Square', 'Triangle Down', 'Triangle Up', 'Star', 'Diamond', 
                   'Hexagon', 'Left Triangle', 'Right Triangle', 'Pentagon']
    shape_mapping = dict(zip(shape_names, available_shapes))
    
    if plot_type == "Hybrid Stacked":
        st.write("**Configure one mass per spectrum:**")
        mass_configs = []
        colors = get_color_palette(palette, len(uploaded_files))
        
        for i, file_name in enumerate(file_names):
            st.write(f"**Spectrum {i+1}: {file_name}**")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mass = st.number_input(f"Mass (Da):", value=15000.0, key=f"hybrid_mass_{i}")
            with col2:
                shape_name = st.selectbox("Shape:", shape_names, key=f"hybrid_shape_{i}")
                shape = shape_mapping[shape_name]
            with col3:
                charge_input = st.text_input("Charge states:", "10,11,12,13", key=f"hybrid_charges_{i}")
            with col4:
                st.write(f"Color: {palette}")
                
            try:
                charge_states = [int(c.strip()) for c in charge_input.split(',') if c.strip()]
                mass_configs.append({
                    'mass': mass,
                    'charge_states': charge_states,
                    'color': colors[i],
                    'shape': shape
                })
            except ValueError:
                st.error(f"Invalid charge states for spectrum {i+1}")
                
    else:
        st.write("**Configure masses to annotate:**")
        num_masses = st.number_input("Number of masses:", 1, 10, 1)
        mass_configs = []
        colors = get_color_palette(palette, num_masses)
        
        for i in range(num_masses):
            st.write(f"**Mass {i+1}:**")
            col1, col2, col3 = st.columns(3)
            
            with col1:
                mass = st.number_input(f"Mass (Da):", value=15000.0, key=f"mass_{i}")
            with col2:
                shape_name = st.selectbox("Shape:", shape_names, key=f"shape_{i}")
                shape = shape_mapping[shape_name]
            with col3:
                charge_input = st.text_input("Charge states:", "10,11,12,13", key=f"charges_{i}")
                
            try:
                charge_states = [int(c.strip()) for c in charge_input.split(',') if c.strip()]
                mass_configs.append({
                    'mass': mass,
                    'charge_states': charge_states,
                    'color': colors[i],
                    'shape': shape
                })
            except ValueError:
                st.error(f"Invalid charge states for mass {i+1}")
    
    # Plot-specific settings
    col1, col2 = st.columns(2)
    with col1:
        plot_title = st.text_input("Plot title:", "Mass Spectrum")
        marker_size = st.number_input("Marker size:", 10, 200, 50)
    with col2:
        show_legend = st.checkbox("Show legend", True)
        legend_pos = st.selectbox("Legend position:", ["upper right", "upper left", "lower right", "lower left"])
    
    # Advanced settings
    with st.expander("Advanced Settings"):
        col1, col2, col3 = st.columns(3)
        with col1:
            threshold = st.slider("Annotation threshold:", 0.01, 1.0, 0.01)
        with col2:
            offset = st.slider("Label offset:", 0.05, 0.5, 0.1)
        with col3:
            if plot_type == "Single Spectrum":
                zoom = st.number_input("Y-axis zoom:", 0.1, 10.0, 1.4)
            else:
                zoom_factors = []
                for i, name in enumerate(file_names):
                    zoom = st.number_input(f"Zoom {name}:", 0.1, 10.0, 1.0, key=f"zoom_{i}")
                    zoom_factors.append(zoom)
    
    # Generate plot
    if st.button("Generate Plot", type="primary"):
        # Save temp files
        temp_paths = []
        for file in uploaded_files:
            temp_path = f"temp_{file.name}"
            with open(temp_path, "wb") as f:
                f.write(file.getbuffer())
            temp_paths.append(temp_path)
        
        # Plot settings
        plot_settings = {
            'width': figure_width, 'height': figure_height, 'font_size': font_size,
            'x_min': x_min, 'x_max': x_max, 'line_width': line_width,
            'line_color': line_color, 'line_style': '-', 'background': background,
            'show_grid': show_grid, 'grid_style': '--', 'grid_alpha': grid_alpha,
            'threshold': threshold, 'offset': offset, 'marker_size': marker_size,
            'show_legend': show_legend, 'legend_pos': legend_pos, 'title': plot_title
        }
        
        if plot_type == "Single Spectrum":
            plot_settings['zoom'] = zoom
            fig = plot_single_spectrum(temp_paths[0], mass_configs, plot_settings)
        else:
            plot_settings['zoom_factors'] = zoom_factors
            plot_settings['titles'] = [f"Spectrum {i+1}" for i in range(len(file_names))]
            is_hybrid = (plot_type == "Hybrid Stacked")
            fig = plot_stacked_spectra(temp_paths, file_names, mass_configs, plot_settings, is_hybrid)
        
        if fig:
            st.pyplot(fig)
            
            # Download
            buf = io.BytesIO()
            fig.savefig(buf, format='png', dpi=300, bbox_inches='tight')
            buf.seek(0)
            
            st.download_button("📊 Download Plot", buf.getvalue(), 
                             f"{plot_type.lower().replace(' ', '_')}_plot.png", "image/png")
        
        # Cleanup
        for temp_path in temp_paths:
            if os.path.exists(temp_path):
                os.remove(temp_path)

# Usage guide
st.markdown("""
<div class="info-card">
    <h3>ℹ️ Usage Guide</h3>
    <p><strong>Single Spectrum:</strong> One file, multiple masses with custom charge states</p>
    <p><strong>Stacked Comparison:</strong> Multiple files, shared annotations across all spectra</p>
    <p><strong>Hybrid Stacked:</strong> Multiple files, each with its own mass and annotations</p>
    <p><strong>Charge States:</strong> Enter as comma-separated numbers (e.g., "10,11,12,13")</p>
</div>
""", unsafe_allow_html=True)