import streamlit as st
import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import io
from scipy import signal
from scipy.signal import savgol_filter
from scipy.interpolate import interp1d
from matplotlib.patches import Rectangle
from matplotlib.ticker import MaxNLocator, MultipleLocator
import matplotlib.patches as mpatches
from myutils import styling

# Apply custom styling
styling.load_custom_css()

# Main header
st.markdown(
    '<div class="main-header">'
    '<h1>📊 Advanced Mass Spectrum Plotting</h1>'
    '<p>Create publication-ready mass spectrum plots with extensive customization and data processing</p>'
    '</div>',
    unsafe_allow_html=True
)

# Info card
st.markdown("""
<div class="info-card">
    <p>Professional mass spectrometry plotting tool with advanced features for publication-quality figures.</p>
    <p><strong>Plot Types:</strong></p>
    <ul>
        <li><strong>Single Spectrum:</strong> Detailed single spectrum with advanced annotations</li>
        <li><strong>Stacked Comparison:</strong> Multiple spectra with shared annotations</li>
        <li><strong>Hybrid Stacked:</strong> Individual spectrum annotations and processing</li>
        <li><strong>Mirror Plot:</strong> Two spectra mirrored for comparison</li>
        <li><strong>Overlay Plot:</strong> Multiple spectra overlaid with transparency</li>
    </ul>
    <p><strong>Advanced Features:</strong> Data smoothing, baseline correction, peak detection, noise filtering, isotope patterns, and more.</p>
</div>
""", unsafe_allow_html=True)

# Enhanced plotting functions
def read_mass_spectrum_file(file_path):
    """Read mass spectrum file with enhanced error handling"""
    try:
        # Try different separators and encodings
        for sep in ['\t', ',', ' ', ';']:
            try:
                df = pd.read_csv(file_path, sep=sep, header=None, on_bad_lines='skip', encoding='utf-8')
                if len(df.columns) >= 2:
                    break
            except:
                continue
        
        # Take first two columns as m/z and intensity
        df = df.iloc[:, :2]
        df.columns = ['m/z', 'intensity']
        df['m/z'] = pd.to_numeric(df['m/z'], errors='coerce')
        df['intensity'] = pd.to_numeric(df['intensity'], errors='coerce')
        df.dropna(inplace=True)
        df = df.sort_values('m/z').reset_index(drop=True)
        return df
    except Exception as e:
        st.error(f"Error reading file {file_path}: {e}")
        return pd.DataFrame(columns=['m/z', 'intensity'])

def process_spectrum_data(df, processing_options):
    """Apply data processing to spectrum"""
    if df.empty:
        return df
    
    processed_df = df.copy()
    
    # Intensity thresholding
    if processing_options.get('intensity_threshold', 0) > 0:
        threshold = processing_options['intensity_threshold'] * processed_df['intensity'].max()
        processed_df = processed_df[processed_df['intensity'] >= threshold]
    
    # Smoothing
    if processing_options.get('smoothing', False):
        window_length = min(processing_options.get('smooth_window', 5), len(processed_df) - 1)
        if window_length >= 3 and window_length % 2 == 1:
            try:
                processed_df['intensity'] = savgol_filter(
                    processed_df['intensity'], 
                    window_length, 
                    processing_options.get('smooth_order', 2)
                )
            except:
                pass
    
    # Baseline correction
    if processing_options.get('baseline_correction', False):
        try:
            baseline = np.percentile(processed_df['intensity'], processing_options.get('baseline_percentile', 5))
            processed_df['intensity'] = processed_df['intensity'] - baseline
            processed_df['intensity'] = np.maximum(processed_df['intensity'], 0)
        except:
            pass
    
    # Normalization
    if processing_options.get('normalize', False):
        norm_type = processing_options.get('normalize_type', 'max')
        if norm_type == 'max':
            max_val = processed_df['intensity'].max()
            if max_val > 0:
                processed_df['intensity'] = processed_df['intensity'] / max_val
        elif norm_type == 'sum':
            sum_val = processed_df['intensity'].sum()
            if sum_val > 0:
                processed_df['intensity'] = processed_df['intensity'] / sum_val
        elif norm_type == 'tic':
            processed_df['intensity'] = processed_df['intensity'] / processed_df['intensity'].sum() * 100
    
    # Binning/resampling
    if processing_options.get('binning', False):
        bin_size = processing_options.get('bin_size', 1.0)
        mz_min, mz_max = processed_df['m/z'].min(), processed_df['m/z'].max()
        bins = np.arange(mz_min, mz_max + bin_size, bin_size)
        processed_df['bin'] = pd.cut(processed_df['m/z'], bins, labels=False)
        processed_df = processed_df.groupby('bin').agg({
            'm/z': 'mean',
            'intensity': processing_options.get('bin_method', 'max')
        }).reset_index(drop=True)
        processed_df.dropna(inplace=True)
    
    return processed_df

def detect_peaks(df, prominence=0.01, distance=50):
    """Detect peaks in spectrum"""
    if df.empty:
        return []
    
    peaks, properties = signal.find_peaks(
        df['intensity'].values,
        prominence=prominence * df['intensity'].max(),
        distance=distance
    )
    
    peak_data = []
    for peak in peaks:
        peak_data.append({
            'm/z': df.iloc[peak]['m/z'],
            'intensity': df.iloc[peak]['intensity'],
            'prominence': properties['prominences'][np.where(peaks == peak)[0][0]]
        })
    
    return peak_data

def get_extended_color_palette(palette_name, n_colors):
    """Extended color palette options"""
    palettes = {
        "Scientific": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf'],
        "Nature": ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#FBAFE4', '#949494', '#ECE133', '#56B4E9', '#F0E442'],
        "Science": ['#CC503E', '#0F8554', '#B24745', '#73AC27', '#C49A00', '#00A1C9', '#8B0000', '#006400', '#FF8C00', '#8A2BE2'],
        "Colorblind Safe": sns.color_palette("colorblind", n_colors),
        "High Contrast": ['#000000', '#FF0000', '#0000FF', '#00FF00', '#FF00FF', '#FFFF00', '#00FFFF', '#800000', '#008000', '#000080'],
        "Pastel": sns.color_palette("pastel", n_colors),
        "Dark": sns.color_palette("dark", n_colors),
        "Viridis": plt.cm.viridis(np.linspace(0, 1, n_colors)),
        "Plasma": plt.cm.plasma(np.linspace(0, 1, n_colors)),
        "Magma": plt.cm.magma(np.linspace(0, 1, n_colors)),
        "Cividis": plt.cm.cividis(np.linspace(0, 1, n_colors)),
        "Spectral": plt.cm.Spectral(np.linspace(0, 1, n_colors)),
        "Cool": plt.cm.cool(np.linspace(0, 1, n_colors)),
        "Hot": plt.cm.hot(np.linspace(0, 1, n_colors)),
        "Autumn": plt.cm.autumn(np.linspace(0, 1, n_colors)),
        "Winter": plt.cm.winter(np.linspace(0, 1, n_colors)),
        "Spring": plt.cm.spring(np.linspace(0, 1, n_colors)),
        "Summer": plt.cm.summer(np.linspace(0, 1, n_colors)),
    }
    
    if palette_name in palettes:
        colors = palettes[palette_name]
        if hasattr(colors, '__iter__') and hasattr(colors[0], '__len__') and len(colors[0]) == 4:  # For colormap colors (RGBA)
            return [colors[i] for i in range(min(n_colors, len(colors)))]
        elif isinstance(colors, list) and isinstance(colors[0], str):  # For hex color strings
            return colors[:n_colors] if len(colors) >= n_colors else colors * ((n_colors // len(colors)) + 1)
        else:  # For seaborn palettes or numpy arrays
            return list(colors)[:n_colors]
    else:
        return sns.color_palette("husl", n_colors)

def plot_enhanced_spectrum(file_paths, file_names, mass_configs, plot_settings, processing_options, plot_type="single"):
    """Enhanced plotting function with manual peak identification and fixed transparent background"""
    
    # Figure setup
    fig_size = (plot_settings['width'], plot_settings['height'])
    
    if plot_type == "mirror":
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=fig_size, dpi=plot_settings['dpi'], sharex=True)
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(figsize=fig_size, dpi=plot_settings['dpi'])
        axes = [ax]
    
    # Fix transparent background handling
    if plot_settings['background'] == "transparent":
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0)
        for axis in axes:
            axis.patch.set_facecolor('none')
            axis.patch.set_alpha(0)
    elif plot_settings['background'] != "white":
        fig.patch.set_facecolor(plot_settings['background'])
        for axis in axes:
            axis.set_facecolor(plot_settings['background'])
    
    plt.rcParams.update({
        'font.size': plot_settings['font_size'],
        'font.family': plot_settings.get('font_family', 'Arial'),
        'font.weight': plot_settings.get('font_weight', 'normal')
    })
    
    # Process and plot data
    data_frames = []
    max_intensity_overall = 0
    
    for i, (file_path, file_name) in enumerate(zip(file_paths, file_names)):
        df = read_mass_spectrum_file(file_path)
        if df.empty:
            continue
        
        # Apply processing
        df = process_spectrum_data(df, processing_options)
        
        # Filter by m/z range
        df = df[(df['m/z'] >= plot_settings['x_min']) & (df['m/z'] <= plot_settings['x_max'])]
        data_frames.append(df)
        
        max_intensity = df['intensity'].max()
        max_intensity_overall = max(max_intensity_overall, max_intensity)
        
        # Select axis for plotting
        current_ax = axes[0] if plot_type != "mirror" else axes[i % 2]
        
        # Plot spectrum based on type
        if plot_type == "single":
            current_ax.plot(df['m/z'], df['intensity'], 
                           color=plot_settings['line_color'], 
                           linewidth=plot_settings['line_width'], 
                           linestyle=plot_settings['line_style'],
                           alpha=plot_settings.get('alpha', 1.0),
                           label=file_name if plot_settings.get('show_file_labels', False) else None)
            
            # Fill under curve if requested
            if plot_settings.get('fill_under', False):
                current_ax.fill_between(df['m/z'], df['intensity'], 
                                      alpha=plot_settings.get('fill_alpha', 0.3),
                                      color=plot_settings['line_color'])
        
        elif plot_type == "stacked":
            df['normalized'] = df['intensity'] / max_intensity * plot_settings['zoom_factors'][i]
            staggered = df['normalized'] + i * plot_settings.get('stack_offset', 1.2)
            
            current_ax.plot(df['m/z'], staggered, 
                           color=plot_settings['line_color'], 
                           linewidth=plot_settings['line_width'], 
                           linestyle=plot_settings['line_style'])
            
            # Spectrum labels
            x_range = plot_settings['x_max'] - plot_settings['x_min']
            label_x = plot_settings['x_max'] - (0.05 * x_range)
            label_text = plot_settings.get('titles', [f"Spectrum {i+1}"])[i]
            current_ax.text(label_x, staggered.iloc[0] + 0.1, label_text, 
                           fontsize=plot_settings['font_size'], ha='right')
        
        elif plot_type == "overlay":
            colors = get_extended_color_palette(plot_settings.get('palette', 'Scientific'), len(file_paths))
            current_ax.plot(df['m/z'], df['intensity'], 
                           color=colors[i], 
                           linewidth=plot_settings['line_width'], 
                           linestyle=plot_settings['line_style'],
                           alpha=plot_settings.get('alpha', 0.7),
                           label=file_name)
        
        elif plot_type == "mirror":
            if i == 0:
                current_ax.plot(df['m/z'], df['intensity'], 
                               color=plot_settings['line_color'], 
                               linewidth=plot_settings['line_width'])
                current_ax.fill_between(df['m/z'], df['intensity'], 
                                      alpha=plot_settings.get('fill_alpha', 0.3),
                                      color=plot_settings['line_color'])
            else:
                current_ax.plot(df['m/z'], -df['intensity'], 
                               color=plot_settings.get('line_color_2', 'red'), 
                               linewidth=plot_settings['line_width'])
                current_ax.fill_between(df['m/z'], -df['intensity'], 
                                      alpha=plot_settings.get('fill_alpha', 0.3),
                                      color=plot_settings.get('line_color_2', 'red'))
                current_ax.axhline(y=0, color='black', linewidth=0.5)
    
    # Peak detection and annotation
    if plot_settings.get('show_peaks', False) and data_frames:
        for i, df in enumerate(data_frames):
            peaks = detect_peaks(df, 
                                prominence=plot_settings.get('peak_prominence', 0.01),
                                distance=plot_settings.get('peak_distance', 50))
            
            current_ax = axes[0] if plot_type != "mirror" else axes[i % 2]
            
            for peak in peaks[:plot_settings.get('max_peaks', 10)]:  # Limit number of peaks
                current_ax.scatter([peak['m/z']], [peak['intensity']], 
                                 color='red', marker='x', s=50, zorder=5)
                if plot_settings.get('show_peak_labels', False):
                    current_ax.annotate(f"{peak['m/z']:.1f}", 
                                      (peak['m/z'], peak['intensity']),
                                      xytext=(5, 5), textcoords='offset points',
                                      fontsize=plot_settings['font_size']-2)
    
    # Mass annotations with correct m/z calculations
    for config in mass_configs:
        if config.get('mode') == 'manual':
            # Manual peak identification
            target_mz = config['mz']
            tolerance = config['tolerance']
            color = config['color']
            shape = config['shape']
            label_text = config['label']
            
            # Find peaks within tolerance for each spectrum
            for df_idx, df in enumerate(data_frames):
                if df.empty:
                    continue
                
                current_ax = axes[0] if plot_type != "mirror" else axes[df_idx % 2]
                
                # Find intensity within tolerance
                peak_data = df[(df['m/z'] >= target_mz - tolerance) & 
                              (df['m/z'] <= target_mz + tolerance)]
                
                if not peak_data.empty:
                    # Get the maximum intensity peak within tolerance
                    max_intensity_idx = peak_data['intensity'].idxmax()
                    peak_mz = df.loc[max_intensity_idx, 'm/z']
                    peak_intensity = df.loc[max_intensity_idx, 'intensity']
                    
                    y_marker = peak_intensity + plot_settings.get('offset', 0.1) * max_intensity_overall
                    y_label = y_marker + plot_settings.get('offset', 0.1) * max_intensity_overall
                    
                    # Annotation line
                    if plot_settings.get('show_annotation_lines', True):
                        current_ax.plot([peak_mz, peak_mz], [peak_intensity, y_marker], 
                                      color=color, linewidth=1, alpha=0.7)
                    
                    # Marker
                    current_ax.scatter([peak_mz], [y_marker], color=color, marker=shape, 
                                     s=plot_settings.get('marker_size', 80), 
                                     edgecolor='black', linewidth=0.5, zorder=5)
                    
                    # Enhanced label styling
                    label_kwargs = {
                        'ha': 'center', 
                        'va': 'bottom',
                        'fontsize': plot_settings.get('label_font_size', plot_settings['font_size']), 
                        'color': color, 
                        'weight': plot_settings.get('label_font_weight', 'bold')
                    }
                    
                    if plot_settings.get('show_label_borders', True):
                        border_style = plot_settings.get('label_border_style', 'round')
                        boxstyle_map = {
                            'round': "round,pad=0.3",
                            'square': "square,pad=0.3", 
                            'sawtooth': "sawtooth,pad=0.3",
                            'circle': "circle,pad=0.3"
                        }
                        
                        bg_color = plot_settings.get('label_background_color', 'white')
                        if bg_color == 'auto':
                            bg_color = 'white' if plot_settings['background'] not in ['transparent', 'white'] else 'lightgray'
                        elif bg_color == 'transparent':
                            bg_color = 'none'
                        
                        border_color = plot_settings.get('label_border_color', 'black')
                        if border_color == 'auto':
                            border_color = color
                        
                        label_kwargs['bbox'] = dict(
                            boxstyle=boxstyle_map.get(border_style, "round,pad=0.3"),
                            facecolor=bg_color,
                            edgecolor=border_color,
                            linewidth=plot_settings.get('label_border_width', 0.5),
                            alpha=plot_settings.get('label_background_alpha', 0.8)
                        )
                    
                    current_ax.text(peak_mz, y_label, label_text, **label_kwargs)
        
        elif config.get('mode') == 'automatic':
            # Existing automatic mass calculation code
            color = config['color']
            shape = config['shape']
            mass = config['mass']
            charge_states = config['charge_states']
            
            for charge in charge_states:
                mz = (mass + charge * 1.007276) / charge
                if plot_settings['x_min'] <= mz <= plot_settings['x_max']:
                    
                    for df_idx, df in enumerate(data_frames):
                        if df.empty:
                            continue
                        
                        current_ax = axes[0] if plot_type != "mirror" else axes[df_idx % 2]
                        
                        width = plot_settings.get('annotation_width', 100) / charge
                        intensity_in_range = df[(df['m/z'] >= mz - width/2) & (df['m/z'] <= mz + width/2)]
                        
                        if not intensity_in_range.empty:
                            peak_intensity = intensity_in_range['intensity'].max()
                            threshold_intensity = plot_settings.get('threshold', 0.01) * max_intensity_overall
                            
                            if peak_intensity >= threshold_intensity:
                                y_marker = peak_intensity + plot_settings.get('offset', 0.1) * max_intensity_overall
                                y_label = y_marker + plot_settings.get('offset', 0.1) * max_intensity_overall
                                
                                if plot_settings.get('show_annotation_lines', True):
                                    current_ax.plot([mz, mz], [peak_intensity, y_marker], 
                                                  color=color, linewidth=1, alpha=0.7)
                                
                                current_ax.scatter([mz], [y_marker], color=color, marker=shape, 
                                                 s=plot_settings.get('marker_size', 80), 
                                                 edgecolor='black', linewidth=0.5, zorder=5)
                                
                                label_text = f'{charge}+'
                                if plot_settings.get('show_mz_labels', False):
                                    label_text += f'\n{mz:.1f}'
                                if plot_settings.get('show_mass_labels', False):
                                    label_text += f'\n{mass:.0f}Da'
                                
                                # Same label styling as manual mode...
                                # [Include the same label styling code as above]
    
    # Enhanced axis formatting and styling with full customization
    for ax_idx, current_ax in enumerate(axes):
        # X-axis settings
        current_ax.set_xlim(plot_settings['x_min'], plot_settings['x_max'])
        
        # X-axis label
        if plot_settings.get('show_x_label', True):
            current_ax.set_xlabel(
                plot_settings.get('x_label_text', 'm/z'), 
                fontsize=plot_settings.get('axis_label_size', 14), 
                weight=plot_settings.get('axis_label_weight', 'bold')
            )
        else:
            current_ax.set_xlabel('')
        
        # Y-axis label
        if plot_type == "mirror" and ax_idx == 1:
            if plot_settings.get('show_y_label', True):
                current_ax.set_ylabel(
                    plot_settings.get('y_label_text', 'Intensity'), 
                    fontsize=plot_settings.get('axis_label_size', 14), 
                    weight=plot_settings.get('axis_label_weight', 'bold')
                )
            current_ax.invert_yaxis()
        elif ax_idx == 0 and plot_settings.get('show_y_label', True):
            current_ax.set_ylabel(
                plot_settings.get('y_label_text', 'Intensity'), 
                fontsize=plot_settings.get('axis_label_size', 14), 
                weight=plot_settings.get('axis_label_weight', 'bold')
            )
        elif not plot_settings.get('show_y_label', True):
            current_ax.set_ylabel('')
        
        # Y-axis limits
        if plot_type == "single":
            if processing_options.get('normalize', False):
                current_ax.set_ylim(0, plot_settings.get('zoom', 1.1))
            else:
                current_ax.set_ylim(-0.05 * max_intensity_overall, 
                                   plot_settings.get('zoom', 1.4) * max_intensity_overall)
        elif plot_type == "stacked":
            current_ax.set_ylim(-0.1, 0.3 + len(file_paths) * plot_settings.get('stack_offset', 1.2))
        
        # Advanced spine visibility controls
        spine_width = plot_settings.get('spine_width', 1.0)
        spine_color = plot_settings.get('spine_color', 'black')
        
        # Set spine visibility
        current_ax.spines['bottom'].set_visible(plot_settings.get('show_bottom_axis', True))
        current_ax.spines['top'].set_visible(plot_settings.get('show_top_axis', False))
        current_ax.spines['left'].set_visible(plot_settings.get('show_left_axis', True))
        current_ax.spines['right'].set_visible(plot_settings.get('show_right_axis', False))
        
        # Set spine styling for visible spines
        for spine_name, spine in current_ax.spines.items():
            if spine.get_visible():
                spine.set_linewidth(spine_width)
                spine.set_color(spine_color)
        
        # Grid
        if plot_settings.get('show_grid', False):
            current_ax.grid(True, linestyle=plot_settings.get('grid_style', '--'), 
                           alpha=plot_settings.get('grid_alpha', 0.3),
                           color=plot_settings.get('grid_color', 'gray'))
        
        # Enhanced tick controls
        if plot_settings.get('custom_x_ticks', False):
            x_tick_spacing = plot_settings.get('x_tick_spacing', 500)
            current_ax.xaxis.set_major_locator(MultipleLocator(x_tick_spacing))
        
        # X-tick labels
        if not plot_settings.get('show_x_tick_labels', True):
            current_ax.set_xticklabels([])
        else:
            current_ax.tick_params(axis='x', labelsize=plot_settings.get('tick_label_size', 12))
        
        # Y-tick controls
        if plot_settings.get('hide_y_ticks', True):
            current_ax.set_yticks([])
        else:
            current_ax.yaxis.set_major_locator(MaxNLocator(nbins=plot_settings.get('y_tick_count', 5)))
            if not plot_settings.get('show_y_tick_labels', True):
                current_ax.set_yticklabels([])
            else:
                current_ax.tick_params(axis='y', labelsize=plot_settings.get('tick_label_size', 12))
    
    # Title and legend
    if plot_settings.get('title', ''):
        fig.suptitle(plot_settings['title'], 
                    fontsize=plot_settings['font_size']+2, 
                    weight=plot_settings.get('title_weight', 'bold'))  # Use the setting instead of hardcoded 'bold'
    
    if plot_settings.get('show_legend', False):
        if plot_type == "overlay":
            axes[0].legend(loc=plot_settings.get('legend_pos', 'upper right'), 
                          frameon=plot_settings.get('legend_frame', False),
                          fancybox=True, shadow=True)
        elif mass_configs:
            legend_elements = []
            for config in mass_configs:
                legend_elements.append(
                    plt.Line2D([0], [0], marker=config['shape'], color='w', 
                              markerfacecolor=config['color'], markersize=8, 
                              label=f"{config['mass']:.0f} Da")
                )
            axes[0].legend(handles=legend_elements, loc=plot_settings.get('legend_pos', 'upper right'),
                          frameon=plot_settings.get('legend_frame', False))
    
    plt.tight_layout()
    return fig

# Streamlit UI
st.markdown('<h3 class="section-header">🎯 Advanced Plot Configuration</h3>', unsafe_allow_html=True)

# Plot type selection
col1, col2 = st.columns(2)
with col1:
    plot_type = st.selectbox("Plot type:", 
                            ["Single Spectrum", "Stacked Comparison", "Hybrid Stacked", 
                             "Mirror Plot", "Overlay Plot"])
with col2:
    output_format = st.selectbox("Output format:", ["PNG", "PDF", "SVG", "EPS"])

# Figure settings
st.markdown('<h3 class="section-header">📐 Figure Settings</h3>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    figure_width = st.number_input("Width (inches):", 2.0, 20.0, 12.0, 0.5)
    figure_height = st.number_input("Height (inches):", 2.0, 15.0, 8.0, 0.5)

with col2:
    dpi = st.selectbox("DPI:", [150, 200, 300, 600], index=2)
    font_size = st.number_input("Font size:", 6, 24, 14)
    
    # Add warning for high DPI + large size combinations
    total_pixels = figure_width * dpi * figure_height * dpi
    if total_pixels > 178956970:
        st.warning(f"⚠️ Large image ({total_pixels:,} pixels). May cause display issues.")

with col3:
    font_family = st.selectbox("Font family:", ["Arial", "Times New Roman", "Helvetica", "DejaVu Sans"])
    font_weight = st.selectbox("Font weight:", ["normal", "bold"])

with col4:
    background = st.selectbox("Background:", ["white", "lightgray", "transparent"])
    
# Enhanced Axis settings
st.markdown('<h3 class="section-header">📊 Axis Settings</h3>', unsafe_allow_html=True)

# Main axis controls
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("**X-Axis Range**")
    x_min = st.number_input("X-axis min:", 0.0, value=50.0, step=50.0)
    x_max = st.number_input("X-axis max:", 100.0, value=5000.0, step=100.0)

with col2:
    st.write("**X-Axis Ticks**")
    custom_x_ticks = st.checkbox("Custom X ticks")
    if custom_x_ticks:
        x_tick_spacing = st.number_input("X tick spacing:", 100, 2000, 500)
    show_x_tick_labels = st.checkbox("Show X tick labels", True)

with col3:
    st.write("**Y-Axis Ticks**")
    hide_y_ticks = st.checkbox("Hide Y ticks", True)
    if not hide_y_ticks:
        show_y_tick_labels = st.checkbox("Show Y tick labels", True)
        y_tick_count = st.number_input("Max Y ticks:", 3, 15, 5)

with col4:
    st.write("**Grid Options**")
    show_grid = st.checkbox("Show grid")
    if show_grid:
        grid_alpha = st.slider("Grid transparency:", 0.0, 1.0, 0.3)
        grid_style = st.selectbox("Grid style:", ["--", "-", ":", "-."])

# Advanced axis visibility controls
st.markdown("**🎛️ Advanced Axis Controls**")
col1, col2, col3, col4 = st.columns(4)

with col1:
    st.write("**Axis Visibility**")
    show_bottom_axis = st.checkbox("Show bottom axis", True)
    show_top_axis = st.checkbox("Show top axis", False)
    show_left_axis = st.checkbox("Show left axis", True)
    show_right_axis = st.checkbox("Show right axis", False)

with col2:
    st.write("**Axis Labels**")
    show_x_label = st.checkbox("Show X-axis label", True)
    if show_x_label:
        x_label_text = st.text_input("X-axis label:", "m/z")
    show_y_label = st.checkbox("Show Y-axis label", True)
    if show_y_label:
        y_label_text = st.text_input("Y-axis label:", "Intensity")

with col3:
    st.write("**Label Styling**")
    axis_label_size = st.number_input("Axis label size:", 8, 24, 14)
    axis_label_weight = st.selectbox("Axis label weight:", ["normal", "bold"], index=1)
    tick_label_size = st.number_input("Tick label size:", 6, 20, 12)

with col4:
    st.write("**Spine Styling**")
    spine_width = st.number_input("Spine width:", 0.1, 3.0, 1.0)
    spine_color = st.selectbox("Spine color:", ["black", "gray", "darkgray", "white"])

# Line and color settings
st.markdown('<h3 class="section-header">🎨 Appearance Settings</h3>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    line_color = st.selectbox("Primary line color:", 
                             ["black", "blue", "red", "green", "purple", "orange", "brown"])
    line_width = st.number_input("Line width:", 0.1, 5.0, 1.5, 0.1)

with col2:
    line_style = st.selectbox("Line style:", ["-", "--", ":", "-."])
    alpha = st.slider("Line transparency:", 0.1, 1.0, 1.0)

with col3:
    fill_under = st.checkbox("Fill under curve")
    if fill_under:
        fill_alpha = st.slider("Fill transparency:", 0.1, 1.0, 0.3)

with col4:
    palette = st.selectbox("Color palette:", 
                          ["Scientific", "Nature", "Science", "Colorblind Safe", "High Contrast",
                           "Pastel", "Dark", "Viridis", "Plasma", "Magma", "Spectral", "Cool", 
                           "Hot", "Autumn", "Winter", "Spring", "Summer"])

# Data processing options
st.markdown('<h3 class="section-header">⚙️ Data Processing</h3>', unsafe_allow_html=True)

with st.expander("📈 Signal Processing", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.write("**Smoothing**")
        smoothing = st.checkbox("Apply smoothing")
        if smoothing:
            smooth_window = st.number_input("Window size:", 3, 51, 5, step=2)
            smooth_order = st.number_input("Polynomial order:", 1, 5, 2)
    
    with col2:
        st.write("**Baseline Correction**")
        baseline_correction = st.checkbox("Baseline correction")
        if baseline_correction:
            baseline_percentile = st.slider("Baseline percentile:", 1, 20, 5)
    
    with col3:
        st.write("**Filtering**")
        intensity_threshold = st.slider("Intensity threshold (%):", 0.0, 10.0, 0.0)

with st.expander("📊 Normalization & Binning", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**Normalization**")
        normalize = st.checkbox("Normalize data")
        if normalize:
            normalize_type = st.selectbox("Normalization type:", ["max", "sum", "tic"])
    
    with col2:
        st.write("**Binning**")
        binning = st.checkbox("Apply binning")
        if binning:
            bin_size = st.number_input("Bin size (m/z):", 0.1, 10.0, 1.0)
            bin_method = st.selectbox("Binning method:", ["max", "mean", "sum"])

# Peak detection
with st.expander("🔍 Peak Detection", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        show_peaks = st.checkbox("Show detected peaks")
        if show_peaks:
            peak_prominence = st.slider("Peak prominence:", 0.001, 0.1, 0.01)
    
    with col2:
        if show_peaks:
            peak_distance = st.number_input("Min peak distance:", 10, 200, 50)
            max_peaks = st.number_input("Max peaks to show:", 5, 50, 10)
    
    with col3:
        if show_peaks:
            show_peak_labels = st.checkbox("Show peak m/z labels")

# File upload
st.markdown('<h3 class="section-header">📁 File Upload</h3>', unsafe_allow_html=True)

if plot_type == "Single Spectrum":
    uploaded_file = st.file_uploader("Upload spectrum file", type=['txt', 'csv', 'tsv'])
    uploaded_files = [uploaded_file] if uploaded_file else []
    file_names = [uploaded_file.name] if uploaded_file else []
else:
    max_files = 2 if plot_type == "Mirror Plot" else 10
    uploaded_files = st.file_uploader(f"Upload spectrum files (max {max_files})", 
                                     type=['txt', 'csv', 'tsv'], 
                                     accept_multiple_files=True)
    file_names = [f.name for f in uploaded_files]
    
    if len(uploaded_files) > max_files:
        st.warning(f"Only the first {max_files} files will be used.")
        uploaded_files = uploaded_files[:max_files]
        file_names = file_names[:max_files]

if uploaded_files and all(f is not None for f in uploaded_files):
    
    # Mass and annotation configuration
    st.markdown('<h3 class="section-header">🔬 Mass Configuration</h3>', unsafe_allow_html=True)
    
    # Enhanced shape options
    available_shapes = ['o', 's', 'v', '^', '<', '>', 'D', 'h', 'p', '*', '+', 'x', '8', 'H']
    shape_names = ['Circle', 'Square', 'Triangle Down', 'Triangle Up', 'Left Triangle', 
                   'Right Triangle', 'Diamond', 'Hexagon', 'Pentagon', 'Star', 'Plus', 
                   'X', 'Octagon', 'Hexagon2']
    shape_mapping = dict(zip(shape_names, available_shapes))
    
    # Number of masses to configure
    if plot_type == "Hybrid Stacked":
        st.write("**Configure one mass per spectrum:**")
        mass_configs = []
        colors = get_extended_color_palette(palette, len(uploaded_files))
        
        for i, file_name in enumerate(file_names):
            with st.expander(f"**Spectrum {i+1}: {file_name}**", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    mass = st.number_input(f"Mass (Da):", value=15000.0, key=f"hybrid_mass_{i}")
                with col2:
                    shape_name = st.selectbox("Shape:", shape_names, key=f"hybrid_shape_{i}")
                    shape = shape_mapping[shape_name]
                with col3:
                    charge_input = st.text_input("Charge states:", "10,11,12,13", key=f"hybrid_charges_{i}")
                with col4:
                    pass
                
                try:
                    charge_states = [int(c.strip()) for c in charge_input.split(',') if c.strip()]
                    mass_configs.append({
                        'mode': 'automatic',
                        'mass': mass,
                        'charge_states': charge_states,
                        'color': colors[i],
                        'shape': shape
                    })
                except ValueError:
                    st.error(f"Invalid charge states for spectrum {i+1}")
    
    else:
        num_masses = st.number_input("Number of masses to annotate:", 1, 10, 1)
        mass_configs = []
        colors = get_extended_color_palette(palette, num_masses)
        
        for i in range(num_masses):
            with st.expander(f"**Mass {i+1}**", expanded=True):
                col1, col2, col3, col4 = st.columns(4)
                
                with col1:
                    mass = st.number_input(f"Mass (Da):", value=15000.0 + i*1000, key=f"mass_{i}")
                with col2:
                    shape_name = st.selectbox("Shape:", shape_names, key=f"shape_{i}")
                    shape = shape_mapping[shape_name]
                with col3:
                    charge_input = st.text_input("Charge states:", "10,11,12,13", key=f"charges_{i}")
                with col4:
                    pass
                
                try:
                    charge_states = [int(c.strip()) for c in charge_input.split(',') if c.strip()]
                    mass_configs.append({
                        'mode': 'automatic',
                        'mass': mass,
                        'charge_states': charge_states,
                        'color': colors[i],
                        'shape': shape
                    })
                except ValueError:
                    st.error(f"Invalid charge states for mass {i+1}")
    
    # Enhanced Annotation Settings with manual peak identification
    st.markdown('<h3 class="section-header">🏷️ Annotation Settings</h3>', unsafe_allow_html=True)

    # Add annotation mode selection
    annotation_mode = st.selectbox("Annotation mode:", 
                                  ["No annotations", "Automatic mass calculations", "Manual peak identification"])

    if annotation_mode == "No annotations":
        mass_configs = []
        
    elif annotation_mode == "Manual peak identification":
        st.markdown("**🎯 Manual Peak Identification**")
        st.info("Manually specify m/z values and their corresponding labels")
        
        num_manual_peaks = st.number_input("Number of peaks to identify:", 0, 20, 0)
        mass_configs = []
        
        if num_manual_peaks > 0:
            colors = get_extended_color_palette(palette, num_manual_peaks)
            
            for i in range(num_manual_peaks):
                with st.expander(f"**Peak {i+1}**", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        mz_value = st.number_input(f"m/z value:", value=1000.0 + i*100, key=f"manual_mz_{i}")
                        label_text = st.text_input(f"Label text:", value=f"Peak {i+1}", key=f"manual_label_{i}")
                    
                    with col2:
                        shape_name = st.selectbox("Shape:", shape_names, key=f"manual_shape_{i}")
                        shape = shape_mapping[shape_name]
                    
                    with col3:
                        color_override = st.checkbox("Custom color", key=f"manual_color_check_{i}")
                        if color_override:
                            custom_color = st.selectbox("Color:", 
                                                       ["red", "blue", "green", "orange", "purple", "black", "brown"],
                                                       key=f"manual_color_{i}")
                            peak_color = custom_color
                        else:
                            peak_color = colors[i]
                    
                    with col4:
                        tolerance = st.number_input(f"Tolerance (±):", value=5.0, key=f"manual_tol_{i}")
                    
                    # Create a special config for manual peaks
                    mass_configs.append({
                        'mode': 'manual',
                        'mz': mz_value,
                        'label': label_text,
                        'tolerance': tolerance,
                        'color': peak_color,
                        'shape': shape
                    })

    elif annotation_mode == "Automatic mass calculations":
        # Existing mass configuration code here...
        # [Keep all the existing mass configuration code from your original file]
        if plot_type == "Hybrid Stacked":
            st.write("**Configure one mass per spectrum:**")
            mass_configs = []
            colors = get_extended_color_palette(palette, len(uploaded_files))
            
            for i, file_name in enumerate(file_names):
                with st.expander(f"**Spectrum {i+1}: {file_name}**", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        mass = st.number_input(f"Mass (Da):", value=15000.0, key=f"hybrid_mass_{i}")
                    with col2:
                        shape_name = st.selectbox("Shape:", shape_names, key=f"hybrid_shape_{i}")
                        shape = shape_mapping[shape_name]
                    with col3:
                        charge_input = st.text_input("Charge states:", "10,11,12,13", key=f"hybrid_charges_{i}")
                    with col4:
                        pass
                    
                    try:
                        charge_states = [int(c.strip()) for c in charge_input.split(',') if c.strip()]
                        mass_configs.append({
                            'mode': 'automatic',
                            'mass': mass,
                            'charge_states': charge_states,
                            'color': colors[i],
                            'shape': shape
                        })
                    except ValueError:
                        st.error(f"Invalid charge states for spectrum {i+1}")
        
        else:
            num_masses = st.number_input("Number of masses to annotate:", 1, 10, 1)
            mass_configs = []
            colors = get_extended_color_palette(palette, num_masses)
            
            for i in range(num_masses):
                with st.expander(f"**Mass {i+1}**", expanded=True):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        mass = st.number_input(f"Mass (Da):", value=15000.0 + i*1000, key=f"mass_{i}")
                    with col2:
                        shape_name = st.selectbox("Shape:", shape_names, key=f"shape_{i}")
                        shape = shape_mapping[shape_name]
                    with col3:
                        charge_input = st.text_input("Charge states:", "10,11,12,13", key=f"charges_{i}")
                    with col4:
                        pass
                    
                    try:
                        charge_states = [int(c.strip()) for c in charge_input.split(',') if c.strip()]
                        mass_configs.append({
                            'mode': 'automatic',
                            'mass': mass,
                            'charge_states': charge_states,
                            'color': colors[i],
                            'shape': shape
                        })
                    except ValueError:
                        st.error(f"Invalid charge states for mass {i+1}")

    # Only show annotation styling options if annotations are enabled
    if annotation_mode != "No annotations":
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.write("**Marker Settings**")
            marker_size = st.number_input("Marker size:", 10, 200, 80)
            if annotation_mode == "Automatic mass calculations":
                threshold = st.slider("Annotation threshold:", 0.001, 1.0, 0.01)
        
        with col2:
            st.write("**Label Positioning**")
            offset = st.slider("Label offset:", 0.01, 0.5, 0.1)
            if annotation_mode == "Automatic mass calculations":
                annotation_width = st.number_input("Annotation width:", 10, 500, 100)
        
        with col3:
            st.write("**Annotation Display**")
            show_annotation_lines = st.checkbox("Show annotation lines", True)
            if annotation_mode == "Automatic mass calculations":
                show_mz_labels = st.checkbox("Show m/z in labels")
                show_mass_labels = st.checkbox("Show mass in labels")

        with col4:
            st.write("**Label Styling**")
            label_font_size = st.number_input("Label font size:", 6, 20, 12)
            label_font_weight = st.selectbox("Label font weight:", ["normal", "bold"], index=1)
            
            show_label_borders = st.checkbox("Show label borders", True)
            if show_label_borders:
                label_border_style = st.selectbox("Border style:", ["round", "square", "sawtooth", "circle"])
                label_border_width = st.number_input("Border width:", 0.1, 3.0, 0.5)
                label_border_color = st.selectbox("Border color:", ["black", "gray", "auto"])
                label_background_alpha = st.slider("Background alpha:", 0.0, 1.0, 0.8)
                label_background_color = st.selectbox("Background color:", ["white", "lightgray", "auto", "transparent"])

# ...existing code until the plotting function...