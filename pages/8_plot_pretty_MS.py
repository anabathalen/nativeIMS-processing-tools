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

class SafeSpectrumData:
    """Safe container for spectrum data with built-in validation"""
    
    def __init__(self, mz_data, intensity_data, name=""):
        self.name = name
        self.mz = np.array(mz_data, dtype=np.float64)
        self.intensity = np.array(intensity_data, dtype=np.float64)
        self._validate_and_clean()
    
    def _validate_and_clean(self):
        """Clean and validate data to prevent astronomical values"""
        # Remove NaN and infinite values
        valid_mask = np.isfinite(self.mz) & np.isfinite(self.intensity)
        self.mz = self.mz[valid_mask]
        self.intensity = self.intensity[valid_mask]
        
        # Remove negative intensities
        positive_mask = self.intensity >= 0
        self.mz = self.mz[positive_mask]
        self.intensity = self.intensity[positive_mask]
        
        # Cap maximum intensity to prevent astronomical plots
        MAX_INTENSITY = 1e6
        if len(self.intensity) > 0:
            max_int = np.max(self.intensity)
            if max_int > MAX_INTENSITY:
                scale_factor = MAX_INTENSITY / max_int
                self.intensity = self.intensity * scale_factor
                st.warning(f"⚠️ Large intensities detected in {self.name}. Scaled down for display.")
        
        # Sort by m/z
        if len(self.mz) > 0:
            sort_idx = np.argsort(self.mz)
            self.mz = self.mz[sort_idx]
            self.intensity = self.intensity[sort_idx]
    
    def filter_mz_range(self, mz_min, mz_max):
        """Return filtered data within m/z range"""
        if len(self.mz) == 0:
            return SafeSpectrumData([], [], self.name)
        
        mask = (self.mz >= mz_min) & (self.mz <= mz_max)
        return SafeSpectrumData(self.mz[mask], self.intensity[mask], self.name)
    
    def get_max_intensity(self):
        """Get maximum intensity safely"""
        return np.max(self.intensity) if len(self.intensity) > 0 else 0.0
    
    def normalize(self, method='max'):
        """Normalize spectrum data"""
        if len(self.intensity) == 0:
            return
        
        if method == 'max':
            max_val = np.max(self.intensity)
            if max_val > 0:
                self.intensity = self.intensity / max_val
        elif method == 'sum':
            sum_val = np.sum(self.intensity)
            if sum_val > 0:
                self.intensity = self.intensity / sum_val

def read_spectrum_file_safe(file_path):
    """Safely read spectrum file and return SafeSpectrumData object"""
    try:
        # Try different separators
        for sep in ['\t', ',', ' ', ';']:
            try:
                df = pd.read_csv(file_path, sep=sep, header=None, on_bad_lines='skip', encoding='utf-8')
                if len(df.columns) >= 2:
                    break
            except:
                continue
        
        if len(df.columns) < 2:
            st.error(f"Could not read file {file_path}")
            return SafeSpectrumData([], [], os.path.basename(file_path))
        
        # Extract m/z and intensity columns
        mz_data = pd.to_numeric(df.iloc[:, 0], errors='coerce').dropna().values
        intensity_data = pd.to_numeric(df.iloc[:, 1], errors='coerce').dropna().values
        
        # Ensure same length
        min_len = min(len(mz_data), len(intensity_data))
        mz_data = mz_data[:min_len]
        intensity_data = intensity_data[:min_len]
        
        return SafeSpectrumData(mz_data, intensity_data, os.path.basename(file_path))
    
    except Exception as e:
        st.error(f"Error reading {file_path}: {e}")
        return SafeSpectrumData([], [], os.path.basename(file_path))

def apply_processing(spectrum_data, options):
    """Apply processing to spectrum data safely"""
    if len(spectrum_data.intensity) == 0:
        return spectrum_data
    
    processed = SafeSpectrumData(spectrum_data.mz.copy(), spectrum_data.intensity.copy(), spectrum_data.name)
    
    # Smoothing
    if options.get('smoothing', False) and len(processed.intensity) > 5:
        window = min(options.get('smooth_window', 5), len(processed.intensity) - 1)
        if window >= 3 and window % 2 == 1:
            try:
                processed.intensity = savgol_filter(processed.intensity, window, options.get('smooth_order', 2))
                processed._validate_and_clean()  # Re-validate after processing
            except:
                pass
    
    # Baseline correction
    if options.get('baseline_correction', False):
        try:
            baseline = np.percentile(processed.intensity, options.get('baseline_percentile', 5))
            processed.intensity = np.maximum(processed.intensity - baseline, 0)
        except:
            pass
    
    # Normalization
    if options.get('normalize', False):
        processed.normalize(options.get('normalize_type', 'max'))
    
    return processed

def calculate_safe_y_limits(spectra_list, plot_type, custom_y_min=None, custom_y_max=None, zoom=1.4, stack_offset=1.2, preserve_baseline=False):
    """Calculate Y-axis limits that will NEVER cause astronomical plots"""
    
    if not spectra_list or all(len(s.intensity) == 0 for s in spectra_list):
        return 0, 1000
    
    # Get maximum intensity across all spectra
    max_intensity = max([s.get_max_intensity() for s in spectra_list if len(s.intensity) > 0])
    
    if max_intensity <= 0:
        return 0, 1000
    
    # Calculate safe limits based on plot type
    if plot_type == "single":
        if custom_y_min is not None and custom_y_max is not None:
            # Validate custom range
            if (np.isfinite(custom_y_min) and np.isfinite(custom_y_max) and 
                custom_y_max > custom_y_min and 
                abs(custom_y_max - custom_y_min) < 1e6 and
                abs(custom_y_max) < 1e6 and abs(custom_y_min) < 1e6):
                return float(custom_y_min), float(custom_y_max)
            else:
                st.warning("⚠️ Invalid custom Y range. Using automatic.")
        
        # 🔧 ENHANCED: Baseline preservation and extended zoom
        if preserve_baseline:
            y_min = 0  # Keep baseline at zero
        else:
            y_min = -max_intensity * 0.1
        
        y_max = max_intensity * min(zoom, 10.0)  # Extended zoom to 10x
        
    elif plot_type == "stacked":
        num_spectra = len([s for s in spectra_list if len(s.intensity) > 0])
        # For stacked plots, we normalize each to 0-1 then stack
        if preserve_baseline:
            y_min = 0
        else:
            y_min = -0.1
        y_max = 1.0 + (num_spectra - 1) * min(stack_offset, 2.0) + 0.5
        
    elif plot_type == "overlay":
        if preserve_baseline:
            y_min = 0
        else:
            y_min = -max_intensity * 0.05
        y_max = max_intensity * min(zoom, 10.0)  # Extended zoom
        
    elif plot_type == "mirror":
        y_min = -max_intensity * min(zoom, 10.0)
        y_max = max_intensity * min(zoom, 10.0)
    
    else:
        y_min = 0
        y_max = max_intensity * min(zoom, 10.0)
    
    # Final safety checks
    y_min = np.clip(y_min, -1e6, 1e6)
    y_max = np.clip(y_max, -1e6, 1e6)
    
    if y_max <= y_min:
        y_max = y_min + 1
    
    if abs(y_max - y_min) > 1e6:
        y_max = y_min + 1e6
    
    return float(y_min), float(y_max)

def add_annotations_safe(ax, spectrum_data, annotations, y_max):
    """Add annotations with safe Y positioning"""
    
    if not annotations or len(spectrum_data.intensity) == 0:
        return
    
    max_intensity = spectrum_data.get_max_intensity()
    
    for annotation in annotations:
        if annotation.get('mode') == 'manual':
            target_mz = annotation['mz']
            tolerance = annotation['tolerance']
            
            # Find peak in range
            mask = (spectrum_data.mz >= target_mz - tolerance) & (spectrum_data.mz <= target_mz + tolerance)
            if not np.any(mask):
                continue
            
            peak_intensities = spectrum_data.intensity[mask]
            peak_mzs = spectrum_data.mz[mask]
            
            if len(peak_intensities) == 0:
                continue
            
            max_idx = np.argmax(peak_intensities)
            peak_mz = peak_mzs[max_idx]
            peak_intensity = peak_intensities[max_idx]
            
            # SAFE annotation positioning - use fixed small offsets
            marker_y = peak_intensity + max_intensity * 0.05  # 5% above peak
            label_y = marker_y + max_intensity * 0.03  # 3% above marker
            
            # Cap at reasonable positions
            marker_y = min(marker_y, y_max * 0.9)
            label_y = min(label_y, y_max * 0.95)
            
            # Draw annotation
            if annotation.get('show_line', True):
                ax.plot([peak_mz, peak_mz], [peak_intensity, marker_y], 
                       color=annotation['color'], linewidth=1, alpha=0.7)
            
            ax.scatter([peak_mz], [marker_y], color=annotation['color'], 
                      marker=annotation['shape'], s=annotation.get('size', 80), 
                      edgecolor='black', linewidth=0.5, zorder=5)
            
            label_kwargs = {
                'ha': 'center', 
                'va': 'bottom', 
                'fontsize': annotation.get('font_size', 12),
                'color': annotation['color'], 
                'weight': annotation.get('weight', 'bold')
            }
            
            if annotation.get('show_label_border', False):
                label_kwargs['bbox'] = dict(
                    boxstyle="round,pad=0.3",
                    facecolor=annotation.get('label_bg_color', 'white'),
                    edgecolor=annotation['color'],
                    linewidth=0.5,
                    alpha=annotation.get('label_alpha', 0.8)
                )
            
            ax.text(peak_mz, label_y, annotation['label'], **label_kwargs)
        
        elif annotation.get('mode') == 'automatic':
            mass = annotation['mass']
            charges = annotation['charge_states']
            threshold = annotation.get('threshold', 0.01)
            
            for charge in charges:
                mz = (mass + charge * 1.007276) / charge
                
                # Check if m/z is in spectrum range
                if mz < np.min(spectrum_data.mz) or mz > np.max(spectrum_data.mz):
                    continue
                
                # Find intensity near this m/z
                width = annotation.get('width', 100) / charge
                mask = (spectrum_data.mz >= mz - width/2) & (spectrum_data.mz <= mz + width/2)
                
                if not np.any(mask):
                    continue
                
                peak_intensity = np.max(spectrum_data.intensity[mask])
                threshold_intensity = threshold * max_intensity
                
                if peak_intensity >= threshold_intensity:
                    # SAFE positioning
                    marker_y = peak_intensity + max_intensity * 0.05
                    label_y = marker_y + max_intensity * 0.03
                    
                    marker_y = min(marker_y, y_max * 0.9)
                    label_y = min(label_y, y_max * 0.95)
                    
                    if annotation.get('show_line', True):
                        ax.plot([mz, mz], [peak_intensity, marker_y], 
                               color=annotation['color'], linewidth=1, alpha=0.7)
                    
                    ax.scatter([mz], [marker_y], color=annotation['color'], 
                              marker=annotation['shape'], s=annotation.get('size', 80), 
                              edgecolor='black', linewidth=0.5, zorder=5)
                    
                    label = f'{charge}+'
                    if annotation.get('show_mz', False):
                        label += f'\n{mz:.1f}'
                    if annotation.get('show_mass', False):
                        label += f'\n{mass:.0f}Da'
                    
                    label_kwargs = {
                        'ha': 'center', 
                        'va': 'bottom',
                        'fontsize': annotation.get('font_size', 12),
                        'color': annotation['color'], 
                        'weight': annotation.get('weight', 'bold')
                    }
                    
                    if annotation.get('show_label_border', False):
                        label_kwargs['bbox'] = dict(
                            boxstyle="round,pad=0.3",
                            facecolor=annotation.get('label_bg_color', 'white'),
                            edgecolor=annotation['color'],
                            linewidth=0.5,
                            alpha=annotation.get('label_alpha', 0.8)
                        )
                    
                    ax.text(mz, label_y, label, **label_kwargs)

def create_safe_plot(spectra_list, plot_settings, annotations, plot_type, spectrum_labels=None, vertical_lines=None, spectrum_colors=None):
    """Create plot with guaranteed safe Y-axis limits"""
    
    # Filter spectra by m/z range
    filtered_spectra = []
    for spectrum in spectra_list:
        filtered = spectrum.filter_mz_range(plot_settings['x_min'], plot_settings['x_max'])
        if len(filtered.intensity) > 0:
            filtered_spectra.append(filtered)
    
    if not filtered_spectra:
        st.error("No data in specified m/z range")
        return None
    
    # Calculate safe Y limits BEFORE creating figure
    y_min, y_max = calculate_safe_y_limits(
        filtered_spectra, 
        plot_type, 
        plot_settings.get('y_min'), 
        plot_settings.get('y_max'),
        plot_settings.get('zoom', 1.4),
        plot_settings.get('stack_offset', 1.2),
        plot_settings.get('preserve_baseline', False)
    )
    
    # Create figure with proper subplot handling
    if plot_type == "mirror" and len(filtered_spectra) >= 2:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(plot_settings['width'], plot_settings['height']), 
                                      dpi=plot_settings['dpi'], sharex=True)
        axes = [ax1, ax2]
    else:
        fig, ax = plt.subplots(figsize=(plot_settings['width'], plot_settings['height']), 
                              dpi=plot_settings['dpi'])
        axes = [ax]
    
    # Set background
    if plot_settings['background'] == 'transparent':
        fig.patch.set_facecolor('none')
        fig.patch.set_alpha(0)
        for axis in axes:
            axis.patch.set_facecolor('none')
            axis.patch.set_alpha(0)
    elif plot_settings['background'] != 'white':
        fig.patch.set_facecolor(plot_settings['background'])
        for axis in axes:
            axis.set_facecolor(plot_settings['background'])
    
    # Font settings
    plt.rcParams.update({
        'font.size': plot_settings['font_size'],
        'font.family': plot_settings.get('font_family', 'Arial'),
        'font.weight': plot_settings.get('font_weight', 'normal')
    })
    
    # Plot data based on type
    if plot_type == "single":
        spectrum = filtered_spectra[0]
        ax = axes[0]
        ax.plot(spectrum.mz, spectrum.intensity, 
               color=plot_settings['line_color'], 
               linewidth=plot_settings['line_width'],
               linestyle=plot_settings.get('line_style', '-'),
               alpha=plot_settings.get('alpha', 1.0))
        
        if plot_settings.get('fill_under', False):
            ax.fill_between(spectrum.mz, spectrum.intensity, 
                           alpha=plot_settings.get('fill_alpha', 0.3),
                           color=plot_settings['line_color'])
    
    elif plot_type == "stacked":
        # 🔧 ENHANCED: Use custom spectrum colors if provided
        if spectrum_colors and len(spectrum_colors) == len(filtered_spectra):
            colors = spectrum_colors
        else:
            colors = get_color_palette(plot_settings.get('palette', 'Scientific'), len(filtered_spectra))
        
        ax = axes[0]
        
        for i, spectrum in enumerate(filtered_spectra):
            # Normalize each spectrum to 0-1
            normalized_spectrum = SafeSpectrumData(spectrum.mz.copy(), spectrum.intensity.copy(), spectrum.name)
            normalized_spectrum.normalize('max')
            
            # Stack with offset
            offset = i * plot_settings.get('stack_offset', 1.2)
            stacked_intensity = normalized_spectrum.intensity + offset
            
            color = colors[i] if i < len(colors) else 'black'
            ax.plot(normalized_spectrum.mz, stacked_intensity,
                   color=color, linewidth=plot_settings['line_width'],
                   linestyle=plot_settings.get('line_style', '-'),
                   alpha=plot_settings.get('alpha', 1.0))
            
            # Add vertical lines for EACH spectrum in stacked plot
            if vertical_lines:
                add_vertical_lines_safe(ax, vertical_lines, plot_settings, plot_type, i, plot_settings.get('stack_offset', 1.2))
            
            # Add custom spectrum labels if provided
            if spectrum_labels and i < len(spectrum_labels) and not spectrum_labels[i].get('hidden', False):
                add_custom_spectrum_label(ax, spectrum_labels[i], stacked_intensity, 
                                        plot_settings, i, offset)
            elif not spectrum_labels:
                # Default labeling
                if len(stacked_intensity) > 0:
                    label_x = plot_settings['x_max'] - (plot_settings['x_max'] - plot_settings['x_min']) * 0.05
                    label_y = np.median(stacked_intensity)
                    ax.text(label_x, label_y, f"Spectrum {i+1}", 
                           ha='right', va='center', fontsize=plot_settings['font_size'])
    
    elif plot_type == "overlay":
        # 🔧 ENHANCED: Use custom spectrum colors if provided
        if spectrum_colors and len(spectrum_colors) == len(filtered_spectra):
            colors = spectrum_colors
        else:
            colors = get_color_palette(plot_settings.get('palette', 'Scientific'), len(filtered_spectra))
        
        ax = axes[0]
        
        for i, spectrum in enumerate(filtered_spectra):
            color = colors[i] if i < len(colors) else 'black'
            ax.plot(spectrum.mz, spectrum.intensity,
                   color=color, linewidth=plot_settings['line_width'],
                   linestyle=plot_settings.get('line_style', '-'),
                   alpha=plot_settings.get('alpha', 0.7),
                   label=spectrum.name)
        
        if plot_settings.get('show_legend', False):
            ax.legend(loc=plot_settings.get('legend_pos', 'upper right'),
                     frameon=plot_settings.get('legend_frame', False))
    
    elif plot_type == "mirror" and len(filtered_spectra) >= 2:
        # 🔧 ENHANCED: Use custom colors for mirror plot
        color1 = spectrum_colors[0] if spectrum_colors and len(spectrum_colors) >= 1 else plot_settings['line_color']
        color2 = spectrum_colors[1] if spectrum_colors and len(spectrum_colors) >= 2 else plot_settings.get('line_color_2', 'red')
        
        # Top spectrum (normal)
        spectrum1 = filtered_spectra[0]
        axes[0].plot(spectrum1.mz, spectrum1.intensity,
                    color=color1, 
                    linewidth=plot_settings['line_width'],
                    linestyle=plot_settings.get('line_style', '-'))
        
        if plot_settings.get('fill_under', False):
            axes[0].fill_between(spectrum1.mz, spectrum1.intensity,
                               alpha=plot_settings.get('fill_alpha', 0.3),
                               color=color1)
        
        # Bottom spectrum (inverted)
        spectrum2 = filtered_spectra[1]
        axes[1].plot(spectrum2.mz, -spectrum2.intensity,
                    color=color2, 
                    linewidth=plot_settings['line_width'],
                    linestyle=plot_settings.get('line_style', '-'))
        
        if plot_settings.get('fill_under', False):
            axes[1].fill_between(spectrum2.mz, -spectrum2.intensity,
                               alpha=plot_settings.get('fill_alpha', 0.3),
                               color=color2)
        
        # Add center line
        axes[1].axhline(y=0, color='black', linewidth=0.5)
        
        # Mirror plot specific Y limits
        max_int1 = spectrum1.get_max_intensity()
        max_int2 = spectrum2.get_max_intensity()
        max_combined = max(max_int1, max_int2)
        
        zoom_factor = min(plot_settings.get('zoom', 1.4), 10.0)
        axes[0].set_ylim(0, max_combined * zoom_factor)
        axes[1].set_ylim(-max_combined * zoom_factor, 0)
    
    # Set axis limits and styling for all axes
    for i, current_ax in enumerate(axes):
        current_ax.set_xlim(plot_settings['x_min'], plot_settings['x_max'])
        
        if plot_type != "mirror":
            current_ax.set_ylim(y_min, y_max)
        
        # Add vertical lines properly for non-stacked plots
        if vertical_lines and plot_type != "stacked":
            add_vertical_lines_safe(current_ax, vertical_lines, plot_settings, plot_type, i)
        
        # Apply styling
        apply_plot_styling(current_ax, plot_settings, is_bottom=(i == len(axes)-1))
    
    # Add annotations to first spectrum
    if annotations and len(filtered_spectra) > 0:
        add_annotations_safe(axes[0], filtered_spectra[0], annotations, 
                           y_max if plot_type != "mirror" else filtered_spectra[0].get_max_intensity() * min(plot_settings.get('zoom', 1.4), 10.0))
    
    # Add title if specified
    if plot_settings.get('title'):
        fig.suptitle(plot_settings['title'], fontsize=plot_settings.get('title_font_size', plot_settings['font_size'] + 2),
                    weight=plot_settings.get('title_weight', 'bold'))
    
    plt.tight_layout()
    return fig

def add_custom_spectrum_label(ax, label_config, stacked_intensity, plot_settings, spectrum_index, offset):
    """Add custom label for stacked spectrum"""
    
    x_range = plot_settings['x_max'] - plot_settings['x_min']
    
    # Determine label position
    if label_config['position'] == "Custom X,Y":
        label_x = label_config['x']
        label_y = offset + label_config['y_offset']
    elif label_config['position'] == "Left":
        label_x = plot_settings['x_min'] + (0.05 * x_range)
        label_y = np.median(stacked_intensity) if len(stacked_intensity) > 0 else offset
    elif label_config['position'] == "Center":
        label_x = plot_settings['x_min'] + (0.5 * x_range)
        label_y = np.median(stacked_intensity) if len(stacked_intensity) > 0 else offset
    elif label_config['position'] == "Right":
        label_x = plot_settings['x_max'] - (0.05 * x_range)
        label_y = np.median(stacked_intensity) if len(stacked_intensity) > 0 else offset
    else:  # Auto (right)
        label_x = plot_settings['x_max'] - (0.05 * x_range)
        label_y = np.median(stacked_intensity) if len(stacked_intensity) > 0 else offset
    
    # Auto-avoid overlap if enabled
    if plot_settings.get('auto_avoid_overlap', True) and spectrum_index > 0:
        label_y += plot_settings.get('vertical_spacing', 0.3) * spectrum_index * 0.1
    
    # Prepare label text
    label_text = label_config['text']
    if plot_settings.get('show_spectrum_numbers', False):
        number = f"{plot_settings.get('number_prefix', '')}{spectrum_index+1}{plot_settings.get('number_suffix', '')}"
        label_text = f"{number} {label_text}" if label_text else number
    
    # Label styling
    label_kwargs = {
        'ha': 'right' if 'right' in label_config['position'].lower() else 'center' if 'center' in label_config['position'].lower() else 'left',
        'va': 'center',
        'fontsize': label_config['size'],
        'color': label_config['color'],
        'weight': label_config['weight'],
        'style': label_config['style'],
        'alpha': plot_settings.get('global_label_alpha', 1.0)
    }
    
    # Add box if requested
    if label_config['show_box']:
        label_kwargs['bbox'] = dict(
            boxstyle="round,pad=0.3",
            facecolor=label_config['box_color'],
            edgecolor=label_config['color'],
            linewidth=0.5,
            alpha=plot_settings.get('global_box_alpha', 0.8)
        )
    
    ax.text(label_x, label_y, label_text, **label_kwargs)

def get_color_palette(palette_name, n_colors):
    """Get color palette for plotting"""
    palettes = {
        "Scientific": ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b'],
        "Nature": ['#0173B2', '#DE8F05', '#029E73', '#CC78BC', '#CA9161', '#FBAFE4'],
        "Colorblind Safe": sns.color_palette("colorblind", n_colors),
        "Viridis": plt.cm.viridis(np.linspace(0, 1, n_colors)),
        "Plasma": plt.cm.plasma(np.linspace(0, 1, n_colors)),
        "Inferno": plt.cm.inferno(np.linspace(0, 1, n_colors)),
    }
    
    if palette_name in palettes:
        colors = palettes[palette_name]
        if isinstance(colors, list):
            return (colors * ((n_colors // len(colors)) + 1))[:n_colors]
        else:
            return list(colors)[:n_colors]
    
    return sns.color_palette("husl", n_colors)

def apply_plot_styling(ax, settings, is_bottom=True):
    """Apply styling to the plot"""
    
    # Labels
    if settings.get('show_x_label', True) and is_bottom:
        ax.set_xlabel(settings.get('x_label_text', 'm/z'), 
                     fontsize=settings.get('axis_label_size', 14),
                     weight=settings.get('axis_label_weight', 'bold'))
    
    if settings.get('show_y_label', True):
        ax.set_ylabel(settings.get('y_label_text', 'Intensity'),
                     fontsize=settings.get('axis_label_size', 14),
                     weight=settings.get('axis_label_weight', 'bold'))
    
    # Grid
    if settings.get('show_grid', False):
        ax.grid(True, linestyle=settings.get('grid_style', '--'),
               alpha=settings.get('grid_alpha', 0.3),
               color=settings.get('grid_color', 'gray'))
    
    # Spines
    spine_settings = {
        'top': settings.get('show_top_axis', False),
        'right': settings.get('show_right_axis', False),
        'bottom': settings.get('show_bottom_axis', True),
        'left': settings.get('show_left_axis', True)
    }
    
    for spine_name, show_spine in spine_settings.items():
        spine = ax.spines[spine_name]
        if show_spine:
            spine.set_visible(True)
            spine.set_linewidth(settings.get('spine_width', 1.0))
            spine.set_color(settings.get('spine_color', 'black'))
        else:
            spine.set_visible(False)
    
    # Ticks
    if settings.get('hide_y_ticks', True):
        ax.set_yticks([])
    elif settings.get('y_tick_count'):
        ax.yaxis.set_major_locator(MaxNLocator(nbins=settings['y_tick_count']))
    
    if not settings.get('show_y_tick_labels', True):
        ax.set_yticklabels([])
    
    if not settings.get('show_x_tick_labels', True):
        ax.set_xticklabels([])
    elif settings.get('custom_x_ticks', False) and settings.get('x_tick_spacing'):
        ax.xaxis.set_major_locator(MultipleLocator(settings['x_tick_spacing']))
    
    # Tick label size
    if settings.get('tick_label_size'):
        ax.tick_params(axis='both', which='major', labelsize=settings['tick_label_size'])

def add_vertical_lines_safe(ax, vertical_lines, plot_settings, plot_type="single", spectrum_index=0, stack_offset=1.2):
    """Add vertical lines with safe positioning for all plot types"""
    
    if not vertical_lines:
        return
    
    y_min, y_max = ax.get_ylim()
    
    for vline in vertical_lines:
        mz_position = vline['mz']
        
        # Check if line is within plot range
        if mz_position < plot_settings['x_min'] or mz_position > plot_settings['x_max']:
            continue
        
        # Adjust Y limits based on plot type and spectrum
        if plot_type == "stacked":
            # For stacked plots, each spectrum is normalized and offset
            # Lines should span from just below the spectrum to just above it
            spectrum_base = spectrum_index * stack_offset
            spectrum_top = spectrum_base + 1.0  # normalized spectra are 0-1
            
            # Add some padding
            line_bottom = spectrum_base - 0.05
            line_top = spectrum_top + 0.05
            
            # Ensure within plot bounds
            line_bottom = max(line_bottom, y_min)
            line_top = min(line_top, y_max)
        elif plot_type == "mirror":
            # For mirror plots, use the actual axis limits
            line_bottom = y_min
            line_top = y_max
        else:
            # For single and overlay plots, use full axis range
            line_bottom = y_min
            line_top = y_max
        
        # Draw vertical line
        if vline.get('style') == 'shaded':
            # Draw shaded region
            width = vline.get('width', 5.0)  # width in m/z units
            x_left = mz_position - width/2
            x_right = mz_position + width/2
            
            if plot_type == "stacked":
                # For stacked plots, only shade the current spectrum's region
                ax.axvspan(x_left, x_right, 
                          ymin=(line_bottom - y_min) / (y_max - y_min),
                          ymax=(line_top - y_min) / (y_max - y_min),
                          color=vline['color'], 
                          alpha=vline.get('alpha', 0.3),
                          zorder=1)
            else:
                # For other plots, shade the full height
                ax.axvspan(x_left, x_right, 
                          color=vline['color'], 
                          alpha=vline.get('alpha', 0.3),
                          zorder=1)
            
            # Optional center line
            if vline.get('show_center_line', True):
                ax.plot([mz_position, mz_position], [line_bottom, line_top],
                       color=vline['color'], 
                       linewidth=vline.get('line_width', 1.0),
                       linestyle=vline.get('line_style', '-'),
                       alpha=vline.get('line_alpha', 0.8),
                       zorder=2)
        else:
            # Draw simple line
            ax.plot([mz_position, mz_position], [line_bottom, line_top],
                   color=vline['color'], 
                   linewidth=vline.get('line_width', 2.0),
                   linestyle=vline.get('line_style', '-'),
                   alpha=vline.get('alpha', 0.8),
                   zorder=2)
        
        # Add label if specified (only once per line, not per spectrum)
        if vline.get('label') and (plot_type != "stacked" or spectrum_index == 0):
            label_y_pos = vline.get('label_position', 0.9)  # 0-1 scale
            
            if plot_type == "stacked" and spectrum_index == 0:
                # For stacked plots, position label at the top
                label_y = y_min + (y_max - y_min) * label_y_pos
            else:
                label_y = y_min + (y_max - y_min) * label_y_pos
            
            label_kwargs = {
                'ha': 'center',
                'va': vline.get('label_va', 'bottom'),
                'fontsize': vline.get('label_font_size', 12),
                'color': vline.get('label_color', vline['color']),
                'weight': vline.get('label_weight', 'bold'),
                'rotation': vline.get('label_rotation', 0)
            }
            
            if vline.get('label_background', False):
                label_kwargs['bbox'] = dict(
                    boxstyle="round,pad=0.3",
                    facecolor=vline.get('label_bg_color', 'white'),
                    edgecolor=vline.get('label_color', vline['color']),
                    linewidth=0.5,
                    alpha=vline.get('label_bg_alpha', 0.9)
                )
            
            ax.text(mz_position, label_y, vline['label'], **label_kwargs)

# Streamlit UI starts here
st.markdown('<h3 class="section-header">🎯 Plot Configuration</h3>', unsafe_allow_html=True)

# Plot type selection
col1, col2 = st.columns(2)
with col1:
    plot_type = st.selectbox("Plot type:", 
                            ["Single Spectrum", "Stacked Comparison", "Overlay Plot", "Mirror Plot"])
with col2:
    output_format = st.selectbox("Output format:", ["PNG", "PDF", "SVG"])

# Figure settings
st.markdown('<h3 class="section-header">📐 Figure Settings</h3>', unsafe_allow_html=True)
col1, col2, col3, col4 = st.columns(4)

with col1:
    figure_width = st.number_input("Width (inches):", 1.0, 30.0, 10.0, 0.5)
    figure_height = st.number_input("Height (inches):", 1.0, 20.0, 6.0, 0.5)

with col2:
    dpi = st.selectbox("DPI:", [150, 200, 300, 400, 600, 800], index=1)
    font_size = st.number_input("Font size:", 6, 24, 14)

with col3:
    font_family = st.selectbox("Font family:", ["Arial", "Times New Roman", "Helvetica", "Calibri"])
    font_weight = st.selectbox("Font weight:", ["normal", "bold"])
    background = st.selectbox("Background:", ["white", "lightgray", "black", "transparent"])

with col4:
    line_color = st.selectbox("Line color:", ["black", "blue", "red", "green", "purple", "orange", "brown"])
    line_width = st.number_input("Line width:", 0.1, 5.0, 1.5, 0.1)
    line_style = st.selectbox("Line style:", ["-", "--", "-.", ":"])
    alpha = st.slider("Line transparency:", 0.1, 1.0, 1.0)

# Advanced styling
with st.expander("🎨 Advanced Styling", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        fill_under = st.checkbox("Fill under curve")
        if fill_under:
            fill_alpha = st.slider("Fill transparency:", 0.1, 1.0, 0.3)
        else:
            fill_alpha = 0.3
        
        if plot_type == "Overlay Plot":
            palette = st.selectbox("Color palette:", 
                                  ["Scientific", "Nature", "Colorblind Safe", "Viridis", "Plasma", "Inferno"])
        else:
            palette = "Scientific"
    
    with col2:
        if plot_type == "Mirror Plot":
            line_color_2 = st.selectbox("Second spectrum color:", 
                                       ["red", "blue", "green", "purple", "orange"])
        else:
            line_color_2 = "red"
    
    with col3:
        if plot_type == "Stacked Comparison":
            stack_offset = st.slider("Stack offset:", 0.5, 3.0, 1.2)
        else:
            stack_offset = 1.2

# Axis settings
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
    zoom = st.slider("Y-axis zoom:", 0.5, 10.0, 1.4)  # 🔧 EXTENDED: Up to 10x zoom
    preserve_baseline = st.checkbox("Preserve baseline at zero", True)  # 🔧 NEW: Baseline preservation

# Axis customization
with st.expander("⚙️ Axis Customization", expanded=False):
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        show_x_label = st.checkbox("Show X label", True)
        if show_x_label:
            x_label_text = st.text_input("X label:", "m/z")
        else:
            x_label_text = "m/z"
        
        show_y_label = st.checkbox("Show Y label", True)
        if show_y_label:
            y_label_text = st.text_input("Y label:", "Intensity")
        else:
            y_label_text = "Intensity"
    
    with col2:
        axis_label_size = st.number_input("Axis label size:", 8, 24, 14)
        axis_label_weight = st.selectbox("Axis label weight:", ["normal", "bold"])
        tick_label_size = st.number_input("Tick label size:", 6, 20, 12)
    
    with col3:
        show_bottom_axis = st.checkbox("Bottom axis", True)
        show_top_axis = st.checkbox("Top axis", False)
        show_left_axis = st.checkbox("Left axis", True)
        show_right_axis = st.checkbox("Right axis", False)
    
    with col4:
        hide_y_ticks = st.checkbox("Hide Y ticks", True)
        show_y_tick_labels = st.checkbox("Show Y tick labels", False)
        show_x_tick_labels = st.checkbox("Show X tick labels", True)
        
        custom_x_ticks = st.checkbox("Custom X tick spacing")
        if custom_x_ticks:
            x_tick_spacing = st.number_input("X tick spacing:", 1.0, 1000.0, 100.0)
        else:
            x_tick_spacing = None
        
        y_tick_count = st.number_input("Y tick count:", 2, 20, 5)

# Spine settings
with st.expander("🖼️ Spine Settings", expanded=False):
    col1, col2 = st.columns(2)
    
    with col1:
        spine_width = st.number_input("Spine width:", 0.1, 5.0, 1.0)
    with col2:
        spine_color = st.selectbox("Spine color:", ["black", "gray", "white"])

# Plot title
with st.expander("📝 Plot Title", expanded=False):
    plot_title = st.text_input("Title:", "")
    if plot_title:
        title_font_size = st.number_input("Title font size:", 8, 30, 16)
        title_weight = st.selectbox("Title weight:", ["normal", "bold"])
    else:
        title_font_size = 16
        title_weight = "bold"

# Legend settings (for overlay plots)
if plot_type == "Overlay Plot":
    with st.expander("📊 Legend Settings", expanded=False):
        show_legend = st.checkbox("Show legend", True)
        if show_legend:
            legend_pos = st.selectbox("Legend position:", 
                                    ["upper right", "upper left", "lower right", "lower left", "center"])
            legend_frame = st.checkbox("Legend frame", False)
        else:
            legend_pos = "upper right"
            legend_frame = False
else:
    show_legend = False
    legend_pos = "upper right"
    legend_frame = False

# File upload
st.markdown('<h3 class="section-header">📁 File Upload</h3>', unsafe_allow_html=True)

if plot_type == "Single Spectrum":
    uploaded_file = st.file_uploader("Upload spectrum file", type=['txt', 'csv', 'tsv'])
    uploaded_files = [uploaded_file] if uploaded_file else []
elif plot_type == "Mirror Plot":
    uploaded_files = st.file_uploader("Upload exactly 2 spectrum files for mirror plot", 
                                     type=['txt', 'csv', 'tsv'], 
                                     accept_multiple_files=True)
    if uploaded_files and len(uploaded_files) != 2:
        st.warning("⚠️ Mirror plot requires exactly 2 files")
else:
    uploaded_files = st.file_uploader("Upload spectrum files", 
                                     type=['txt', 'csv', 'tsv'], 
                                     accept_multiple_files=True)

# Get file names for labeling
file_names = [f.name for f in uploaded_files if f is not None]

# Individual spectrum labeling for stacked plots
spectrum_labels = []
if uploaded_files and all(f is not None for f in uploaded_files) and plot_type == "Stacked Comparison":
    
    st.markdown('<h3 class="section-header">🏷️ Individual Spectrum Labels</h3>', unsafe_allow_html=True)
    
    for i, file_name in enumerate(file_names):
        with st.expander(f"📊 Spectrum {i+1}: {file_name}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                use_custom_label = st.checkbox("Custom label", key=f"custom_label_{i}")
                if use_custom_label:
                    custom_label = st.text_input("Label text:", value=f"Spectrum {i+1}", key=f"label_text_{i}")
                else:
                    custom_label = file_name.replace('.txt', '').replace('.csv', '').replace('.tsv', '')
                
            with col2:
                label_position = st.selectbox("Label position:", 
                                            ["Auto (right)", "Custom X,Y", "Left", "Center", "Right", "Hidden"], 
                                            key=f"label_pos_{i}")
                
                if label_position == "Custom X,Y":
                    label_x = st.number_input("X position (m/z):", value=float(x_max * 0.95), key=f"label_x_{i}")
                    label_y_offset = st.number_input("Y offset from spectrum:", value=0.0, step=0.1, key=f"label_y_{i}")
                else:
                    label_x = None
                    label_y_offset = 0.0
            
            with col3:
                label_color = st.selectbox("Label color:", 
                                         ["black", "red", "blue", "green", "purple", "orange", "brown", "gray"],
                                         key=f"label_color_{i}")
                label_size = st.number_input("Label size:", 6, 20, font_size, key=f"label_size_{i}")
            
            with col4:
                label_weight = st.selectbox("Label weight:", ["normal", "bold"], index=1, key=f"label_weight_{i}")
                label_style = st.selectbox("Label style:", ["normal", "italic"], key=f"label_style_{i}")
                show_label_box = st.checkbox("Label box", key=f"label_box_{i}")
                if show_label_box:
                    box_color = st.selectbox("Box color:", ["white", "lightgray", "yellow", "lightblue"], key=f"box_color_{i}")
                else:
                    box_color = "white"
            
            spectrum_labels.append({
                'text': custom_label,
                'position': label_position,
                'x': label_x,
                'y_offset': label_y_offset,
                'color': label_color,
                'size': label_size,
                'weight': label_weight,
                'style': label_style,
                'show_box': show_label_box,
                'box_color': box_color,
                'hidden': label_position == "Hidden"
            })
    
    # Global label settings
    st.markdown("**🎨 Global Label Settings**")
    col1, col2, col3 = st.columns(3)
    
    with col1:
        global_label_alpha = st.slider("Label transparency:", 0.1, 1.0, 1.0)
        global_box_alpha = st.slider("Box transparency:", 0.1, 1.0, 0.8)
    
    with col2:
        auto_avoid_overlap = st.checkbox("Auto-avoid overlap", True)
        vertical_spacing = st.number_input("Vertical spacing:", 0.1, 2.0, 0.3)
    
    with col3:
        show_spectrum_numbers = st.checkbox("Show spectrum numbers")
        if show_spectrum_numbers:
            number_prefix = st.text_input("Number prefix:", "")
            number_suffix = st.text_input("Number suffix:", "")
        else:
            number_prefix = ""
            number_suffix = ""

else:
    global_label_alpha = 1.0
    global_box_alpha = 0.8
    auto_avoid_overlap = True
    vertical_spacing = 0.3
    show_spectrum_numbers = False
    number_prefix = ""
    number_suffix = ""

# 🔧 NEW: Vertical Lines
vertical_lines = []
st.markdown('<h3 class="section-header">📏 Vertical Reference Lines</h3>', unsafe_allow_html=True)

with st.expander("➕ Add Vertical Lines", expanded=False):
    num_vlines = st.number_input("Number of vertical lines:", 0, 20, 0)
    
    for i in range(num_vlines):
        with st.expander(f"Line {i+1}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                vline_mz = st.number_input("m/z position:", value=1000.0 + i*500, key=f"vline_mz_{i}")
                vline_style = st.selectbox("Style:", ["simple", "shaded"], key=f"vline_style_{i}")
            
            with col2:
                vline_color = st.selectbox("Color:", 
                                         ["red", "blue", "green", "orange", "purple", "gray", "black"], 
                                         key=f"vline_color_{i}")
                
                if vline_style == "shaded":
                    vline_width = st.number_input("Shaded width (m/z):", 1.0, 100.0, 10.0, key=f"vline_width_{i}")
                    vline_alpha = st.slider("Shade transparency:", 0.1, 1.0, 0.3, key=f"vline_alpha_{i}")
                else:
                    vline_width = 0
                    vline_alpha = 0.8
            
            with col3:
                vline_line_width = st.number_input("Line width:", 0.5, 5.0, 2.0, key=f"vline_lw_{i}")
                vline_line_style = st.selectbox("Line style:", ["-", "--", "-.", ":"], key=f"vline_ls_{i}")
                
                if vline_style == "shaded":
                    show_center_line = st.checkbox("Show center line", True, key=f"vline_center_{i}")
                    line_alpha = st.slider("Line transparency:", 0.1, 1.0, 0.8, key=f"vline_line_alpha_{i}")
                else:
                    show_center_line = True
                    line_alpha = vline_alpha
            
            with col4:
                vline_label = st.text_input("Label (optional):", "", key=f"vline_label_{i}")
                
                if vline_label:
                    label_position = st.slider("Label Y position:", 0.0, 1.0, 0.9, key=f"vline_label_pos_{i}")
                    label_font_size = st.number_input("Label font size:", 6, 24, 12, key=f"vline_label_font_{i}")
                    label_rotation = st.selectbox("Label rotation:", [0, 45, 90], key=f"vline_label_rot_{i}")
                    label_background = st.checkbox("Label background", True, key=f"vline_label_bg_{i}")
                    
                    if label_background:
                        label_bg_color = st.selectbox("Background color:", 
                                                    ["white", "lightgray", "yellow", "lightblue"], 
                                                    key=f"vline_bg_color_{i}")
                        label_bg_alpha = st.slider("Background alpha:", 0.1, 1.0, 0.9, key=f"vline_bg_alpha_{i}")
                    else:
                        label_bg_color = "white"
                        label_bg_alpha = 0.9
                else:
                    label_position = 0.9
                    label_font_size = 12
                    label_rotation = 0
                    label_background = False
                    label_bg_color = "white"
                    label_bg_alpha = 0.9
            
            vertical_lines.append({
                'mz': vline_mz,
                'color': vline_color,
                'style': vline_style,
                'width': vline_width,
                'alpha': vline_alpha,
                'line_width': vline_line_width,
                'line_style': vline_line_style,
                'line_alpha': line_alpha,
                'show_center_line': show_center_line,
                'label': vline_label,
                'label_position': label_position,
                'label_font_size': label_font_size,
                'label_color': vline_color,
                'label_weight': 'bold',
                'label_rotation': label_rotation,
                'label_va': 'bottom',
                'label_background': label_background,
                'label_bg_color': label_bg_color,
                'label_bg_alpha': label_bg_alpha
            })

# Annotations
annotations = []
st.markdown('<h3 class="section-header">🏷️ Peak Annotations</h3>', unsafe_allow_html=True)

annotation_mode = st.selectbox("Annotation mode:", 
                              ["No annotations", "Manual peaks", "Automatic mass"])

if annotation_mode == "Manual peaks":
    num_peaks = st.number_input("Number of peaks:", 0, 20, 0)
    
    for i in range(num_peaks):
        with st.expander(f"Peak {i+1}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mz_val = st.number_input("m/z:", value=1000.0 + i*100, key=f"mz_{i}")
                tolerance = st.number_input("Tolerance:", value=5.0, key=f"tol_{i}")
            
            with col2:
                label = st.text_input("Label:", value=f"Peak {i+1}", key=f"label_{i}")
                color = st.selectbox("Color:", ["red", "blue", "green", "orange", "purple", "brown"], key=f"color_{i}")
            
            with col3:
                shape = st.selectbox("Shape:", ["o", "s", "^", "D", "v", "*"], key=f"shape_{i}")
                marker_size = st.number_input("Marker size:", 20, 200, 80, key=f"size_{i}")
            
            with col4:
                label_font_size = st.number_input("Label font size:", 6, 24, 12, key=f"label_font_{i}")
                show_label_border = st.checkbox("Label border", key=f"border_{i}")
                if show_label_border:
                    label_bg_color = st.selectbox("Label bg:", ["white", "yellow", "lightgray"], key=f"bg_{i}")
                    label_alpha = st.slider("Label alpha:", 0.1, 1.0, 0.8, key=f"alpha_{i}")
                else:
                    label_bg_color = "white"
                    label_alpha = 0.8
            
            annotations.append({
                'mode': 'manual',
                'mz': mz_val,
                'tolerance': tolerance,
                'label': label,
                'color': color,
                'shape': shape,
                'show_line': True,
                'size': marker_size,
                'font_size': label_font_size,
                'show_label_border': show_label_border,
                'label_bg_color': label_bg_color,
                'label_alpha': label_alpha,
                'weight': 'bold'
            })

elif annotation_mode == "Automatic mass":
    num_masses = st.number_input("Number of masses:", 1, 10, 1)
    
    for i in range(num_masses):
        with st.expander(f"Mass {i+1}", expanded=True):
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                mass = st.number_input("Mass (Da):", value=15000.0 + i*1000, key=f"mass_{i}")
                charge_input = st.text_input("Charges:", "10,11,12", key=f"charges_{i}")
            
            with col2:
                threshold = st.slider("Threshold:", 0.001, 1.0, 0.01, key=f"thresh_{i}")
                color = st.selectbox("Color:", ["red", "blue", "green", "orange", "purple", "brown"], key=f"auto_color_{i}")
            
            with col3:
                shape = st.selectbox("Shape:", ["o", "s", "^", "D", "v", "*"], key=f"auto_shape_{i}")
                marker_size = st.number_input("Marker size:", 20, 200, 80, key=f"auto_size_{i}")
            
            with col4:
                show_mz = st.checkbox("Show m/z", key=f"show_mz_{i}")
                show_mass = st.checkbox("Show mass", key=f"show_mass_{i}")
                label_font_size = st.number_input("Label font size:", 6, 24, 12, key=f"auto_label_font_{i}")
                width = st.number_input("Search width:", 10, 500, 100, key=f"width_{i}")
            
            try:
                charges = [int(c.strip()) for c in charge_input.split(',')]
                annotations.append({
                    'mode': 'automatic',
                    'mass': mass,
                    'charge_states': charges,
                    'threshold': threshold,
                    'color': color,
                    'shape': shape,
                    'show_line': True,
                    'size': marker_size,
                    'font_size': label_font_size,
                    'width': width,
                    'show_mz': show_mz,
                    'show_mass': show_mass,
                    'show_label_border': False,
                    'label_bg_color': 'white',
                    'label_alpha': 0.8,
                    'weight': 'bold'
                })
            except:
                st.error(f"Invalid charge states for mass {i+1}")

# Data processing
with st.expander("⚙️ Data Processing", expanded=False):
    col1, col2, col3 = st.columns(3)
    
    with col1:
        smoothing = st.checkbox("Apply smoothing")
        if smoothing:
            smooth_window = st.number_input("Window size:", 3, 51, 5, step=2)
            smooth_order = st.number_input("Polynomial order:", 1, 5, 2)
        else:
            smooth_window = 5
            smooth_order = 2
    
    with col2:
        baseline_correction = st.checkbox("Baseline correction")
        if baseline_correction:
            baseline_percentile = st.slider("Baseline percentile:", 1, 20, 5)
        else:
            baseline_percentile = 5
    
    with col3:
        normalize = st.checkbox("Normalize data")
        if normalize:
            normalize_type = st.selectbox("Normalization:", ["max", "sum"])
        else:
            normalize_type = "max"

# Generate plot
if uploaded_files and all(f is not None for f in uploaded_files):
    if st.button("🎨 Generate Plot", type="primary", key="generate_plot_main"):
        with st.spinner("Generating plot..."):
            try:
                # Save files temporarily and read data
                file_paths = []
                for uploaded_file in uploaded_files:
                    temp_path = f"temp_{uploaded_file.name}"
                    with open(temp_path, "wb") as f:
                        f.write(uploaded_file.getbuffer())
                    file_paths.append(temp_path)
                
                # Read spectra
                spectra = []
                for file_path in file_paths:
                    spectrum = read_spectrum_file_safe(file_path)
                    if len(spectrum.intensity) > 0:
                        # Apply processing
                        processing_options = {
                            'smoothing': smoothing,
                            'smooth_window': smooth_window,
                            'smooth_order': smooth_order,
                            'baseline_correction': baseline_correction,
                            'baseline_percentile': baseline_percentile,
                            'normalize': normalize,
                            'normalize_type': normalize_type
                        }
                        processed_spectrum = apply_processing(spectrum, processing_options)
                        spectra.append(processed_spectrum)
                
                if not spectra:
                    st.error("No valid spectra loaded")
                else:
                    # Prepare plot settings
                    plot_settings = {
                        'width': figure_width,
                        'height': figure_height,
                        'dpi': dpi,
                        'font_size': font_size,
                        'font_family': font_family,
                        'font_weight': font_weight,
                        'background': background,
                        'x_min': x_min,
                        'x_max': x_max,
                        'y_min': y_min,
                        'y_max': y_max,
                        'line_color': line_color,
                        'line_width': line_width,
                        'line_style': line_style,
                        'alpha': alpha,
                        'fill_under': fill_under,
                        'fill_alpha': fill_alpha,
                        'zoom': zoom,
                        'preserve_baseline': preserve_baseline,  # 🔧 NEW
                        'stack_offset': stack_offset,
                        'show_grid': show_grid,
                        'grid_alpha': grid_alpha,
                        'grid_style': grid_style,
                        'grid_color': grid_color,
                        'show_x_label': show_x_label,
                        'show_y_label': show_y_label,
                        'x_label_text': x_label_text,
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
                        'spine_color': spine_color,
                        'palette': palette,
                        'line_color_2': line_color_2,
                        'show_legend': show_legend,
                        'legend_pos': legend_pos,
                        'legend_frame': legend_frame,
                        'title': plot_title,
                        'title_font_size': title_font_size,
                        'title_weight': title_weight,
                        # Stacked plot specific settings
                        'global_label_alpha': global_label_alpha,
                        'global_box_alpha': global_box_alpha,
                        'auto_avoid_overlap': auto_avoid_overlap,
                        'vertical_spacing': vertical_spacing,
                        'show_spectrum_numbers': show_spectrum_numbers,
                        'number_prefix': number_prefix,
                        'number_suffix': number_suffix,
                    }
                    
                    # Map plot type
                    plot_type_map = {
                        "Single Spectrum": "single",
                        "Stacked Comparison": "stacked",
                        "Overlay Plot": "overlay",
                        "Mirror Plot": "mirror"
                    }
                    
                    # Create plot
                    fig = create_safe_plot(spectra, plot_settings, annotations, 
                                         plot_type_map[plot_type], spectrum_labels, vertical_lines, spectrum_colors)
                    
                    if fig:
                        # Display plot
                        st.pyplot(fig)
                        
                        # Download options
                        st.markdown("### 📥 Download Options")
                        col1, col2, col3 = st.columns(3)
                        
                        with col1:
                            # PNG download
                            buf_png = io.BytesIO()
                            facecolor = 'none' if background == 'transparent' else background
                            fig.savefig(buf_png, format='png', dpi=dpi, bbox_inches='tight',
                                       facecolor=facecolor, transparent=(background == 'transparent'))
                            buf_png.seek(0)
                            
                            file_size_mb = len(buf_png.getvalue()) / (1024 * 1024)
                            st.download_button(f"📊 Download PNG ({dpi} DPI, {file_size_mb:.1f}MB)", 
                                             buf_png.getvalue(),
                                             f"{plot_type.lower().replace(' ', '_')}_plot.png", "image/png")
                        
                        with col2:
                            # PDF download
                            buf_pdf = io.BytesIO()
                            fig.savefig(buf_pdf, format='pdf', bbox_inches='tight',
                                       facecolor=facecolor, transparent=(background == 'transparent'))
                            buf_pdf.seek(0)
                            
                            st.download_button("📄 Download PDF", buf_pdf.getvalue(),
                                             f"{plot_type.lower().replace(' ', '_')}_plot.pdf", "application/pdf")
                        
                        with col3:
                            # SVG download
                            buf_svg = io.BytesIO()
                            fig.savefig(buf_svg, format='svg', bbox_inches='tight',
                                       facecolor=facecolor, transparent=(background == 'transparent'))
                            buf_svg.seek(0)
                            
                            st.download_button("🎨 Download SVG", buf_svg.getvalue(),
                                             f"{plot_type.lower().replace(' ', '_')}_plot.svg", "image/svg+xml")
                
                # Clean up
                for file_path in file_paths:
                    try:
                        os.remove(file_path)
                    except:
                        pass
                        
            except Exception as e:
                st.error(f"Error: {str(e)}")
                import traceback
                st.code(traceback.format_exc())

else:
    st.info("👆 Please upload spectrum files to begin plotting.")