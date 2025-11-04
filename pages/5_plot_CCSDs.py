import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from myutils import styling
from scipy.signal import argrelextrema

class CCSDPlotter:
    @staticmethod
    def load_calibrated_csv(file) -> pd.DataFrame:
        df = pd.read_csv(file)
        required_cols = {"Charge", "CCS", "CCS Std.Dev.", "Scaled_Intensity", "Normalized_Intensity", "m/z"}
        if not required_cols.issubset(df.columns):
            st.error(f"CSV file missing required columns: {required_cols - set(df.columns)}")
            return None
        df = df[df["CCS Std.Dev."] < 0.5 * df["CCS"]].copy()
        return df

    @staticmethod
    def load_gaussian_fits(file) -> pd.DataFrame:
        """Load Gaussian fit parameters CSV."""
        df = pd.read_csv(file)
        required = {
            "Charge","Peak_Number","Peak_Type","Amplitude","Center_CCS","Sigma"
        }
        missing = required - set(df.columns)
        if missing:
            st.error(f"Gaussian fits CSV missing required columns: {missing}")
            return None
        # Keep only Gaussian peaks; coerce numeric
        df = df[df["Peak_Type"].str.lower() == "gaussian"].copy()
        for col in ["Charge","Amplitude","Center_CCS","Sigma"]:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        df = df.dropna(subset=["Charge","Amplitude","Center_CCS","Sigma"])
        # Ensure integer charge
        df["Charge"] = df["Charge"].astype(int)
        return df

    @staticmethod
    def plot_ccs_traces(
        cal_df: pd.DataFrame,
        selected_charges,
        trace_palette,
         fig_width,
         fig_height,
         fig_dpi,
         font_size,
         line_thickness,
         plot_mode,
         use_scaled,
         ccs_min,
         ccs_max,
         ccs_label_values,
         font_family,
         bg_option,
         file_name,
         show_dashed_lines,
         show_ccs_labels,
         label_vertical_pos,
         label_orientation,
         shade_under,
         black_lines,
         gaussian_fits_df: pd.DataFrame = None,
         show_gaussian_fits: bool = False,
         gaussian_palette: list | None = None,
         shade_gaussians: bool = False
     ):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=fig_dpi)
        plt.rcParams.update({'font.family': font_family})
        ccs_grid = np.arange(ccs_min, ccs_max + 1, 1.0)
        interpolated_traces = []
        maxima_info = []
        def compute_gaussian_components(charge_val: int):
            """Return list of individual Gaussian component curves for this charge across ccs_grid."""
            if gaussian_fits_df is None or gaussian_fits_df.empty:
                return []
            sub = gaussian_fits_df[gaussian_fits_df["Charge"] == int(charge_val)]
            if sub.empty:
                return []
            comps = []
            # Keep consistent order by Peak_Number so component color index is repeatable across charges
            for _, r in sub.sort_values(["Peak_Number"]).iterrows():
                amp = float(r["Amplitude"])
                cen = float(r["Center_CCS"])
                sig = float(r["Sigma"])
                if sig <= 0 or not np.isfinite([amp, cen, sig]).all():
                    continue
                g = amp * np.exp(-0.5 * ((ccs_grid - cen) / sig) ** 2)
                comps.append(g)
            if not comps:
                return []
            if not use_scaled:
                summed = np.sum(comps, axis=0)
                m = float(np.max(summed)) if np.isfinite(summed).any() and np.max(summed) > 0 else 0.0
                if m > 0:
                    comps = [c / m for c in comps]
            return comps

        if plot_mode == "Summed":
            for i, (charge, group) in enumerate(cal_df.groupby("Charge")):
                group_sorted = group.sort_values("CCS")
                y_values = group_sorted["Scaled_Intensity"] if use_scaled else group_sorted["Normalized_Intensity"]
                interp = np.interp(ccs_grid, group_sorted["CCS"], y_values, left=0, right=0)
                interpolated_traces.append(interp)
                # If user picked palette "Black" the entries are already 'black'
                line_color = "black" if black_lines else trace_palette[i]
                ax.plot(ccs_grid, interp, color=line_color, label=f"{int(charge)}+", linewidth=line_thickness)
                if shade_under:
                    ax.fill_between(ccs_grid, 0, interp, color=trace_palette[i], alpha=0.3)
                # Overlay Gaussian components per charge (individual curves)
                if show_gaussian_fits:
                    comps = compute_gaussian_components(charge)
                    # Color each component by index using gaussian_palette (fallback to line color)
                    for j, comp in enumerate(comps):
                        comp_color = (gaussian_palette[j % len(gaussian_palette)]
                                      if gaussian_palette and len(gaussian_palette) > 0
                                      else line_color)
                        ax.plot(
                            ccs_grid, comp,
                            color=comp_color, linestyle="--", linewidth=max(line_thickness, 1.0),
                            alpha=0.9
                        )
                        if shade_gaussians:
                            ax.fill_between(ccs_grid, 0, comp, color=comp_color, alpha=0.15)
            total_trace = np.sum(interpolated_traces, axis=0) if interpolated_traces else np.array([])
            if total_trace.size:
                ax.plot(ccs_grid, total_trace, color="black", linewidth=line_thickness, label="Summed")
                maxima_idx = argrelextrema(total_trace, np.greater)[0]
                maxima_ccs = ccs_grid[maxima_idx]
                maxima_vals = total_trace[maxima_idx]
                maxima_info.append(("Summed", list(zip(maxima_ccs, maxima_vals))))
            ax.legend(fontsize=font_size, frameon=False)

        elif plot_mode == "Stacked":
            interpolated = {}
            base_max = 0.0
            for charge, group in cal_df.groupby("Charge"):
                group_sorted = group.sort_values("CCS")
                y_values = group_sorted["Scaled_Intensity"] if use_scaled else group_sorted["Normalized_Intensity"]
                interp = np.interp(ccs_grid, group_sorted["CCS"], y_values, left=0, right=0)
                interpolated[int(charge)] = interp
                if interp.size:
                    base_max = max(base_max, float(interp.max()))
            if not use_scaled:
                gap = 0.10 * base_max if base_max > 0 else 0.10
                offset_step = (base_max if base_max > 0 else 1.0) + gap
                def offset_for(i): return i * offset_step
            else:
                offset_unit = 1.0 / max(len(interpolated), 1)
                def offset_for(i): return i * offset_unit * base_max

            for i, charge in enumerate(sorted(interpolated.keys())):
                interp = interpolated[charge]
                offset = offset_for(i)
                offset_interp = interp + offset
                line_color = "black" if black_lines else trace_palette[i]
                ax.plot(ccs_grid, offset_interp, color=line_color, linewidth=line_thickness)
                if shade_under:
                    ax.fill_between(ccs_grid, offset, offset_interp, color=trace_palette[i], alpha=0.3)
                # Overlay Gaussian components per charge at same offset
                if show_gaussian_fits:
                    comps = compute_gaussian_components(charge)
                    for j, comp in enumerate(comps):
                        comp_color = (gaussian_palette[j % len(gaussian_palette)]
                                      if gaussian_palette and len(gaussian_palette) > 0
                                      else line_color)
                        ax.plot(
                            ccs_grid, comp + offset,
                            color=comp_color, linestyle="--", linewidth=max(line_thickness, 1.0),
                            alpha=0.9
                        )
                        if shade_gaussians:
                            ax.fill_between(ccs_grid, offset, comp + offset, color=comp_color, alpha=0.15)
                label_x = ccs_min + (ccs_max - ccs_min) * 0.05
                label_y = offset + (base_max * 0.05 if base_max > 0 else 0.05)
                ax.text(label_x, label_y, f"{int(charge)}+", fontsize=font_size,
                        verticalalignment="bottom", horizontalalignment="left", color=trace_palette[i])
                maxima_idx = argrelextrema(interp, np.greater)[0]
                maxima_ccs = ccs_grid[maxima_idx]
                maxima_vals = interp[maxima_idx]
                maxima_info.append((f"{int(charge)}+", list(zip(maxima_ccs, maxima_vals))))

        # Draw all user-specified CCS label lines and annotate them inside the plot area if enabled
        ylim = ax.get_ylim()
        y_label = ylim[0] + label_vertical_pos * (ylim[1] - ylim[0])
        for ccs_label_value in ccs_label_values:
            if ccs_label_value > 0:
                if show_dashed_lines:
                    ax.axvline(ccs_label_value, color="black", linewidth=1.0, linestyle="--")
                if show_ccs_labels:
                    ax.text(
                        ccs_label_value,
                        y_label,
                        f"{ccs_label_value:.1f}",
                        rotation=90 if label_orientation == "Vertical" else 0,
                        verticalalignment='top' if label_orientation == "Vertical" else 'center',
                        horizontalalignment='right' if label_orientation == "Vertical" else 'center',
                        fontsize=font_size,
                        color="black",
                        backgroundcolor="white" if bg_option == "White" else "none"
                    )

        ax.set_xlim([ccs_min, ccs_max])
        ax.set_xlabel("CCS (Å²)", fontsize=font_size)
        ax.set_yticks([])
        ax.grid(False)
        for label in ax.get_xticklabels():
            label.set_fontsize(font_size)
        for spine in ax.spines.values():
            spine.set_edgecolor("black")
            spine.set_linewidth(1.5)
        st.pyplot(fig)

        # Show local maxima info
        st.markdown("**Local maxima (CCS, Intensity):**")
        for label, maxima in maxima_info:
            if maxima:
                st.write(f"{label}: " + ", ".join([f"({ccs:.1f}, {val:.2f})" for ccs, val in maxima]))
            else:
                st.write(f"{label}: None found")

        fig_buffer = BytesIO()
        transparent = (bg_option == "Transparent")
        fig.savefig(fig_buffer, format='png', dpi=fig_dpi, bbox_inches='tight', transparent=transparent)
        fig_buffer.seek(0)
        st.download_button(
            "Download CCS Plot as PNG",
            data=fig_buffer,
            file_name=file_name if file_name.endswith('.png') else file_name + '.png',
            mime="image/png",
            key="ccs_download"
        )

class UI:
    @staticmethod
    def show_main_header():
        st.markdown("""
        <div class="main-header">
            <h1>Plot Calibrated & Scaled IMS Data</h1>
            <p>Upload the CSV files generated from the previous step. This page allows you to plot the scaled CCSDs for selected charge states, either stacked or summed. No further normalization or scaling is performed here.</p>
        </div>
        """, unsafe_allow_html=True)

    @staticmethod
    def show_upload_section():
        st.markdown("""
        <div class="section-card">
            <div class="section-header">📁 Step 1: Upload Calibrated & Scaled CSV File</div>
        </div>
        """, unsafe_allow_html=True)
        return st.file_uploader("Upload calibrated & scaled CSV file", type="csv")

    @staticmethod
    def show_plot_options(charges):
        st.markdown("""
        <div class="section-card">
            <div class="section-header">🎨 Step 2: Plot Options</div>
        </div>
        """, unsafe_allow_html=True)
        trace_palette_choice = st.selectbox(
            "Trace color palette (raw data)",
            list(sns.palettes.SEABORN_PALETTES.keys()) + ["Black"]
        )
        fig_width = st.slider("Figure width", min_value=2, max_value=20, value=6)
        fig_height = st.slider("Figure height", min_value=2, max_value=20, value=4)
        fig_dpi = st.slider("Figure DPI", min_value=100, max_value=1000, value=300)
        font_size = st.slider("Font size", min_value=5, max_value=24, value=12)
        line_thickness = st.slider("Line thickness", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        plot_mode = st.radio("Display Mode", ["Summed", "Stacked"])
        use_scaled = st.radio("Use Scaled or Normalized Intensities?", ["Scaled", "Normalized"]) == "Scaled"
        font_family = st.selectbox("Font family", ["DejaVu Sans", "Arial", "Times New Roman", "Calibri", "Verdana"])
        bg_option = st.radio("Background", ["White", "Transparent"])
        file_name = st.text_input("PNG file name", value="ccs_plot.png")
        show_dashed_lines = st.checkbox("Show dashed CCS lines", value=True)
        show_ccs_labels = st.checkbox("Show CCS value labels inside plot", value=True)
        label_vertical_pos = st.slider("Label vertical position (0 = bottom, 1 = top)", min_value=0.0, max_value=1.0, value=0.95)
        label_orientation = st.radio("Label orientation", ["Vertical", "Horizontal"])
        shade_under = st.checkbox("Shade under the curves", value=True)  # <-- added
        black_lines = st.checkbox("Use black lines for traces", value=False)  # <-- added
        return (trace_palette_choice, fig_width, fig_height, fig_dpi, font_size, line_thickness, plot_mode, use_scaled, 
                font_family, bg_option, file_name, show_dashed_lines, show_ccs_labels, label_vertical_pos, label_orientation,
                shade_under, black_lines)  # <-- updated

def main():
    styling.load_custom_css()
    UI.show_main_header()
    cal_file = UI.show_upload_section()
    if not cal_file:
        st.info("Please upload a calibrated & scaled CSV file to continue.")
        return

    cal_df = CCSDPlotter.load_calibrated_csv(cal_file)
    if cal_df is None or cal_df.empty:
        return

    all_charges = sorted(cal_df["Charge"].unique())
    selected_charges = st.multiselect("Select charge states to include", all_charges, default=all_charges)
    cal_df = cal_df[cal_df["Charge"].isin(selected_charges)]

    st.dataframe(cal_df)

    # Optional: Gaussian fits overlay
    st.markdown("### Optional Gaussian Fits")
    fits_file = st.file_uploader("Upload Gaussian fits CSV (Amplitude/Center_CCS/Sigma)", type="csv", key="fits_csv")
    show_gaussian_fits = st.checkbox("Overlay Gaussian fits", value=False)
    gaussian_fits_df = None
    shade_gaussians = False
    gaussian_palette_choice = None
    gaussian_palette = None
    if fits_file and show_gaussian_fits:
        gaussian_fits_df = CCSDPlotter.load_gaussian_fits(fits_file)
        if gaussian_fits_df is not None and not gaussian_fits_df.empty:
            # Filter to selected charges only
            gaussian_fits_df = gaussian_fits_df[gaussian_fits_df["Charge"].isin([int(c) for c in selected_charges])]
            # Options for Gaussian components
            gaussian_palette_choice = st.selectbox(
                "Gaussian component palette",
                list(sns.palettes.SEABORN_PALETTES.keys()) + ["Black"],
                index=0,
                key="gaussian_palette_choice"
            )
            shade_gaussians = st.checkbox("Shade under Gaussian components", value=False, key="shade_gaussian_components")

    (trace_palette_choice, fig_width, fig_height, fig_dpi, font_size, line_thickness, plot_mode, use_scaled, 
     font_family, bg_option, file_name, show_dashed_lines, show_ccs_labels, label_vertical_pos, label_orientation,
     shade_under, black_lines) = UI.show_plot_options(selected_charges)
    ccs_min = float(np.floor(cal_df["CCS"].min()))
    ccs_max = float(np.ceil(cal_df["CCS"].max()))
    ccs_min_input = st.number_input("CCS x-axis min", value=ccs_min)
    ccs_max_input = st.number_input("CCS x-axis max", value=ccs_max)
    ccs_label_str = st.text_input("Optional CCS label positions (comma-separated, e.g. 1200,1350)", value="")
    ccs_label_values = []
    if ccs_label_str.strip():
        try:
            ccs_label_values = [float(x.strip()) for x in ccs_label_str.split(",") if x.strip()]
        except Exception:
            st.warning("Could not parse CCS label positions. Please enter comma-separated numbers.")

    # Build trace palette
    if trace_palette_choice == "Black":
        trace_palette = ["black"] * max(1, len(selected_charges))
    else:
        trace_palette = sns.color_palette(trace_palette_choice, n_colors=len(selected_charges))
    
    # Build gaussian palette if applicable; cycle per-peak across charges
    if gaussian_fits_df is not None and not gaussian_fits_df.empty and gaussian_palette_choice:
        # Estimate number of components (use unique Peak_Number across file)
        try:
            n_comps = int(max(gaussian_fits_df.groupby('Charge')['Peak_Number'].nunique().max(), 1))
        except Exception:
            n_comps = int(max(gaussian_fits_df['Peak_Number'].nunique(), 1))
        if gaussian_palette_choice == "Black":
            gaussian_palette = ["black"] * max(1, n_comps)
        else:
            gaussian_palette = sns.color_palette(gaussian_palette_choice, n_colors=max(1, n_comps))
 
    st.subheader("Intensity vs CCS")
    CCSDPlotter.plot_ccs_traces(
        cal_df, selected_charges, trace_palette, fig_width, fig_height, fig_dpi,
        font_size, line_thickness, plot_mode, use_scaled, ccs_min_input, ccs_max_input, ccs_label_values,
        font_family, bg_option, file_name, show_dashed_lines, show_ccs_labels, label_vertical_pos, label_orientation,
        shade_under, black_lines,
        gaussian_fits_df=gaussian_fits_df,
        show_gaussian_fits=show_gaussian_fits,
        gaussian_palette=gaussian_palette,
        shade_gaussians=shade_gaussians
    )

if __name__ == "__main__":
    main()