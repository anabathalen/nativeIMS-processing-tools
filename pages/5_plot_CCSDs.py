import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from io import BytesIO
from myutils import styling

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
    def plot_ccs_traces(
        cal_df: pd.DataFrame,
        selected_charges,
        palette,
        fig_width,
        fig_height,
        fig_dpi,
        font_size,
        line_thickness,
        plot_mode,
        use_scaled,
        ccs_min,
        ccs_max,
        ccs_label_value
    ):
        fig, ax = plt.subplots(figsize=(fig_width, fig_height), dpi=fig_dpi)
        ccs_grid = np.arange(ccs_min, ccs_max + 1, 1.0)
        interpolated_traces = []

        if plot_mode == "Summed":
            for i, (charge, group) in enumerate(cal_df.groupby("Charge")):
                group_sorted = group.sort_values("CCS")
                y_values = group_sorted["Scaled_Intensity"] if use_scaled else group_sorted["Normalized_Intensity"]
                interp = np.interp(ccs_grid, group_sorted["CCS"], y_values, left=0, right=0)
                interpolated_traces.append(interp)
                ax.plot(ccs_grid, interp, color=palette[i], label=f"{int(charge)}+", linewidth=line_thickness)
                ax.fill_between(ccs_grid, 0, interp, color=palette[i], alpha=0.3)
            total_trace = np.sum(interpolated_traces, axis=0)
            ax.plot(ccs_grid, total_trace, color="black", linewidth=line_thickness, label="Summed")
            ax.legend(fontsize=font_size, frameon=False)
        elif plot_mode == "Stacked":
            offset_unit = 1.0 / len(selected_charges)
            base_max = 0
            interpolated = {}
            for charge, group in cal_df.groupby("Charge"):
                group_sorted = group.sort_values("CCS")
                y_values = group_sorted["Scaled_Intensity"] if use_scaled else group_sorted["Normalized_Intensity"]
                interp = np.interp(ccs_grid, group_sorted["CCS"], y_values, left=0, right=0)
                interpolated[charge] = interp
                base_max = max(base_max, interp.max())
            for i, charge in enumerate(sorted(interpolated.keys())):
                interp = interpolated[charge]
                offset = i * offset_unit * base_max
                offset_interp = interp + offset
                ax.plot(ccs_grid, offset_interp, color=palette[i], linewidth=line_thickness)
                ax.fill_between(ccs_grid, offset, offset_interp, color=palette[i], alpha=0.3)
                label_x = ccs_min + (ccs_max - ccs_min) * 0.05
                label_y = offset + base_max * 0.05
                ax.text(label_x, label_y, f"{int(charge)}+", fontsize=font_size,
                        verticalalignment="bottom", horizontalalignment="left", color=palette[i])

        if ccs_label_value > 0:
            ax.axvline(ccs_label_value, color="black", linewidth=1.0, linestyle="--")
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
        fig_buffer = BytesIO()
        fig.savefig(fig_buffer, format='png', dpi=fig_dpi, bbox_inches='tight')
        fig_buffer.seek(0)
        st.download_button("Download CCS Plot as PNG", data=fig_buffer, file_name="ccs_plot.png", mime="image/png", key="ccs_download")

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
        palette_choice = st.selectbox("Choose a color palette", list(sns.palettes.SEABORN_PALETTES.keys()))
        fig_width = st.slider("Figure width", min_value=2, max_value=20, value=6)
        fig_height = st.slider("Figure height", min_value=2, max_value=20, value=4)
        fig_dpi = st.slider("Figure DPI", min_value=100, max_value=1000, value=300)
        font_size = st.slider("Font size", min_value=5, max_value=24, value=12)
        line_thickness = st.slider("Line thickness", min_value=0.1, max_value=5.0, value=1.0, step=0.1)
        plot_mode = st.radio("Display Mode", ["Summed", "Stacked"])
        use_scaled = st.radio("Use Scaled or Normalized Intensities?", ["Scaled", "Normalized"]) == "Scaled"
        return palette_choice, fig_width, fig_height, fig_dpi, font_size, line_thickness, plot_mode, use_scaled

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

    palette_choice, fig_width, fig_height, fig_dpi, font_size, line_thickness, plot_mode, use_scaled = UI.show_plot_options(selected_charges)
    ccs_min = float(np.floor(cal_df["CCS"].min()))
    ccs_max = float(np.ceil(cal_df["CCS"].max()))
    ccs_min_input = st.number_input("CCS x-axis min", value=ccs_min)
    ccs_max_input = st.number_input("CCS x-axis max", value=ccs_max)
    ccs_label_value = st.number_input("Optional CCS label position (leave blank if unused)", value=0.0, step=1.0, format="%.1f")
    palette = sns.color_palette(palette_choice, n_colors=len(selected_charges))

    st.subheader("Intensity vs CCS")
    CCSDPlotter.plot_ccs_traces(
        cal_df, selected_charges, palette, fig_width, fig_height, fig_dpi,
        font_size, line_thickness, plot_mode, use_scaled, ccs_min_input, ccs_max_input, ccs_label_value
    )

if __name__ == "__main__":
    main()