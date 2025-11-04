import streamlit as st
from pathlib import Path

def load_custom_css(css_file: str = "static/styles.css"):
    try:
        app_root = Path(__file__).resolve().parents[1]  # project root (nativeIMS-processing-tools/)
        css_path = (app_root / css_file).resolve()
        if not css_path.exists():
            st.warning(f"CSS file not found at {css_path}. Using default minimal styles.")
            st.markdown("<style>:root{--dummy:0}</style>", unsafe_allow_html=True)
            return

        css_text = css_path.read_text(encoding="utf-8")
        st.markdown(f"<style>{css_text}</style>", unsafe_allow_html=True)
    except Exception as e:
        st.warning(f"Failed to load CSS: {e}")
        st.markdown("<style>:root{--dummy:0}</style>", unsafe_allow_html=True)