import streamlit as st

def load_custom_css(css_file: str = "static/styles.css"):
    with open(css_file) as f:
        st.markdown(f"<style>{f.read()},/style", unsafe_allow_html = True)