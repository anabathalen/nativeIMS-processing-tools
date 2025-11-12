import os
import numpy as np
import pandas as pd
import streamlit as st
import io
import zipfile
import shutil
from pathlib import Path

# handles uploaded zip files returning the names of the sub-folders and defining a temporary directory to store the data
def handle_zip_upload(uploaded_file):
    temp_dir = '/tmp/extracted_zip/'

    if os.path.exists(temp_dir):
        for root, dirs, files in os.walk(temp_dir, topdown=False):
            for name in files:
                os.remove(os.path.join(root, name))
            for name in dirs:
                os.rmdir(os.path.join(root, name))

    os.makedirs(temp_dir, exist_ok = True)

    with zipfile.ZipFile(uploaded_file, 'r') as zip_ref:
        zip_ref.extractall(temp_dir)

    folders = [f for f in os.listdir(temp_dir) if os.path.isdir(os.path.join(temp_dir, f))]
    if not folders:
        st.error("No folders found in the ZIP file :(")
    return folders, temp_dir

# read and store the bush database in its current state
def read_bush():
    file_path = os.path.join(os.path.dirname(__file__), '../data/bush.csv')
    if os.path.exists(file_path):
        bush_df = pd.read_csv(file_path)
    else:
        st.error(f"'{file_path}' not found. Make sure 'bush.csv' is in the 'data' folder on GitHub.")
        bush_df = pd.DataFrame() # make an empty dataframe if there is no bush.csv to be found
    return bush_df

# clear all data
def clear_cache():
    """Remove temp data or cached files, but do NOT call st.experimental_rerun() here."""
    temp_dir = Path('/tmp/extracted_zip/')  # or whatever temp dir you're using
    if temp_dir.exists() and temp_dir.is_dir():
        shutil.rmtree(temp_dir, ignore_errors=True)