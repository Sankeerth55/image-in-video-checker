import os
import shutil
import tempfile
from datetime import datetime
import streamlit as st

# Define all working directories
PROJECT_DIRS = ["temp", "logs", "cache", "assets/models"]

def ensure_project_dirs():
    """
    Ensures that all required directories exist.
    Creates them if missing.
    """
    for d in PROJECT_DIRS:
        os.makedirs(d, exist_ok=True)


def save_upload(uploaded_file, subdir="temp") -> str:
    """
    Saves an uploaded Streamlit file uploader object to a permanent path.
    Returns the saved file path.
    """
    if uploaded_file is None:
        raise ValueError("No file uploaded")

    # Ensure directory exists
    ensure_project_dirs()
    save_dir = os.path.join(os.getcwd(), subdir)
    os.makedirs(save_dir, exist_ok=True)

    # Generate safe file name
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    safe_name = f"{timestamp}_{uploaded_file.name.replace(' ', '_')}"
    save_path = os.path.join(save_dir, safe_name)

    # Save file to disk
    with open(save_path, "wb") as f:
        f.write(uploaded_file.getbuffer())

    return save_path


def clear_temp_files(subdir="temp"):
    """
    Clears temporary uploaded files to free disk space.
    """
    temp_path = os.path.join(os.getcwd(), subdir)
    if os.path.exists(temp_path):
        try:
            shutil.rmtree(temp_path)
            os.makedirs(temp_path, exist_ok=True)
        except Exception as e:
            st.warning(f"Unable to clear temporary files: {e}")


def list_files(subdir="temp"):
    """
    Returns a list of files in a given directory.
    """
    folder = os.path.join(os.getcwd(), subdir)
    if not os.path.exists(folder):
        return []
    return [os.path.join(folder, f) for f in os.listdir(folder)]
