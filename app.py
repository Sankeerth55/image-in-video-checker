import os
import time
import traceback
from datetime import timedelta
from typing import List
import streamlit as st
import cv2
import numpy as np

from config import APP_NAME, MAX_WIDTH
from utils.file_handler import ensure_project_dirs, save_upload
from utils.image_matcher import ImageMatcher, MatchParams
from utils.video_processor import VideoProcessor, DetectionEvent

# ---------------- Streamlit Setup ----------------
st.set_page_config(
    page_title=APP_NAME,
    page_icon="🎯",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------- Theme Selection ----------------
with st.sidebar:
    st.header("🧭 Theme Settings")
    selected_theme = st.radio("Choose Theme:", ["🌞 Light", "🌙 Dark"], index=0)

# ---------------- Apply Theme ----------------
# Default light theme
if selected_theme == "🌞 Light":
    bg_color = "#f9f9fb"
    text_color = "#111"
    card_color = "#ffffff"
    gradient = "linear-gradient(135deg, #667eea 0%, #764ba2 100%)"
else:
    bg_color = "#0e1117"
    text_color = "#ffffff"
    card_color = "#1c1f26"
    gradient = "linear-gradient(135deg, #3a3b8f 0%, #5a189a 100%)"

# ---------------- Load CSS (Render-safe path) ----------------
css_path = os.path.join(os.path.dirname(__file__), "assets", "styles.css")
if os.path.exists(css_path):
    with open(css_path, "r", encoding="utf-8") as f:
        css_content = f.read()
        st.markdown(f"<style>{css_content}</style>", unsafe_allow_html=True)

# ---------------- Inject Theme Styling ----------------
st.markdown(
    f"""
    <style>
        body {{
            background-color: {bg_color};
            color: {text_color};
        }}
        .main-header {{
            background: {gradient};
            color: white;
        }}
        .stApp {{
            background-color: {bg_color};
        }}
        div[data-testid="stSidebar"] {{
            background-color: {card_color};
        }}
        .stButton>button {{
            background: {gradient} !important;
            color: white !important;
            border: none;
        }}
        .stProgress>div>div>div>div {{
            background: {gradient};
        }}
    </style>
    """,
    unsafe_allow_html=True,
)

# ---------------- Upload Config ----------------
st.session_state.setdefault("maxUploadSize", 200)
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = "200"
os.environ["STREAMLIT_SERVER_MAX_MESSAGE_SIZE"] = "200"

ensure_project_dirs()

# ---------------- Header ----------------
st.markdown(
    f"""
    <div class="main-header" style="text-align:center;padding:2rem;border-radius:12px;margin-bottom:2rem;">
      <h1>{APP_NAME} 🎯</h1>
      <p>This app detects whether your uploaded image appears anywhere in your uploaded video using advanced face recognition.</p>
    </div>
    """,
    unsafe_allow_html=True,
)

# ---------------- Sidebar Instructions ----------------
with st.sidebar:
    st.header("ℹ️ How to Use")
    st.markdown(
        """
        1. **Upload a clear image** — a face from your video.
        2. **Upload your video** — up to 200 MB (Render limit).
        3. Click **Start Checking** — scanning begins.
        4. See results below when done.
        """
    )

# ---------------- Upload Section ----------------
st.subheader("📤 Upload Files")

col_img, col_vid = st.columns(2)
ref_image_file, video_file = None, None

with col_img:
    ref_image_file = st.file_uploader(
        "Upload Reference Image (PNG/JPG/JPEG)", type=["png", "jpg", "jpeg"], key="ref_image"
    )
    if ref_image_file:
        ref_image_path = save_upload(ref_image_file, subdir="temp")
        st.image(ref_image_path, caption="Reference Image", use_column_width=True)

with col_vid:
    video_file = st.file_uploader(
        "Upload Video (MP4/MKV/MOV/AVI)", type=["mp4", "mkv", "mov", "avi"], key="video"
    )
    if video_file:
        video_path = save_upload(video_file, subdir="temp")
        st.video(video_path)

st.markdown("---")

# ---------------- Run Button ----------------
run_btn = st.button("🔎 Start Checking", type="primary", use_container_width=True)

# ---------------- Helpers ----------------
def _count_total_frames(path: str) -> int:
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        return 0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    cap.release()
    return total

# ---------------- Processing ----------------
if run_btn:
    if not ref_image_file or not video_file:
        st.error("Please upload both a reference image and a video before running.")
        st.stop()

    try:
        matcher_params = MatchParams(match_thresh=0.55, max_width=MAX_WIDTH)
        matcher = ImageMatcher(ref_image_path, params=matcher_params, model_dir=os.path.join("assets", "models"))

        processor = VideoProcessor(matcher=matcher, step=1, no_skip=True)

        total_frames = _count_total_frames(video_path)
        st.info("🚀 Scanning your video... Please wait until the process completes.")

        status_placeholder = st.container()
        with status_placeholder:
            st.markdown("<h4 style='color:crimson;'>Processing video…</h4>", unsafe_allow_html=True)
            progress = st.progress(0.0, text="Initializing...")

        start = time.time()
        processed = 0
        results: List[DetectionEvent] = []

        for out in processor.scan(video_path):
            processed = out.frames_processed

            if out.match is not None:
                results.append(out.match)

            elapsed = time.time() - start
            eta = (elapsed / processed) * (max(total_frames, processed) - processed) if processed else 0
            denom = total_frames if total_frames > 0 else processed + 1
            progress.progress(
                min(processed / denom, 1.0),
                text=f"Scanning… {processed}/{total_frames or '—'} frames | ETA: {str(timedelta(seconds=int(eta)))}"
            )

        status_placeholder.empty()
        st.markdown("<h4 style='color:green;'>✅ Scan Complete</h4>", unsafe_allow_html=True)

        # ---------------- Result Summary ----------------
        st.markdown("### 📊 Final Result")

        if results:
            timestamps = sorted({int(r.timestamp_sec) for r in results})
            timestamps_td = [str(timedelta(seconds=int(t))) for t in timestamps]

            st.success(f"🎉 Image found in the video ({len(timestamps_td)} unique occurrence(s))!")
            st.markdown("#### 🕓 Detected Timestamps:")
            for i, t in enumerate(timestamps_td, start=1):
                st.markdown(f"**{i}.** {t}")
        else:
            st.error("❌ Image not found in the video. Try using a clearer or larger face image.")

    except Exception as e:
        st.error("❌ Something went wrong while processing.")
        st.code("".join(traceback.format_exception(e)))
