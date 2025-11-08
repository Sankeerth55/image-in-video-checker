import os

# --------------------------------------------------
# 💡 Core Application Configuration
# --------------------------------------------------

APP_NAME = "Image in Video Checker"

# Maximum width used for frame/image processing
# (Higher = more accuracy, lower = more speed)
MAX_WIDTH = 960

# Folder setup
TEMP_DIR = os.path.join(os.getcwd(), "temp")
LOG_DIR = os.path.join(os.getcwd(), "logs")
CACHE_DIR = os.path.join(os.getcwd(), "cache")
MODEL_DIR = os.path.join(os.getcwd(), "assets", "models")

# Streamlit and video limits
MAX_UPLOAD_SIZE_MB = 1024  # 1 GB
os.environ["STREAMLIT_SERVER_MAX_UPLOAD_SIZE"] = str(MAX_UPLOAD_SIZE_MB)
os.environ["STREAMLIT_SERVER_MAX_MESSAGE_SIZE"] = str(MAX_UPLOAD_SIZE_MB)

# --------------------------------------------------
# 🧠 Model / Detection Settings
# --------------------------------------------------
# You can tune these globally if needed:
FACE_MATCH_THRESHOLD = 0.55   # default similarity threshold
FRAME_STEP = 1                # process every frame (increase for speed)
SCORE_THRESHOLD = 0.85        # YuNet face detector confidence
NMS_THRESHOLD = 0.3           # Non-maximum suppression
TOP_K = 5000                  # keep top detections per frame

# --------------------------------------------------
# Utility
# --------------------------------------------------
def ensure_model_dirs():
    """Ensures model directory exists before loading YuNet/SFace."""
    os.makedirs(MODEL_DIR, exist_ok=True)
    return MODEL_DIR
