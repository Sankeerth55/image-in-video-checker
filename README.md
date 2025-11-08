# 🧠 Image-in-Video Checker

### 🎥 AI-Powered Streamlit Web App  
Detect whether an **image, object, or face** appears inside any **local video** or **YouTube video** — using **Computer Vision + CLIP + OpenCV**, running 100% **locally and free**.

---

## 🌟 Features

✅ **Upload or Stream Videos**
- Supports local video uploads up to **1GB**
- Streams YouTube videos **directly (no download)** using `yt-dlp`

✅ **Smart Image Matching**
- Detects exact or partial matches of any **image or screenshot**
- Handles **logos, faces, and objects**
- Uses **CLIP (OpenAI)** for deep similarity + **ORB/SFace** fallback for local matching

✅ **AI-Powered Face Search**
- Upload a face photo and check where it appears in the video

✅ **Performance-Optimized**
- Efficient **frame sampling** to analyze large files without crashing
- Early stop when match is found (optional)
- Memory-safe with dynamic frame resizing

✅ **Modern Streamlit UI**
- Interactive controls for threshold, FPS, frame skip, and match mode
- Detailed timestamped results with preview images

---

## 🧩 Project Structure

image-in-video-checker/
│
├── app.py # Main Streamlit application
├── requirements.txt # Dependencies
├── utils/
│ ├── init.py
│ ├── video_utils.py # Video extraction (local + YouTube)
│ └── image_matcher.py # CLIP + Face/ORB matcher
│
├── .gitignore # Ignore temp/cache/media files
└── README.md # Documentation

---

## ⚙️ Installation & Setup

### 🧾 1. Clone the project
```bash
git clone https://github.com/yourusername/image-in-video-checker.git
cd image-in-video-checker
🐍 2. Create a Python environment
# (Recommended: Python 3.11)
python -m venv venv
source venv/bin/activate      # On Linux/Mac
venv\Scripts\activate         # On Windows

📦 3. Install dependencies
pip install -r requirements.txt


If you get a dependency error (rare), upgrade pip first:

python -m pip install --upgrade pip

🚀 Running the App

Run Streamlit:

streamlit run app.py


Your app will launch at:
👉 http://localhost:8501

💻 How to Use
1️⃣ Choose your video source

Upload Local Video: Supports .mp4, .avi, .mkv, .mov, etc. (up to 1GB)

YouTube URL: Paste a valid YouTube link — the app will stream it live (no full download)

2️⃣ Upload the target image or face

Supports all major image formats: .png, .jpg, .jpeg, .bmp, .gif, .webp

Can be a face, logo, or screenshot from the video

3️⃣ Configure analysis (sidebar)

Match Mode: Generic (logo/object) or Face

Threshold: Sensitivity (lower = more matches)

Sampling FPS / Step: Control how many frames per second are analyzed

Early Exit: Stop when first confident match found (faster)

4️⃣ Click 🔍 Analyze Video

App extracts frames → runs AI detection → shows timestamps and preview images.

🧠 How It Works
Component	Description
Streamlit	Interactive web interface for uploads and results
OpenCV	Extracts frames from local and YouTube streams
yt-dlp	Fetches direct streaming URLs from YouTube (no download)
CLIP (OpenAI)	Deep neural model to compute semantic image similarity
ORB/FLANN (OpenCV)	Local keypoint-based matcher for small/local objects
SFace (OpenCV DNN)	Lightweight, free model for face embeddings
Torch & Transformers	Backend frameworks for CLIP
NumPy / PIL / Scikit-image	Image preprocessing & handling
⚡ Example Usage
🧩 Example 1 – Logo in a video

Upload a Coca-Cola logo → check if it appears in a 10-minute ad compilation.

😊 Example 2 – Face detection

Upload a person’s face photo → check where they appear in a movie or CCTV clip.

🔗 Example 3 – YouTube link

Paste:

https://www.youtube.com/watch?v=abc123xyz


Then upload an image → app streams and searches directly!

🧰 Troubleshooting
Issue	Fix
“File exceeds limit”	Create .streamlit/config.toml and set maxUploadSize = 1024
“CUDA not available”	CLIP runs fine on CPU, just slower
“No face detected”	Try a clearer face with front view
“YouTube stream error”	Ensure you have the latest yt-dlp version
App too slow on large videos	Increase frame step or lower FPS sampling
📁 Optional: .streamlit/config.toml

To allow larger file uploads:

[server]
maxUploadSize = 1024

🧠 Technical Highlights

Streaming Frame Extraction:
Uses yt-dlp to fetch direct media URL → OpenCV decodes frames on the fly.

Dynamic Sampling:
Extracts frames based on desired FPS and frame step to optimize performance.

Hybrid Matching:

CLIP for global semantic similarity

ORB/FLANN for structural verification

Face embeddings via OpenCV DNN (SFace)

Responsive Web UI:
CSS-enhanced Streamlit layout with real-time progress tracking and collapsible results.

🛡️ License

This project is open-source under the MIT License
.
You’re free to use, modify, and share — attribution appreciated. 🙌

💬 Acknowledgments

OpenAI CLIP

Streamlit

OpenCV

yt-dlp