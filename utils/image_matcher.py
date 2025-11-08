from dataclasses import dataclass
from typing import Optional, Tuple, Dict
import os
import cv2
import numpy as np
import urllib.request

# ------------------------------------------------------------
# Model download setup
# ------------------------------------------------------------
MODELS = {
    "yunet": {
        "fname": "face_detection_yunet_2023mar.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_detection_yunet/face_detection_yunet_2023mar.onnx",
    },
    "sface": {
        "fname": "face_recognition_sface_2021dec.onnx",
        "url": "https://github.com/opencv/opencv_zoo/raw/main/models/face_recognition_sface/face_recognition_sface_2021dec.onnx",
    },
}


def _ensure_models(model_dir: str) -> Dict[str, str]:
    """Ensure YuNet and SFace models exist locally (auto-download if missing)."""
    os.makedirs(model_dir, exist_ok=True)
    out = {}
    for k, v in MODELS.items():
        fpath = os.path.join(model_dir, v["fname"])
        if not os.path.exists(fpath):
            try:
                urllib.request.urlretrieve(v["url"], fpath)
            except Exception:
                pass  # ignore if offline; will raise clearer error below
        out[k] = fpath
    return out


# ------------------------------------------------------------
# Config dataclass
# ------------------------------------------------------------
@dataclass
class MatchParams:
    # Face detector config
    score_thresh: float = 0.85
    nms_thresh: float = 0.3
    top_k: int = 5000
    # Recognition threshold (cosine similarity)
    match_thresh: float = 0.55
    # Max width for resize (affects speed/accuracy tradeoff)
    max_width: int = 960


# ------------------------------------------------------------
# ImageMatcher Class
# ------------------------------------------------------------
class ImageMatcher:
    """
    Fast, robust face matcher using YuNet + SFace.
    Handles both CPU and GPU frames (UMat compatible).
    """

    def __init__(self, ref_image_path: str, params: Optional[MatchParams] = None, model_dir: str = "assets/models"):
        self.params = params or MatchParams()
        self.model_dir = model_dir
        models = _ensure_models(model_dir)
        self.det_model_path = models["yunet"]
        self.rec_model_path = models["sface"]

        # Initialize detector and recognizer
        if not (os.path.exists(self.det_model_path) and os.path.exists(self.rec_model_path)):
            raise FileNotFoundError(
                "Required ONNX models not found. Ensure YuNet and SFace exist in assets/models, "
                "or let the app auto-download them."
            )

        self.detector = cv2.FaceDetectorYN.create(
            self.det_model_path, "", (320, 320),
            score_threshold=self.params.score_thresh,
            nms_threshold=self.params.nms_thresh,
            top_k=self.params.top_k
        )
        self.recognizer = cv2.FaceRecognizerSF.create(self.rec_model_path, "")

        # Load and process reference image
        ref_bgr = cv2.imread(ref_image_path)
        if ref_bgr is None:
            raise ValueError(f"Cannot read reference image: {ref_image_path}")

        ref_bgr = self._resize_max_width(ref_bgr, self.params.max_width)
        self.ref_face, self.ref_embedding = self._detect_and_embed(ref_bgr)
        if self.ref_embedding is None:
            raise ValueError("No face found in reference image. Please upload a clear, front-facing photo.")

    # --------------------------------------------------------
    # Utility: Resize (with UMat safety)
    # --------------------------------------------------------
    def _resize_max_width(self, img, max_w: int) -> np.ndarray:
        """Resize image safely, supporting UMat and None handling."""
        if img is None:
            raise ValueError("Received None frame for resizing.")

        if isinstance(img, cv2.UMat):
            try:
                arr = img.get()
                if arr is None:
                    raise ValueError("UMat conversion failed.")
                img = arr
            except Exception:
                # If conversion fails, skip gracefully
                pass

        if img is None or not hasattr(img, "shape"):
            raise ValueError("Invalid image passed to resize.")

        h, w = img.shape[:2]
        if w <= max_w:
            return img

        scale = max_w / float(w)
        try:
            return cv2.resize(img, (max_w, int(h * scale)), interpolation=cv2.INTER_AREA)
        except Exception:
            return img  # fallback if resize fails

    # --------------------------------------------------------
    # Detect faces
    # --------------------------------------------------------
    def _detect_faces(self, bgr: np.ndarray) -> np.ndarray:
        h, w = bgr.shape[:2]
        self.detector.setInputSize((w, h))
        dets = self.detector.detect(bgr)[1]
        if dets is None or len(dets) == 0:
            return np.empty((0, 15), dtype=np.float32)
        return dets

    # --------------------------------------------------------
    # Select best face
    # --------------------------------------------------------
    def _best_face(self, dets: np.ndarray) -> Optional[np.ndarray]:
        if dets is None or len(dets) == 0:
            return None
        dets = dets[np.argsort(-dets[:, 4])]
        return dets[0]

    # --------------------------------------------------------
    # Compute embeddings
    # --------------------------------------------------------
    def _embed(self, bgr: np.ndarray, det: np.ndarray) -> Optional[np.ndarray]:
        x, y, w, h = det[:4].astype(int)
        landmarks = det[5:].reshape(5, 2)
        aligned = self.recognizer.alignCrop(bgr, det[:4], landmarks)
        if aligned is None:
            return None
        feat = self.recognizer.feature(aligned)
        return feat

    # --------------------------------------------------------
    # Detect & Embed
    # --------------------------------------------------------
    def _detect_and_embed(self, bgr: np.ndarray) -> Tuple[Optional[np.ndarray], Optional[np.ndarray]]:
        dets = self._detect_faces(bgr)
        best = self._best_face(dets)
        if best is None:
            return None, None
        feat = self._embed(bgr, best)
        return best, feat

    # --------------------------------------------------------
    # Similarity Computation
    # --------------------------------------------------------
    def _similarity(self, f1: np.ndarray, f2: np.ndarray) -> float:
        return float(self.recognizer.match(f1, f2, cv2.FaceRecognizerSF_FR_COSINE))

    # --------------------------------------------------------
    # Main Match Function
    # --------------------------------------------------------
    def match(self, frame_bgr: np.ndarray) -> Tuple[bool, Dict]:
        """Compare input frame vs reference embedding."""
        if frame_bgr is None:
            return False, {"overlay_bgr": None, "inliers": 0, "confidence": 0.0, "good_matches": 0}

        # Convert UMat → ndarray
        if isinstance(frame_bgr, cv2.UMat):
            try:
                arr = frame_bgr.get()
                if arr is not None:
                    frame_bgr = arr
            except Exception:
                pass

        frame_bgr = self._resize_max_width(frame_bgr, self.params.max_width)
        dets = self._detect_faces(frame_bgr)
        if dets is None or len(dets) == 0:
            return False, {"overlay_bgr": frame_bgr, "inliers": 0, "confidence": 0.0, "good_matches": 0}

        best_sim = 0.0
        best_det = None
        for det in dets:
            feat = self._embed(frame_bgr, det)
            if feat is None:
                continue
            sim = self._similarity(self.ref_embedding, feat)
            if sim > best_sim:
                best_sim = sim
                best_det = det

        vis = frame_bgr.copy()
        if best_det is not None:
            x, y, w, h = best_det[:4].astype(int)
            cv2.rectangle(
                vis, (x, y), (x + w, y + h),
                (0, 255, 0) if best_sim >= self.params.match_thresh else (0, 0, 255),
                2
            )
            cv2.putText(
                vis, f"sim={best_sim:.3f}", (x, max(0, y - 8)),
                cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2, cv2.LINE_AA
            )

        is_match = best_sim >= self.params.match_thresh
        return is_match, {
            "overlay_bgr": vis,
            "inliers": int(best_sim * 1000),
            "good_matches": int(best_sim * 1000),
            "confidence": best_sim,
        }
