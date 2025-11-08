from dataclasses import dataclass
from typing import Generator, Optional
import cv2
import time
import numpy as np

# Try enabling OpenCL, but don't fail if not available
try:
    cv2.ocl.setUseOpenCL(True)
except Exception:
    pass

@dataclass
class DetectionEvent:
    timestamp_sec: float
    frame_index: int
    confidence: float
    inliers: int
    overlay_bgr: any  # numpy array

@dataclass
class ScanStep:
    frames_processed: int
    fps: float
    match: Optional[DetectionEvent]


class VideoProcessor:
    """
    Robust, GPU-aware video scanner.
    Keeps same accuracy and UI while being defensive about UMat/resize.
    """
    def __init__(self, matcher, step:int = 5, no_skip: bool = True, max_width: int = 480):
        self.matcher = matcher
        self.step = max(1, int(step))
        self.no_skip = no_skip
        self.max_width = int(max_width)

    def _resize_fast(self, frame, max_width: int):
        """Resize frame, preferring GPU (UMat) but falling back to CPU safely."""
        if frame is None:
            return None

        # If frame is a UMat already, get ndarray
        if isinstance(frame, cv2.UMat):
            try:
                arr = frame.get()
                if arr is not None:
                    frame = arr
                else:
                    # fallback: continue below to CPU resize
                    pass
            except Exception:
                # fallback to CPU path
                pass

        h, w = frame.shape[:2]
        if w <= max_width:
            return frame

        scale = max_width / float(w)

        # Try GPU-accelerated resize if available, but guard against failures
        if cv2.ocl.haveOpenCL():
            try:
                u_frame = cv2.UMat(frame)
                resized = cv2.UMat()
                cv2.resize(u_frame, (max_width, int(h * scale)), dst=resized, interpolation=cv2.INTER_LINEAR)
                out = resized.get()
                if out is not None:
                    return out
                # else fall through to CPU resize
            except Exception:
                # ignore and fall back to CPU
                pass

        # CPU fallback (guaranteed)
        try:
            return cv2.resize(frame, (max_width, int(h * scale)), interpolation=cv2.INTER_LINEAR)
        except Exception:
            # In case resize fails (very rare), return original frame as last resort
            return frame

    def scan(self, video_path: str) -> Generator[ScanStep, None, None]:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Cannot open video: {video_path}")

        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        fps_native = float(cap.get(cv2.CAP_PROP_FPS) or 30.0)
        frame_idx = 0
        frames_processed = 0
        avg_fps = 0.0
        t0 = time.time()

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                frame_idx += 1

                # Skip frames for speed
                if frame_idx % self.step != 0:
                    continue

                # Resize with safe fallback
                frame_small = self._resize_fast(frame, self.max_width)
                if frame_small is None:
                    # Skip this frame robustly if resize failed
                    continue

                # Match face (matcher expects BGR ndarray)
                ok, info = self.matcher.match(frame_small)

                match_evt = None
                if ok:
                    ts = frame_idx / max(fps_native, 1e-6)
                    match_evt = DetectionEvent(
                        timestamp_sec=ts,
                        frame_index=frame_idx,
                        confidence=float(info.get("confidence", 0.0)),
                        inliers=int(info.get("inliers", 0)),
                        overlay_bgr=info.get("overlay_bgr", frame_small),
                    )

                frames_processed += 1
                elapsed = time.time() - t0
                if elapsed > 0:
                    inst_fps = frames_processed / elapsed
                    avg_fps = 0.9 * avg_fps + 0.1 * inst_fps if avg_fps else inst_fps

                yield ScanStep(frames_processed=frames_processed, fps=avg_fps, match=match_evt)

        finally:
            cap.release()
