import os
import time
import pathlib

import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import mediapipe as mp
import av

# ──────────────────────────────────────────────
# 1. SILENCE NOISY LOGS
# ──────────────────────────────────────────────
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# ──────────────────────────────────────────────
# 2. PATHS
# ──────────────────────────────────────────────
BASE_DIR   = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = str(BASE_DIR / "hand_landmarker.task")

# ──────────────────────────────────────────────
# 3. RTC CONFIG  (more STUN = faster ICE)
# ──────────────────────────────────────────────
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {"urls": ["stun:stun1.l.google.com:19302"]},
        {"urls": ["stun:stun2.l.google.com:19302"]},
        {"urls": ["stun:stun3.l.google.com:19302"]},
        {"urls": ["stun:stun4.l.google.com:19302"]},
        {"urls": ["stun:stun.cloudflare.com:3478"]},
    ]
})

# ──────────────────────────────────────────────
# 4. VIDEO PROCESSOR
# ──────────────────────────────────────────────
class TremorProcessor(VideoProcessorBase):          # ← inherit VideoProcessorBase
    def __init__(self):
        if not os.path.exists(MODEL_PATH):
            st.error(f"Model file not found: {MODEL_PATH}")
            self.landmarker = None
            return

        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")

        if self.landmarker is None:
            return av.VideoFrame.from_ndarray(img, format="bgr24")

        # ── Process ──────────────────────────────
        img_rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)

        # Use wall-clock time — avoids all frame.pts math bugs
        timestamp_ms = int(time.time() * 1000)

        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

        # ── Draw landmarks ───────────────────────
        if result.hand_landmarks:
            ih, iw, _ = img.shape
            for landmarks in result.hand_landmarks:
                for lm in landmarks:
                    cx, cy = int(lm.x * iw), int(lm.y * ih)
                    cv2.circle(img, (cx, cy), 5, (0, 255, 0), -1)

        return av.VideoFrame.from_ndarray(img, format="bgr24")

# ──────────────────────────────────────────────
# 5. STREAMLIT UI
# ──────────────────────────────────────────────
st.set_page_config(page_title="NeuroScan AI", layout="centered")
st.title("🫨 NeuroScan AI")
st.caption("Real-time hand tremor detection via webcam")

if not os.path.exists(MODEL_PATH):
    st.error(f"Missing model file: `hand_landmarker.task` not found in `{BASE_DIR}`")
    st.info("Make sure `hand_landmarker.task` is committed to the root of your GitHub repo.")
    st.stop()

st.info("⏳ Camera may take 15–30 seconds to connect on first load. Please wait after clicking START.")

webrtc_streamer(
    key="neuroscan",
    video_processor_factory=TremorProcessor,
    rtc_configuration=RTC_CONFIG,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
