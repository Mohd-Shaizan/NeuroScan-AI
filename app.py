import os
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pathlib
import urllib.request

# 1. LOGGING & PATHS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

BASE_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# 2. AUTOMATIC MODEL DOWNLOADER
# This ensures the .task file exists before the AI starts
def ensure_model_exists():
    if not os.path.exists(MODEL_PATH):
        url = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
        try:
            with st.spinner("Downloading AI Model... Please wait."):
                urllib.request.urlretrieve(url, MODEL_PATH)
        except Exception as e:
            st.error(f"Failed to download model: {e}")

# 3. WEB_RTC CONFIG (The fix for your 'sendto' error)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

class TremorProcessor:
    def __init__(self):
        # Initialize the AI inside the processor
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # AI Processing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        timestamp_ms = int(frame.pts * 1000 / frame.time_base.denominator)
        
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Visualization
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                for lm in landmarks:
                    ih, iw, _ = img.shape
                    cv2.circle(img, (int(lm.x * iw), int(lm.y * ih)), 5, (0, 255, 0), -1)

        return frame.from_ndarray(img, format="bgr24")

# 4. STREAMLIT UI
st.title("🫨 NeuroScan AI")
ensure_model_exists()

webrtc_streamer(
    key="neuroscan",
    mode=WebRtcMode.SENDRECV,
    rtc_configuration=RTC_CONFIG,
    video_processor_factory=TremorProcessor,
    media_stream_constraints={"video": True, "audio": False},
    async_processing=True,
)
