import os
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
import pathlib

# 1. SILENCE LOGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# 2. PATHS (Pointing to your uploaded file)
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# 3. WEB_RTC CONFIG
# This is the "Phone Book" that helps the browser find the server
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

class TremorProcessor:
    def __init__(self):
        # Initialize MediaPipe once when the stream starts
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        
        # Verify file exists to prevent crash
        if not os.path.exists(MODEL_PATH):
            print(f"CRITICAL ERROR: {MODEL_PATH} not found!")
            self.landmarker = None
            return

        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO,
            num_hands=1
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)

    def recv(self, frame):
         img = frame.to_ndarray(format="bgr24")
        
        if self.landmarker is None:
            return frame.from_ndarray(img, format="bgr24")
    
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # ✅ Correct timestamp calculation
        timestamp_ms = int(frame.pts * float(frame.time_base) * 1000)
        
        result = self.landmarker.detect_for_video(mp_image, timestamp_ms)        
        # Draw Landmarks
        if result.hand_landmarks:
            for landmarks in result.hand_landmarks:
                for lm in landmarks:
                    ih, iw, _ = img.shape
                    cv2.circle(img, (int(lm.x * iw), int(lm.y * ih)), 5, (0, 255, 0), -1)

        return frame.from_ndarray(img, format="bgr24")

# 4. STREAMLIT UI
st.set_page_config(page_title="NeuroScan AI", layout="centered")
st.title("🫨 NeuroScan AI")

# Check if file uploaded correctly before showing the UI
if not os.path.exists(MODEL_PATH):
    st.error(f"Missing File: hand_landmarker.task not found in {BASE_DIR}")
    st.info("Please ensure the file is uploaded to your GitHub main folder.")
else:
    webrtc_streamer(
        key="neuroscan",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=TremorProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )
