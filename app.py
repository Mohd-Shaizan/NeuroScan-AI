import os
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration, WebRtcMode
import mediapipe as mp
import pathlib
import time  # ← Fix for Roadblock 1: Missing import

# 1. SILENCE LOGS
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# 2. PATHS 
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# 3. WEB_RTC CONFIG
RTC_CONFIG = RTCConfiguration({
    "iceServers": [
        {"urls": ["stun:stun.l.google.com:19302"]},
        {
            "urls": ["turn:openrelay.metered.ca:80"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
        {
            "urls": ["turn:openrelay.metered.ca:443?transport=tcp"],
            "username": "openrelayproject",
            "credential": "openrelayproject",
        },
    ]
})
class TremorProcessor(VideoProcessorBase):
    def __init__(self):
        from mediapipe.tasks import python as mp_python
        from mediapipe.tasks.python import vision as mp_vision
        
        if not os.path.exists(MODEL_PATH):
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

        # Fix for Roadblock 2: Define img_rgb before using it
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Wrap in try-except to prevent the 'NoneType' sendto crash if MediaPipe blips
        try:
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
            timestamp_ms = int(time.time() * 1000) 
            result = self.landmarker.detect_for_video(mp_image, timestamp_ms)

            if result.hand_landmarks:
                for landmarks in result.hand_landmarks:
                    for lm in landmarks:
                        ih, iw, _ = img.shape
                        cv2.circle(img, (int(lm.x * iw), int(lm.y * ih)), 5, (0, 255, 0), -1)
        except Exception as e:
            # Silently catch frame processing errors to keep the WebRTC thread alive
            pass

        return frame.from_ndarray(img, format="bgr24")

# 4. STREAMLIT UI
st.set_page_config(page_title="NeuroScan AI", layout="centered")
st.title("🫨 NeuroScan AI")

if not os.path.exists(MODEL_PATH):
    st.error(f"Missing File: hand_landmarker.task not found!")
    st.info("Ensure the file is in the root of your GitHub repo.")
else:
    # Change async_processing to True at the bottom of your app.py
    webrtc_streamer(
        key="neuroscan-v2", # Changing the key helps reset the state
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=TremorProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True, # Allows the AI to run without freezing the video
    )
