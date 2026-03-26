import os
import cv2
import numpy as np
import streamlit as st
from streamlit_webrtc import webrtc_streamer, WebRtcMode, RTCConfiguration
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
import pathlib
import queue

# 1. Silence unnecessary logs from MediaPipe/TF
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
import absl.logging
absl.logging.set_verbosity(absl.logging.ERROR)

# 2. Setup Paths
BASE_DIR = pathlib.Path(__file__).parent.resolve()
MODEL_PATH = os.path.join(BASE_DIR, "hand_landmarker.task")

# 3. WebRTC Configuration (The fix for your "sendto" error)
RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302", "stun:stun1.l.google.com:19302"]}]}
)

class TremorProcessor:
    def __init__(self):
        # Initialize MediaPipe inside the processor thread
        base_options = mp_python.BaseOptions(model_asset_path=MODEL_PATH)
        options = mp_vision.HandLandmarkerOptions(
            base_options=base_options,
            running_mode=mp_vision.RunningMode.VIDEO, # Consistent for WebRTC frames
            num_hands=1,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(options)
        self.result_queue = queue.Queue()

    def recv(self, frame):
        img = frame.to_ndarray(format="bgr24")
        
        # Convert to RGB for MediaPipe
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img_rgb)
        
        # Process Frame
        timestamp_ms = int(frame.pts * 1000 / frame.time_base.denominator)
        detection_result = self.landmarker.detect_for_video(mp_image, timestamp_ms)
        
        # Logic for tremor visualization (Drawing landmarks)
        if detection_result.hand_landmarks:
            for landmarks in detection_result.hand_landmarks:
                for landmark in landmarks:
                    x = int(landmark.x * img.shape[1])
                    y = int(landmark.y * img.shape[0])
                    cv2.circle(img, (x, y), 2, (0, 255, 0), -1)

        return frame.from_ndarray(img, format="bgr24")

# 4. Streamlit UI
st.set_page_config(page_title="NeuroScan AI", layout="wide")
st.title("🫨 NeuroScan AI: Tremor Quantification")

col_cam, col_stats = st.columns([2, 1])

with col_cam:
    st.write("Live Analysis Feed")
    webrtc_streamer(
        key="neuroscan",
        mode=WebRtcMode.SENDRECV,
        rtc_configuration=RTC_CONFIG,
        video_processor_factory=TremorProcessor,
        media_stream_constraints={"video": True, "audio": False},
        async_processing=True,
    )

with col_stats:
    st.subheader("Metrics")
    st.info("Frequency and Power data will appear here during live tracking.")
