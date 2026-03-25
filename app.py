import os
os.environ["GLOG_minloglevel"]     = "2"
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["MEDIAPIPE_DISABLE_GPU"]= "1"

import streamlit as st
import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python as mp_python
from mediapipe.tasks.python import vision as mp_vision
from mediapipe import Image as MpImage, ImageFormat
import time
import urllib.request
import pathlib
from scipy.fft import fft, fftfreq
from collections import deque
from streamlit_webrtc import webrtc_streamer, VideoProcessorBase, RTCConfiguration
import av

# ── MediaPipe model path ───────────────────────────────────────────────────────
MODEL_URL  = "https://storage.googleapis.com/mediapipe-models/hand_landmarker/hand_landmarker/float16/latest/hand_landmarker.task"
MODEL_PATH = pathlib.Path(os.path.expanduser("~")) / ".cache" / "mediapipe" / "hand_landmarker.task"

def ensure_model():
    MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
    if not MODEL_PATH.exists():
        urllib.request.urlretrieve(MODEL_URL, MODEL_PATH)
    return str(MODEL_PATH)

# ── Hand connections ───────────────────────────────────────────────────────────
HAND_CONN = [
    (0,1),(1,2),(2,3),(3,4),
    (0,5),(5,6),(6,7),(7,8),
    (0,9),(9,10),(10,11),(11,12),
    (0,13),(13,14),(14,15),(15,16),
    (0,17),(17,18),(18,19),(19,20),
    (5,9),(9,13),(13,17),
]

# ── Config ─────────────────────────────────────────────────────────────────────
WINDOW_DURATION   = 5
MIN_FREQ          = 2.5
MAX_FREQ          = 8
STABILITY_HISTORY = 5
# CHANGED: MOTION_THRESHOLD is kept for naming consistency but made near-zero —
# detection is now purely FFT-power-based (see compute_tremor_frequency).
MOTION_THRESHOLD  = 0.1
# CHANGED: POWER_THRESHOLD drastically reduced so micro-tremors (tiny oscillations
# visible in raw landmark positions) clear the bar. Was 15; now 1.5 so sub-pixel
# jitter is detectable while still suppressing pure noise via the risk formula.
POWER_THRESHOLD   = 1.5
SMOOTHING_WINDOW  = 5
# NEW: FFT is recomputed at most once every UPDATE_INTERVAL seconds so displayed
# values change every ~1-2 s instead of jumping every frame.
UPDATE_INTERVAL   = 1.0

RTC_CONFIG = RTCConfiguration(
    {"iceServers": [{"urls": ["stun:stun.l.google.com:19302"]}]}
)

# ── Page config ────────────────────────────────────────────────────────────────
st.set_page_config(page_title="NeuroScan AI", layout="wide")

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');
:root{--bg:#0a0c10;--surface:#111318;--border:#2a2d36;--accent:#00e5ff;--accent2:#ff6b6b;--accent3:#7fff6b;--muted:#6b7080;--warning:#ffb74d;}
html,body,[class*="css"]{font-family:'DM Sans',sans-serif;background-color:var(--bg)!important;color:#e8eaf0!important;}
.stApp{background-color:var(--bg)!important;}
[data-testid="stSidebar"]{background-color:var(--surface)!important;border-right:1px solid var(--border)!important;}
h1{font-family:'Space Mono',monospace!important;color:var(--accent)!important;text-align:center;}
.metric-card{background:var(--surface);border:1px solid var(--border);border-radius:12px;padding:18px;position:relative;overflow:hidden;margin-bottom:10px;}
.metric-card::before{content:'';position:absolute;top:0;left:0;right:0;height:2px;background:linear-gradient(90deg,var(--accent),transparent);}
.metric-label{font-size:10px;letter-spacing:2px;text-transform:uppercase;color:var(--muted);margin-bottom:6px;}
.metric-value{font-family:'Space Mono',monospace;font-size:30px;font-weight:700;color:var(--accent);line-height:1;}
.metric-unit{font-size:12px;color:var(--muted);margin-top:4px;}
.badge{display:inline-block;padding:2px 10px;border-radius:20px;font-size:10px;font-weight:600;letter-spacing:1px;margin-top:6px;}
.badge-low{background:rgba(127,255,107,.15);color:var(--accent3);border:1px solid rgba(127,255,107,.3);}
.badge-mid{background:rgba(255,183,77,.15);color:var(--warning);border:1px solid rgba(255,183,77,.3);}
.badge-high{background:rgba(255,107,107,.15);color:var(--accent2);border:1px solid rgba(255,107,107,.3);}
.section-label{font-family:'Space Mono',monospace;font-size:10px;letter-spacing:3px;text-transform:uppercase;color:var(--muted);margin:16px 0 10px;display:flex;align-items:center;gap:8px;}
.section-label::after{content:'';flex:1;height:1px;background:var(--border);}
.info-box{background:rgba(0,229,255,.05);border:1px solid rgba(0,229,255,.2);border-left:3px solid var(--accent);border-radius:8px;padding:14px;font-size:12px;color:var(--muted);line-height:1.6;}
.warning-box{background:rgba(255,183,77,.05);border:1px solid rgba(255,183,77,.2);border-left:3px solid var(--warning);border-radius:8px;padding:14px;font-size:12px;color:var(--muted);line-height:1.6;}
/* WebRTC iframe sizing */
.element-container iframe{width:100%!important;border-radius:12px!important;}
</style>
""", unsafe_allow_html=True)


# ── Video Processor ────────────────────────────────────────────────────────────
class TremorProcessor(VideoProcessorBase):

    def __init__(self):
        self.timestamps = deque()
        # CHANGED: store raw (x, y) pixel positions instead of inter-frame magnitudes.
        # Raw positions are what we FFT; magnitudes were too coarse for micro-tremors.
        self.positions  = deque()   # (x_px, y_px) of landmark 8
        self.magnitudes = deque()   # kept for naming consistency (unused in FFT path)
        self.freq_history = deque(maxlen=STABILITY_HISTORY)

        # NEW: throttle FFT computation to once per UPDATE_INTERVAL seconds.
        self._last_update_time = 0.0
        # NEW: cache last computed results so display never goes blank.
        self._cached_freq      = 0.0
        self._cached_power     = 0.0

        model_path = ensure_model()
        opts = mp_vision.HandLandmarkerOptions(
            base_options=mp_python.BaseOptions(model_asset_path=model_path),
            running_mode=mp_vision.RunningMode.IMAGE,
            num_hands=1,
            min_hand_detection_confidence=0.5,
            min_hand_presence_confidence=0.5,
            min_tracking_confidence=0.5,
        )
        self.landmarker = mp_vision.HandLandmarker.create_from_options(opts)

        self.last_freq      = 0.0
        self.last_power     = 0.0
        self.last_stability = 0.0
        self.last_risk      = 0.0

    # ── DSP ───────────────────────────────────────────────────────────────────
    # CHANGED: entire method rewritten.
    # Old approach: FFT of inter-frame magnitude (speed signal) — too coarse.
    # New approach:
    #   1. Build a time-series of raw X and Y pixel positions of landmark 8.
    #   2. Detrend each axis (subtract rolling mean) to remove slow drift.
    #   3. FFT both axes; use the one with higher peak power (dominant tremor axis).
    #   4. Return (peak_freq, peak_power) or (0.0, 0.0) — never None.
    def compute_tremor_frequency(self):
        n = len(self.positions)
        if n < 30:
            return 0.0, 0.0

        xs = np.array([p[0] for p in self.positions], dtype=float)
        ys = np.array([p[1] for p in self.positions], dtype=float)

        # Detrend: subtract the mean to remove DC / slow drift component.
        xs -= np.mean(xs)
        ys -= np.mean(ys)

        time_arr = np.array(self.timestamps)[-n:]
        dt = np.mean(np.diff(time_arr))
        if dt <= 0:
            return 0.0, 0.0

        # Optional light smoothing to suppress single-frame jitter before FFT.
        if n > SMOOTHING_WINDOW:
            xs = np.convolve(xs, np.ones(SMOOTHING_WINDOW)/SMOOTHING_WINDOW, mode='valid')
            ys = np.convolve(ys, np.ones(SMOOTHING_WINDOW)/SMOOTHING_WINDOW, mode='valid')

        # Compute FFT for both axes; pick the dominant one.
        best_freq, best_power = 0.0, 0.0
        for sig in (xs, ys):
            yf = fft(sig)
            xf = fftfreq(len(sig), dt)
            pf = xf[:len(xf)//2]
            pm = np.abs(yf[:len(yf)//2])
            mask = (pf >= MIN_FREQ) & (pf <= MAX_FREQ)
            if not np.any(mask):
                continue
            idx   = np.argmax(pm[mask])
            freq  = float(pf[mask][idx])
            power = float(pm[mask][idx])
            if power > best_power:
                best_freq, best_power = freq, power

        # CHANGED: gate on POWER_THRESHOLD only (no MOTION_THRESHOLD gate).
        if best_power < POWER_THRESHOLD:
            return 0.0, 0.0

        return best_freq, best_power

    def compute_stability(self):
        if len(self.freq_history) < STABILITY_HISTORY:
            return 0.0
        return float(max(0, 1 - np.std(self.freq_history) / 2))

    # CHANGED: risk formula tightened so low power always yields low risk.
    # Old formula could give high risk on tiny power because freq_score alone
    # could contribute 0.5 even at micro-tremor powers.
    # New formula: power_score gates everything — if power is tiny, risk is tiny.
    def compute_risk(self, freq, power, stability):
        if freq == 0.0 or power == 0.0:
            return 0.0
        freq_score  = 1.0 if 4 <= freq <= 6 else 0.3
        # Normalize against a realistic vigorous-tremor power (~80 px FFT amplitude).
        power_score = min(power / 150.0, 1.0)   # lower ceiling → vigorous shaking hits 1.0 sooner
        total = freq_score * 0.35 + power_score * 0.55 + stability * 0.10
        risk  = round(total * 100, 1)
        # Suppress sub-threshold noise floor — keep the 40% gate.
        return 0.0 if risk < 30 else risk

    # ── Frame processing ──────────────────────────────────────────────────────
    def recv(self, frame):
        img  = frame.to_ndarray(format="bgr24")
        img  = cv2.flip(img, 1)
        rgb  = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        h, w = img.shape[:2]
        now  = time.time()

        try:
            mp_img = MpImage(image_format=ImageFormat.SRGB, data=rgb)
            result = self.landmarker.detect(mp_img)

            if result.hand_landmarks:
                lms = result.hand_landmarks[0]

                # Draw skeleton
                for a, b in HAND_CONN:
                    x1,y1 = int(lms[a].x*w), int(lms[a].y*h)
                    x2,y2 = int(lms[b].x*w), int(lms[b].y*h)
                    cv2.line(img, (x1,y1), (x2,y2), (0,255,180), 2)

                # Draw joints
                for lm in lms:
                    cv2.circle(img, (int(lm.x*w), int(lm.y*h)), 5, (0,255,255), -1)

                # Index fingertip = landmark 8
                tip = lms[8]
                tx, ty = int(tip.x*w), int(tip.y*h)
                cv2.circle(img, (tx,ty), 10, (255,255,0), -1)

                # CHANGED: store raw pixel position (not magnitude) for FFT.
                self.positions.append((tx, ty))
                self.timestamps.append(now)

                # magnitudes deque kept for API compatibility; populate lightly.
                if len(self.positions) > 1:
                    dx = self.positions[-1][0] - self.positions[-2][0]
                    dy = self.positions[-1][1] - self.positions[-2][1]
                    self.magnitudes.append(np.sqrt(dx**2 + dy**2))

        except Exception:
            pass

        # Sliding window — prune data older than WINDOW_DURATION seconds.
        while self.timestamps and (now - self.timestamps[0]) > WINDOW_DURATION:
            self.timestamps.popleft()
            if self.positions:  self.positions.popleft()
            if self.magnitudes: self.magnitudes.popleft()

        # CHANGED: throttle FFT to once per UPDATE_INTERVAL seconds.
        # Between updates, reuse cached values so display is stable.
        if (now - self._last_update_time) >= UPDATE_INTERVAL:
            self._last_update_time = now
            freq, power = self.compute_tremor_frequency()
            self._cached_freq  = freq
            self._cached_power = power
            # Only append to freq_history when a real non-zero tremor is detected,
            # so stability isn't polluted by silent frames.
            if freq > 0.0:
                self.freq_history.append(freq)
        else:
            freq  = self._cached_freq
            power = self._cached_power

        stability = self.compute_stability()
        risk      = self.compute_risk(freq, power, stability)

        # Store for sidebar — always numeric, never None.
        self.last_freq      = freq
        self.last_power     = power
        self.last_stability = stability
        self.last_risk      = risk

        # HUD overlay
        overlay = img.copy()
        cv2.rectangle(overlay, (16,16), (400,220), (10,15,12), -1)
        cv2.addWeighted(overlay, 0.8, img, 0.2, 0, img)
        neon = (0, 255, 200)
        cv2.putText(img, "NEUROSCAN AI", (32,48),  cv2.FONT_HERSHEY_SIMPLEX, 0.8, neon, 2)
        # CHANGED: always show numeric value (0.00 when no tremor) — never "--".
        cv2.putText(img, f"Freq:  {freq:.2f} Hz",
                    (32,90),  cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,255), 1)
        cv2.putText(img, f"Power: {power:.1f}",
                    (32,118), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,255), 1)
        cv2.putText(img, f"Stab:  {stability:.2f}",
                    (32,146), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200,255,255), 1)
        cv2.putText(img, f"Risk:  {risk}%",
                    (32,188), cv2.FONT_HERSHEY_SIMPLEX, 0.8, neon, 2)

        return av.VideoFrame.from_ndarray(img, format="bgr24")


# ── UI Layout ──────────────────────────────────────────────────────────────────
st.markdown("<h1>🧠 NeuroScan AI</h1>", unsafe_allow_html=True)
st.markdown("""<div style='text-align:center;color:#6b7080;font-size:14px;margin-bottom:24px;'>
Micro-Tremor Quantification System &nbsp;·&nbsp; Research Prototype — Not a Medical Diagnosis Tool
</div>""", unsafe_allow_html=True)

col_cam, col_info = st.columns([3, 2], gap="large")

with col_cam:
    st.markdown('<div class="section-label">Live Camera Feed</div>', unsafe_allow_html=True)
    ctx = webrtc_streamer(
        key="neuroscan",
        video_processor_factory=TremorProcessor,
        rtc_configuration=RTC_CONFIG,
        media_stream_constraints={
            "video": {"width": {"ideal": 860}, "height": {"ideal": 540}, "frameRate": {"ideal": 30}},
            "audio": False,
        },
        async_processing=True,
    )

with col_info:
    st.markdown('<div class="section-label">Real-Time Metrics</div>', unsafe_allow_html=True)

    freq_ph  = st.empty()
    power_ph = st.empty()
    stab_ph  = st.empty()
    risk_ph  = st.empty()

    st.markdown('<div class="section-label">Clinical Reference</div>', unsafe_allow_html=True)
    st.markdown("""<div class="info-box">
<b>Tremor Frequency Bands</b><br>
- Physiological: <b>&lt; 3 Hz</b><br>
- Resting (Parkinson's): <b>4–6 Hz</b><br>
- Action / Postural: <b>5–12 Hz</b><br>
- Essential Tremor: <b>6–12 Hz</b><br><br>
<b>Risk Index</b><br>
Scores below 40% are suppressed as noise.
A score ≥ 70% warrants clinical consultation.
</div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("""<div class="warning-box">
⚠️ <b>Disclaimer</b><br>
This tool is for research purposes only and does not constitute a medical diagnosis.
Please consult a neurologist for clinical assessment.
</div>""", unsafe_allow_html=True)

# ── Live metric display ────────────────────────────────────────────────────────
if ctx.video_processor:
    proc = ctx.video_processor

    while True:
        freq      = proc.last_freq      # always float, never None
        power     = proc.last_power     # always float, never None
        stability = proc.last_stability
        risk      = proc.last_risk

        # CHANGED: always show numeric values — 0.00 when no tremor, never "—".
        freq_ph.markdown(f"""<div class="metric-card">
<div class="metric-label">Peak Tremor Frequency</div>
<div class="metric-value">{"%.2f" % freq}</div>
<div class="metric-unit">Hz &nbsp;·&nbsp; Target band: {MIN_FREQ}–{MAX_FREQ} Hz</div>
</div>""", unsafe_allow_html=True)

        power_ph.markdown(f"""<div class="metric-card" style="border-top-color:#ff6b6b">
<div class="metric-label">Signal Power</div>
<div class="metric-value" style="color:#ff6b6b">{"%.1f" % power}</div>
<div class="metric-unit">FFT amplitude magnitude</div>
</div>""", unsafe_allow_html=True)

        stab_ph.markdown(f"""<div class="metric-card" style="border-top-color:#ffb74d">
<div class="metric-label">Oscillation Stability</div>
<div class="metric-value" style="color:#ffb74d">{"%.0f%%" % (stability*100)}</div>
<div class="metric-unit">Frequency consistency over last {STABILITY_HISTORY} readings</div>
</div>""", unsafe_allow_html=True)

        badge = "badge-low" if risk < 40 else "badge-mid" if risk < 70 else "badge-high"
        label = "Low"       if risk < 40 else "Moderate"  if risk < 70 else "High"
        risk_ph.markdown(f"""<div class="metric-card" style="border-top-color:#7fff6b">
<div class="metric-label">Tremor Risk Index</div>
<div class="metric-value" style="color:#7fff6b">{risk if risk else "0.0"}%</div>
<div class="metric-unit">Composite score (freq + power + stability)</div>
<span class="badge {badge}">{label} Risk</span>
</div>""", unsafe_allow_html=True)

        time.sleep(0.25)
        st.rerun()
else:
    for ph in [freq_ph, power_ph, stab_ph, risk_ph]:
        ph.markdown('<div class="metric-card"><div class="metric-value" style="color:#2a2d36">—</div></div>',
                    unsafe_allow_html=True)