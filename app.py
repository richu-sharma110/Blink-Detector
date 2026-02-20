import streamlit as st
import cv2
import mediapipe as mp
import numpy as np
import time
import winsound

# ================== Streamlit UI ==================
st.set_page_config(page_title="Blink Detector", layout="wide")

# Dark Theme CSS
st.markdown("""
<style>
body {
    background-color: #0e1117;
    color: white;
}
.stButton>button {
    background-color: #1f77b4;
    color: white;
    border-radius: 8px;
    padding: 10px 20px;
}
</style>
""", unsafe_allow_html=True)

st.title("👁️ Blink Detector ")
st.write("Blink Counter + Sleep Alert System")

start = st.button("▶ Start Camera")
stop = st.button("⏹ Stop Camera")

night_mode = st.toggle("🌙 Night Mode (Low-light enhancement)")

frame_placeholder = st.empty()

col1, col2, col3 = st.columns(3)
blink_text = col1.metric("Total Blinks", 0)
eye_state_text = col2.metric("Eye State", "Open")
timer_text = col3.metric("Eye Close Time", "0 sec")

# ================== MediaPipe Setup ==================
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Eye landmark indexes (MediaPipe)
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]

def eye_aspect_ratio(eye):
    A = np.linalg.norm(eye[1] - eye[5])
    B = np.linalg.norm(eye[2] - eye[4])
    C = np.linalg.norm(eye[0] - eye[3])
    ear = (A + B) / (2.0 * C)
    return ear

EAR_THRESHOLD = 0.1
CLOSE_TIME_THRESHOLD = 2.80   # seconds

# ================== ADD-ON: Night Mode Enhancement ==================
def enhance_low_light(frame):
    lab = cv2.cvtColor(frame, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    enhanced = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return enhanced

# ================== State Variables ==================
blink_count = 0
eye_closed_start = None
is_eye_closed = False

# ================== Camera Loop ==================
if start:
    cap = cv2.VideoCapture(0)

    while cap.isOpened():
        if stop:
            break

        ret, frame = cap.read()
        if not ret:
            st.error("Camera not working")
            break

        frame = cv2.flip(frame, 1)

        if night_mode:
            frame = enhance_low_light(frame)

        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        eye_state = "Open"
        close_time = 0

        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                h, w, _ = frame.shape

                left_eye = []
                right_eye = []

                for i in LEFT_EYE:
                    x = int(face_landmarks.landmark[i].x * w)
                    y = int(face_landmarks.landmark[i].y * h)
                    left_eye.append([x, y])

                for i in RIGHT_EYE:
                    x = int(face_landmarks.landmark[i].x * w)
                    y = int(face_landmarks.landmark[i].y * h)
                    right_eye.append([x, y])

                left_eye = np.array(left_eye)
                right_eye = np.array(right_eye)

                left_ear = eye_aspect_ratio(left_eye)
                right_ear = eye_aspect_ratio(right_eye)
                ear = (left_ear + right_ear) / 2

                if ear < EAR_THRESHOLD:
                    eye_state = "Closed"

                    if not is_eye_closed:
                        is_eye_closed = True
                        eye_closed_start = time.time()

                    close_time = int(time.time() - eye_closed_start)

                    if close_time >= CLOSE_TIME_THRESHOLD:
                        winsound.Beep(1000, 500)

                else:
                    if is_eye_closed:
                        blink_count += 1

                    is_eye_closed = False
                    eye_closed_start = None
                    eye_state = "Open"

                cv2.putText(frame, f"Blinks: {blink_count}", (30, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

                cv2.putText(frame, f"Eye: {eye_state}", (30, 90),
                            cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

        frame_placeholder.image(frame, channels="BGR", use_container_width=True)

        blink_text.metric("Total Blinks", blink_count)
        eye_state_text.metric("Eye State", eye_state)
        timer_text.metric("Eye Close Time", f"{close_time} sec")

    cap.release()
    cv2.destroyAllWindows()