# ============================================================
# DRIVER MONITORING SYSTEM
# Detects drowsiness (closed eyes) and yawning using webcam
# Press 'Q' to quit the program
# ============================================================

# ---------- Step 1: Import Libraries ----------
import cv2                          # OpenCV - for webcam and image processing
import mediapipe as mp              # MediaPipe - for detecting face landmarks
import numpy as np                  # NumPy - for math calculations
import winsound                     # WinSound - for alert beep (Windows only)

# ---------- Step 2: Setup Face Detection ----------
face_mesh = mp.solutions.face_mesh.FaceMesh(
    max_num_faces=1,                # detect only 1 face (the driver)
    min_detection_confidence=0.5,   # 50% confidence needed to detect face
    min_tracking_confidence=0.5     # 50% confidence needed to track face
)

# ---------- Step 3: Define Eye and Mouth Landmark Points ----------
# These are specific point numbers on the face mesh (468 points total)
# Each eye has 6 key points used to calculate if eye is open or closed
RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # right eye corner points
LEFT_EYE  = [362, 385, 387, 263, 373, 380]  # left eye corner points

# ---------- Step 4: Set Thresholds ----------
EAR_THRESHOLD   = 0.20   # if eye ratio goes below this → eyes are closed
MAR_THRESHOLD   = 0.60   # if mouth ratio goes above this → mouth is open (yawn)
DROWSY_FRAMES   = 15     # eyes must be closed for 15 frames to trigger alert
closed_counter  = 0       # counts how many frames eyes stayed closed

# ---------- Step 5: Helper Function - Eye Aspect Ratio (EAR) ----------
def get_ear(landmarks, eye_points, w, h):
    """Calculate how open/closed an eye is using 6 landmark points."""
    # Convert normalized landmarks to pixel coordinates
    pts = []
    for i in eye_points:
        x = int(landmarks[i].x * w)   # x position in pixels
        y = int(landmarks[i].y * h)   # y position in pixels
        pts.append(np.array([x, y]))

    # Vertical eye distances (top-to-bottom of eye)
    vertical_1 = np.linalg.norm(pts[1] - pts[5])   # distance point 1 to 5
    vertical_2 = np.linalg.norm(pts[2] - pts[4])   # distance point 2 to 4
    # Horizontal eye distance (corner-to-corner of eye)
    horizontal = np.linalg.norm(pts[0] - pts[3])   # distance point 0 to 3

    # EAR formula: average of vertical distances / horizontal distance
    ear = (vertical_1 + vertical_2) / (2.0 * horizontal)
    return ear

# ---------- Step 6: Helper Function - Mouth Aspect Ratio (MAR) ----------
def get_mar(landmarks, w, h):
    """Calculate how open/closed the mouth is."""
    # Get 4 mouth points: top lip, bottom lip, left corner, right corner
    top    = np.array([int(landmarks[13].x * w),  int(landmarks[13].y * h)])
    bottom = np.array([int(landmarks[14].x * w),  int(landmarks[14].y * h)])
    left   = np.array([int(landmarks[78].x * w),  int(landmarks[78].y * h)])
    right  = np.array([int(landmarks[308].x * w), int(landmarks[308].y * h)])

    # MAR = mouth height / mouth width
    mar = np.linalg.norm(top - bottom) / np.linalg.norm(left - right)
    return mar

# ---------- Step 7: Start Webcam ----------
cam = cv2.VideoCapture(0)           # open default webcam (camera index 0)

print("Driver Monitoring System Started! Press 'Q' to quit.")

# ---------- Step 8: Main Loop - Process Each Frame ----------
while cam.isOpened():
    success, frame = cam.read()      # read one frame from webcam
    if not success:                  # if frame not captured, skip
        break

    frame = cv2.flip(frame, 1)       # flip horizontally (mirror effect)
    h, w, _ = frame.shape            # get frame height and width
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # convert BGR to RGB for MediaPipe

    # Run face mesh detection on the frame
    result = face_mesh.process(rgb)

    # Default status
    status = "ACTIVE - Driver is Alert"
    color  = (0, 200, 0)            # green color

    # ---------- Step 9: If Face Detected, Analyze It ----------
    if result.multi_face_landmarks:
        face = result.multi_face_landmarks[0].landmark  # get first face's landmarks

        # Calculate EAR for both eyes and average them
        left_ear  = get_ear(face, LEFT_EYE, w, h)
        right_ear = get_ear(face, RIGHT_EYE, w, h)
        ear = (left_ear + right_ear) / 2.0

        # Calculate MAR for mouth
        mar = get_mar(face, w, h)

        # --- Check Drowsiness ---
        if ear < EAR_THRESHOLD:              # eyes are closed
            closed_counter += 1              # increment counter
            if closed_counter >= DROWSY_FRAMES:  # closed too long
                status = "⚠ DROWSY! WAKE UP!"
                color  = (0, 0, 255)         # red color
                winsound.Beep(1000, 200)     # beep at 1000Hz for 200ms
        else:
            closed_counter = 0               # reset counter if eyes open

        # --- Check Yawning ---
        if mar > MAR_THRESHOLD:              # mouth is wide open
            status = "⚠ YAWNING DETECTED!"
            color  = (0, 165, 255)           # orange color

        # --- Draw Eye Points on Frame ---
        for idx in LEFT_EYE + RIGHT_EYE:
            cx = int(face[idx].x * w)
            cy = int(face[idx].y * h)
            cv2.circle(frame, (cx, cy), 2, (0, 255, 255), -1)  # yellow dots

        # --- Show EAR and MAR Values ---
        cv2.putText(frame, f"EAR: {ear:.2f}", (10, 70),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(frame, f"MAR: {mar:.2f}", (10, 100),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        status = "NO FACE DETECTED"
        color  = (0, 0, 255)                 # red color

    # ---------- Step 10: Display Status on Screen ----------
    cv2.putText(frame, status, (10, 35),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, color, 2)
    cv2.rectangle(frame, (0, 0), (w - 1, h - 1), color, 3)  # border
    cv2.imshow("Driver Monitoring System", frame)

    # ---------- Step 11: Check for Quit Key ----------
    if cv2.waitKey(1) & 0xFF == ord('q'):    # press Q to quit
        break

# ---------- Step 12: Cleanup ----------
cam.release()                        # release the webcam
cv2.destroyAllWindows()              # close all OpenCV windows
print("System stopped.")
