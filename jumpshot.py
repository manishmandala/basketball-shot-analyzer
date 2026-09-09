# ============================================================
# BASKETBALL SHOOTING FORM ANALYZER
# ============================================================

import time

import cv2
import mediapipe as mp
import numpy as np

# --- SETUP ---
mp_pose = mp.solutions.pose
mp_draw = mp.solutions.drawing_utils
pose = mp_pose.Pose(min_detection_confidence=0.7,
                    min_tracking_confidence=0.7)

# --- ANGLE CALCULATOR ---
# Takes 3 points, returns the angle at the middle point
def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    radians = np.arctan2(c[1]-b[1], c[0]-b[0]) - \
              np.arctan2(a[1]-b[1], a[0]-b[0])
    angle = np.abs(radians * 180.0 / np.pi)
    if angle > 180.0:
        angle = 360 - angle
    return angle

# --- METRIC DISPLAY ---
# Draws a labeled metric on screen in green (good) or red (bad)
def draw_metric(frame, label, value, pos, good):
    color = (0, 255, 0) if good else (0, 0, 255)
    text = f'{label}: {value}'
    (text_w, text_h), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.45, 1)
    cv2.rectangle(frame,
                  (pos[0], pos[1] - text_h - 8),
                  (pos[0] + text_w + 10, pos[1] + 6),
                  (0, 0, 0), -1)
    cv2.putText(frame, text,
                pos, cv2.FONT_HERSHEY_SIMPLEX,
                0.45, color, 1)

# --- FORM ANALYZER ---
# Extracts landmarks, calculates angles, returns metrics dict
def analyze_shooting_form(landmarks, w, h, side='right'):
    
    if side == 'right':
        shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
        elbow    = [landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ELBOW.value].y * h]
        wrist    = [landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_WRIST.value].y * h]
        hip      = [landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_HIP.value].y * h]
        knee     = [landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_KNEE.value].y * h]
        ankle    = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                    landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]
    else:
        shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
        elbow    = [landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_ELBOW.value].y * h]
        wrist    = [landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_WRIST.value].y * h]
        hip      = [landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_HIP.value].y * h]
        knee     = [landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_KNEE.value].y * h]
        ankle    = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                    landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]

    # Both sides for stance width
    l_shoulder = [landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].x * w,
                  landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER.value].y * h]
    r_shoulder = [landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].x * w,
                  landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER.value].y * h]
    l_ankle    = [landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].x * w,
                  landmarks[mp_pose.PoseLandmark.LEFT_ANKLE.value].y * h]
    r_ankle    = [landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].x * w,
                  landmarks[mp_pose.PoseLandmark.RIGHT_ANKLE.value].y * h]

    # Calculate angles
    elbow_angle    = calculate_angle(shoulder, elbow, wrist)
    knee_angle     = calculate_angle(hip, knee, ankle)
    shoulder_angle = calculate_angle([elbow[0], hip[1]], shoulder, elbow)

    # Stance width ratio
    shoulder_width = abs(r_shoulder[0] - l_shoulder[0])
    ankle_width    = abs(r_ankle[0] - l_ankle[0])
    stance_ratio   = round(ankle_width / shoulder_width, 2) if shoulder_width > 0 else 0

    # Wrist above elbow = good release position
    wrist_above_elbow = wrist[1] < elbow[1]

    return {
        'elbow_angle':    int(elbow_angle),
        'elbow_good':     80 <= elbow_angle <= 110,
        'knee_angle':     int(knee_angle),
        'knee_good':      knee_angle >= 140,
        'shoulder_angle': int(shoulder_angle),
        'shoulder_good':  shoulder_angle >= 70,
        'stance_ratio':   stance_ratio,
        'stance_good':    0.8 <= stance_ratio <= 1.4,
        'wrist_good':     wrist_above_elbow,
        'wrist_status':   'HIGH' if wrist_above_elbow else 'LOW',
        'score': int(sum([
            80 <= elbow_angle <= 110,
            knee_angle >= 140,
            shoulder_angle >= 70,
            0.8 <= stance_ratio <= 1.4,
            wrist_above_elbow
        ]) / 5 * 100)
    }

# --- MAIN LOOP ---
cap = cv2.VideoCapture(0)
if not cap.isOpened():
    raise SystemExit('Could not open webcam (index 0). Check it is connected and not in use by another app.')
SHOOTING_SIDE = 'right'    # change to 'left' if needed

# --- FPS TRACKING ---
# Measures actual end-to-end pipeline speed (capture + pose estimation +
# metric calculation + rendering) so real-time performance can be
# reported with a real number instead of a guess.
prev_frame_time = time.time()
fps_history = []
WARMUP_FRAMES = 30  # skip the first second or so while the camera/model settle

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    current_frame_time = time.time()
    elapsed = current_frame_time - prev_frame_time
    fps = 1.0 / elapsed if elapsed > 0 else 0.0
    prev_frame_time = current_frame_time
    fps_history.append(fps)

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb)

    if results.pose_landmarks:
        mp_draw.draw_landmarks(frame, results.pose_landmarks,
                               mp_pose.POSE_CONNECTIONS)

        metrics = analyze_shooting_form(
            results.pose_landmarks.landmark, w, h, SHOOTING_SIDE
        )

        draw_metric(frame, 'Elbow angle',
                    f"{metrics['elbow_angle']}deg (want 80-110)",
                    (10, 50), metrics['elbow_good'])
        draw_metric(frame, 'Knee bend',
                    f"{metrics['knee_angle']}deg (want 140+)",
                    (10, 100), metrics['knee_good'])
        draw_metric(frame, 'Elbow height',
                    f"{metrics['shoulder_angle']}deg (want 70+)",
                    (10, 150), metrics['shoulder_good'])
        draw_metric(frame, 'Stance width',
                    f"ratio {metrics['stance_ratio']} (want 0.8-1.4)",
                    (10, 200), metrics['stance_good'])
        draw_metric(frame, 'Wrist position',
                    metrics['wrist_status'],
                    (10, 250), metrics['wrist_good'])

        # Score display
        score = metrics['score']
        score_color = (0, 255, 0) if score >= 80 else \
                      (0, 165, 255) if score >= 60 else \
                      (0, 0, 255)
        score_text = f'SCORE: {score}'
        (text_w, text_h), _ = cv2.getTextSize(score_text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        margin = 10
        box_left = w - text_w - margin * 3
        cv2.rectangle(frame, (box_left, 10), (w - margin, 20 + text_h + margin), (0, 0, 0), -1)
        cv2.putText(frame, score_text,
                    (box_left + margin, 20 + text_h),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, score_color, 2)

    else:
        cv2.putText(frame, 'Stand back - full body needed',
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    0.55, (255, 255, 0), 1)

    cv2.putText(frame, f'FPS: {fps:.1f}',
                (10, h-15), cv2.FONT_HERSHEY_SIMPLEX,
                0.45, (255, 255, 255), 1)

    cv2.imshow('Basketball Form Analyzer', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        cap.release()
        cv2.destroyAllWindows()
        stable_fps = fps_history[WARMUP_FRAMES:] if len(fps_history) > WARMUP_FRAMES else fps_history
        if stable_fps:
            avg_fps = sum(stable_fps) / len(stable_fps)
            print(f'Average FPS over {len(stable_fps)} frames: {avg_fps:.1f}')
        quit()