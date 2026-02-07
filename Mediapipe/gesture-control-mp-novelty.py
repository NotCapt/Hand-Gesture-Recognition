import cv2
import mediapipe as mp
import time
from collections import deque
import os

cap = cv2.VideoCapture(0)

mp_hands = mp.solutions.hands
mp_draw = mp.solutions.drawing_utils

hands = mp_hands.Hands(
    max_num_hands=2, 
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)


gesture_buffer = deque(maxlen=5)
last_command_time = 0
last_selfie_time = 0

COOLDOWN = 1.5
SELFIE_COOLDOWN = 3  

# Helper functions
def get_hand_bbox(landmarks, w, h):
    xs = [lm.x * w for lm in landmarks]
    ys = [lm.y * h for lm in landmarks]
    return min(xs), min(ys), max(xs), max(ys)

def detect_environment(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 80, 160)
    edge_density = edges.mean()

    # Color variance 
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    saturation_mean = hsv[:, :, 1].mean()

    # Brightness
    brightness_mean = gray.mean()

    # Decision logic 
    if edge_density > 20 or saturation_mean < 40:
        return "INDOOR"
    else:
        return "OUTDOOR"

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    h, w, _ = frame.shape

    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(rgb)

    gesture = "NONE"
    environment = detect_environment(frame)

    if result.multi_hand_landmarks:

        # Single hand operations
        if len(result.multi_hand_landmarks) == 1:
            hand_landmarks = result.multi_hand_landmarks[0]
            mp_draw.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

            landmarks = hand_landmarks.landmark
            wrist = landmarks[0]
            middle_tip = landmarks[12]

            dx = middle_tip.x - wrist.x
            dy = middle_tip.y - wrist.y

            if abs(dx) > abs(dy):
                gesture = "RIGHT" if dx > 0 else "LEFT"
            else:
                gesture = "UP" if dy < 0 else "DOWN"

            gesture_buffer.append(gesture)

        # Multi hand operations
        elif len(result.multi_hand_landmarks) == 2:
            hand1 = result.multi_hand_landmarks[0].landmark
            hand2 = result.multi_hand_landmarks[1].landmark

            mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[0], mp_hands.HAND_CONNECTIONS)
            mp_draw.draw_landmarks(frame, result.multi_hand_landmarks[1], mp_hands.HAND_CONNECTIONS)

            x1_min, y1_min, x1_max, y1_max = get_hand_bbox(hand1, w, h)
            x2_min, y2_min, x2_max, y2_max = get_hand_bbox(hand2, w, h)

            box_width = abs(x2_min - x1_max)
            box_height = abs(y2_max - y1_min)

            if box_width > 150 and box_height > 150:
                if time.time() - last_selfie_time > SELFIE_COOLDOWN:
                    if not os.path.exists("selfies"):
                        os.makedirs("selfies")
                    filename = f"selfies/selfie_{int(time.time())}.jpg"
                    cv2.imwrite(filename, frame)
                    print(" SELFIE CAPTURED:", filename)
                    last_selfie_time = time.time()

    # Stability check + output
    if len(gesture_buffer) == gesture_buffer.maxlen:
        if gesture_buffer.count(gesture_buffer[0]) == gesture_buffer.maxlen:
            stable_gesture = gesture_buffer[0]
            current_time = time.time()

            if current_time - last_command_time > COOLDOWN:
                if stable_gesture == "UP":
                    speed = "SLOW" if environment == "INDOOR" else "FAST"
                    print(f"COMMAND: UP | SPEED: {speed} | ENV: {environment}")
                else:
                    print("COMMAND:", stable_gesture)

                last_command_time = current_time

    # Display
    cv2.putText(frame, f"Gesture: {gesture}", (20, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.putText(frame, f"Environment: {environment}", (20, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)

    cv2.imshow("Gesture Control with Novelty", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
