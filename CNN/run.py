import cv2
import numpy as np
from tensorflow.keras.models import load_model
from utils import process_frame, IMG_SIZE

# Config
MODEL_PATH = "gesture_model_best.h5"
CLASSES = ["down", "left", "right", "up"] 


model = load_model(MODEL_PATH)
cap = cv2.VideoCapture(0)

buffer = []
BUFFER_SIZE = 5

while True:
    ret, frame = cap.read()
    if not ret: break
    frame = cv2.flip(frame, 1)

    # 1. Process Frame (Shared Logic)
    mask, (x1, y1, x2, y2) = process_frame(frame)

    # 2. Prepare for Model
    # Normalize and reshape: (1, 128, 128, 1)
    img_input = (mask.astype(np.float32) / 255.0).reshape(1, IMG_SIZE, IMG_SIZE, 1)

    # 3. Predict
    preds = model.predict(img_input, verbose=0)[0]
    idx = np.argmax(preds)
    conf = preds[idx]

    # 4. Smooth Predictions
    buffer.append(idx)
    if len(buffer) > BUFFER_SIZE: buffer.pop(0)
    
    # Take the most frequent prediction in the buffer
    smooth_idx = max(set(buffer), key=buffer.count)
    label = CLASSES[smooth_idx].upper()

    # 5. Display
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"{label} ({conf:.2f})", (x1, y1 - 10), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

    cv2.imshow("Gesture", frame)
    cv2.imshow("Model Input", mask)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()