import cv2
import numpy as np
from tensorflow import keras
import sys



# Configuration
MODEL_PATH = "gesture_model_best.h5"  
IMG_SIZE = 128
CLASS_NAMES = ["down", "left", "right", "up"]

# Load the trained model
model = keras.models.load_model(MODEL_PATH)

print(f"Model input shape: {model.input_shape}")
print(f"Model output shape: {model.output_shape}")


# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print(" Could not open webcam")
    sys.exit(1)

# Colors for each gesture 
colors = {
    "down": (0, 0, 255),      # Red
    "up": (0, 255, 0),        # Green
    "left": (255, 0, 0),      # Blue
    "right": (0, 255, 255)    # Yellow
}

frame_count = 0
smoothing_buffer = []
SMOOTHING_SIZE = 5

while True:
    ret, frame = cap.read()
    if not ret:
        print("✗ Failed to grab frame")
        break
    
    # Flip frame 
    frame = cv2.flip(frame, 1)
    
    h, w, _ = frame.shape
    
    # Define ROI 
    roi_y1, roi_y2 = int(0.3 * h), int(0.7 * h)
    roi_x1, roi_x2 = int(0.3 * w), int(0.7 * w)
    
    # Draw rectangle for ROI
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 3)
    
    # Extract ROI
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Convert ROI to HSV for skin detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV 
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations 
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    
    # Resize mask to model input size
    mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    
    # Normalize to [0, 1] 
    mask_normalized = mask_resized / 255.0
    
    # Reshape for model input: (1, 128, 128, 1)
    mask_input = mask_normalized.reshape(1, IMG_SIZE, IMG_SIZE, 1)
    
    # Predict
    predictions = model.predict(mask_input, verbose=0)[0]
    pred_idx = np.argmax(predictions)
    pred_label = CLASS_NAMES[pred_idx]
    confidence = predictions[pred_idx]
    
    # Smoothing with buffer
    smoothing_buffer.append(pred_idx)
    if len(smoothing_buffer) > SMOOTHING_SIZE:
        smoothing_buffer.pop(0)
    
    # Get most common prediction
    if len(smoothing_buffer) >= 3:
        smoothed_idx = max(set(smoothing_buffer), key=smoothing_buffer.count)
        smoothed_label = CLASS_NAMES[smoothed_idx]
        smoothed_confidence = predictions[smoothed_idx]
    else:
        smoothed_label = pred_label
        smoothed_confidence = confidence
    
    # Print predictions 
    frame_count += 1
    if frame_count % 15 == 0:
        pred_str = " | ".join([f"{CLASS_NAMES[i]}: {predictions[i]*100:.1f}%" 
                               for i in range(len(CLASS_NAMES))])
        print(f"[Frame {frame_count:4d}] {pred_str} => {pred_label}")
    
    # Display prediction on frame
    color = colors.get(smoothed_label, (255, 255, 255))
    
    # Main gesture text
    text = smoothed_label.upper()
    conf_text = f"{smoothed_confidence*100:.1f}%"
    
    cv2.putText(frame, text, (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 
                2, color, 4)
    cv2.putText(frame, conf_text, (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 
                1, color, 2)
    

    # Show frames
    cv2.imshow("Gesture Recognition", frame)
    
    # Show processed mask
    mask_display = cv2.resize(mask, (300, 300))
    cv2.imshow("Processed Mask (Model Input)", mask_display)
    
    # Quit on 'q'
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        print("\nQuitting")
        break

cap.release()
cv2.destroyAllWindows()

