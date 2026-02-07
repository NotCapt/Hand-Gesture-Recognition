import cv2
import numpy as np
import os
import sys

# ex - python generate_data.py dataset/train/0 1000

# Get command line arguments
if len(sys.argv) != 3:
    print("Usage: python generate_data.py <directory> <num_images>")
    print("Example: python generate_data.py dataset/train/0 1000")
    sys.exit(1)

directory = sys.argv[1]
num_images = int(sys.argv[2])

# Create directory if it doesn't exist
os.makedirs(directory, exist_ok=True)

# Image dimensions
IMG_SIZE = 128

# Start video capture
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam")
    sys.exit(1)

print(f"Saving to: {directory}")
print(f"Number of images to capture: {num_images}")

count = 0

while True:
    ret, frame = cap.read()
    if not ret:
        print("Failed to grab frame")
        break
    
    # Flipping frame
    frame = cv2.flip(frame, 1)
    
    h, w, _ = frame.shape
    
    # Define ROI
    roi_y1, roi_y2 = int(0.3 * h), int(0.7 * h)
    roi_x1, roi_x2 = int(0.3 * w), int(0.7 * w)
    
    # Draw rectangle for ROI
    cv2.rectangle(frame, (roi_x1, roi_y1), (roi_x2, roi_y2), (0, 255, 0), 2)
    
    # Extract ROI
    roi = frame[roi_y1:roi_y2, roi_x1:roi_x2]
    
    # Convert ROI to HSV for skin detection
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    
    # Define skin color range in HSV
    # These values work well for most skin tones
    lower_skin = np.array([0, 20, 70], dtype=np.uint8)
    upper_skin = np.array([20, 255, 255], dtype=np.uint8)
    
    # Create mask for skin color
    mask = cv2.inRange(hsv, lower_skin, upper_skin)
    
    # Apply morphological operations to clean up the mask
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    
    # Resize mask to target size
    mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    
    # Display info on frame
    info_text = f"Captured: {count}/{num_images}"
    cv2.putText(frame, info_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 
                1, (0, 255, 0), 2)
    cv2.putText(frame, "Press SPACE to capture", (20, h - 20), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    
    # Show frames
    cv2.imshow("Data Collection - Original", frame)
    cv2.imshow("Data Collection - Processed Mask", mask)
    
    key = cv2.waitKey(1) & 0xFF
    
    # Capture image on SPACE key
    if key == ord(' ') and count < num_images:
        # Save the mask
        filename = os.path.join(directory, f"gesture_{count}.jpg")
        cv2.imwrite(filename, mask_resized)
        count += 1
        print(f"Saved: {filename} ({count}/{num_images})")
        
        if count >= num_images:
            print(f"\n Successfully captured {num_images} images!")
            print(f"  Saved to: {directory}")
            break
    
    # Quit on 'q'
    elif key == ord('q'):
        print(f"\nQuitting... Captured {count}/{num_images} images")
        break

cap.release()
cv2.destroyAllWindows()

