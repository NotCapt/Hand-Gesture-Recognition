import cv2
import numpy as np

# Configuration Constants
IMG_SIZE = 128
HSV_LOWER = np.array([0, 20, 70], dtype=np.uint8)
HSV_UPPER = np.array([20, 255, 255], dtype=np.uint8)

def get_roi_coords(h, w):
    # Fetches the coordinates of ROI(region of interest)
    y1, y2 = int(0.3 * h), int(0.7 * h)
    x1, x2 = int(0.3 * w), int(0.7 * w)
    return x1, y1, x2, y2

def process_frame(frame):
    """Basically takes a raw frame, extracts the ROI, detects skin, 
    and returns the cleaned mask and ROI coords for drawing."""
    
    h, w, _ = frame.shape
    x1, y1, x2, y2 = get_roi_coords(h, w)
    
    # Extract ROI
    roi = frame[y1:y2, x1:x2]
    
    # Convert to HSV and Mask
    hsv = cv2.cvtColor(roi, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, HSV_LOWER, HSV_UPPER)
    
    # Morphological Operations (Cleaning)
    kernel = np.ones((3, 3), np.uint8)
    mask = cv2.dilate(mask, kernel, iterations=4)
    mask = cv2.GaussianBlur(mask, (5, 5), 100)
    
    # Resize to target size for Model/Storage
    mask_resized = cv2.resize(mask, (IMG_SIZE, IMG_SIZE))
    
    return mask_resized, (x1, y1, x2, y2)