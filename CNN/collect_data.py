import cv2
import os
import sys
from utils import process_frame  

if len(sys.argv) != 3:
    print("Usage: python collect_data.py <dir_path> <num_images>")
    sys.exit(1)
# example - python collect_data.py dataset/train/0 1000
save_dir = sys.argv[1]
num_images = int(sys.argv[2])
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0

print(f"Starting capture. Target: {num_images} images in '{save_dir}'")
print("Press 'SPACE' to capture, 'q' to quit.")

while True:
    ret, frame = cap.read()
    if not ret: break
    
    frame = cv2.flip(frame, 1)
    
    # Use the shared processing function.
    mask, (x1, y1, x2, y2) = process_frame(frame)
    
    # Visualization
    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(frame, f"Count: {count}/{num_images}", (20, 40), 
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
    
    cv2.imshow("Original", frame)
    cv2.imshow("Mask", mask)
    
    key = cv2.waitKey(1) & 0xFF
    if key == ord(' '):
        if count < num_images:
            path = os.path.join(save_dir, f"{count}.jpg")
            cv2.imwrite(path, mask)
            print(f"Saved: {path}")
            count += 1
        else:
            print("Target reached!")
            break
            
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()