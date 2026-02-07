# Hand Gesture Recognition

Real-time hand gesture recognition using **MediaPipe** and a **CNN model** with webcam input.

## Features
- Live gesture detection via webcam
- MediaPipe hand landmark extraction
- CNN-based gesture classification

---
This project recognizes the following custom gestures:

| Gesture | Meaning / Use |
|---------|----------------|
| 👆 **Up** | Raise the drone / upward movement |
| 👇 **Down** | Lower the drone / downward movement |
| 👈 **Left** | Left |
| 👉 **Right** | Right |
| ✌️✌️ **Selfie Gesture** | Caputure current frame |
---


### Indoor vs Outdoor Behavior

- The **Up gesture** behaves differently depending on environment:  
  - **Fast Up** for outdoor/long-distance elevation  
  - **Slow Up** for indoor/controlled elevation  
- Selfie gesture is optimized for close-range detection and image framing  
- Works in varying lighting, but performance improves with consistent background

##  Dataset Information

This project uses a custom image dataset with:

- 🟢 **800 training images per class**  
- 🔵 **200 validation images per class**  

Each class corresponds to one gesture type, ensuring balanced training and validation.


## Setup
```bash
pip install opencv-python mediapipe tensorflow numpy
```

## Run
```bash
python gesture_recognition_webcam.py
python gesture-control-mp-novelty.py
```

## Notes
- Trained model file (`.h5`) was not included due to GitHub size limits
- Download the trained `.h5` model file here:
  https://drive.google.com/file/d/1fyz9HPvx79bt94aem5t_uzUx__Q3omjS/view?usp=sharing
- Place downloaded models inside the `CNN/` folder

