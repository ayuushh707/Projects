# Projects
# Projects

## Hand-tracked hologram MVP (Python)

This starter project gives you a webcam-based hand tracker that controls a simple
"hologram" visualization.

### Features
- Tracks one hand with MediaPipe Hands (21 landmarks)
- Pinch gesture (thumb + index) to grab/release object
- Move hand while pinching to move the hologram
- Wrist orientation rotates the hologram indicator
- Smoothed tracking to reduce jitter

### Setup
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run
```bash
python hand_hologram_mvp.py
```

### Controls
- `q` to quit
- Pinch thumb + index finger to grab and move the hologram

### Next upgrades
- Swap glow circle for a real 3D model in Unity/Three.js
- Add gesture classes (open palm, fist, swipe)
- Use a depth camera for stronger tracking and occlusion
hand_hologr
