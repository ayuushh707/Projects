"""Hand-tracked hologram-style object controller (MVP).

This script uses a webcam + MediaPipe Hands to track one hand and control a
simple glowing circle overlay:
- Pinch (thumb + index close) to grab the object
- Move hand while pinching to move the object
- Rotate wrist to rotate object
- Spread thumb/index while pinching to scale object

Run:
    python hand_hologram_mvp.py
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import mediapipe as mp
import numpy as np


@dataclass
class HologramState:
    x: int
    y: int
    size: float = 70.0
    angle_deg: float = 0.0
    grabbed: bool = False


def euclidean(a: tuple[float, float], b: tuple[float, float]) -> float:
    return math.hypot(a[0] - b[0], a[1] - b[1])


def smooth_point(
    previous: tuple[float, float] | None,
    current: tuple[float, float],
    alpha: float = 0.30,
) -> tuple[float, float]:
    """Exponential moving average smoothing for less jitter."""
    if previous is None:
        return current
    return (
        previous[0] * (1 - alpha) + current[0] * alpha,
        previous[1] * (1 - alpha) + current[1] * alpha,
    )


def draw_hologram(frame: np.ndarray, state: HologramState) -> None:
    """Draw a glowing sci-fi style circle as a fake hologram."""
    overlay = frame.copy()

    # Outer glow
    for r, a in [(int(state.size * 1.5), 20), (int(state.size * 1.2), 35), (int(state.size), 70)]:
        cv2.circle(overlay, (state.x, state.y), r, (255, 255, 0), -1)
        frame[:] = cv2.addWeighted(overlay, a / 255.0, frame, 1 - a / 255.0, 0)

    # Main ring
    cv2.circle(frame, (state.x, state.y), int(state.size), (255, 255, 0), 2)

    # Rotate indicator line
    theta = math.radians(state.angle_deg)
    x2 = int(state.x + state.size * math.cos(theta))
    y2 = int(state.y + state.size * math.sin(theta))
    cv2.line(frame, (state.x, state.y), (x2, y2), (0, 255, 255), 3)

    label = "GRABBED" if state.grabbed else "OPEN"
    cv2.putText(
        frame,
        f"{label} | size={state.size:.0f} | angle={state.angle_deg:.0f}",
        (20, 35),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )


def main() -> None:
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("Could not open webcam. Check camera permissions/device.")

    ok, frame = cap.read()
    if not ok:
        raise RuntimeError("Webcam opened but no frames were returned.")

    h, w = frame.shape[:2]
    holo = HologramState(x=w // 2, y=h // 2)

    prev_hand_center: tuple[float, float] | None = None
    baseline_pinch: float | None = None

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:
        while True:
            ok, frame = cap.read()
            if not ok:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                pts = []
                for lm in landmarks.landmark:
                    pts.append((lm.x * frame.shape[1], lm.y * frame.shape[0]))

                wrist = pts[0]
                thumb_tip = pts[4]
                index_tip = pts[8]
                middle_mcp = pts[9]

                hand_center_raw = (
                    (wrist[0] + middle_mcp[0]) / 2,
                    (wrist[1] + middle_mcp[1]) / 2,
                )
                hand_center = smooth_point(prev_hand_center, hand_center_raw, alpha=0.35)
                prev_hand_center = hand_center

                pinch_px = euclidean(thumb_tip, index_tip)
                pinch_threshold = 45

                holo.grabbed = pinch_px < pinch_threshold

                if holo.grabbed:
                    holo.x = int(hand_center[0])
                    holo.y = int(hand_center[1])

                    if baseline_pinch is None:
                        baseline_pinch = pinch_px
                    else:
                        scale_ratio = pinch_px / max(baseline_pinch, 1e-5)
                        holo.size = float(np.clip(holo.size * (0.92 + 0.08 * scale_ratio), 30, 180))

                    dx = middle_mcp[0] - wrist[0]
                    dy = middle_mcp[1] - wrist[1]
                    holo.angle_deg = (math.degrees(math.atan2(dy, dx)) + 360) % 360
                else:
                    baseline_pinch = None

                cv2.putText(
                    frame,
                    f"pinch distance: {pinch_px:.1f}px",
                    (20, 70),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.7,
                    (200, 255, 200),
                    2,
                    cv2.LINE_AA,
                )
            else:
                prev_hand_center = None
                baseline_pinch = None

            draw_hologram(frame, holo)

            cv2.putText(
                frame,
                "Q = quit | pinch to grab/move",
                (20, frame.shape[0] - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

            cv2.imshow("Hand Hologram MVP", frame)
            key = cv2.waitKey(1) & 0xFF
            if key == ord("q"):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
requirements.txt
