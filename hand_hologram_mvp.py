"""Hand-tracked hologram-style object controller (MVP)."""

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


def euclidean(a, b):
    return math.hypot(a[0] - b[0], a[1] - b[1])


def smooth_point(previous, current, alpha=0.30):
    if previous is None:
        return current
    return (
        previous[0] * (1 - alpha) + current[0] * alpha,
        previous[1] * (1 - alpha) + current[1] * alpha,
    )


def draw_hologram(frame, state):
    overlay = frame.copy()

    for r, a in [(int(state.size * 1.5), 20), (int(state.size * 1.2), 35), (int(state.size), 70)]:
        cv2.circle(overlay, (state.x, state.y), r, (255, 255, 0), -1)
        frame[:] = cv2.addWeighted(overlay, a / 255.0, frame, 1 - a / 255.0, 0)

    cv2.circle(frame, (state.x, state.y), int(state.size), (255, 255, 0), 2)

    theta = math.radians(state.angle_deg)
    x2 = int(state.x + state.size * math.cos(theta))
    y2 = int(state.y + state.size * math.sin(theta))
    cv2.line(frame, (state.x, state.y), (x2, y2), (0, 255, 255), 3)

    label = "GRABBED" if state.grabbed else "OPEN"
    cv2.putText(frame, f"{label} | size={state.size:.0f} | angle={state.angle_deg:.0f}",
                (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def main():
    mp_hands = mp.solutions.hands
    mp_draw = mp.solutions.drawing_utils

    cap = cv2.VideoCapture(0)

    if not cap.isOpened():
        print("Camera not working")
        return

    ret, frame = cap.read()
    h, w = frame.shape[:2]

    holo = HologramState(x=w // 2, y=h // 2)

    prev_hand_center = None
    baseline_pinch = None

    with mp_hands.Hands(
        model_complexity=1,
        max_num_hands=1,
        min_detection_confidence=0.6,
        min_tracking_confidence=0.6,
    ) as hands:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            result = hands.process(rgb)

            if result.multi_hand_landmarks:
                landmarks = result.multi_hand_landmarks[0]
                mp_draw.draw_landmarks(frame, landmarks, mp_hands.HAND_CONNECTIONS)

                pts = [(lm.x * frame.shape[1], lm.y * frame.shape[0]) for lm in landmarks.landmark]

                wrist = pts[0]
                thumb_tip = pts[4]
                index_tip = pts[8]
                middle_mcp = pts[9]

                hand_center_raw = (
                    (wrist[0] + middle_mcp[0]) / 2,
                    (wrist[1] + middle_mcp[1]) / 2,
                )

                hand_center = smooth_point(prev_hand_center, hand_center_raw)
                prev_hand_center = hand_center

                pinch_px = euclidean(thumb_tip, index_tip)
                holo.grabbed = pinch_px < 45

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

            else:
                prev_hand_center = None
                baseline_pinch = None

            draw_hologram(frame, holo)

            cv2.putText(frame, "Q = Quit", (20, frame.shape[0] - 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)

            cv2.imshow("Hand Hologram", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
