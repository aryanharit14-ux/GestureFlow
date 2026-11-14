"""Live preview: webcam + hand landmarks + latency. Press q or Esc to quit."""
import time
from pathlib import Path

import cv2

from gestureflow.capture.camera import Camera
from gestureflow.perception.hand_tracker import HAND_CONNECTIONS, HandTracker

MODEL = Path(__file__).resolve().parents[1] / "models" / "hand_landmarker.task"


def draw(frame, hand) -> None:
    h, w = frame.shape[:2]
    pts = [(int(x * w), int(y * h)) for x, y, _ in hand.landmarks]
    for a, b in HAND_CONNECTIONS:
        cv2.line(frame, pts[a], pts[b], (0, 255, 0), 2)
    for p in pts:
        cv2.circle(frame, p, 4, (0, 0, 255), -1)


def main() -> None:
    last_id = -1
    last_ts = -1
    fps_t0, fps_n, fps = time.monotonic(), 0, 0.0
    with Camera() as cam, HandTracker(MODEL) as tracker:
        while True:
            item = cam.read(last_id)
            if item is None:
                if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                    break
                continue
            last_id, frame = item
            frame = cv2.flip(frame, 1)  # mirror view; MediaPipe expects selfie orientation

            ts = max(int(time.monotonic() * 1000), last_ts + 1)
            last_ts = ts
            t0 = time.perf_counter()
            hands = tracker.detect(frame, ts)
            latency_ms = (time.perf_counter() - t0) * 1000

            for hand in hands:
                draw(frame, hand)

            fps_n += 1
            if time.monotonic() - fps_t0 >= 1.0:
                fps, fps_n, fps_t0 = fps_n, 0, time.monotonic()
            label = f"{fps:.0f} FPS | inference {latency_ms:.1f} ms | hands: {len(hands)}"
            cv2.putText(frame, label, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                        (255, 255, 255), 2)
            cv2.imshow("GestureFlow preview", frame)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
