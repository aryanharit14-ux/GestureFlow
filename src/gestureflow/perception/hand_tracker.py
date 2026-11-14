"""MediaPipe HandLandmarker wrapper (Tasks API, video mode)."""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import mediapipe as mp
import numpy as np
from mediapipe.tasks.python import BaseOptions, vision

# Standard 21-landmark hand skeleton, defined locally so we don't depend on legacy APIs.
HAND_CONNECTIONS: tuple[tuple[int, int], ...] = (
    (0, 1), (1, 2), (2, 3), (3, 4),
    (0, 5), (5, 6), (6, 7), (7, 8),
    (5, 9), (9, 10), (10, 11), (11, 12),
    (9, 13), (13, 14), (14, 15), (15, 16),
    (13, 17), (17, 18), (18, 19), (19, 20),
    (0, 17),
)


@dataclass(frozen=True)
class HandResult:
    handedness: str          # "Left" / "Right" as seen in the (mirrored) image
    score: float             # handedness confidence
    landmarks: np.ndarray    # (21, 3): x, y normalized to the image, z relative depth
    world_landmarks: np.ndarray  # (21, 3): metric coordinates centred on the hand


class HandTracker:
    def __init__(self, model_path: str | Path, max_hands: int = 1,
                 min_detection: float = 0.6, min_presence: float = 0.6,
                 min_tracking: float = 0.6) -> None:
        model_path = Path(model_path)
        if not model_path.exists():
            raise FileNotFoundError(
                f"{model_path} not found. Run: python scripts/download_models.py"
            )
        options = vision.HandLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(model_path)),
            running_mode=vision.RunningMode.VIDEO,
            num_hands=max_hands,
            min_hand_detection_confidence=min_detection,
            min_hand_presence_confidence=min_presence,
            min_tracking_confidence=min_tracking,
        )
        self._landmarker = vision.HandLandmarker.create_from_options(options)

    def detect(self, frame_bgr: np.ndarray, timestamp_ms: int) -> list[HandResult]:
        """timestamp_ms must be strictly increasing across calls."""
        rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)
        result = self._landmarker.detect_for_video(image, timestamp_ms)
        hands: list[HandResult] = []
        for i, lms in enumerate(result.hand_landmarks):
            cat = result.handedness[i][0]
            hands.append(
                HandResult(
                    handedness=cat.category_name,
                    score=float(cat.score),
                    landmarks=np.array([[p.x, p.y, p.z] for p in lms], dtype=np.float32),
                    world_landmarks=np.array(
                        [[p.x, p.y, p.z] for p in result.hand_world_landmarks[i]],
                        dtype=np.float32,
                    ),
                )
            )
        return hands

    def close(self) -> None:
        self._landmarker.close()

    def __enter__(self) -> HandTracker:
        return self

    def __exit__(self, *exc: object) -> None:
        self.close()
