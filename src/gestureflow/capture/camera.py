"""Threaded webcam capture that always exposes the newest frame (stale frames are dropped)."""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np


class Camera:
    def __init__(self, index: int = 0, width: int = 640, height: int = 480) -> None:
        self._index = index
        self._width = width
        self._height = height
        self._cap: cv2.VideoCapture | None = None
        self._thread: threading.Thread | None = None
        self._lock = threading.Lock()
        self._frame: np.ndarray | None = None
        self._frame_id = 0
        self._running = False

    def start(self) -> None:
        cap = cv2.VideoCapture(self._index)
        if not cap.isOpened():
            raise RuntimeError(
                f"Could not open camera {self._index}. On macOS, allow camera access for "
                "your terminal / VS Code in System Settings > Privacy & Security > Camera."
            )
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, self._width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self._height)
        self._cap = cap
        self._running = True
        self._thread = threading.Thread(target=self._loop, name="camera", daemon=True)
        self._thread.start()

    def _loop(self) -> None:
        assert self._cap is not None
        while self._running:
            ok, frame = self._cap.read()
            if not ok:
                time.sleep(0.01)
                continue
            with self._lock:
                self._frame = frame
                self._frame_id += 1

    def read(self, last_id: int = -1) -> tuple[int, np.ndarray] | None:
        """Return (frame_id, frame) if a frame newer than last_id exists, else None."""
        with self._lock:
            if self._frame is None or self._frame_id == last_id:
                return None
            return self._frame_id, self._frame.copy()

    def stop(self) -> None:
        self._running = False
        if self._thread is not None:
            self._thread.join(timeout=1.0)
        if self._cap is not None:
            self._cap.release()
        self._cap = None
        self._thread = None

    def __enter__(self) -> Camera:
        self.start()
        return self

    def __exit__(self, *exc: object) -> None:
        self.stop()
