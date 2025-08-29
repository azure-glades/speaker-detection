"""
FaceTracker
Detects and tracks all visible faces in a frame sequence.
Returns bounding boxes (x1, y1, x2, y2) in absolute pixel coordinates.

Dependencies
------------
pip install opencv-python mediapipe numpy
"""

from typing import List, Tuple, Optional
import numpy as np
import cv2
import mediapipe as mp


class FaceTracker:
    def __init__(
        self,
        model_selection: int = 1,          # 0=close-range, 1=full-range
        min_detection_confidence: float = 0.5,
        smooth_boxes: bool = True,         # simple exponential smoothing
        smoothing_alpha: float = 0.3,
    ):
        self.smooth_boxes = smooth_boxes
        self.alpha = smoothing_alpha
        self._prev_boxes: List[Tuple[int, int, int, int]] = []

        # MediaPipe Face Detection (lightweight; no landmarks)
        self.detector = mp.solutions.face_detection.FaceDetection(
            model_selection=model_selection,
            min_detection_confidence=min_detection_confidence,
        )

    # ------------------------------------------------------------------ #
    def __call__(self, frame: np.ndarray) -> List[Tuple[int, int, int, int]]:
        """
        Parameters
        ----------
        frame : np.ndarray
            BGR image (H, W, 3).

        Returns
        -------
        boxes : list[tuple]
            [(x1, y1, x2, y2), ...] in pixel coordinates.
            Boxes are sorted left→right to maintain consistent ordering.
        """
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.detector.process(rgb)

        h, w = frame.shape[:2]
        boxes: List[Tuple[int, int, int, int]] = []

        if results.detections:
            for det in results.detections:
                bbox = det.location_data.relative_bounding_box
                x1 = max(0, int(bbox.xmin * w))
                y1 = max(0, int(bbox.ymin * h))
                x2 = min(w, int((bbox.xmin + bbox.width) * w))
                y2 = min(h, int((bbox.ymin + bbox.height) * h))
                boxes.append((x1, y1, x2, y2))

        # Sort by leftmost coordinate so indexing is stable
        boxes = sorted(boxes, key=lambda b: b[0])

        # Optional smoothing (very simple)
        if self.smooth_boxes and self._prev_boxes:
            if len(boxes) == len(self._prev_boxes):
                boxes = [
                    tuple(
                        int(a * self.alpha + b * (1 - self.alpha))
                        for a, b in zip(curr, prev)
                    )
                    for curr, prev in zip(boxes, self._prev_boxes)
                ]

        self._prev_boxes = boxes
        return boxes