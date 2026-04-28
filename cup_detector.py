from __future__ import annotations

from dataclasses import dataclass
import importlib
from typing import List, Sequence, Tuple

import cv2
import numpy as np


@dataclass(frozen=True)
class CupDetection:
    cup_id: int
    obb: Sequence[float]


def _sort_cup_detections(
    detections: List[CupDetection], 
    row_y_threshold: float = 50.0
) -> List[CupDetection]:
    """
    內部使用的排序函式：
    將偵測到的杯子動態分排，並依照「先下排再上排、同排由左至右」的順序回傳。
    """
    if not detections:
        return []

    def get_center(d: CupDetection) -> Tuple[float, float]:
        pts = np.array(d.obb).reshape(4, 2)
        return pts.mean(axis=0)

    # 1. 依 Y 座標遞增排序 (由上到下)
    sorted_by_y = sorted(detections, key=lambda d: get_center(d)[1])
    
    # 2. 動態分排
    rows = []
    current_row = [sorted_by_y[0]]
    
    for d in sorted_by_y[1:]:
        prev_y = get_center(current_row[-1])[1]
        curr_y = get_center(d)[1]
        
        # 若 Y 座標落差大於閾值，視為換排
        if curr_y - prev_y > row_y_threshold:
            rows.append(current_row)
            current_row = [d]
        else:
            current_row.append(d)
    rows.append(current_row)
    
    # 3. 各排內部依 X 座標遞增排序 (由左至右)
    # reverse = True 可以改成由右至左
    for row in rows:
        row.sort(key=lambda d: get_center(d)[0], reverse = True)
    
    # 4. 反轉陣列，達成「先下排，再上排」
    rows.reverse()
    
    return [d for row in rows for d in row]


class CupDetector:
    """YOLO-OBB cup detector used by left arm. Lazily imports ultralytics."""

    def __init__(self, model_path: str, row_y_threshold: float = 50.0):
        self.model_path = model_path
        self._model = None
        self.row_y_threshold = row_y_threshold

    def _ensure_model(self):
        if self._model is None:
            ultralytics_module = importlib.import_module('ultralytics')
            self._model = ultralytics_module.YOLO(self.model_path)

    def detect(self, image_bgr: np.ndarray) -> List[CupDetection]:
        """Detect cups and return them in the specific picking order."""
        self._ensure_model()
        results = self._model(image_bgr)
        if not results:
            return []

        obb = getattr(results[0], 'obb', None)
        if obb is None or obb.xyxyxyxy is None:
            return []

        raw_detections: List[CupDetection] = []
        for i, points in enumerate(obb.xyxyxyxy.cpu().numpy(), start=1):
            flat_points = np.array(points, dtype=float).reshape(-1).tolist()
            raw_detections.append(CupDetection(cup_id=i, obb=flat_points))

        ordered_detections = _sort_cup_detections(raw_detections, self.row_y_threshold)
        return ordered_detections

    def detect_with_visualization(
        self, image_bgr: np.ndarray
    ) -> Tuple[List[CupDetection], np.ndarray]:
        """Detect cups and return detections with visualization image."""
        detections = self.detect(image_bgr)
        vis_image = self._draw_detections(image_bgr, detections)
        return detections, vis_image

    def _draw_detections(
        self, image_bgr: np.ndarray, detections: List[CupDetection]
    ) -> np.ndarray:
        """Draw detection boxes and highlight the picking order."""
        img_vis = image_bgr.copy()
        for pick_order, detection in enumerate(detections, start=1):
            obb = np.array(detection.obb, dtype=np.float32).reshape(4, 2)
            pts = np.int32(obb)
            cv2.polylines(img_vis, [pts], True, (0, 255, 0), 2)
            center = pts.mean(axis=0)
            label = f"Pick {pick_order} (ID:{detection.cup_id})"
            cv2.putText(
                img_vis,
                label,
                tuple(center.astype(int)),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
            )
        return img_vis
