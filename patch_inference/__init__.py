"""
SAHI-like Patch-Based Detection Module
YOLOv11 with patch-based inference for small object detection
"""

from .detector import run_patch_detection, save_detections_to_csv
from .visualizer import create_detection_gif

__all__ = ["run_patch_detection", "create_detection_gif", "save_detections_to_csv"]
__version__ = "1.0.0"

