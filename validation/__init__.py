"""
Validation package for YOLO model evaluation.

This package provides:
- SAHI-based validation with sliced inference
- Ultralytics standard validation
- Comparison between different validation methods
- Error analysis and visualization
"""

from validation.sahi_validator import SAHIValidator
from validation.ultralytics_validator import UltralyticsValidator
from validation.comparator import ValidationComparator

__all__ = [
    "SAHIValidator",
    "UltralyticsValidator",
    "ValidationComparator",
]

__version__ = "1.0.0"

