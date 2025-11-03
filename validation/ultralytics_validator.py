"""
Ultralytics standard validation.

This module provides a validator class for standard Ultralytics model evaluation.
"""

import json
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any

from ultralytics import YOLO


@dataclass
class UltralyticsValidationConfig:
    """Ultralytics validation configuration"""
    
    model_path: str = "runs/train/train12/weights/best.pt"
    data_yaml: str = "Dataset/YOLODataset_test_with_label/data.yaml"
    output_dir: str = "runs/val"
    experiment_name: str = "ultralytics_val"
    auto_increment: bool = True
    
    # Validation parameters
    imgsz: int = 640
    conf: float = 0.25
    iou: float = 0.6
    split: str = "val"
    batch: int = 1
    device: str = "0"
    
    # Output options
    save_json: bool = True
    save_hybrid: bool = False
    verbose: int = 1


class UltralyticsValidator:
    """Ultralytics standard validator"""
    
    def __init__(self, config: UltralyticsValidationConfig):
        """
        Args:
            config: Validation configuration
        """
        self.config = config
        self.model: Optional[YOLO] = None
        self.results: Optional[Any] = None
        
        # Output directory setup
        if self.config.auto_increment:
            from validation.utils.coco_converter import get_incremental_dir
            self.output_dir = str(get_incremental_dir(
                self.config.output_dir,
                self.config.experiment_name
            ))
        else:
            self.output_dir = str(Path(self.config.output_dir) / self.config.experiment_name)
        
        Path(self.output_dir).mkdir(parents=True, exist_ok=True)
    
    def load_model(self) -> None:
        """Load Ultralytics model"""
        if self.config.verbose >= 1:
            print(f"モデルを読み込み中: {self.config.model_path}")
        
        self.model = YOLO(self.config.model_path)
        
        if self.config.verbose >= 1:
            print("モデルの読み込みが完了しました")
            print(f"  Task: {self.model.task}")
            print(f"  Names: {self.model.names}")
    
    def validate(self) -> Dict[str, Any]:
        """Run validation
        
        Returns:
            Validation results dictionary
        """
        if self.model is None:
            self.load_model()
        
        if self.config.verbose >= 1:
            print("Validationを実行中...")
        
        # Run validation
        self.results = self.model.val(
            data=self.config.data_yaml,
            split=self.config.split,
            imgsz=self.config.imgsz,
            conf=self.config.conf,
            iou=self.config.iou,
            batch=self.config.batch,
            device=self.config.device,
            save_json=self.config.save_json,
            save_hybrid=self.config.save_hybrid,
            project=self.config.output_dir,
            name=self.config.experiment_name,
            exist_ok=True,
        )
        
        # Extract metrics
        metrics = {
            "box_map": float(self.results.box.map),
            "box_map50": float(self.results.box.map50),
            "box_map75": float(self.results.box.map75),
            "box_precision": float(self.results.box.p[0]),
            "box_recall": float(self.results.box.r[0]),
        }
        
        # Segmentation metrics (if available)
        if hasattr(self.results, 'seg') and self.results.seg is not None:
            metrics.update({
                "segm_map": float(self.results.seg.map),
                "segm_map50": float(self.results.seg.map50),
                "segm_map75": float(self.results.seg.map75),
                "segm_precision": float(self.results.seg.p[0]),
                "segm_recall": float(self.results.seg.r[0]),
            })
        
        # Save metrics to JSON
        metrics_file = Path(self.output_dir) / "metrics.json"
        with open(metrics_file, 'w', encoding='utf-8') as f:
            json.dump(metrics, f, indent=2, ensure_ascii=False)
        
        if self.config.verbose >= 1:
            print(f"\nValidation完了")
            print(f"結果を保存しました: {self.output_dir}")
            self._print_metrics(metrics)
        
        return {
            "output_dir": self.output_dir,
            "metrics": metrics,
            "results": self.results,
        }
    
    def _print_metrics(self, metrics: Dict[str, Any]) -> None:
        """Print metrics in a formatted way"""
        print("\n=== メトリクス ===")
        
        if "segm_map" in metrics:
            print("\n[Segmentation]")
            print(f"  mAP50-95: {metrics['segm_map']:.4f} ({metrics['segm_map']*100:.2f}%)")
            print(f"  mAP50:    {metrics['segm_map50']:.4f} ({metrics['segm_map50']*100:.2f}%)")
            print(f"  mAP75:    {metrics['segm_map75']:.4f} ({metrics['segm_map75']*100:.2f}%)")
            print(f"  Precision: {metrics['segm_precision']:.4f}")
            print(f"  Recall:    {metrics['segm_recall']:.4f}")
        
        print("\n[Bounding Box]")
        print(f"  mAP50-95: {metrics['box_map']:.4f} ({metrics['box_map']*100:.2f}%)")
        print(f"  mAP50:    {metrics['box_map50']:.4f} ({metrics['box_map50']*100:.2f}%)")
        print(f"  mAP75:    {metrics['box_map75']:.4f} ({metrics['box_map75']*100:.2f}%)")
        print(f"  Precision: {metrics['box_precision']:.4f}")
        print(f"  Recall:    {metrics['box_recall']:.4f}")

