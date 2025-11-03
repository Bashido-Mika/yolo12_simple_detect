"""
COCO format conversion utilities.

This module provides utilities for converting YOLO format datasets to COCO format.
"""

import json
import yaml
from pathlib import Path
from PIL import Image
from tqdm import tqdm


def get_incremental_dir(base_dir: str, name: str = "exp") -> Path:
    """連番ディレクトリを取得する
    
    Args:
        base_dir: ベースディレクトリ (例: "runs/val")
        name: 実験名 (例: "exp")
        
    Returns:
        インクリメントされたディレクトリパス (例: "runs/val/exp", "runs/val/exp2", ...)
    """
    base_path = Path(base_dir)
    
    # 最初の候補（番号なし）
    candidate = base_path / name
    if not candidate.exists():
        return candidate
    
    # 番号付きの候補を探す
    i = 2
    while True:
        candidate = base_path / f"{name}{i}"
        if not candidate.exists():
            return candidate
        i += 1


class YOLOToCOCOConverter:
    """YOLO形式のデータセットをCOCO形式に変換するクラス"""

    def __init__(self, yolo_dataset_dir: str, split: str = "val"):
        """
        Args:
            yolo_dataset_dir: YOLO形式データセットのディレクトリ
            split: "train" または "val"
        """
        self.yolo_dataset_dir = Path(yolo_dataset_dir)
        self.split = split
        self.data_yaml_path = self.yolo_dataset_dir / "data.yaml"
        
        # data.yamlを読み込み
        if not self.data_yaml_path.exists():
            raise FileNotFoundError(f"data.yamlが見つかりません: {self.data_yaml_path}")
        
        with open(self.data_yaml_path, "r", encoding="utf-8") as f:
            self.data_config = yaml.safe_load(f)
        
        self.class_names = self.data_config.get("names", [])
        self.num_classes = self.data_config.get("nc", len(self.class_names))
        
        # パス設定
        self.images_dir = self.yolo_dataset_dir / "images" / split
        self.labels_dir = self.yolo_dataset_dir / "labels" / split
        
        if not self.images_dir.exists():
            raise FileNotFoundError(f"画像ディレクトリが見つかりません: {self.images_dir}")
        if not self.labels_dir.exists():
            raise FileNotFoundError(f"ラベルディレクトリが見つかりません: {self.labels_dir}")

    def convert(self, output_json_path: str) -> str:
        """YOLO形式をCOCO形式に変換
        
        Args:
            output_json_path: 出力するCOCO JSONファイルのパス
            
        Returns:
            出力ファイルのパス
        """
        # COCO構造を初期化
        coco_dict = {
            "info": {
                "description": "Converted from YOLO format",
                "version": "1.0",
                "year": 2024,
            },
            "licenses": [],
            "images": [],
            "annotations": [],
            "categories": [],
        }
        
        # カテゴリを追加
        for idx, class_name in enumerate(self.class_names):
            coco_dict["categories"].append({
                "id": idx,
                "name": class_name,
                "supercategory": class_name,
            })
        
        # 画像とアノテーションを処理
        annotation_id = 0
        image_id = 0
        
        image_files = sorted(list(self.images_dir.glob("*")))
        image_files = [f for f in image_files if f.suffix.lower() in [".jpg", ".jpeg", ".png", ".bmp"]]
        
        for image_file in tqdm(image_files, desc="Converting YOLO to COCO"):
            # 画像情報を取得
            try:
                img = Image.open(image_file)
                width, height = img.size
            except Exception as e:
                print(f"警告: 画像を読み込めませんでした {image_file}: {e}")
                continue
            
            # 画像エントリを追加（file_nameは画像ファイル名のみ）
            coco_image = {
                "id": image_id,
                "file_name": image_file.name,
                "width": width,
                "height": height,
            }
            coco_dict["images"].append(coco_image)
            
            # ラベルファイルを読み込み
            label_file = self.labels_dir / f"{image_file.stem}.txt"
            if not label_file.exists():
                image_id += 1
                continue
            
            with open(label_file, "r", encoding="utf-8") as f:
                lines = f.readlines()
            
            for line in lines:
                line = line.strip()
                if not line:
                    continue
                
                parts = line.split()
                if len(parts) < 3:
                    continue
                
                class_id = int(parts[0])
                # 正規化されたポリゴン座標（x1, y1, x2, y2, ...）
                polygon_coords = [float(x) for x in parts[1:]]
                
                if len(polygon_coords) < 6:  # 最低3点（6座標）必要
                    continue
                
                # 正規化を解除して絶対座標に変換
                absolute_coords = []
                for i in range(0, len(polygon_coords), 2):
                    if i + 1 < len(polygon_coords):
                        x = polygon_coords[i] * width
                        y = polygon_coords[i + 1] * height
                        absolute_coords.extend([x, y])
                
                if len(absolute_coords) < 6:
                    continue
                
                # バウンディングボックスを計算
                x_coords = absolute_coords[::2]
                y_coords = absolute_coords[1::2]
                xmin = max(0, min(x_coords))
                ymin = max(0, min(y_coords))
                xmax = min(width, max(x_coords))
                ymax = min(height, max(y_coords))
                
                bbox = [xmin, ymin, xmax - xmin, ymax - ymin]
                area = (xmax - xmin) * (ymax - ymin)
                
                # アノテーションを追加
                coco_annotation = {
                    "id": annotation_id,
                    "image_id": image_id,
                    "category_id": class_id,
                    "segmentation": [absolute_coords],
                    "area": area,
                    "bbox": bbox,
                    "iscrowd": 0,
                }
                coco_dict["annotations"].append(coco_annotation)
                annotation_id += 1
            
            image_id += 1
        
        # JSONファイルに保存
        output_path = Path(output_json_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(output_path, "w", encoding="utf-8") as f:
            json.dump(coco_dict, f, indent=2, ensure_ascii=False)
        
        return str(output_path)

