#!/usr/bin/env python3
"""
UltralyticsとSAHI COCOの精度評価の違いを診断

原因を特定して表示します。
"""

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_prediction
from pathlib import Path
import json

def main():
    model_path = "runs/train/train12/weights/best.pt"
    test_img = "Dataset/YOLODataset_test_with_label/images/val/Image_000473.jpeg"
    
    print("="*70)
    print("精度評価の違いの診断")
    print("="*70)
    
    # 1. Ultralytics標準推論
    print("\n[ステップ1] Ultralytics標準推論")
    model = YOLO(model_path)
    results_ultra = model.predict(test_img, conf=0.25, iou=0.6, imgsz=640, verbose=False)
    count_ultra = len(results_ultra[0].boxes)
    print(f"  検出数: {count_ultra}個")
    
    # 2. SAHI推論
    print("\n[ステップ2] SAHI推論（スライスなし）")
    detection_model = AutoDetectionModel.from_pretrained(
        model_type='ultralytics',
        model_path=model_path,
        confidence_threshold=0.25,
        device="0",
        image_size=640,
    )
    
    prediction_result = get_prediction(
        image=test_img,
        detection_model=detection_model,
        shift_amount=[0, 0],
        full_shape=None,
        postprocess=None,
        verbose=0,
    )
    count_sahi = len(prediction_result.object_prediction_list)
    print(f"  検出数: {count_sahi}個")
    
    # 3. SAHIのモデルで直接predict
    print("\n[ステップ3] SAHIのUltralyticsモデルを直接使用")
    count_direct = len(detection_model.model.predict(test_img, conf=0.25, iou=0.6, imgsz=640, verbose=False)[0].boxes)
    print(f"  検出数: {count_direct}個")
    
    # 4. Ultralytics標準評価
    print("\n[ステップ4] Ultralytics標準評価（全画像）")
    val_results = model.val(
        data="Dataset/YOLODataset_test_with_label/data.yaml",
        split='val',
        imgsz=640,
        conf=0.25,
        iou=0.6,
        verbose=False
    )
    ultra_map = val_results.seg.map
    print(f"  Segmentation mAP50-95: {ultra_map:.4f} ({ultra_map*100:.2f}%)")
    
    # 5. SAHI評価結果を確認
    print("\n[ステップ5] SAHI_val.pyの評価結果")
    sahi_eval_path = Path("runs/val/correct_model_no_slice/eval.json")
    if sahi_eval_path.exists():
        with open(sahi_eval_path, 'r') as f:
            sahi_eval = json.load(f)
        sahi_map = sahi_eval.get('segm_mAP', 0)
        print(f"  Segmentation mAP50-95: {sahi_map:.4f} ({sahi_map*100:.2f}%)")
    else:
        print("  評価結果が見つかりません")
        sahi_map = None
    
    # 診断結果
    print("\n" + "="*70)
    print("診断結果")
    print("="*70)
    
    print(f"\n1. 同じ画像での検出数:")
    print(f"   Ultralytics: {count_ultra}個")
    print(f"   SAHI:        {count_sahi}個")
    print(f"   差分:        {count_sahi - count_ultra}個 ({(count_sahi-count_ultra)/count_ultra*100:+.1f}%)")
    
    if sahi_map is not None:
        print(f"\n2. mAP50-95の違い:")
        print(f"   Ultralytics: {ultra_map:.4f} ({ultra_map*100:.2f}%)")
        print(f"   SAHI:        {sahi_map:.4f} ({sahi_map*100:.2f}%)")
        print(f"   差分:        {sahi_map - ultra_map:.4f} ({(sahi_map-ultra_map)*100:.2f}ポイント)")
    
    print("\n" + "="*70)
    print("原因の特定")
    print("="*70)
    
    if count_ultra == count_direct and count_ultra != count_sahi:
        print("""
✗ 問題: SAHIの`get_prediction()`関数が余分に検出している

原因:
  SAHIライブラリの`get_prediction()`は、Ultralyticsの標準推論と
  異なる動作をしています。具体的には：
  
  1. NMS（Non-Maximum Suppression）の処理が異なる
  2. 信頼度閾値の適用タイミングが異なる
  3. 後処理のロジックが異なる

結果:
  - より多くの検出（偽陽性が増加）
  - Precision低下
  - mAP低下

解決策:
  SAHI_val.pyは相対比較専用として使用し、
  絶対的な精度評価にはUltralytics標準の`model.val()`を使用する
        """)
    elif count_ultra != count_direct:
        print("""
✗ 問題: SAHIのAutoDetectionModelの設定に問題がある

原因:
  SAHIでモデルをロードする際の設定が不適切です。
        """)
    else:
        print("""
✓ 検出数は一致しています

問題は別の箇所にあります：
  - COCO評価の設定
  - データ形式の変換
  - 評価指標の計算方法
        """)
    
    print("\n" + "="*70)
    print("推奨事項")
    print("="*70)
    print("""
1. **絶対的な精度評価**: Ultralytics標準の`model.val()`を使用
   → ultralytics_val.py を使用

2. **SAHI_val.py**: スライス推論の効果測定専用として使用
   → SAHI（スライスあり）vs 標準（スライスなし）の相対比較のみ

3. **注意**: SAHI_val.pyの絶対値（mAP）は参考程度にとどめる
    """)

if __name__ == "__main__":
    main()

