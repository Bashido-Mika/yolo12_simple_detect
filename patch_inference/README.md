# 🔍 SAHI-like Patch-Based Detection

YOLOv11を使用したパッチベースの物体検出・セグメンテーションモジュール

## 📋 機能

- ✨ **パッチベース推論**: 大きな画像を小さなパッチに分割して高精度検出
- 🎯 **インスタンスセグメンテーション**: マスク付き検出
- 🚀 **バッチ推論**: 高速化のためのバッチ処理サポート
- 🎬 **GIFアニメーション**: 検出過程の可視化
- 🔄 **自動NMS**: 重複検出の除去

## 📁 ディレクトリ構造

```
patch_inference/
├── __init__.py          # モジュール初期化
├── detector.py          # 検出処理
├── visualizer.py        # GIF作成
└── README.md            # このファイル
```

## 🔧 使用方法

### Python APIとして使用

```python
from patch_inference import run_patch_detection, create_detection_gif

# 検出実行
processed_images, output_dir = run_patch_detection(
    model_path="best.pt",
    source_path="images/",
    output_dir="runs/detect/results",
    shape_x=400,
    shape_y=400,
    overlap_x=30,
    overlap_y=30,
    conf_threshold=0.5,
    batch_inference=True
)

# GIF作成
gif_path = create_detection_gif(
    image_path="image.jpg",
    model_path="best.pt",
    output_dir="runs/detect/gif",
    shape_x=400,
    shape_y=400,
    overlap_x=30,
    overlap_y=30,
    fps=30
)
```

### CLIツールとして使用

メインスクリプト `sahi_detect_cli.py` を使用:

```bash
# 基本的な検出
python sahi_detect_cli.py -m best.pt -s images/

# GIF動画も作成
python sahi_detect_cli.py -m best.pt -s image.jpg --create-gif

# パラメータカスタマイズ
python sahi_detect_cli.py -m best.pt -s images/ \
    --shape-x 512 --shape-y 512 \
    --overlap-x 40 --conf 0.6
```

## ⚙️ パラメータ

| パラメータ | デフォルト | 説明 |
|-----------|----------|------|
| `shape_x` | 400 | パッチの幅 |
| `shape_y` | 400 | パッチの高さ |
| `overlap_x` | 30 | X軸オーバーラップ (%) |
| `overlap_y` | 30 | Y軸オーバーラップ (%) |
| `conf_threshold` | 0.5 | 信頼度閾値 |
| `imgsz` | 640 | YOLO入力サイズ |
| `nms_threshold` | 0.3 | NMS閾値 |
| `batch_inference` | True | バッチ推論の有効化 |

## 📦 依存関係

- `ultralytics` - YOLOv11
- `patched-yolo-infer` - パッチベース推論
- `opencv-python` - 画像処理
- `numpy` - 数値計算
- `imageio` - GIF作成
- `tqdm` - プログレスバー

## 🎨 GIFアニメーションの仕組み

1. **元画像表示**: 最初に元の画像を表示
2. **パッチスキャン**: 各パッチを順番にスライド表示
   - 薄い青色でパッチを強調
   - オーバーラップ部分は濃く表示
3. **検出結果**: 各パッチの検出結果をフェードイン
4. **NMS前**: すべてのパッチの検出結果を表示
5. **最終結果**: NMS適用後の最終結果を表示

## 📝 ライセンス

このモジュールは [YOLO-Patch-Based-Inference](https://github.com/Koldim2001/YOLO-Patch-Based-Inference) を使用しています。

