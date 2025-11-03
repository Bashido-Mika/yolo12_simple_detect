# YOLO12 Detect Project

YOLOv12を使用した物体検出・セグメンテーションプロジェクト

## 📋 目次

- [プロジェクト概要](#プロジェクト概要)
- [ディレクトリ構造](#ディレクトリ構造)
- [機能](#機能)
  - [Validation Module](#validation-module)
  - [Patch-Based Detection](#patch-based-detection)
  - [SAHI CLI Tool](#sahi-cli-tool)
- [インストール](#インストール)
- [使用方法](#使用方法)
- [依存関係](#依存関係)
- [参考資料](#参考資料)

---

## プロジェクト概要

このプロジェクトは、YOLOv12を使用した物体検出・セグメンテーションのための統合ツールセットです。以下の主要機能を提供します：

- **Validation Module**: YOLOモデルの評価を統一的に行うモジュール
- **Patch-Based Detection**: 大きな画像を小さなパッチに分割して高精度検出
- **SAHI CLI Tool**: パッチベース物体検出のCLIツール

---

## ディレクトリ構造

```
yolo12_detect/
├── validation/                  # Validation Module
│   ├── __init__.py
│   ├── ultralytics_validator.py
│   ├── sahi_validator.py
│   ├── comparator.py
│   └── utils/
├── patch_inference/             # Patch-Based Detection
│   ├── __init__.py
│   ├── detector.py
│   └── visualizer.py
├── sahi_detect_cli.py           # SAHI CLI Tool
├── validate.py                  # メイン評価エントリーポイント（統合CLI）
└── Dataset/                     # データセット
```

---

## 機能

### Validation Module

YOLOモデルの評価を統一的に行うためのモジュール群

#### 使い方

##### 1. Ultralytics標準評価

```bash
# 基本的な使い方
python validate.py ultralytics --model runs/train/train12/weights/best.pt

# オプション指定
python validate.py ultralytics \
  --model runs/train/train12/weights/best.pt \
  --data Dataset/YOLODataset_test_with_label/data.yaml \
  --name my_validation \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.6

# JSON結果を保存
python validate.py ultralytics -m best.pt --save-json

# データセット分割を指定（train/val/test）
python validate.py ultralytics -m best.pt --split val

# デバイスを指定
python validate.py ultralytics -m best.pt --device cuda:0
```

**出力:**
- `runs/val/my_validation/metrics.json` - 評価メトリクス
- `runs/val/my_validation/predictions.json` - 予測結果（`--save-json`使用時）

##### 2. SAHI評価

```bash
# 基本的な使い方（YOLO形式データセットから自動変換）
python validate.py sahi --yolo-dataset Dataset/YOLODataset_test_with_label

# 短縮形を使用
python validate.py sahi -y Dataset/YOLODataset_test_with_label

# カスタムスライスサイズ
python validate.py sahi \
  -y Dataset/YOLODataset_test_with_label \
  --slice-height 640 \
  --slice-width 640

# エラー解析付き
python validate.py sahi \
  --yolo-dataset Dataset/YOLODataset_test_with_label \
  --error-analysis

# 可視化画像を保存
python validate.py sahi \
  --yolo-dataset Dataset/YOLODataset_test_with_label \
  --export-visuals

# 予測のみ実行（評価をスキップ、predictions.jsonのみ生成）
python validate.py sahi \
  --yolo-dataset Dataset/YOLODataset_test_with_label \
  --predict-only

# COCO形式データセットを使用
python validate.py sahi \
  --dataset dataset.json \
  --images images/

# 標準推論のみ（スライス推論なし）
python validate.py sahi \
  --yolo-dataset Dataset/YOLODataset_test_with_label \
  --no-sliced-prediction

# 詳細な設定例
python validate.py sahi \
  --yolo-dataset Dataset/YOLODataset_test_with_label \
  --slice-height 512 \
  --slice-width 512 \
  --overlap-height-ratio 0.5 \
  --overlap-width-ratio 0.5 \
  --postprocess-type GREEDYNMM \
  --postprocess-match-metric IOS \
  --postprocess-match-threshold 0.5 \
  --classwise \
  --export-visuals \
  --max-visual-samples 10
```

**出力:**
- `runs/val/sahi_val/eval.json` - 評価メトリクス
- `runs/val/sahi_val/predictions.json` - 予測結果
- `runs/val/sahi_val/segm/` - エラー解析プロット（`--error-analysis`使用時）
- `runs/val/sahi_val/visuals/` - 可視化画像（`--export-visuals`使用時）

##### 3. 比較モード ⭐ おすすめ

```bash
# Ultralytics vs SAHI の自動比較（基本）
python validate.py compare --yolo-dataset Dataset/YOLODataset_test_with_label

# エラー解析と可視化付き
python validate.py compare \
  --yolo-dataset Dataset/YOLODataset_test_with_label \
  --name comparison_test \
  --error-analysis \
  --export-visuals

# 詳細なパラメータ指定
python validate.py compare \
  --yolo-dataset Dataset/YOLODataset_test_with_label \
  --imgsz 640 \
  --conf 0.25 \
  --iou 0.6 \
  --slice-height 512 \
  --slice-width 512 \
  --error-analysis \
  --export-visuals
```

**動作:**
1. ✅ Ultralyticsで評価実行（自動）
2. ✅ SAHI評価を実行（自動）
3. ✅ 結果を比較してレポート生成（自動）

**出力:**
```
runs/val/comparison_test/
├── ultralytics/           # Ultralytics評価結果
│   ├── metrics.json
│   └── predictions.json
├── sahi/                  # SAHI評価結果
│   ├── eval.json
│   ├── predictions.json
│   └── segm/             # エラー解析プロット
└── README.txt            # 比較サマリー
```

#### 出力形式

**Ultralytics評価結果 (`metrics.json`)**

```json
{
  "box_map": 0.4329,
  "box_map50": 0.6972,
  "box_map75": 0.4722,
  "box_precision": 0.6953,
  "box_recall": 0.6762,
  "segm_map": 0.3627,
  "segm_map50": 0.6920,
  "segm_map75": 0.3550,
  "segm_precision": 0.6885,
  "segm_recall": 0.6710
}
```

#### 開発状況

| モジュール | 状態 | 説明 |
|-----------|------|------|
| `ultralytics_validator.py` | ✅ 完成 | Ultralytics標準評価 |
| `sahi_validator.py` | ✅ 完成 | SAHI評価（ラッパー） |
| `utils/coco_converter.py` | ✅ 完成 | YOLO→COCO変換 |
| `utils/visualization.py` | ✅ 完成 | 比較画像生成 |
| `utils/metrics.py` | ✅ 完成 | メトリクス計算 |
| `comparator.py` | ✅ 完成 | 自動比較ロジック |

---

### Patch-Based Detection

YOLOv11を使用したパッチベースの物体検出・セグメンテーションモジュール

#### 機能

- ✨ **パッチベース推論**: 大きな画像を小さなパッチに分割して高精度検出
- 🎯 **インスタンスセグメンテーション**: マスク付き検出
- 🚀 **バッチ推論**: 高速化のためのバッチ処理サポート
- 🎬 **GIFアニメーション**: 検出過程の可視化
- 🔄 **自動NMS**: 重複検出の除去

#### 使用方法

##### Python APIとして使用

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

#### パラメータ

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

#### GIFアニメーションの仕組み

1. **元画像表示**: 最初に元の画像を表示
2. **パッチスキャン**: 各パッチを順番にスライド表示
   - 薄い青色でパッチを強調
   - オーバーラップ部分は濃く表示
3. **検出結果**: 各パッチの検出結果をフェードイン
4. **NMS前**: すべてのパッチの検出結果を表示
5. **最終結果**: NMS適用後の最終結果を表示

---

### SAHI CLI Tool

YOLOv11を使用したパッチベース物体検出・セグメンテーションのCLIツール

#### クイックスタート

```bash
# 単一画像の検出
uv run sahi_detect_cli.py -m runs/train/train12/weights/best.pt -s detect_images/image.jpg

# ディレクトリ内のすべての画像を検出
uv run sahi_detect_cli.py -m runs/train/train12/weights/best.pt -s detect_images/

# GIF動画も作成
uv run sahi_detect_cli.py -m best.pt -s image.jpg --create-gif
```

#### コマンドライン引数

##### 必須引数

| 引数 | 短縮形 | 説明 | 例 |
|------|--------|------|-----|
| `--model` | `-m` | YOLOモデルのパス | `best.pt` |
| `--source` | `-s` | 画像ファイルまたはディレクトリ | `images/` |

##### オプション引数

**出力設定**

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--output` / `-o` | `runs/detect/sahi_results` | 検出結果の出力先 |
| `--gif-output` | `runs/detect/sahi_gif` | GIF動画の出力先 |

**パッチ設定**

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--shape-x` | `400` | パッチの幅 |
| `--shape-y` | `400` | パッチの高さ |
| `--overlap-x` | `30` | X軸オーバーラップ (%) |
| `--overlap-y` | `30` | Y軸オーバーラップ (%) |

**推論設定**

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--conf` | `0.5` | 信頼度閾値 |
| `--imgsz` | `640` | YOLO入力サイズ |
| `--nms-threshold` | `0.3` | NMS閾値 |
| `--no-batch-inference` | - | バッチ推論を無効化（メモリ節約） |

**GIF設定**

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--create-gif` | - | GIF動画を作成 |
| `--gif-fps` | `30` | GIFのフレームレート |

#### 使用例

##### 例1: 基本的な検出

```bash
uv run sahi_detect_cli.py \
    --model runs/train/train12/weights/best.pt \
    --source detect_images/
```

##### 例2: パラメータをカスタマイズ

```bash
uv run sahi_detect_cli.py \
    -m best.pt \
    -s images/ \
    --shape-x 512 \
    --shape-y 512 \
    --overlap-x 40 \
    --conf 0.6 \
    --nms-threshold 0.2
```

##### 例3: GIF動画を作成

```bash
uv run sahi_detect_cli.py \
    -m best.pt \
    -s image.jpg \
    --create-gif \
    --gif-fps 30
```

##### 例4: メモリ節約モード（バッチ推論なし）

```bash
uv run sahi_detect_cli.py \
    -m best.pt \
    -s images/ \
    --no-batch-inference
```

#### パラメータの選び方

**パッチサイズ (`--shape-x`, `--shape-y`)**

- **小さい (320-400)**: 小さな物体の検出に有効、処理時間が長い
- **大きい (512-640)**: 高速処理、大きな物体向け

**オーバーラップ (`--overlap-x`, `--overlap-y`)**

- **小さい (10-20%)**: 高速処理、境界での見逃しが増える可能性
- **大きい (30-50%)**: 高精度、処理時間が長い

**信頼度閾値 (`--conf`)**

- **低い (0.3-0.4)**: より多くの検出、誤検出も増える
- **高い (0.6-0.7)**: 確実な検出のみ、見逃しが増える可能性

**NMS閾値 (`--nms-threshold`)**

- **低い (0.1-0.2)**: 重複を厳しく除去、検出数が減る
- **高い (0.4-0.5)**: 重複を許容、検出数が増える

#### トラブルシューティング

**メモリ不足エラー**

```bash
# バッチ推論を無効化
uv run sahi_detect_cli.py -m best.pt -s images/ --no-batch-inference

# パッチサイズを小さくする
uv run sahi_detect_cli.py -m best.pt -s images/ --shape-x 320 --shape-y 320
```

**処理が遅い**

```bash
# パッチサイズを大きくする
uv run sahi_detect_cli.py -m best.pt -s images/ --shape-x 512 --shape-y 512

# オーバーラップを減らす
uv run sahi_detect_cli.py -m best.pt -s images/ --overlap-x 20 --overlap-y 20
```

**小さな物体が検出されない**

```bash
# パッチサイズを小さくする
uv run sahi_detect_cli.py -m best.pt -s images/ --shape-x 320 --shape-y 320

# オーバーラップを増やす
uv run sahi_detect_cli.py -m best.pt -s images/ --overlap-x 40 --overlap-y 40

# 信頼度閾値を下げる
uv run sahi_detect_cli.py -m best.pt -s images/ --conf 0.3
```

---

## インストール

### 依存関係のインストール

```bash
pip install ultralytics
pip install sahi
pip install patched-yolo-infer
pip install opencv-python
pip install numpy
pip install imageio
pip install tqdm
```

または、`uv`を使用する場合:

```bash
uv sync
```

---

## 使用方法

### クイックスタート

#### Validation（評価）

```bash
# Ultralytics標準評価
python validate.py ultralytics -m runs/train/train12/weights/best.pt

# SAHI評価（スライス推論）
python validate.py sahi -y Dataset/YOLODataset_test_with_label

# 比較モード（Ultralytics vs SAHI）
python validate.py compare -y Dataset/YOLODataset_test_with_label --error-analysis
```

#### 検出（推論）

```bash
# パッチベース検出
uv run sahi_detect_cli.py -m best.pt -s detect_images/
```

詳細な使用方法は各セクションを参照してください：

- [Validation Module](#validation-module) - 詳細な使用例とパラメータ
- [Patch-Based Detection](#patch-based-detection)
- [SAHI CLI Tool](#sahi-cli-tool)

---

## 依存関係

### 主要ライブラリ

- `ultralytics` - YOLOv12/YOLOv11
- `sahi` - Slicing Aided Hyper Inference
- `patched-yolo-infer` - パッチベース推論ライブラリ
- `opencv-python` - 画像処理
- `numpy` - 数値計算
- `imageio` - GIF作成
- `tqdm` - プログレスバー

### 外部ライブラリ

#### SAHI

**Slicing Aided Hyper Inference** - 大規模物体検出とインスタンスセグメンテーションのための軽量ビジョンライブラリ

- 公式リポジトリ: [obss/sahi](https://github.com/obss/sahi)
- インストール: `pip install sahi`
- ドキュメント: [SAHI Documentation](https://github.com/obss/sahi)

#### YOLO-Patch-Based-Inference

**YOLO-Patch-Based-Inference** - インスタンスセグメンテーションタスク向けのSAHIライクな推論ライブラリ

- 公式リポジトリ: [Koldim2001/YOLO-Patch-Based-Inference](https://github.com/Koldim2001/YOLO-Patch-Based-Inference)
- インストール: `pip install patched-yolo-infer`
- サポートモデル: YOLOv8, YOLOv8-seg, YOLOv9, YOLOv9-seg, YOLOv10, YOLO11, YOLO11-seg, YOLO12, YOLO12-seg, FastSAM, RTDETR

---

## 参考資料

### 公式ドキュメント

- [Ultralytics YOLOv12](https://docs.ultralytics.com/) - YOLO公式ドキュメント
- [SAHI](https://github.com/obss/sahi) - Slicing Aided Hyper Inference
- [YOLO-Patch-Based-Inference](https://github.com/Koldim2001/YOLO-Patch-Based-Inference) - ベースライブラリ

### 関連ファイル

- `validate.py` - 統合評価CLI（全ての評価機能を統合）
- `diagnose_difference.py` - 精度差異の診断スクリプト
- `VALIDATION_GUIDE.md` - バリデーションガイド
- `SAHI_vs_PatchBased_Inference_Comparison.md` - 比較ドキュメント

---

## ライセンス

このプロジェクトは教育・研究目的で作成されています。

