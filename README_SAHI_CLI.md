# 🎯 SAHI-like Detection CLI Tool

YOLOv11を使用したパッチベース物体検出・セグメンテーションのCLIツール

## ✨ 主な機能

- 🔄 **2つのバックエンド**: `patched-yolo-infer`（高速）と公式`sahi`（標準）
- 📊 **CSV出力**: 検出結果をCSV形式で保存
- 🎬 **GIF生成**: 検出過程のアニメーション作成（patched-yoloのみ）
- 🖼️ **可視化**: バウンディングボックスとマスクの描画
- 🎛️ **柔軟な設定**: スライスサイズ、オーバーラップ、閾値など

## 🚀 クイックスタート

### 基本的な使い方

```powershell
# 単一画像の検出（デフォルト: patched-yolo）
uv run sahi_detect_cli.py -m runs/train/train12/weights/best.pt -s detect_images/image.jpg

# ディレクトリ内のすべての画像を検出
uv run sahi_detect_cli.py -m runs/train/train12/weights/best.pt -s detect_images/

# 公式SAHIバックエンドを使用
uv run sahi_detect_cli.py -m best.pt -s detect_images/ --backend sahi --save-csv

# GIF動画も作成（patched-yoloのみ）
uv run sahi_detect_cli.py -m best.pt -s image.jpg --create-gif
```

## 📁 プロジェクト構造

```
yolo12_detect/
├── sahi_detect_cli.py         # メインCLIスクリプト
├── patch_inference/           # patched-yoloモジュール
│   ├── __init__.py
│   ├── detector.py            # 検出処理（patched-yolo）
│   ├── visualizer.py          # GIF作成
│   └── README.md              # モジュールドキュメント
├── runs/
│   └── detect/
│       ├── sahi_results/      # 検出結果（自動連番）
│       └── sahi_gif/          # GIF動画（自動連番）
└── detect_images/             # 入力画像

注: 公式SAHIバックエンドは外部ライブラリ（sahi）を直接使用します
```

## 💻 コマンドライン引数

### 必須引数

| 引数 | 短縮形 | 説明 | 例 |
|------|--------|------|-----|
| `--model` | `-m` | YOLOモデルのパス | `best.pt` |
| `--source` | `-s` | 画像ファイルまたはディレクトリ | `images/` |

### オプション引数

#### 出力設定

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--output` / `-o` | `runs/detect/sahi_results` | 検出結果の出力先 |
| `--gif-output` | `runs/detect/sahi_gif` | GIF動画の出力先 |

#### パッチ設定

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--shape-x` | `400` | パッチの幅 |
| `--shape-y` | `400` | パッチの高さ |
| `--overlap-x` | `30` | X軸オーバーラップ (%) |
| `--overlap-y` | `30` | Y軸オーバーラップ (%) |

#### 推論設定

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--conf` | `0.5` | 信頼度閾値 |
| `--imgsz` | `640` | YOLO入力サイズ |
| `--device` | `0` | デバイス指定 (例: cpu, cuda:0, 0, 1) |
| `--nms-threshold` | `0.1` | NMS閾値（patched-yoloのみ） |
| `--no-batch-inference` | - | バッチ推論を無効化（patched-yoloのみ） |

#### バックエンド選択

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--backend` | `patched-yolo` | 推論バックエンド: `patched-yolo` (高速) または `sahi` (公式) |

#### GIF設定

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--create-gif` | - | GIF動画を作成 |
| `--gif-fps` | `30` | GIFのフレームレート |

#### CSV出力

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--save-csv` | - | 検出結果をCSVファイルに保存 |
| `--csv-path` | `{output_dir}/detections.csv` | CSV出力パス |

#### その他

| 引数 | 短縮形 | 説明 |
|------|--------|------|
| `--quiet` | `-q` | 詳細出力を抑制 |

## 📖 使用例

### 例1: 基本的な検出（patched-yolo）

```powershell
# デフォルトバックエンド（高速）
uv run sahi_detect_cli.py `
    --model runs/train/train12/weights/best.pt `
    --source detect_images/
```

### 例2: 公式SAHIバックエンドを使用

```powershell
# 標準的なSAHI実装
uv run sahi_detect_cli.py `
    -m best.pt `
    -s images/ `
    --backend sahi `
    --save-csv
```

### 例3: パラメータをカスタマイズ

```powershell
# スライスサイズとオーバーラップを調整
uv run sahi_detect_cli.py `
    -m best.pt `
    -s images/ `
    --backend sahi `
    --shape-x 640 `
    --shape-y 640 `
    --overlap-x 30 `
    --overlap-y 30 `
    --conf 0.5 `
    --save-csv
```

### 例4: GIF動画を作成（patched-yoloのみ）

```powershell
# 検出過程をGIFアニメーションで可視化
uv run sahi_detect_cli.py `
    -m best.pt `
    -s image.jpg `
    --create-gif `
    --gif-fps 30
```

### 例5: メモリ節約モード（patched-yoloのみ）

```powershell
# バッチ推論を無効化
uv run sahi_detect_cli.py `
    -m best.pt `
    -s images/ `
    --no-batch-inference
```

### 例6: 静かに実行（詳細出力なし）

```powershell
uv run sahi_detect_cli.py `
    -m best.pt `
    -s images/ `
    --quiet
```

### 例7: CSV形式で検出結果を保存

```powershell
# 検出カウントをCSVに保存
uv run sahi_detect_cli.py `
    -m best.pt `
    -s images/ `
    --backend sahi `
    --save-csv
```

**CSVファイル形式（detection_counts.csv）**:
```csv
image_name,total,class1,class2,class3,...
image1.jpg,15,5,7,3,...
image2.jpg,8,2,4,2,...
TOTAL,23,7,11,5,...
```

### 例8: デバイス指定

```powershell
# CPU強制
uv run sahi_detect_cli.py `
    -m best.pt `
    -s images/ `
    --device cpu `
    --save-csv

# GPU 1を使用
uv run sahi_detect_cli.py `
    -m best.pt `
    -s images/ `
    --device 1 `
    --save-csv
```

## 🔄 バックエンドの選択

### patched-yolo（デフォルト）

**特徴**:
- ✅ 高速処理
- ✅ バッチ推論対応
- ✅ GIFアニメーション生成
- ✅ メモリ最適化

**使用例**:
```powershell
uv run sahi_detect_cli.py -m best.pt -s images/ --save-csv
```

### sahi（公式）

**特徴**:
- ✅ [公式SAHI実装](https://github.com/obss/sahi)
- ✅ 標準的なアルゴリズム
- ✅ コミュニティサポート
- ✅ 研究・論文での使用に最適

**使用例**:
```powershell
uv run sahi_detect_cli.py -m best.pt -s images/ --backend sahi --save-csv
```

### バックエンド比較

| 機能 | patched-yolo | sahi |
|------|-------------|------|
| **検出精度** | 高い | 高い |
| **処理速度** | ⚡ 高速 | 標準 |
| **GIF生成** | ✅ | ❌ |
| **バッチ推論** | ✅ | ❌ |
| **CSV出力** | ✅ | ✅ |
| **標準互換** | 部分的 | ✅ 完全 |

---

## 🎬 GIF動画について（patched-yoloのみ）

`--create-gif` オプションを使用すると、検出過程を可視化したGIF動画が作成されます。

### GIFの内容

1. **元画像**: 最初に元の画像を表示
2. **パッチスキャン**: 各パッチを順番にスライド表示
   - 薄い青色でパッチを強調
   - オーバーラップ部分は濃く表示
3. **検出プロセス**: 各パッチの検出結果をリアルタイム表示
4. **NMS前**: すべてのパッチの検出結果
5. **最終結果**: NMS適用後の最終結果

### 出力先

GIF動画は専用ディレクトリに保存され、実行の度に自動で連番が付きます：

```
runs/detect/sahi_gif/          # 初回
runs/detect/sahi_gif2/         # 2回目
runs/detect/sahi_gif3/         # 3回目
...
```

**注意**: GIF生成は`patched-yolo`バックエンドのみ対応しています。

## ⚙️ パラメータの選び方

### パッチサイズ (`--shape-x`, `--shape-y`)

| サイズ | 用途 | 処理速度 | 精度 |
|--------|------|---------|------|
| **320-400** | 小物体検出 | 🐢 遅い | ⭐⭐⭐⭐⭐ |
| **512-640** | バランス型 | ⚡ 普通 | ⭐⭐⭐⭐ |
| **800+** | 大物体・高速 | 🚀 高速 | ⭐⭐⭐ |

**デフォルト**: 400×400（小物体向け）

### オーバーラップ (`--overlap-x`, `--overlap-y`)

| 比率 | 用途 | 処理速度 | 境界検出 |
|------|------|---------|---------|
| **10-20%** | 高速処理 | 🚀 高速 | ⭐⭐ |
| **25-35%** | バランス型 | ⚡ 普通 | ⭐⭐⭐⭐ |
| **40-50%** | 高精度 | 🐢 遅い | ⭐⭐⭐⭐⭐ |

**デフォルト**: 30-40%（バランス型）

### 信頼度閾値 (`--conf`)

| 閾値 | 検出数 | 偽陽性 | 用途 |
|------|--------|--------|------|
| **0.3-0.4** | 多い | 増える | 見逃し防止 |
| **0.5** | 標準 | 標準 | バランス型 |
| **0.6-0.7** | 少ない | 減る | 高精度 |

**デフォルト**: 0.5（標準）

### NMS閾値 (`--nms-threshold`)（patched-yoloのみ）

| 閾値 | 重複除去 | 検出数 |
|------|---------|--------|
| **0.1-0.2** | 厳しい | 減る |
| **0.3** | 標準 | 標準 |
| **0.4-0.5** | 緩い | 増える |

**デフォルト**: 0.1（厳しめ）

---

## 🎯 シーン別推奨設定

### シーン1: 小物体を逃したくない

```powershell
uv run sahi_detect_cli.py `
  -m best.pt `
  -s images/ `
  --backend sahi `
  --shape-x 512 `
  --shape-y 512 `
  --overlap-x 40 `
  --overlap-y 40 `
  --conf 0.3 `
  --save-csv
```

### シーン2: 偽陽性を減らしたい

```powershell
uv run sahi_detect_cli.py `
  -m best.pt `
  -s images/ `
  --backend sahi `
  --conf 0.6 `
  --save-csv
```

### シーン3: 大量の画像を高速処理

```powershell
uv run sahi_detect_cli.py `
  -m best.pt `
  -s images/ `
  --backend sahi `
  --shape-x 800 `
  --shape-y 800 `
  --overlap-x 20 `
  --overlap-y 20 `
  --save-csv
```

### シーン4: バランス型（最もおすすめ）

```powershell
uv run sahi_detect_cli.py `
  -m best.pt `
  -s images/ `
  --backend sahi `
  --shape-x 640 `
  --shape-y 640 `
  --overlap-x 30 `
  --overlap-y 30 `
  --conf 0.5 `
  --box-thickness 2 `
  --show-class-labels `
  --save-csv
```

## 🔧 トラブルシューティング

### メモリ不足エラー

```powershell
# バッチ推論を無効化（patched-yoloのみ）
uv run sahi_detect_cli.py -m best.pt -s images/ --no-batch-inference

# パッチサイズを小さくする
uv run sahi_detect_cli.py -m best.pt -s images/ --shape-x 320 --shape-y 320

# CPU強制
uv run sahi_detect_cli.py -m best.pt -s images/ --device cpu
```

### 処理が遅い

```powershell
# パッチサイズを大きくする
uv run sahi_detect_cli.py -m best.pt -s images/ --shape-x 640 --shape-y 640

# オーバーラップを減らす
uv run sahi_detect_cli.py -m best.pt -s images/ --overlap-x 20 --overlap-y 20

# patched-yoloバックエンドを使用
uv run sahi_detect_cli.py -m best.pt -s images/ --backend patched-yolo
```

### 重複検出が多い

```powershell
# オーバーラップを減らす
uv run sahi_detect_cli.py `
  -m best.pt `
  -s images/ `
  --backend sahi `
  --overlap-x 20 `
  --overlap-y 20

# スライスサイズを大きくする
uv run sahi_detect_cli.py `
  -m best.pt `
  -s images/ `
  --backend sahi `
  --shape-x 800 `
  --shape-y 800
```

### 検出漏れが多い

```powershell
# 信頼度閾値を下げる
uv run sahi_detect_cli.py -m best.pt -s images/ --conf 0.3

# オーバーラップを増やす
uv run sahi_detect_cli.py `
  -m best.pt `
  -s images/ `
  --overlap-x 40 `
  --overlap-y 40

# スライスサイズを小さくする
uv run sahi_detect_cli.py `
  -m best.pt `
  -s images/ `
  --shape-x 512 `
  --shape-y 512
```

### 小さな物体が検出されない

```powershell
# パッチサイズを小さくする
uv run sahi_detect_cli.py -m best.pt -s images/ --shape-x 320 --shape-y 320

# オーバーラップを増やす
uv run sahi_detect_cli.py -m best.pt -s images/ --overlap-x 40 --overlap-y 40

# 信頼度閾値を下げる
uv run sahi_detect_cli.py -m best.pt -s images/ --conf 0.3
```

## 📦 依存関係

### 共通
- `ultralytics` - YOLOv11/v8エンジン
- `opencv-python` - 画像処理
- `numpy` - 数値計算
- `tqdm` - プログレスバー

### patched-yoloバックエンド
- `patched-yolo-infer` - パッチベース推論ライブラリ
- `imageio` - GIF作成

### sahiバックエンド
- `sahi` - 公式SAHI実装

**インストール**:
```powershell
# 基本パッケージ
uv pip install ultralytics opencv-python numpy tqdm

# patched-yolo用
uv pip install patched-yolo-infer imageio

# 公式SAHI用
uv pip install sahi
```

## 📝 Python APIとしても使用可能

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
    batch_inference=True,
    verbose=True
)

print(f"検出完了: {len(processed_images)}枚")
print(f"保存先: {output_dir}")

# GIF作成
gif_path = create_detection_gif(
    image_path="image.jpg",
    model_path="best.pt",
    output_dir="runs/detect/gif",
    shape_x=400,
    shape_y=400,
    overlap_x=30,
    overlap_y=30,
    fps=30,
    verbose=True
)

print(f"GIF作成完了: {gif_path}")
```

## 🎓 参考

- [公式SAHI](https://github.com/obss/sahi) - Slicing Aided Hyper Inference（公式実装）
- [YOLO-Patch-Based-Inference](https://github.com/Koldim2001/YOLO-Patch-Based-Inference) - patched-yoloベースライブラリ
- [Ultralytics YOLOv11](https://docs.ultralytics.com/) - YOLO公式ドキュメント
- [SAHI論文](https://arxiv.org/abs/2202.06934) - Slicing Aided Hyper Inference for Small Object Detection

## 📄 ライセンス

このツールは教育・研究目的で作成されています。

