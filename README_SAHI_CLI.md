# 🎯 SAHI-like Detection CLI Tool

YOLOv11を使用したパッチベース物体検出・セグメンテーションのCLIツール

## 🚀 クイックスタート

### 基本的な使い方

```bash
# 単一画像の検出
uv run sahi_detect_cli.py -m runs/train/train12/weights/best.pt -s detect_images/image.jpg

# ディレクトリ内のすべての画像を検出
uv run sahi_detect_cli.py -m runs/train/train12/weights/best.pt -s detect_images/

# GIF動画も作成
uv run sahi_detect_cli.py -m best.pt -s image.jpg --create-gif
```

## 📁 プロジェクト構造

```
yolo12_detect/
├── sahi_detect_cli.py         # メインCLIスクリプト
├── patch_inference/           # モジュールディレクトリ
│   ├── __init__.py
│   ├── detector.py            # 検出処理
│   ├── visualizer.py          # GIF作成
│   └── README.md              # モジュールドキュメント
├── runs/
│   └── detect/
│       ├── sahi_results/      # 検出結果（自動連番）
│       └── sahi_gif/          # GIF動画（自動連番）
└── detect_images/             # 入力画像
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
| `--nms-threshold` | `0.3` | NMS閾値 |
| `--no-batch-inference` | - | バッチ推論を無効化（メモリ節約） |

#### GIF設定

| 引数 | デフォルト | 説明 |
|------|-----------|------|
| `--create-gif` | - | GIF動画を作成 |
| `--gif-fps` | `30` | GIFのフレームレート |

#### その他

| 引数 | 短縮形 | 説明 |
|------|--------|------|
| `--quiet` | `-q` | 詳細出力を抑制 |

## 📖 使用例

### 例1: 基本的な検出

```bash
uv run sahi_detect_cli.py \
    --model runs/train/train12/weights/best.pt \
    --source detect_images/
```

### 例2: パラメータをカスタマイズ

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

### 例3: GIF動画を作成

```bash
uv run sahi_detect_cli.py \
    -m best.pt \
    -s image.jpg \
    --create-gif \
    --gif-fps 30
```

### 例4: メモリ節約モード（バッチ推論なし）

```bash
uv run sahi_detect_cli.py \
    -m best.pt \
    -s images/ \
    --no-batch-inference
```

### 例5: 静かに実行（詳細出力なし）

```bash
uv run sahi_detect_cli.py \
    -m best.pt \
    -s images/ \
    --quiet
```

## 🎬 GIF動画について

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

## ⚙️ パラメータの選び方

### パッチサイズ (`--shape-x`, `--shape-y`)

- **小さい (320-400)**: 小さな物体の検出に有効、処理時間が長い
- **大きい (512-640)**: 高速処理、大きな物体向け

### オーバーラップ (`--overlap-x`, `--overlap-y`)

- **小さい (10-20%)**: 高速処理、境界での見逃しが増える可能性
- **大きい (30-50%)**: 高精度、処理時間が長い

### 信頼度閾値 (`--conf`)

- **低い (0.3-0.4)**: より多くの検出、誤検出も増える
- **高い (0.6-0.7)**: 確実な検出のみ、見逃しが増える可能性

### NMS閾値 (`--nms-threshold`)

- **低い (0.1-0.2)**: 重複を厳しく除去、検出数が減る
- **高い (0.4-0.5)**: 重複を許容、検出数が増える

## 🔧 トラブルシューティング

### メモリ不足エラー

```bash
# バッチ推論を無効化
uv run sahi_detect_cli.py -m best.pt -s images/ --no-batch-inference

# パッチサイズを小さくする
uv run sahi_detect_cli.py -m best.pt -s images/ --shape-x 320 --shape-y 320
```

### 処理が遅い

```bash
# パッチサイズを大きくする
uv run sahi_detect_cli.py -m best.pt -s images/ --shape-x 512 --shape-y 512

# オーバーラップを減らす
uv run sahi_detect_cli.py -m best.pt -s images/ --overlap-x 20 --overlap-y 20
```

### 小さな物体が検出されない

```bash
# パッチサイズを小さくする
uv run sahi_detect_cli.py -m best.pt -s images/ --shape-x 320 --shape-y 320

# オーバーラップを増やす
uv run sahi_detect_cli.py -m best.pt -s images/ --overlap-x 40 --overlap-y 40

# 信頼度閾値を下げる
uv run sahi_detect_cli.py -m best.pt -s images/ --conf 0.3
```

## 📦 依存関係

- `ultralytics` - YOLOv11
- `patched-yolo-infer` - パッチベース推論ライブラリ
- `opencv-python` - 画像処理
- `numpy` - 数値計算
- `imageio` - GIF作成
- `tqdm` - プログレスバー

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

- [YOLO-Patch-Based-Inference](https://github.com/Koldim2001/YOLO-Patch-Based-Inference) - ベースライブラリ
- [Ultralytics YOLOv11](https://docs.ultralytics.com/) - YOLO公式ドキュメント
- [SAHI](https://github.com/obss/sahi) - Slicing Aided Hyper Inference

## 📄 ライセンス

このツールは教育・研究目的で作成されています。

