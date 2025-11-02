# 🎬 GIF作成の詳細な処理フロー

## 📊 現在の問題

**症状**: 
- 通常検出: 62個
- GIF用検出: 63個（1個多い）
- 最終フレームに不要なバウンディングボックスが残る

**原因**:
バッチ推論と順次推論で、推論結果がわずかに異なるため

---

## 🔄 GIF作成の処理ステップ

### ステップ1: モデル読み込みと画像準備
```python
model = YOLO(model_path)
test_img = cv2.imread(image_path)  # 元画像 (958x1065)
```

### ステップ2: パッチベース推論
```python
element_crops = MakeCropsDetectThem(
    image=test_img,
    model=model,
    batch_inference=True,  # ← ここが重要！
    ...
)
```

**重要ポイント**:
- `batch_inference=True`: すべてのパッチを一度に推論（GPU効率的、結果が安定）
- `batch_inference=False`: パッチを1つずつ推論（わずかに結果が異なる）

**内部処理**:
1. 画像を `working_img` (680x960) にリサイズ
2. パッチに分割（例: 6個）
3. 各パッチでYOLO推論
4. 結果を元の座標系に戻す

### ステップ3: NMS（重複除去）
```python
result = CombineDetections(
    element_crops, 
    nms_threshold=0.2,        # IoU閾値
    class_agnostic_nms=True,  # クラス間でもNMS適用
)
```

**処理内容**:
- NMS前: 122個の検出
- NMS後: 62個の検出（60個削減）

### ステップ4: 最終画像の生成
```python
# 完全に新しいクリーンな画像を読み込み
clean_img_for_final = cv2.imread(image_path)
clean_img_for_final = cv2.resize(clean_img_for_final, (680, 960))

# NMS後の結果のみを描画
final_img = visualize_results(
    img=clean_img_for_final,  # ← クリーンな画像
    boxes=result.filtered_boxes,  # ← NMS後の62個のみ
    ...
)
```

**重要ポイント**:
- 完全に新しい画像から描画開始
- NMS後の結果（62個）のみを描画
- 古い検出結果は含まれない

### ステップ5: フレーム生成

#### 5-1. 元画像 (15フレーム)
```python
frames.extend([base_rgb] * 15)
```

#### 5-2. パッチスキャン（各パッチごと）
```python
for idx, crop in element_crops.crops:
    # スライドアニメーション（8フレーム）
    # 検出中（5フレーム）
    # フェードアウト（8フレーム）
    # 停止（2フレーム）
    # 検出結果フェードイン（8フレーム）
    # 停止（2フレーム）
```

#### 5-3. NMS前の全検出（15フレーム）
```python
# すべての検出（122個）を表示
nms_before_img = draw_detections(working_img, all_detections)
frames.extend([nms_before_img_rgb] * 15)
```

#### 5-4. 遷移（12フレーム）
```python
for step in range(11):  # 最後の1フレームを除く
    # パッチグリッドをフェードアウト
    # 最終結果（62個）をフェードイン
    transition = blend(clean_img, final_img, alpha)
    frames.append(transition)

# 最終フレームは完全にfinal_img（62個のみ）
frames.append(final_rgb)  # ← ここが最終フレーム！
```

#### 5-5. 最終結果（60フレーム）
```python
# NMS後の結果（62個）を長く表示
frames.extend([final_rgb] * 60)
```

### ステップ6: リサイズとGIF保存
```python
# 元のサイズ (958x1065) にリサイズ
resized_frames = [cv2.resize(f, (958, 1065)) for f in frames]

# GIF保存
imageio.mimsave(gif_path, resized_frames, fps=30)
```

---

## 🔍 問題の診断フロー

### 問題1: 検出数が異なる
**原因**: `batch_inference`の設定が異なる

**修正**:
```python
# 修正前
batch_inference=False  # 順次処理 → 63個

# 修正後
batch_inference=True   # バッチ処理 → 62個 ✅
```

### 問題2: 最終フレームに不要なボックス
**原因**: 遷移時に古い画像をブレンドしていた

**修正**:
```python
# 修正前
# NMS前の画像（122個付き）から遷移
transition = blend(nms_before_img, final_img)  # ❌

# 修正後
# クリーンな画像から遷移
transition = blend(clean_img, final_img)  # ✅
```

### 問題3: 遷移の最後が不完全
**原因**: ループの最後が `alpha=1.0` にならない

**修正**:
```python
# 修正前
for step in range(12):  # alpha = 11/12 = 0.917 が最大 ❌
    ...

# 修正後
for step in range(11):  # alpha = 10/11 = 0.909
    ...
frames.append(final_rgb)  # 完全なfinal_img ✅
```

---

## ✅ 正しい結果の確認方法

### 1. 検出数の確認
```bash
uv run sahi_detect_cli.py -m best.pt -s image.jpg
# 出力: pod: 62個

uv run sahi_detect_cli.py -m best.pt -s image.jpg --create-gif
# 出力: 🎯 検出数: 62個
```
→ **両方が62個なら正しい** ✅

### 2. 画像の確認
```bash
uv run debug_detection.py
```
出力を確認:
```
通常検出: 62個
GIF用検出: 62個
✅ 検出数は一致
✅ すべての座標が一致
画像差分の合計: 0 (または非常に小さい値)
✅ 完全に一致！
```

### 3. GIFの目視確認
1. 最初: 元画像のみ
2. 中盤: パッチスキャン + 検出が追加される
3. NMS前: 多くの検出（122個）
4. **最終**: 少ない検出（62個のみ）← ここをチェック！

---

## 🎯 最終チェックリスト

- [ ] `batch_inference=True` に設定
- [ ] 検出数が通常検出と一致（62個）
- [ ] 最終フレームがクリーンな画像から生成
- [ ] 遷移の最後が完全に `final_rgb`
- [ ] GIF最終フレームに不要なボックスがない

---

## 📝 現在のコード状態

### visualizer.py (Line 182-196)
```python
element_crops = MakeCropsDetectThem(
    image=test_img,
    model=model,
    segment=True,
    shape_x=shape_x,
    shape_y=shape_y,
    overlap_x=overlap_x,
    overlap_y=overlap_y,
    conf=conf_threshold,
    imgsz=imgsz,
    show_crops=False,
    memory_optimize=False,
    batch_inference=True,  # ✅ バッチ処理
    show_processing_status=False,
)
```

### visualizer.py (Line 436-458)
```python
# 遷移フレーム
for step in range(FRAME_TRANSITION - 1):  # 11フレーム
    alpha = (step + 1) / FRAME_TRANSITION
    ...
    transition = cv2.addWeighted(clean_img, 1-alpha, final_rgb, alpha, 0)
    frames.append(transition)

# 最終フレーム（完全にfinal_rgb）
frames.append(final_rgb.copy())  # ✅ 完全な最終結果

# 最終結果を長く表示
frames.extend([final_rgb.copy()] * FRAME_FINAL)  # 60フレーム
```

---

## 🐛 デバッグのポイント

最新のGIFが正しいか確認:
```bash
# 最新のGIFを確認
ls -lt runs/detect/sahi_gif*/Image_000003_detection.gif | head -1

# デバッグスクリプトが最新のGIFを参照しているか確認
uv run debug_detection.py
```

もしまだ問題がある場合:
1. 古いGIFを削除: `rm -rf runs/detect/sahi_gif*`
2. 新規作成: `uv run sahi_detect_cli.py ... --create-gif`
3. 再確認: `uv run debug_detection.py`

