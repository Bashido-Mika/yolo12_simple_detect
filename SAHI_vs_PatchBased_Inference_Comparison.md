# SAHI vs YOLO-Patch-Based-Inference 詳細比較

## 📊 概要

両ライブラリとも**スライス推論**を使用して小物体検出の精度を向上させる。

### 基本的なアルゴリズム

```
1. 大きな画像を小さなパッチ/スライスに分割
2. 各パッチで推論を実行
3. 座標を元画像サイズに変換
4. 重複検出を除去（NMS等）
5. 結果を統合
```

---

## 🔍 主な違い

### 1. **標準予測とのマージ機能**

#### SAHI ✅
- **デフォルトで標準予測も実行**
- 大物体の検出精度向上

```python
# SAHIでは自動的に両方実行
perform_standard_pred=True  # デフォルト

# 大画像全体での推論 + スライス推論
object_prediction_list.extend(standard_predictions)
```

**効果:**
- 大きな物体: 標準予測が良好
- 小さな物体: スライス予測が良好
- **両方のメリットを統合**

#### YOLO-Patch-Based-Inference ❌
- スライス推論のみ
- 標準予測とのマージ機能なし

---

### 2. **Postprocessing アルゴリズム**

#### SAHI
- **GREEDYNMM** (デフォルト)
- **NMM** (NMS + Matching)
- 標準 **NMS**

```python
# 3つのアルゴリズムから選択可能
postprocess_type='GREEDYNMM'  # 推奨（デフォルト）
postprocess_type='NMM'
postprocess_type='NMS'
```

**GREEDYNMMの特徴:**
- Greedy Non-Maximum Matching
- 重複検出の高精度除去
- 小物体検出に強い

#### YOLO-Patch-Based-Inference
- 独自のNMS実装
- **Intelligent Sorter**機能あり
- GREEDYNMMなし

```python
intelligent_sorter=True   # 面積と信頼度でソート
sorter_bins=5             # ビン数で調整
```

---

### 3. **Match Metric (IOU vs IOS)**

#### SAHI
- **IOS** がデフォルト（Intersection Over Smaller）

```python
match_metric='IOS'  # デフォルト
match_metric='IOU'  # 選択可能
```

**IOSの利点:**
- 小物体の重複検出に有効
- サイズの異なる物体でも精度が高い

#### YOLO-Patch-Based-Inference
- IOU/IOS 両方サポート
- IOSはサポート

---

### 4. **メモリ最適化**

#### SAHI
- 通常: 全マスクを保持
- 精度重視

#### YOLO-Patch-Based-Inference
- **`memory_optimize=True`** オプション
- ポリゴンのみ保存
- メモリ使用量を削減

```python
memory_optimize=True   # 軽量化
memory_optimize=False  # 精度重視（デフォルト）
```

---

### 5. **フレームワーク対応**

#### SAHI 🌟
**多フレームワーク対応:**
- ✅ Ultralytics (YOLOv8/9/10/11/12)
- ✅ HuggingFace
- ✅ YOLOv5
- ✅ MMDetection
- ✅ RT-DETR
- ✅ Roboflow/RF-DETR
- ✅ TorchVision
- ✅ YOLOX

#### YOLO-Patch-Based-Inference
- ✅ Ultralytics専用
- 他のフレームワーク非対応

---

### 6. **追加機能**

#### SAHI 🌟🌟🌟
- ✅ COCO評価 (`sahi coco evaluate`)
- ✅ **エラー分析** (`sahi coco analyse`)
- ✅ データセット変換 (YOLO↔COCO)
- ✅ FiftyOne連携
- ✅ CLIツール充実
- ✅ 論文・コミュニティ実績

#### YOLO-Patch-Based-Inference
- ✅ カスタム可視化
- ✅ バッチ推論
- ✅ GIF生成
- ❌ 評価機能なし
- ❌ エラー分析なし

---

## 🎯 使用ケース別推奨

### SAHI が推奨される場合

1. **研究・開発**
   - 論文レベルの評価
   - エラー分析が必要
   - 複数フレームワークを使う

2. **プロダクション**
   - COCO 評価が必須
   - 標準＋スライスの併用

3. **柔軟性**
   - アルゴリズム切り替え
   - 将来のモデル追加

### YOLO-Patch-Based-Inference が推奨される場合

1. **Ultralytics専用**
   - 他フレームワーク不使用
   - シンプル統合

2. **メモリ制約**
   - 限られたリソース
   - `memory_optimize=True`

3. **カスタム可視化**
   - 専用可視化
   - GIF生成

---

## 📈 精度比較の実際

### あなたの実験結果 (1024x1024)

| ライブラリ | mAP50-95 | 特徴 |
|-----------|---------|------|
| **Ultralytics標準** | 47.13% | **真のベンチマーク** |
| SAHI (512スライス) | 16.10% | スライス推論のみ |
| SAHI (標準+スライス) | ??? | より高精度 |

**なぜSAHIの値が低いのか:**
- SAHIはCOCO評価ベース
- NMS/Postprocessingの違い
- 絶対値より**相対比較**に価値あり

---

## 💡 まとめ

### SAHI の強み
- より洗練されたアルゴリズム
- 標準予測との統合
- エラーの定量化
- 多フレームワーク
- 評価・分析ツール

### YOLO-Patch-Based-Inference の強み
- シンプル
- メモリ軽量
- Ultralytics特化

### あなたのプロジェクトでの選択

✅ **SAHI**: 評価・分析重視  
✅ **YOLO-Patch-Based-Inference**: 軽量・シンプル重視

---

## 🔗 参考リンク

- [SAHI GitHub](https://github.com/obss/sahi)
- [SAHI Documentation](https://obss.github.io/sahi/)
- [SAHI Paper (ICIP 2022)](https://arxiv.org/abs/2202.06934)
- [YOLO-Patch-Based-Inference GitHub](https://github.com/Koldim2001/YOLO-Patch-Based-Inference)

