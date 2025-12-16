"""
SAHI validation module.

This module provides SAHI-based validation with sliced inference for YOLO models.
"""

import json
import sys
import time
import csv
from pathlib import Path
from dataclasses import dataclass, field
from typing import Optional, Dict, Any
from collections import defaultdict, Counter

import numpy as np
from tqdm import tqdm

from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction
from sahi.scripts.coco_evaluation import evaluate as coco_evaluate
from sahi.scripts.coco_error_analysis import analyse as coco_analyse
from sahi.utils.coco import Coco

from validation.utils.coco_converter import YOLOToCOCOConverter, get_incremental_dir


@dataclass
class SAHIValidationConfig:
    """SAHI validation configuration"""
    
    model_path: str = "runs/train/train12/weights/best.pt"
    yolo_dataset_dir: Optional[str] = None
    dataset_json_path: Optional[str] = None
    image_dir: Optional[str] = None
    output_dir: str = "runs/val"
    experiment_name: str = "sahi_val"
    auto_increment: bool = True
    
    # Model parameters
    confidence_threshold: float = 0.25
    device: str = "0"
    image_size: int = 640
    
    # Sliced prediction
    use_sliced_prediction: bool = True
    compare_with_standard: bool = False
    slice_height: int = 512
    slice_width: int = 512
    overlap_height_ratio: float = 0.5
    overlap_width_ratio: float = 0.5
    
    # Postprocessing
    postprocess_type: str = "GREEDYNMM"
    postprocess_match_metric: str = "IOS"
    postprocess_match_threshold: float = 0.5
    postprocess_class_agnostic: bool = False
    
    # Evaluation
    iou_thrs: Optional[list[float]] = None
    max_detections: int = 500
    classwise: bool = False
    areas: list[int] = field(default_factory=lambda: [1024, 9216, 10000000000])
    
    # Visualization
    export_visuals: bool = False
    visual_export_dir: Optional[str] = None
    visual_bbox_thickness: int = 2
    visual_text_size: float = 0.5
    visual_hide_labels: bool = False
    visual_hide_conf: bool = False
    max_visual_samples: Optional[int] = None
    
    # Error analysis
    error_analysis: bool = False
    
    # Prediction only mode
    predict_only: bool = False
    
    # CSV export
    save_csv: bool = False
    csv_path: Optional[str] = None
    
    # Other
    verbose: int = 1
    
    def __post_init__(self):
        """設定の検証"""
        if not Path(self.model_path).exists():
            raise FileNotFoundError(f"モデルファイルが見つかりません: {self.model_path}")
        
        # 自動インクリメント機能
        if self.auto_increment:
            self.output_dir = str(get_incremental_dir(self.output_dir, self.experiment_name))
        
        # YOLO形式の場合は変換を実行
        if self.yolo_dataset_dir:
            if not Path(self.yolo_dataset_dir).exists():
                raise FileNotFoundError(f"YOLOデータセットディレクトリが見つかりません: {self.yolo_dataset_dir}")
            
            # 一時的なCOCO JSONファイルを作成
            if self.dataset_json_path is None:
                temp_dir = Path(self.output_dir) / "temp"
                temp_dir.mkdir(parents=True, exist_ok=True)
                self.dataset_json_path = str(temp_dir / "dataset.json")
            
            # 変換を実行（まだ変換されていない場合、または古い形式の場合）
            need_conversion = True
            if Path(self.dataset_json_path).exists():
                # 既存のJSONファイルをチェックして、古い形式かどうか確認
                try:
                    with open(self.dataset_json_path, "r", encoding="utf-8") as f:
                        existing_json = json.load(f)
                    # file_nameにスラッシュが含まれている場合は古い形式
                    if "images" in existing_json and len(existing_json["images"]) > 0:
                        first_file_name = existing_json["images"][0].get("file_name", "")
                        if "/" in first_file_name or "\\" in first_file_name:
                            # 古い形式なので削除して再生成
                            if self.verbose >= 1:
                                print(f"古い形式のCOCO JSONを検出しました。再生成します: {self.dataset_json_path}")
                            Path(self.dataset_json_path).unlink()
                        else:
                            # 新しい形式なので再生成不要
                            need_conversion = False
                except Exception as e:
                    if self.verbose >= 1:
                        print(f"既存のJSONファイルの読み込みエラー: {e}。再生成します。")
            
            if need_conversion:
                if self.verbose >= 1:
                    print(f"YOLO形式からCOCO形式に変換中...")
                converter = YOLOToCOCOConverter(self.yolo_dataset_dir, split="val")
                if self.verbose >= 1:
                    print(f"  使用するYAML: {converter.data_yaml_path.name}")
                self.dataset_json_path = converter.convert(self.dataset_json_path)
            
            # 画像ディレクトリを設定
            if self.image_dir is None:
                self.image_dir = str(Path(self.yolo_dataset_dir) / "images" / "val")
        else:
            # COCO形式の場合
            if self.dataset_json_path and not Path(self.dataset_json_path).exists():
                raise FileNotFoundError(f"データセットJSONが見つかりません: {self.dataset_json_path}")
            
            if self.image_dir and not Path(self.image_dir).exists():
                raise FileNotFoundError(f"画像ディレクトリが見つかりません: {self.image_dir}")
        
        # 可視化ディレクトリの設定
        if self.export_visuals:
            if self.visual_export_dir is None:
                self.visual_export_dir = str(Path(self.output_dir) / "visuals")
            Path(self.visual_export_dir).mkdir(parents=True, exist_ok=True)


class SAHISegmentationValidator:
    """SAHIを使用したセグメンテーションvalidationクラス"""

    def __init__(self, config: SAHIValidationConfig):
        """
        Args:
            config: SAHIValidationConfig
                Validation設定
        """
        self.config = config
        self.detection_model: Optional[AutoDetectionModel] = None
        self.coco: Optional[Coco] = None
        self.predictions_json: list[dict] = []

    def load_model(self) -> None:
        """モデルを読み込む"""
        if self.config.verbose >= 1:
            print(f"モデルを読み込み中: {self.config.model_path}")

        self.detection_model = AutoDetectionModel.from_pretrained(
            model_type="ultralytics",
            model_path=self.config.model_path,
            confidence_threshold=self.config.confidence_threshold,
            device=self.config.device,
            load_at_init=True,
            image_size=self.config.image_size,
        )

        # セグメンテーションモデルか確認
        if not self.detection_model.has_mask:
            raise ValueError(
                "指定されたモデルはセグメンテーションモデルではありません。"
                "セグメンテーションモデル（例: yolo11n-seg.pt）を使用してください。"
            )

        if self.config.verbose >= 1:
            print("モデルの読み込みが完了しました")

    def load_dataset(self) -> None:
        """データセットを読み込む"""
        if self.config.verbose >= 1:
            print(f"データセットを読み込み中: {self.config.dataset_json_path}")

        self.coco = Coco.from_coco_dict_or_path(self.config.dataset_json_path)

        if self.config.verbose >= 1:
            print(f"データセット読み込み完了: {len(self.coco.images)}枚の画像")

    def predict_images(self) -> None:
        """画像に対して推論を実行"""
        if self.detection_model is None:
            raise RuntimeError("モデルが読み込まれていません。load_model()を先に実行してください。")
        if self.coco is None:
            raise RuntimeError("データセットが読み込まれていません。load_dataset()を先に実行してください。")

        if self.config.verbose >= 1:
            print("推論を開始します...")

        self.predictions_json = []
        self.detection_stats = defaultdict(lambda: defaultdict(int))  # 画像ごとのクラス別カウント
        
        image_paths = [
            str(Path(self.config.image_dir) / coco_image.file_name)
            for coco_image in self.coco.images
        ]

        iterator = tqdm(enumerate(self.coco.images), total=len(self.coco.images), disable=self.config.verbose == 0)

        success_count = 0
        error_count = 0
        visual_count = 0  # 可視化した画像の数

        for img_idx, coco_image in iterator:
            image_path = image_paths[img_idx]

            if not Path(image_path).exists():
                if self.config.verbose >= 1:
                    tqdm.write(f"警告: 画像が見つかりません: {image_path}")
                error_count += 1
                continue

            try:
                # 詳細モードで処理中の画像を表示
                if self.config.verbose >= 2:
                    tqdm.write(f"処理中: {Path(image_path).name}")
                # 推論実行
                if self.config.use_sliced_prediction:
                    prediction_result = get_sliced_prediction(
                        image=image_path,
                        detection_model=self.detection_model,
                        slice_height=self.config.slice_height,
                        slice_width=self.config.slice_width,
                        overlap_height_ratio=self.config.overlap_height_ratio,
                        overlap_width_ratio=self.config.overlap_width_ratio,
                        perform_standard_pred=True,
                        postprocess_type=self.config.postprocess_type,
                        postprocess_match_metric=self.config.postprocess_match_metric,
                        postprocess_match_threshold=self.config.postprocess_match_threshold,
                        postprocess_class_agnostic=self.config.postprocess_class_agnostic,
                        verbose=0,
                    )
                else:
                    prediction_result = get_prediction(
                        image=image_path,
                        detection_model=self.detection_model,
                        shift_amount=[0, 0],
                        full_shape=None,
                        postprocess=None,
                        verbose=0,
                    )

                # COCO形式に変換とカウント
                image_name = Path(image_path).name
                for object_prediction in prediction_result.object_prediction_list:
                    coco_prediction = object_prediction.to_coco_prediction(image_id=coco_image.id)
                    coco_prediction_json = coco_prediction.json
                    if coco_prediction_json.get("bbox") or coco_prediction_json.get("segmentation"):
                        self.predictions_json.append(coco_prediction_json)
                        # クラスごとのカウント
                        class_name = object_prediction.category.name
                        self.detection_stats[image_name][class_name] += 1
                        self.detection_stats[image_name]['total'] += 1
                
                # 可視化の実行
                if self.config.export_visuals:
                    # max_visual_samplesが設定されている場合、その数まで可視化
                    should_visualize = (
                        self.config.max_visual_samples is None or
                        visual_count < self.config.max_visual_samples
                    )
                    
                    if should_visualize:
                        try:
                            # 可視化用のファイル名を生成（拡張子なし、export_visualsが自動的に.pngを追加）
                            image_filename = Path(image_path).stem
                            visual_filename = f"{image_filename}_prediction"
                            
                            # 可視化を実行して保存
                            prediction_result.export_visuals(
                                export_dir=self.config.visual_export_dir,
                                file_name=visual_filename,
                                text_size=self.config.visual_text_size,
                                rect_th=self.config.visual_bbox_thickness,
                                hide_labels=self.config.visual_hide_labels,
                                hide_conf=self.config.visual_hide_conf,
                            )
                            
                            visual_count += 1
                            
                            if self.config.verbose >= 2:
                                tqdm.write(f"可視化保存: {visual_filename}.png")
                        except Exception as e:
                            # エラーを常に表示（verbose >= 1）
                            if self.config.verbose >= 1:
                                tqdm.write(f"可視化エラー {Path(image_path).name}: {str(e)}")
                            if self.config.verbose >= 2:
                                import traceback
                                tqdm.write(traceback.format_exc())
                
                success_count += 1
                
            except KeyboardInterrupt:
                if self.config.verbose >= 1:
                    print("\n\n推論が中断されました（Ctrl+C）")
                    print(f"処理済み: {success_count + error_count}/{len(self.coco.images)}枚")
                raise
            except Exception as e:
                error_count += 1
                error_msg = f"エラー: 画像 {Path(image_path).name} の推論に失敗しました: {str(e)}"
                print(f"\n{error_msg}", file=sys.stderr)
                sys.stderr.flush()
                if self.config.verbose >= 2:
                    import traceback
                    print(traceback.format_exc(), file=sys.stderr)
                    sys.stderr.flush()
                continue

        if self.config.verbose >= 1:
            print(f"\n推論完了: {len(self.predictions_json)}個の予測結果")
            print(f"成功: {success_count}枚, エラー: {error_count}枚")
            if self.config.export_visuals:
                print(f"可視化: {visual_count}枚 (保存先: {self.config.visual_export_dir})")
        
        # CSV保存
        if self.config.save_csv:
            self.save_detection_csv()

    def save_detection_csv(self) -> None:
        """検出結果をCSVファイルに保存"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)
        
        csv_file = output_path / (self.config.csv_path if self.config.csv_path else "detection_counts.csv")
        
        # 全クラス名を収集
        all_classes = set()
        for image_stats in self.detection_stats.values():
            all_classes.update(k for k in image_stats.keys() if k != 'total')
        all_classes = sorted(all_classes)
        
        # CSVヘッダー
        fieldnames = ['image_name', 'total'] + all_classes
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            # 各画像の検出数を書き込み
            for image_name in sorted(self.detection_stats.keys()):
                row = {'image_name': image_name}
                row['total'] = self.detection_stats[image_name].get('total', 0)
                for class_name in all_classes:
                    row[class_name] = self.detection_stats[image_name].get(class_name, 0)
                writer.writerow(row)
            
            # 合計行を追加
            total_row = {'image_name': 'TOTAL'}
            total_row['total'] = sum(stats.get('total', 0) for stats in self.detection_stats.values())
            for class_name in all_classes:
                total_row[class_name] = sum(stats.get(class_name, 0) for stats in self.detection_stats.values())
            writer.writerow(total_row)
        
        if self.config.verbose >= 1:
            print(f"📊 検出カウントCSVを保存しました: {csv_file}")
            print(f"   総検出数: {total_row['total']}個")
            print(f"   画像数: {len(self.detection_stats)}枚")

    def save_predictions(self) -> str:
        """予測結果をJSONファイルに保存"""
        output_path = Path(self.config.output_dir)
        output_path.mkdir(parents=True, exist_ok=True)

        predictions_file = output_path / "predictions.json"
        with open(predictions_file, "w", encoding="utf-8") as f:
            json.dump(self.predictions_json, f, indent=2)

        if self.config.verbose >= 1:
            print(f"予測結果を保存しました: {predictions_file}")

        return str(predictions_file)

    def evaluate(self, predictions_file: str) -> dict:
        """COCO評価を実行（Precision/Recall含む）"""
        if self.config.verbose >= 1:
            print("評価を実行中...")

        # 基本的なCOCO評価
        result = coco_evaluate(
            dataset_json_path=self.config.dataset_json_path,
            result_json_path=predictions_file,
            out_dir=self.config.output_dir,
            type="segm",  # セグメンテーション評価
            classwise=self.config.classwise,
            max_detections=self.config.max_detections,
            iou_thrs=self.config.iou_thrs,
            areas=self.config.areas,
            return_dict=True,
        )

        # Precision/Recallを計算
        pr_metrics = self._calculate_precision_recall(predictions_file)
        result['eval_results'].update(pr_metrics)

        if self.config.verbose >= 1:
            print(f"評価結果を保存しました: {result['export_path']}")
        
        # エラー解析プロットを生成
        if self.config.error_analysis:
            if self.config.verbose >= 1:
                print("\nエラー解析プロットを生成中...")
            
            try:
                error_analysis_result = coco_analyse(
                    dataset_json_path=self.config.dataset_json_path,
                    result_json_path=predictions_file,
                    out_dir=self.config.output_dir,
                    type="segm",
                    no_extraplots=False,  # 追加のプロットも生成
                    areas=self.config.areas,
                    max_detections=self.config.max_detections,
                    return_dict=True,
                )
                
                if self.config.verbose >= 1:
                    print(f"エラー解析プロットを保存しました: {self.config.output_dir}")
                    print("\n[エラー解析の見方]")
                    print("  C75: 0.75 IoU閾値での結果")
                    print("  C50: 0.50 IoU閾値での結果")
                    print("  Loc: ローカリゼーションエラーを無視した結果")
                    print("  Sim: スーパーカテゴリの誤検出を無視した結果")
                    print("  Oth: すべてのカテゴリ混同を無視した結果")
                    print("  BG:  すべての偽陽性を無視した結果")
                    print("  FN:  すべての偽陰性を無視した結果")
                    print("\n[改善のポテンシャル]")
                    print("  C75-C50, C50-Loc: より正確なBBox予測での改善可能性")
                    print("  Loc-Sim: スーパーカテゴリ混同の修正での改善可能性")
                    print("  Loc-Oth: カテゴリ混同の修正での改善可能性")
                    print("  Oth-BG:  偽陽性の修正での改善可能性")
                    print("  BG-FN:   偽陰性の修正での改善可能性")
                
                result['error_analysis'] = error_analysis_result
            except Exception as e:
                if self.config.verbose >= 1:
                    print(f"警告: エラー解析に失敗しました: {e}")
                    if self.config.verbose >= 2:
                        import traceback
                        traceback.print_exc()

        return result
    
    def _calculate_precision_recall(self, predictions_file: str) -> dict:
        """Precision/Recallを計算（Ultralyticsスタイル）"""
        from pycocotools.coco import COCO
        from pycocotools.cocoeval import COCOeval
        
        # COCO評価の準備
        coco_gt = COCO(self.config.dataset_json_path)
        coco_dt = coco_gt.loadRes(predictions_file)
        
        # セグメンテーション評価
        coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
        coco_eval.params.maxDets = [self.config.max_detections]
        coco_eval.evaluate()
        coco_eval.accumulate()
        
        # Precision/Recallを取得
        # precision: [TxRxKxAxM] - T=IoU閾値, R=Recall閾値, K=カテゴリ, A=エリア, M=maxDets
        precision = coco_eval.eval['precision']
        recall = coco_eval.eval['recall']
        
        # IoU=0.5でのPrecision/Recall（mAP50相当）
        # IoU閾値のインデックス: 0=0.5, 5=0.75
        iou_50_idx = 0  # IoU=0.5
        
        # 全カテゴリ、全エリア、maxDetsでの平均
        # precision[iou_idx, :, :, area_idx, maxdet_idx]
        p_iou50 = precision[iou_50_idx, :, :, 0, 0]  # area=all
        p_iou50 = p_iou50[p_iou50 > -1]  # -1を除外
        precision_50 = np.mean(p_iou50) if len(p_iou50) > 0 else 0.0
        
        # IoU=0.5:0.95での平均Precision
        p_all = precision[:, :, :, 0, 0]  # 全IoU、area=all
        p_all = p_all[p_all > -1]
        precision_avg = np.mean(p_all) if len(p_all) > 0 else 0.0
        
        # Recall（IoU=0.5:0.95）
        # recall: [TxKxAxM]
        r_all = recall[:, :, 0, 0]  # 全IoU、area=all
        r_all = r_all[r_all > -1]
        recall_avg = np.mean(r_all) if len(r_all) > 0 else 0.0
        
        # IoU=0.5でのRecall
        r_iou50 = recall[iou_50_idx, :, 0, 0]
        r_iou50 = r_iou50[r_iou50 > -1]
        recall_50 = np.mean(r_iou50) if len(r_iou50) > 0 else 0.0
        
        # ボックス評価も追加
        coco_eval_box = COCOeval(coco_gt, coco_dt, 'bbox')
        coco_eval_box.params.maxDets = [self.config.max_detections]
        coco_eval_box.evaluate()
        coco_eval_box.accumulate()
        
        precision_box = coco_eval_box.eval['precision']
        recall_box = coco_eval_box.eval['recall']
        
        p_box_iou50 = precision_box[iou_50_idx, :, :, 0, 0]
        p_box_iou50 = p_box_iou50[p_box_iou50 > -1]
        precision_box_50 = np.mean(p_box_iou50) if len(p_box_iou50) > 0 else 0.0
        
        p_box_all = precision_box[:, :, :, 0, 0]
        p_box_all = p_box_all[p_box_all > -1]
        precision_box_avg = np.mean(p_box_all) if len(p_box_all) > 0 else 0.0
        
        r_box_all = recall_box[:, :, 0, 0]
        r_box_all = r_box_all[r_box_all > -1]
        recall_box_avg = np.mean(r_box_all) if len(r_box_all) > 0 else 0.0
        
        r_box_iou50 = recall_box[iou_50_idx, :, 0, 0]
        r_box_iou50 = r_box_iou50[r_box_iou50 > -1]
        recall_box_50 = np.mean(r_box_iou50) if len(r_box_iou50) > 0 else 0.0
        
        return {
            # セグメンテーション（Mask）
            "segm_precision": precision_avg,
            "segm_recall": recall_avg,
            "segm_precision50": precision_50,
            "segm_recall50": recall_50,
            # バウンディングボックス（Box）
            "bbox_precision": precision_box_avg,
            "bbox_recall": recall_box_avg,
            "bbox_precision50": precision_box_50,
            "bbox_recall50": recall_box_50,
        }

    def run(self) -> dict:
        """Validationを実行"""
        start_time = time.time()

        try:
            # モデル読み込み
            self.load_model()

            # データセット読み込み
            self.load_dataset()

            # 推論実行
            self.predict_images()

            # 予測結果が0件の場合の警告
            if len(self.predictions_json) == 0:
                print("\n警告: 予測結果が0件です。すべての画像で推論に失敗したか、検出がありませんでした。", file=sys.stderr)

            # 予測結果保存
            predictions_file = self.save_predictions()
            
            # 予測のみモードの場合はここで終了
            if self.config.predict_only:
                elapsed_time = time.time() - start_time
                if self.config.verbose >= 1:
                    print(f"\n予測完了 (経過時間: {elapsed_time:.2f}秒)")
                    print(f"\nCOCO形式の予測結果を保存しました: {predictions_file}")
                    print("このファイルを使って以下が実行できます:")
                    print(f"  1. 評価: sahi coco evaluate --dataset_json_path {self.config.dataset_json_path} --result_json_path {predictions_file}")
                    print(f"  2. エラー解析: sahi coco analyse --dataset_json_path {self.config.dataset_json_path} --result_json_path {predictions_file}")
                
                return {
                    "predictions_file": predictions_file,
                    "num_predictions": len(self.predictions_json),
                    "elapsed_time": elapsed_time
                }

            # 評価実行
            eval_result = self.evaluate(predictions_file)
            
            # eval.jsonを更新（Precision/Recallを追加）
            eval_json_path = Path(self.config.output_dir) / "eval.json"
            with open(eval_json_path, 'w', encoding='utf-8') as f:
                json.dump(eval_result['eval_results'], f, indent=4)

            elapsed_time = time.time() - start_time
            if self.config.verbose >= 1:
                print(f"\nValidation完了 (経過時間: {elapsed_time:.2f}秒)")

            return eval_result

        except KeyboardInterrupt:
            print("\n\nValidationが中断されました", file=sys.stderr)
            sys.exit(130)  # Standard exit code for SIGINT
        except Exception as e:
            print(f"\nエラーが発生しました: {e}", file=sys.stderr)
            if self.config.verbose >= 2:
                import traceback
                print(traceback.format_exc(), file=sys.stderr)
            raise


class SAHIValidator:
    """SAHI validator (wrapper around SAHISegmentationValidator)"""
    
    def __init__(self, config: SAHIValidationConfig):
        """
        Args:
            config: SAHI validation configuration
        """
        self.config = config
        self.validator = SAHISegmentationValidator(self.config)
        self.results: Optional[Dict[str, Any]] = None
    
    def validate(self) -> Dict[str, Any]:
        """Run SAHI validation
        
        Returns:
            Validation results dictionary
        """
        if self.config.verbose >= 1:
            print("\n" + "="*70)
            print("SAHI Validationを実行中...")
            print("="*70)
        
        # Run validation
        self.results = self.validator.run()
        
        if self.config.verbose >= 1:
            print("\n" + "="*70)
            print("✅ SAHI Validation完了")
            print("="*70)
            self._print_results()
        
        return {
            "output_dir": self.config.output_dir,
            "results": self.results,
            "eval_results": self.results.get("eval_results", {}),
        }
    
    def _print_results(self) -> None:
        """Print validation results"""
        if self.results and "eval_results" in self.results:
            eval_results = self.results["eval_results"]
            
            print("\n=== メトリクス ===")
            
            # Segmentation metrics
            if "segm_mAP" in eval_results:
                print("\n[Segmentation]")
                print(f"  mAP50-95: {eval_results.get('segm_mAP', 0):.4f} ({eval_results.get('segm_mAP', 0)*100:.2f}%)")
                print(f"  mAP50:    {eval_results.get('segm_mAP50', 0):.4f} ({eval_results.get('segm_mAP50', 0)*100:.2f}%)")
                print(f"  mAP75:    {eval_results.get('segm_mAP75', 0):.4f} ({eval_results.get('segm_mAP75', 0)*100:.2f}%)")
                
                if "segm_precision" in eval_results:
                    print(f"  Precision: {eval_results.get('segm_precision', 0):.4f}")
                if "segm_recall" in eval_results:
                    print(f"  Recall:    {eval_results.get('segm_recall', 0):.4f}")
            
            # Bounding Box metrics
            if "bbox_mAP" in eval_results:
                print("\n[Bounding Box]")
                print(f"  mAP50-95: {eval_results.get('bbox_mAP', 0):.4f} ({eval_results.get('bbox_mAP', 0)*100:.2f}%)")
                print(f"  mAP50:    {eval_results.get('bbox_mAP50', 0):.4f} ({eval_results.get('bbox_mAP50', 0)*100:.2f}%)")
                
                if "bbox_precision" in eval_results:
                    print(f"  Precision: {eval_results.get('bbox_precision', 0):.4f}")
                if "bbox_recall" in eval_results:
                    print(f"  Recall:    {eval_results.get('bbox_recall', 0):.4f}")
