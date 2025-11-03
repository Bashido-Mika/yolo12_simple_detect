"""
Metrics calculation utilities.

This module provides utilities for calculating precision, recall, and other metrics.
"""

import numpy as np
from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


def calculate_precision_recall(
    dataset_json_path: str,
    predictions_file: str,
    max_detections: int = 500
) -> dict:
    """Precision/Recallを計算（Ultralyticsスタイル）
    
    Args:
        dataset_json_path: COCO形式のデータセットJSON
        predictions_file: COCO形式の予測結果JSON
        max_detections: 最大検出数
        
    Returns:
        Precision/Recallのメトリクス辞書
    """
    # COCO評価の準備
    coco_gt = COCO(dataset_json_path)
    coco_dt = coco_gt.loadRes(predictions_file)
    
    # セグメンテーション評価
    coco_eval = COCOeval(coco_gt, coco_dt, 'segm')
    coco_eval.params.maxDets = [max_detections]
    coco_eval.evaluate()
    coco_eval.accumulate()
    
    # Precision/Recallを抽出
    precision = coco_eval.eval['precision']  # [T, R, K, A, M]
    recall = coco_eval.eval['recall']  # [T, K, A, M]
    
    # mAP50-95 (全IoU閾値の平均)
    precision_avg = np.mean(precision[:, :, :, 0, -1])
    recall_avg = np.mean(recall[:, :, 0, -1])
    
    # mAP50 (IoU=0.50のみ)
    precision_50 = np.mean(precision[0, :, :, 0, -1])
    recall_50 = np.mean(recall[0, :, 0, -1])
    
    # BBox評価（参考）
    coco_eval_box = COCOeval(coco_gt, coco_dt, 'bbox')
    coco_eval_box.params.maxDets = [max_detections]
    coco_eval_box.evaluate()
    coco_eval_box.accumulate()
    
    precision_box = coco_eval_box.eval['precision']
    recall_box = coco_eval_box.eval['recall']
    
    precision_box_avg = np.mean(precision_box[:, :, :, 0, -1])
    recall_box_avg = np.mean(recall_box[:, :, 0, -1])
    
    precision_box_50 = np.mean(precision_box[0, :, :, 0, -1])
    recall_box_50 = np.mean(recall_box[0, :, 0, -1])
    
    return {
        "segm_precision": float(precision_avg),
        "segm_recall": float(recall_avg),
        "segm_precision50": float(precision_50),
        "segm_recall50": float(recall_50),
        "bbox_precision": float(precision_box_avg),
        "bbox_recall": float(recall_box_avg),
        "bbox_precision50": float(precision_box_50),
        "bbox_recall50": float(recall_box_50),
    }

