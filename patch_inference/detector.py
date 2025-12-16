"""
Patch-based detection module
"""

import cv2
import os
import csv
from pathlib import Path
from collections import Counter
from ultralytics import YOLO
from patched_yolo_infer import MakeCropsDetectThem, CombineDetections, visualize_results


def save_detections_to_csv(detections_data, output_path, verbose=True):
    """
    検出結果をCSVファイルに保存
    
    Args:
        detections_data: 検出結果のリスト
        output_path: CSV出力パス
        verbose: 詳細出力
    """
    if not detections_data:
        if verbose:
            print("⚠️  保存する検出結果がありません")
        return
    
    # CSVヘッダー
    fieldnames = [
        'image_name',
        'object_id',
        'class_name',
        'class_id',
        'confidence',
        'x1',
        'y1',
        'x2',
        'y2',
        'width',
        'height',
        'center_x',
        'center_y'
    ]
    
    with open(output_path, 'w', newline='', encoding='utf-8') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        
        total_detections = 0
        for img_data in detections_data:
            image_name = Path(img_data['path']).name
            result = img_data['result']
            
            boxes = result.filtered_boxes
            confidences = result.filtered_confidences
            classes_ids = result.filtered_classes_id
            classes_names = result.filtered_classes_names
            
            for obj_id, (box, conf, cls_id, cls_name) in enumerate(
                zip(boxes, confidences, classes_ids, classes_names), start=1
            ):
                x1, y1, x2, y2 = box
                width = x2 - x1
                height = y2 - y1
                center_x = (x1 + x2) / 2
                center_y = (y1 + y2) / 2
                
                writer.writerow({
                    'image_name': image_name,
                    'object_id': obj_id,
                    'class_name': cls_name,
                    'class_id': int(cls_id),
                    'confidence': float(conf),
                    'x1': float(x1),
                    'y1': float(y1),
                    'x2': float(x2),
                    'y2': float(y2),
                    'width': float(width),
                    'height': float(height),
                    'center_x': float(center_x),
                    'center_y': float(center_y)
                })
                total_detections += 1
    
    if verbose:
        print(f"📊 CSV保存完了: {output_path}")
        print(f"   総検出数: {total_detections}個")


def run_patch_detection(
    model_path,
    source_path,
    output_dir,
    shape_x=400,
    shape_y=400,
    overlap_x=30,
    overlap_y=30,
    conf_threshold=0.5,
    imgsz=640,
    nms_threshold=0.3,
    batch_inference=True,
    verbose=True,
    show_boxes=True,
    show_class=False,
    show_confidences=True,
    fill_mask=True,
    alpha=0.7,
    thickness=2,
    font_scale=2.0,
    random_object_colors=True,
    save_csv=False,
    csv_path=None
):
    """
    パッチベースの検出を実行
    
    Args:
        model_path: YOLOモデルのパス
        source_path: 画像ファイルまたはディレクトリのパス
        output_dir: 出力ディレクトリ
        shape_x: パッチの幅
        shape_y: パッチの高さ
        overlap_x: X軸オーバーラップ (%)
        overlap_y: Y軸オーバーラップ (%)
        conf_threshold: 信頼度閾値
        imgsz: YOLO入力サイズ
        nms_threshold: NMS閾値
        batch_inference: バッチ推論を有効化
        verbose: 詳細出力
        show_boxes: バウンディングボックスを描画するか
        show_class: クラスラベルを表示するか
        show_confidences: 信頼度スコアを表示するか
        fill_mask: マスクを塗りつぶすか
        alpha: マスク透明度
        thickness: バウンディングボックスの線幅
        font_scale: ラベル表示時の文字サイズ
        random_object_colors: オブジェクトごとにランダムな色を使用するか
        save_csv: CSV形式で検出結果を保存するか
        csv_path: CSV出力パス（Noneの場合は自動生成）

    Returns:
        処理した画像のリスト, 出力ディレクトリ
    """
    # モデル読み込み
    if verbose:
        print(f"📦 モデル読み込み: {model_path}")
    model = YOLO(model_path)
    
    # 出力ディレクトリを連番で作成
    base_dir = output_dir
    i = 1
    while True:
        output_dir = f"{base_dir}{i}" if i > 1 else base_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            break
        i += 1
    
    # 画像ファイルのリストを取得
    if os.path.isfile(source_path):
        image_files = [source_path]
    else:
        image_files = []
        for ext in ['*.jpg', '*.jpeg', '*.png', '*.bmp']:
            image_files.extend(Path(source_path).glob(ext))
        image_files = sorted([str(f) for f in image_files])
    
    if not image_files:
        print(f"❌ 画像が見つかりません: {source_path}")
        return []
    
    if verbose:
        print(f"\n⚙️  設定:")
        print(f"  パッチサイズ: {shape_x}x{shape_y}")
        print(f"  オーバーラップ: {overlap_x}% x {overlap_y}%")
        print(f"  信頼度閾値: {conf_threshold}")
        print(f"  NMS閾値: {nms_threshold}")
        print(f"  バッチ推論: {'有効' if batch_inference else '無効'}")
        print(f"  画像数: {len(image_files)}枚")
        print(f"  保存先: {output_dir}\n")
    
    # 各画像に対して推論を実行
    processed_images = []
    
    for idx, img_path in enumerate(image_files):
        if verbose:
            print(f"[{idx+1}/{len(image_files)}] {Path(img_path).name}")
        
        img = cv2.imread(img_path)
        if img is None:
            print(f"  ❌ エラー: 読み込み失敗")
            continue
        
        # パッチベース推論
        element_crops = MakeCropsDetectThem(
            image=img,
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
            batch_inference=batch_inference,
            show_processing_status=False,
        )
        
        # 結果結合とNMS
        result = CombineDetections(
            element_crops, 
            nms_threshold=nms_threshold,
            class_agnostic_nms=True,  # クラス間でもNMSを適用
        )
        
        # 検出結果
        confidences = result.filtered_confidences
        boxes = result.filtered_boxes
        masks = result.filtered_masks
        classes_ids = result.filtered_classes_id
        classes_names = result.filtered_classes_names
        
        # 結果表示
        if verbose:
            class_counts = Counter(classes_names)
            for class_name, count in class_counts.items():
                print(f"  {class_name}: {count}個")
        
        # 可視化
        result_img = visualize_results(
            img=img,
            boxes=boxes,
            classes_ids=classes_ids,
            confidences=confidences,
            classes_names=classes_names,
            masks=masks,
            segment=True,
            show_boxes=show_boxes,
            show_class=show_class,
            fill_mask=fill_mask,
            alpha=alpha,
            thickness=thickness,
            font_scale=font_scale,
            show_confidences=show_confidences,
            return_image_array=True,
            random_object_colors=random_object_colors,
        )
        
        # 保存
        output_path = os.path.join(output_dir, Path(img_path).name)
        cv2.imwrite(output_path, result_img)
        
        processed_images.append({
            'path': img_path,
            'output_path': output_path,
            'detections': len(confidences),
            'element_crops': element_crops,
            'result': result
        })
    
    if verbose:
        print(f"\n✅ 完了！結果: {output_dir}")
    
    # CSV保存
    if save_csv and processed_images:
        if csv_path is None:
            csv_path = os.path.join(output_dir, 'detections.csv')
        save_detections_to_csv(processed_images, csv_path, verbose=verbose)
    
    return processed_images, output_dir

