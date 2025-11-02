"""
GIF Animation visualizer for patch-based detection
"""

import cv2
import numpy as np
import os
from pathlib import Path
from tqdm import tqdm
import imageio
from ultralytics import YOLO
from patched_yolo_infer import MakeCropsDetectThem, CombineDetections, visualize_results


# ========== アニメーション設定 ==========
FRAME_INITIAL = 15       # 初期画像
FRAME_FIRST_FADE = 10    # 最初のパッチフェードイン
FRAME_SLIDE = 8          # パッチスライド
FRAME_DETECT = 5         # 検出中
FRAME_FADE_IN = 8        # 検出結果フェードイン
FRAME_PAUSE = 2          # 検出後の停止
FRAME_NMS_BEFORE = 15    # NMS前
FRAME_TRANSITION = 12    # 遷移
FRAME_FINAL = 60         # 最終結果

ALPHA_PAST = 0.15        # 過去のパッチ
ALPHA_CURRENT = 0.75     # 現在のパッチ


def draw_detections(img, detections_list, scale_factor_x=1.0, scale_factor_y=1.0, box_thickness=1):
    """
    検出結果（ボックス+マスク）を描画
    scale_factor_x: X座標のスケール係数（working_img → 元サイズ）
    scale_factor_y: Y座標のスケール係数（working_img → 元サイズ）
    box_thickness: バウンディングボックスの線幅
    """
    result = img.copy()
    for boxes, masks in detections_list:
        for i, box in enumerate(boxes):
            # 座標をスケール変換（X軸とY軸で異なるスケール係数を使用）
            x1, y1, x2, y2 = box
            x1 = int(x1 * scale_factor_x)
            y1 = int(y1 * scale_factor_y)
            x2 = int(x2 * scale_factor_x)
            y2 = int(y2 * scale_factor_y)
            
            cv2.rectangle(result, (x1, y1), (x2, y2), (0, 255, 0), max(1, box_thickness))
            
            if i < len(masks):
                mask = masks[i]
                if mask.shape != result.shape[:2]:
                    mask = cv2.resize(mask, (result.shape[1], result.shape[0]),
                                    interpolation=cv2.INTER_NEAREST)
                colored_mask = np.zeros_like(result)
                colored_mask[mask > 0] = [0, 200, 0]
                result = cv2.addWeighted(result, 1, colored_mask, 0.35, 0)
    return result


def get_patch_regions(crops):
    """すべてのパッチの座標を取得（リサイズ後の画像ベース）"""
    patch_regions = []
    for crop in crops:
        h, w = crop.source_image_resized.shape[:2]
        x_min = max(0, int(crop.x_start))
        y_min = max(0, int(crop.y_start))
        crop_h, crop_w = crop.crop.shape[:2]
        x_max = min(w, int(crop.x_start + crop_w))
        y_max = min(h, int(crop.y_start + crop_h))
        patch_regions.append((x_min, y_min, x_max, y_max))
    return patch_regions


def create_overlap_map(patch_regions, img_shape):
    """オーバーラップマップを作成"""
    h, w = img_shape[:2]
    overlap_map = np.zeros((h, w), dtype=np.uint8)
    for x_min, y_min, x_max, y_max in patch_regions:
        overlap_map[y_min:y_max, x_min:x_max] += 1
    return overlap_map


def draw_single_patch(img, region, overlap_map, alpha, show_overlap=True):
    """
    単一のパッチを描画（オーバーラップ部分は濃く）
    注：渡された画像を変更せず、新しい画像を返します
    """
    h, w = img.shape[:2]
    x_min, y_min, x_max, y_max = region
    
    # 座標を画像サイズ内にクリップ
    x_min = max(0, min(x_min, w))
    y_min = max(0, min(y_min, h))
    x_max = max(0, min(x_max, w))
    y_max = max(0, min(y_max, h))
    
    # 空のパッチは無視
    if x_max <= x_min or y_max <= y_min:
        return img
    
    result = img
    
    if show_overlap:
        # オーバーラップマップに基づいて描画
        patch_area = overlap_map[y_min:y_max, x_min:x_max]
        
        # 通常部分（オーバーラップなし）- 薄い青色
        normal_mask = (patch_area == 1)
        if np.any(normal_mask):
            overlay = np.zeros_like(img)
            patch_img = overlay[y_min:y_max, x_min:x_max]
            # サイズが一致することを確認
            if patch_img.shape[:2] == normal_mask.shape[:2]:
                patch_img[normal_mask] = [255, 200, 150]  # 薄い青色
                result = cv2.addWeighted(result, 1.0, overlay, alpha * 0.6, 0)
        
        # オーバーラップ部分（より濃く）- 濃い青色
        overlap_mask = (patch_area > 1)
        if np.any(overlap_mask):
            overlay = np.zeros_like(img)
            patch_img = overlay[y_min:y_max, x_min:x_max]
            # サイズが一致することを確認
            if patch_img.shape[:2] == overlap_mask.shape[:2]:
                patch_img[overlap_mask] = [255, 150, 100]  # 濃い青色
                result = cv2.addWeighted(result, 1.0, overlay, alpha * 1.2, 0)
    else:
        # 通常描画 - 薄い青色
        overlay = np.zeros_like(img)
        cv2.rectangle(overlay, (x_min, y_min), (x_max, y_max), (255, 180, 120), -1)
        result = cv2.addWeighted(result, 1.0, overlay, alpha, 0)
    
    # 境界線 - 薄い青色
    cv2.rectangle(result, (x_min, y_min), (x_max, y_max), (255, 200, 150), 2)
    
    return result


def draw_multiple_patches(img, patch_regions, overlap_map, highlight_indices):
    """
    複数のパッチを描画
    """
    result = img.copy()
    
    if not highlight_indices:
        return result
    
    for idx, alpha in highlight_indices.items():
        if idx < len(patch_regions):
            result = draw_single_patch(result, patch_regions[idx], overlap_map, alpha, show_overlap=True)
    
    return result


def create_detection_gif(
    image_path,
    model_path,
    output_dir,
    shape_x=400,
    shape_y=400,
    overlap_x=30,
    overlap_y=30,
    conf_threshold=0.5,
    imgsz=640,
    nms_threshold=0.3,
    fps=30,
    verbose=True,
    box_thickness=1,
    show_boxes=False,
    show_class=False,
    show_confidences=False,
    fill_mask=True,
    alpha=0.7,
    font_scale=2.0,
    random_object_colors=True,
    final_show_boxes=None
):
    """
    検出過程のGIFアニメーションを作成
    
    Args:
        image_path: 画像パス
        model_path: YOLOモデルのパス
        output_dir: 出力ディレクトリ
        shape_x: パッチの幅
        shape_y: パッチの高さ
        overlap_x: X軸オーバーラップ (%)
        overlap_y: Y軸オーバーラップ (%)
        conf_threshold: 信頼度閾値
        imgsz: YOLO入力サイズ
        nms_threshold: NMS閾値
        fps: フレームレート
        verbose: 詳細出力
        box_thickness: GIF内で描画するバウンディングボックスの線幅
        show_boxes: 最終結果にバウンディングボックスを表示するか
        show_class: 最終結果にクラス名を表示するか
        show_confidences: 最終結果に信頼度を表示するか
        fill_mask: 最終結果でマスクを塗りつぶすか
        alpha: 最終結果マスクの透過度
        font_scale: 最終結果ラベルの文字サイズ
        random_object_colors: 最終結果でオブジェクト毎にランダム色を使用するか
        final_show_boxes: 最終フレームでバウンディングボックスを表示するか（Noneの場合は自動判定）
    
    Returns:
        保存されたGIFのパス
    """
    # 出力ディレクトリを連番で作成
    base_dir = output_dir
    i = 1
    while True:
        output_dir = f"{base_dir}{i}" if i > 1 else base_dir
        if not os.path.exists(output_dir):
            os.makedirs(output_dir, exist_ok=True)
            break
        i += 1
    
    if verbose:
        print(f"🎬 高品質GIF作成中: {Path(image_path).name}")
    
    # モデル読み込み
    model = YOLO(model_path)
    
    # 画像読み込み
    test_img = cv2.imread(image_path)
    if test_img is None:
        raise ValueError(f"画像が読み込めません: {image_path}")
    
    # パッチベース推論（バッチ処理で通常検出と同じ結果を保証）
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
        batch_inference=True,  # バッチ処理で通常検出と一致させる
        show_processing_status=False,
        resize_initial_size=True,
    )
    
    # フレームリスト（RGB形式）
    frames = []
    
    # リサイズされた画像を使用（パッチ座標の取得用）
    working_img = element_crops.crops[0].source_image_resized
    
    # スケール係数を計算（working_img → test_img）
    # X座標とY座標でそれぞれ異なるスケール係数を使用
    scale_factor_x = test_img.shape[1] / working_img.shape[1]
    scale_factor_y = test_img.shape[0] / working_img.shape[0]

    gif_box_thickness = max(1, int(box_thickness))

    # 検出結果の座標スケール係数
    if getattr(element_crops, "resize_initial_size", True):
        detection_scale_x = 1.0
        detection_scale_y = 1.0
    else:
        detection_scale_x = scale_factor_x
        detection_scale_y = scale_factor_y
    
    # パッチ情報を取得（working_imgサイズ）
    patch_regions_working = get_patch_regions(element_crops.crops)
    
    # パッチ領域を元サイズにスケール変換（幅と高さで異なるスケール係数を使用）
    patch_regions = []
    for x_min, y_min, x_max, y_max in patch_regions_working:
        scaled_x_min = int(x_min * scale_factor_x)
        scaled_y_min = int(y_min * scale_factor_y)
        scaled_x_max = int(x_max * scale_factor_x)
        scaled_y_max = int(y_max * scale_factor_y)
        
        # 画像サイズ内にクリップ
        scaled_x_min = max(0, min(scaled_x_min, test_img.shape[1]))
        scaled_y_min = max(0, min(scaled_y_min, test_img.shape[0]))
        scaled_x_max = max(0, min(scaled_x_max, test_img.shape[1]))
        scaled_y_max = max(0, min(scaled_y_max, test_img.shape[0]))
        
        patch_regions.append((scaled_x_min, scaled_y_min, scaled_x_max, scaled_y_max))
    
    overlap_map = create_overlap_map(patch_regions, test_img.shape)
    num_patches = len(patch_regions)
    
    if verbose:
        print(f"  パッチ数: {num_patches}個")
        print(f"  元画像サイズ: {test_img.shape[1]}x{test_img.shape[0]}")
        print(f"  スケール係数: X={scale_factor_x:.3f}, Y={scale_factor_y:.3f}")
    
    # ========== ステップ1: 元画像 ==========
    if verbose:
        print("  ステップ1: 元画像表示")
    base_rgb = cv2.cvtColor(test_img, cv2.COLOR_BGR2RGB)
    frames.extend([base_rgb.copy() for _ in range(FRAME_INITIAL)])
    
    # ========== ステップ2: 各パッチをスライドしながら検出 ==========
    if verbose:
        print("  ステップ2: パッチスキャンと検出（スライド表示）")
    
    # 検出結果を保存（累積描画しない）
    all_detections = []  # [(boxes, masks), ...]
    
    # 処理済みパッチを追跡
    processed_patches = []
    
    # プログレスバー付きでパッチ処理
    pbar = tqdm(enumerate(element_crops.crops), total=num_patches, 
                desc="  パッチ処理", unit="patch", ncols=100, disable=not verbose)
    
    for idx, crop in pbar:
        if verbose:
            pbar.set_postfix({"パッチ": f"{idx+1}/{num_patches}"})
        
        # 前のパッチから現在のパッチへスライド
        if idx > 0:
            # スライド動作（前のパッチをフェードアウト、次のパッチをフェードイン）
            for step in range(FRAME_SLIDE):
                t = (step + 1) / FRAME_SLIDE
                
                # 前のパッチの透明度を下げる
                prev_alpha = ALPHA_CURRENT * (1 - t)
                # 現在のパッチの透明度を上げる
                curr_alpha = ALPHA_CURRENT * t
                
                highlight_dict = {}
                # それ以前のパッチ
                for past_idx in processed_patches[:-1]:
                    highlight_dict[past_idx] = ALPHA_PAST
                # 前のパッチ
                if prev_alpha > 0.05:
                    highlight_dict[idx - 1] = prev_alpha
                # 現在のパッチ
                if curr_alpha > 0.05:
                    highlight_dict[idx] = curr_alpha
                
                # 検出結果を描画（座標をスケール変換）
                display_img = draw_detections(
                    test_img,
                    all_detections,
                    detection_scale_x,
                    detection_scale_y,
                    gif_box_thickness
                )
                
                highlight_img = draw_multiple_patches(
                    display_img, patch_regions, overlap_map,
                    highlight_dict
                )
                frames.append(cv2.cvtColor(highlight_img, cv2.COLOR_BGR2RGB))
        else:
            # 最初のパッチ（フェードイン）
            for step in range(FRAME_FIRST_FADE):
                patch_alpha = 0.05 + (0.65 * (step + 1) / FRAME_FIRST_FADE)
                
                highlight_dict = {idx: patch_alpha}
                
                highlight_img = draw_multiple_patches(
                    test_img, patch_regions, overlap_map,
                    highlight_dict
                )
                frames.append(cv2.cvtColor(highlight_img, cv2.COLOR_BGR2RGB))
        
        # パッチ検出中（現在のパッチを強調）
        for _ in range(FRAME_DETECT):
            highlight_dict = {}
            for past_idx in processed_patches:
                highlight_dict[past_idx] = ALPHA_PAST
            highlight_dict[idx] = ALPHA_CURRENT  # 検出中は明るく
            
            display_img = draw_detections(
                test_img,
                all_detections,
                detection_scale_x,
                detection_scale_y,
                gif_box_thickness
            )
            
            highlight_img = draw_multiple_patches(
                display_img, patch_regions, overlap_map,
                highlight_dict
            )
            frames.append(cv2.cvtColor(highlight_img, cv2.COLOR_BGR2RGB))
        
        # このパッチを処理済みリストに追加
        processed_patches.append(idx)
        
        # パッチをゆっくり薄くする（フェードアウト）
        for step in range(FRAME_SLIDE):
            patch_alpha = ALPHA_CURRENT - ((ALPHA_CURRENT - ALPHA_PAST) * (step + 1) / FRAME_SLIDE)
            
            highlight_dict = {}
            for past_idx in processed_patches[:-1]:  # 前のパッチ
                highlight_dict[past_idx] = ALPHA_PAST
            highlight_dict[idx] = patch_alpha  # 現在のパッチをフェードアウト
            
            display_img = draw_detections(
                test_img,
                all_detections,
                detection_scale_x,
                detection_scale_y,
                gif_box_thickness
            )
            
            highlight_img = draw_multiple_patches(
                display_img, patch_regions, overlap_map,
                highlight_dict
            )
            frames.append(cv2.cvtColor(highlight_img, cv2.COLOR_BGR2RGB))
        
        # 少し停止（検出前）
        for _ in range(FRAME_PAUSE):
            highlight_dict = {}
            for past_idx in processed_patches:
                highlight_dict[past_idx] = ALPHA_PAST
            
            display_img = draw_detections(
                test_img,
                all_detections,
                detection_scale_x,
                detection_scale_y,
                gif_box_thickness
            )
            
            highlight_img = draw_multiple_patches(
                display_img, patch_regions, overlap_map,
                highlight_dict
            )
            frames.append(cv2.cvtColor(highlight_img, cv2.COLOR_BGR2RGB))
        
        # 検出結果を累積画像に追加
        det_boxes = []
        det_masks = []
        if len(crop.detected_xyxy_real) > 0:
            for i, box in enumerate(crop.detected_xyxy_real):
                det_boxes.append(box)
                if len(crop.detected_masks_real) > 0 and i < len(crop.detected_masks_real):
                    det_masks.append(crop.detected_masks_real[i])
        
        if det_boxes:
            all_detections.append((det_boxes, det_masks))
        
        # 検出結果をフェードイン
        if det_boxes:
            for step in range(FRAME_FADE_IN):
                det_alpha = (step + 1) / FRAME_FADE_IN
                
                # 既存の検出結果
                display_img_base = draw_detections(
                    test_img,
                    all_detections[:-1],
                    detection_scale_x,
                    detection_scale_y,
                    gif_box_thickness
                )
                
                # 新しい検出結果をフェードイン
                new_detections_img = draw_detections(
                    test_img,
                    [all_detections[-1]],
                    detection_scale_x,
                    detection_scale_y,
                    gif_box_thickness
                )
                
                blended = cv2.addWeighted(display_img_base, 1 - det_alpha, new_detections_img, det_alpha, 0)
                
                highlight_dict = {}
                for past_idx in processed_patches:
                    highlight_dict[past_idx] = ALPHA_PAST
                
                highlight_img = draw_multiple_patches(
                    blended, patch_regions, overlap_map,
                    highlight_dict
                )
                frames.append(cv2.cvtColor(highlight_img, cv2.COLOR_BGR2RGB))
        
        # 検出結果が追加された後の停止
        for _ in range(FRAME_PAUSE):
            highlight_dict = {}
            for past_idx in processed_patches:
                highlight_dict[past_idx] = ALPHA_PAST
            
            display_img = draw_detections(
                test_img,
                all_detections,
                detection_scale_x,
                detection_scale_y,
                gif_box_thickness
            )
            
            highlight_img = draw_multiple_patches(
                display_img, patch_regions, overlap_map,
                highlight_dict
            )
            frames.append(cv2.cvtColor(highlight_img, cv2.COLOR_BGR2RGB))
    
    pbar.close()
    
    # ========== ステップ3: NMS前の全検出結果 ==========
    if verbose:
        print("  ステップ3: NMS前の全検出結果")
    
    # すべての検出結果を描画
    nms_before_img = draw_detections(
        test_img,
        all_detections,
        detection_scale_x,
        detection_scale_y,
        gif_box_thickness
    )
    
    all_patches_dict = {i: 0.12 for i in range(num_patches)}
    final_with_grid = draw_multiple_patches(nms_before_img, patch_regions, overlap_map, 
                                            all_patches_dict)
    frames.extend([cv2.cvtColor(final_with_grid, cv2.COLOR_BGR2RGB) for _ in range(FRAME_NMS_BEFORE)])
    
    # NMS前の検出をフェードアウト（パッチグリッドは残す）
    clean_base_temp = test_img.copy()
    grid_only = draw_multiple_patches(clean_base_temp, patch_regions, overlap_map,
                                     all_patches_dict)
    grid_only_rgb = cv2.cvtColor(grid_only, cv2.COLOR_BGR2RGB)
    
    for step in range(8):  # 8フレームでフェードアウト
        fade_alpha = 1.0 - ((step + 1) / 8.0)  # 1.0 → 0.0
        
        # NMS前の検出をフェードアウト
        fade_out = cv2.addWeighted(
            grid_only_rgb, 1 - fade_alpha,
            cv2.cvtColor(final_with_grid, cv2.COLOR_BGR2RGB), fade_alpha, 0
        )
        frames.append(fade_out)
    
    # パッチグリッドのみを数フレーム表示（完全にクリーン）
    frames.extend([grid_only_rgb.copy() for _ in range(8)])
    
    # ========== ステップ4: NMS適用と最終結果 ==========
    if verbose:
        print("  ステップ4: NMS適用と最終結果")
    
    result = CombineDetections(
        element_crops, 
        nms_threshold=nms_threshold,
        class_agnostic_nms=True,  # クラス間でもNMSを適用
    )
    
    if verbose:
        print(f"  NMS前の検出数: {len([b for det in all_detections for b in det[0]])}個")
        print(f"  NMS後の検出数: {len(result.filtered_boxes)}個")
    
    # 完全に新しいクリーンな画像を用意（元画像から直接コピー）
    clean_img_for_final = test_img.copy()
    
    # クリーンな画像でNMS後の結果を可視化（bboxなし、マスクのみ）
    final_show_boxes_flag = (
        final_show_boxes
        if final_show_boxes is not None
        else (show_boxes and not fill_mask)
    )

    final_img = visualize_results(
        img=clean_img_for_final,
        boxes=result.filtered_boxes,
        classes_ids=result.filtered_classes_id,
        confidences=result.filtered_confidences,
        classes_names=result.filtered_classes_names,
        masks=result.filtered_masks,
        segment=True,
        show_boxes=final_show_boxes_flag,
        show_class=show_class,
        fill_mask=fill_mask,
        alpha=alpha,
        thickness=max(1, int(box_thickness)),
        font_scale=font_scale,
        show_confidences=show_confidences,
        return_image_array=True,
        random_object_colors=random_object_colors,
    )
    
    final_rgb = cv2.cvtColor(final_img, cv2.COLOR_BGR2RGB)
    
    # パッチグリッドをフェードアウトしながら最終結果へ遷移
    for step in range(FRAME_TRANSITION - 1):  # 最後の1フレームを除く
        transition_alpha = (step + 1) / FRAME_TRANSITION
        
        # パッチグリッドの透明度を下げる
        fading_patches_dict = {i: 0.12 * (1 - transition_alpha) for i in range(num_patches)}
        
        # クリーンな画像にパッチグリッドを重ねる
        clean_base = test_img.copy()
        grid_on_clean = draw_multiple_patches(clean_base, patch_regions, overlap_map,
                                             fading_patches_dict)
        
        # 最終結果へブレンド
        transition = cv2.addWeighted(
            cv2.cvtColor(grid_on_clean, cv2.COLOR_BGR2RGB), 1 - transition_alpha,
            final_rgb, transition_alpha, 0
        )
        frames.append(transition)
    
    # 最終フレームは完全にfinal_rgbのみ（遷移の最後）
    frames.append(final_rgb.copy())
    
    # 最終結果を長めに表示
    frames.extend([final_rgb.copy() for _ in range(FRAME_FINAL)])
    
    # ========== GIF保存 ==========
    if verbose:
        print("  ステップ5: GIF保存中...")
    
    # フレームは既に元サイズで作成されているため、リサイズ不要
    image_basename = Path(image_path).stem
    gif_path = os.path.join(output_dir, f"{image_basename}_detection.gif")
    
    if verbose:
        print("  GIF書き込み中...")
    with tqdm(total=100, desc="  GIF保存", unit="%", ncols=100, disable=not verbose) as pbar:
        imageio.mimsave(
            gif_path,
            frames,
            fps=fps,
            loop=0
        )
        pbar.n = 100
        pbar.refresh()
    
    if verbose:
        print(f"\n✅ GIF作成完了!")
        print(f"   📁 保存先: {gif_path}")
        print(f"   📊 総フレーム数: {len(frames)}枚")
        print(f"   ⏱️  再生時間: 約{len(frames)/fps:.1f}秒")
        print(f"   🎯 検出数: {len(result.filtered_confidences)}個")
        print(f"   📐 画像サイズ: {test_img.shape[1]}x{test_img.shape[0]}")
    
    return gif_path

