#!/usr/bin/env python3
"""
SAHI-like Patch-Based Detection CLI
YOLOv11 with patch-based inference for small object detection and segmentation

Usage:
    python sahi_detect_cli.py --model runs/train/train12/weights/best.pt --source detect_images/
    python sahi_detect_cli.py -m best.pt -s image.jpg --create-gif
"""

import argparse
import sys
from pathlib import Path
from patch_inference import run_patch_detection, create_detection_gif


def parse_args():
    """コマンドライン引数をパース"""
    parser = argparse.ArgumentParser(
        description="SAHI-like Patch-Based Detection with YOLOv11",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
例:
  # 基本的な検出
  python sahi_detect_cli.py --model runs/train/train12/weights/best.pt --source detect_images/
  
  # GIF動画も作成
  python sahi_detect_cli.py -m best.pt -s image.jpg --create-gif
  
  # パッチサイズとオーバーラップをカスタマイズ
  python sahi_detect_cli.py -m best.pt -s images/ --shape-x 512 --shape-y 512 --overlap-x 40
  
  # バッチ推論を無効化（メモリ節約）
  python sahi_detect_cli.py -m best.pt -s images/ --no-batch-inference
        """
    )
    
    # 必須引数
    parser.add_argument(
        '--model', '-m',
        type=str,
        default='runs/train/train12/weights/best.pt',
        #required=True,
        help='YOLOモデルのパス (例: best.pt, yolo11n-seg.pt)'
    )
    
    parser.add_argument(
        '--source', '-s',
        type=str,
        default='detect_images/',
        #required=True,
        help='画像ファイルまたはディレクトリのパス'
    )
    
    # 出力設定
    parser.add_argument(
        '--output', '-o',
        type=str,
        default='runs/detect/sahi_results',
        help='検出結果の出力ディレクトリ (デフォルト: runs/detect/sahi_results)'
    )
    
    parser.add_argument(
        '--gif-output',
        type=str,
        default='runs/detect/sahi_gif',
        help='GIF動画の出力ディレクトリ (デフォルト: runs/detect/sahi_gif)'
    )
    
    # パッチ設定
    parser.add_argument(
        '--shape-x',
        type=int,
        default=400,
        help='パッチの幅 (デフォルト: 400)'
    )
    
    parser.add_argument(
        '--shape-y',
        type=int,
        default=400,
        help='パッチの高さ (デフォルト: 400)'
    )
    
    parser.add_argument(
        '--overlap-x',
        type=int,
        default=30,
        help='X軸オーバーラップ (%%) (デフォルト: 30)'
    )
    
    parser.add_argument(
        '--overlap-y',
        type=int,
        default=40,
        help='Y軸オーバーラップ (%%) (デフォルト: 30)'
    )
    
    # 推論設定
    parser.add_argument(
        '--conf',
        type=float,
        default=0.5,
        help='信頼度閾値 (デフォルト: 0.5)'
    )
    
    parser.add_argument(
        '--imgsz',
        type=int,
        default=640,
        help='YOLOの入力画像サイズ (デフォルト: 640)'
    )
    
    parser.add_argument(
        '--nms-threshold',
        type=float,
        default=0.1,
        help='NMS閾値 (デフォルト: 0.1)'
    )
    
    parser.add_argument(
        '--no-batch-inference',
        action='store_true',
        help='バッチ推論を無効化（メモリ節約）'
    )
    
    # GIF作成
    parser.add_argument(
        '--create-gif',
        action='store_true',
        help='検出過程のGIF動画を作成'
    )
    
    parser.add_argument(
        '--gif-fps',
        type=int,
        default=30,
        help='GIFのフレームレート (デフォルト: 30)'
    )
    
    # その他
    parser.add_argument(
        '--quiet', '-q',
        action='store_true',
        help='詳細出力を抑制'
    )

    # 可視化設定
    parser.add_argument(
        '--box-thickness',
        type=int,
        default=1,
        help='検出結果のバウンディングボックスの線幅 (ピクセル)'
    )

    parser.add_argument(
        '--gif-box-thickness',
        type=int,
        default=1,
        help='GIFアニメーション中に描画するバウンディングボックスの線幅 (ピクセル)'
    )

    parser.add_argument(
        '--font-scale',
        type=float,
        default=2.0,
        help='ラベル表示時の文字サイズ'
    )

    parser.add_argument(
        '--mask-alpha',
        type=float,
        default=0.7,
        help='マスク描画時の透過度 (0.0〜1.0)'
    )

    parser.add_argument(
        '--no-fill-mask',
        action='store_true',
        help='マスクの塗りつぶしを無効化'
    )

    parser.add_argument(
        '--hide-boxes',
        action='store_true',
        help='バウンディングボックスを非表示にする'
    )

    parser.add_argument(
        '--show-class-labels',
        action='store_true',
        help='クラス名ラベルを描画する'
    )

    parser.add_argument(
        '--no-confidences',
        action='store_true',
        help='信頼度スコアの表示を無効化する'
    )

    parser.add_argument(
        '--no-random-colors',
        action='store_true',
        help='オブジェクトごとのランダム色割り当てを無効化する'
    )

    parser.add_argument(
        '--gif-mask-alpha',
        type=float,
        default=None,
        help='GIF用のマスク透過度 (指定しない場合は --mask-alpha の値を使用)'
    )

    parser.add_argument(
        '--gif-font-scale',
        type=float,
        default=None,
        help='GIF用のラベル文字サイズ (指定しない場合は --font-scale の値を使用)'
    )

    parser.add_argument(
        '--gif-show-boxes',
        dest='gif_show_boxes',
        action='store_true',
        help='GIFでバウンディングボックスを表示'
    )
    parser.add_argument(
        '--gif-hide-boxes',
        dest='gif_show_boxes',
        action='store_false',
        help='GIFでバウンディングボックスを非表示'
    )
    parser.set_defaults(gif_show_boxes=None)

    parser.add_argument(
        '--gif-show-class-labels',
        dest='gif_show_class_labels',
        action='store_true',
        help='GIFでクラスラベルを表示'
    )
    parser.add_argument(
        '--gif-hide-class-labels',
        dest='gif_show_class_labels',
        action='store_false',
        help='GIFでクラスラベルを非表示'
    )
    parser.set_defaults(gif_show_class_labels=None)

    parser.add_argument(
        '--gif-show-confidences',
        dest='gif_show_confidences',
        action='store_true',
        help='GIFで信頼度スコアを表示'
    )
    parser.add_argument(
        '--gif-no-confidences',
        dest='gif_show_confidences',
        action='store_false',
        help='GIFで信頼度スコアを非表示'
    )
    parser.set_defaults(gif_show_confidences=None)

    parser.add_argument(
        '--gif-fill-mask',
        dest='gif_fill_mask',
        action='store_true',
        help='GIFでマスクを塗りつぶす'
    )
    parser.add_argument(
        '--gif-no-fill-mask',
        dest='gif_fill_mask',
        action='store_false',
        help='GIFでマスクの塗りつぶしを無効化'
    )
    parser.set_defaults(gif_fill_mask=None)

    parser.add_argument(
        '--gif-random-colors',
        dest='gif_random_colors',
        action='store_true',
        help='GIFでオブジェクトごとにランダム色を使用'
    )
    parser.add_argument(
        '--gif-no-random-colors',
        dest='gif_random_colors',
        action='store_false',
        help='GIFでランダム色を無効化'
    )
    parser.set_defaults(gif_random_colors=None)

    parser.add_argument(
        '--final-show-boxes',
        action='store_true',
        help='GIFの最終フレームでもバウンディングボックスを表示する'
    )

    return parser.parse_args()


def main():
    """メイン処理"""
    args = parse_args()
    
    # 引数検証
    if not Path(args.model).exists():
        print(f"❌ エラー: モデルが見つかりません: {args.model}")
        sys.exit(1)
    
    if not Path(args.source).exists():
        print(f"❌ エラー: 画像ソースが見つかりません: {args.source}")
        sys.exit(1)
    
    verbose = not args.quiet
    
    if verbose:
        print("=" * 60)
        print("🚀 SAHI-like Patch-Based Detection")
        print("=" * 60)
    
    # パッチベース検出を実行
    try:
        processed_images, output_dir = run_patch_detection(
            model_path=args.model,
            source_path=args.source,
            output_dir=args.output,
            shape_x=args.shape_x,
            shape_y=args.shape_y,
            overlap_x=args.overlap_x,
            overlap_y=args.overlap_y,
            conf_threshold=args.conf,
            imgsz=args.imgsz,
            nms_threshold=args.nms_threshold,
            batch_inference=not args.no_batch_inference,
            verbose=verbose,
            show_boxes=not args.hide_boxes,
            show_class=args.show_class_labels,
            show_confidences=not args.no_confidences,
            fill_mask=not args.no_fill_mask,
            alpha=args.mask_alpha,
            thickness=max(1, args.box_thickness),
            font_scale=args.font_scale,
            random_object_colors=not args.no_random_colors
        )
        
        if not processed_images:
            print("⚠️  警告: 処理された画像がありません")
            sys.exit(0)
        
        # GIF作成
        if args.create_gif:
            if verbose:
                print("\n" + "=" * 60)
                print("🎬 GIF動画作成")
                print("=" * 60 + "\n")
            
            # 最初の画像でGIFを作成
            first_image = processed_images[0]['path']

            detection_show_boxes = not args.hide_boxes
            detection_show_class = args.show_class_labels
            detection_show_confidences = not args.no_confidences
            detection_fill_mask = not args.no_fill_mask
            detection_random_colors = not args.no_random_colors

            gif_show_boxes = (
                detection_show_boxes if args.gif_show_boxes is None else args.gif_show_boxes
            )
            gif_show_class = (
                detection_show_class
                if args.gif_show_class_labels is None
                else args.gif_show_class_labels
            )
            gif_show_confidences = (
                detection_show_confidences
                if args.gif_show_confidences is None
                else args.gif_show_confidences
            )
            gif_fill_mask = (
                detection_fill_mask if args.gif_fill_mask is None else args.gif_fill_mask
            )
            gif_random_colors = (
                detection_random_colors
                if args.gif_random_colors is None
                else args.gif_random_colors
            )
            gif_mask_alpha = (
                args.mask_alpha if args.gif_mask_alpha is None else args.gif_mask_alpha
            )
            gif_font_scale = (
                args.font_scale if args.gif_font_scale is None else args.gif_font_scale
            )

            gif_path = create_detection_gif(
                image_path=first_image,
                model_path=args.model,
                output_dir=args.gif_output,
                shape_x=args.shape_x,
                shape_y=args.shape_y,
                overlap_x=args.overlap_x,
                overlap_y=args.overlap_y,
                conf_threshold=args.conf,
                imgsz=args.imgsz,
                nms_threshold=args.nms_threshold,
                fps=args.gif_fps,
                verbose=verbose,
                box_thickness=max(1, args.gif_box_thickness),
                show_boxes=gif_show_boxes,
                show_class=gif_show_class,
                show_confidences=gif_show_confidences,
                fill_mask=gif_fill_mask,
                alpha=gif_mask_alpha,
                font_scale=gif_font_scale,
                random_object_colors=gif_random_colors,
                final_show_boxes=args.final_show_boxes
            )
        
        if verbose:
            print("\n" + "=" * 60)
            print("✅ すべての処理が完了しました！")
            print("=" * 60)
    
    except KeyboardInterrupt:
        print("\n\n⚠️  処理が中断されました")
        sys.exit(1)
    except Exception as e:
        print(f"\n❌ エラーが発生しました: {e}")
        if verbose:
            import traceback
            traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

