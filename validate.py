#!/usr/bin/env python3
"""
Unified Validation CLI

YOLOモデルの評価を統一的に実行するCLI

Usage:
    # Ultralyticsでの標準評価
    python validate.py ultralytics --model best.pt --data data.yaml
    
    # SAHIでのスライス評価
    python validate.py sahi --model best.pt --yolo-dataset Dataset/YOLODataset_test_with_label
    
    # 比較モード（UltralyticsとSAHIの両方）
    python validate.py compare --model best.pt --yolo-dataset Dataset/YOLODataset_test_with_label
"""

import argparse
import sys
from pathlib import Path

# Import validators
from validation import UltralyticsValidator, SAHIValidator, ValidationComparator
from validation.ultralytics_validator import UltralyticsValidationConfig
from validation.sahi_validator import SAHIValidationConfig


def parse_args():
    """Parse command line arguments"""
    parser = argparse.ArgumentParser(
        description="Unified Validation CLI for YOLO models",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
詳細な使用例はREADME.mdを参照してください。
        """
    )
    
    subparsers = parser.add_subparsers(dest='mode', help='Validation mode', required=True)
    
    # ========================================
    # Ultralytics mode
    # ========================================
    ultralytics_parser = subparsers.add_parser('ultralytics', help='Ultralytics standard validation')
    ultralytics_parser.add_argument('--model', '-m', type=str, default='runs/train/train12/weights/best.pt', help='Model path')
    ultralytics_parser.add_argument('--data', '-d', type=str, default='Dataset/YOLODataset_test_with_label/data.yaml', help='Data YAML path')
    ultralytics_parser.add_argument('--imgsz', type=int, default=640, help='Image size')
    ultralytics_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    ultralytics_parser.add_argument('--iou', type=float, default=0.6, help='IoU threshold')
    ultralytics_parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset split')
    ultralytics_parser.add_argument('--device', type=str, default='0', help='Device (default: 0, examples: cpu, cuda:0)')
    ultralytics_parser.add_argument('--save-json', action='store_true', help='Save results as JSON')
    ultralytics_parser.add_argument('--save-hybrid', action='store_true', help='Save hybrid labels')
    ultralytics_parser.add_argument('--output', '-o', type=str, default='runs/val', help='Output directory')
    ultralytics_parser.add_argument('--name', type=str, default='ultralytics_val', help='Experiment name')
    ultralytics_parser.add_argument('--no-increment', action='store_true', help='Disable auto-increment (overwrite existing directory)')
    ultralytics_parser.add_argument('--verbose', '-v', type=int, default=1, choices=[0, 1, 2], help='Verbosity level (0: none, 1: normal, 2: detailed)')
    
    # ========================================
    # SAHI mode
    # ========================================
    sahi_parser = subparsers.add_parser('sahi', help='SAHI validation with sliced inference')
    sahi_parser.add_argument('--model', '-m', type=str, default='runs/train/train12/weights/best.pt', help='Model path')
    sahi_parser.add_argument('--yolo-dataset', '-y', type=str, default=None, help='YOLO dataset directory')
    sahi_parser.add_argument('--dataset', '-d', type=str, default=None, help='COCO format dataset JSON file path')
    sahi_parser.add_argument('--images', '-i', type=str, default=None, help='Image directory path')
    sahi_parser.add_argument('--output', '-o', type=str, default='runs/val', help='Output directory')
    sahi_parser.add_argument('--name', type=str, default='sahi_val', help='Experiment name')
    sahi_parser.add_argument('--no-increment', action='store_true', help='Disable auto-increment (overwrite existing directory)')
    
    # Model parameters
    sahi_parser.add_argument('--confidence-threshold', type=float, default=0.25, help='Confidence threshold')
    sahi_parser.add_argument('--device', type=str, default='0', help='Device (default: 0, examples: cpu, cuda:0)')
    sahi_parser.add_argument('--image-size', type=int, default=640, help='Model input image size')
    
    # Sliced prediction parameters
    sahi_parser.add_argument('--slice-height', type=int, default=512, help='Slice height')
    sahi_parser.add_argument('--slice-width', type=int, default=512, help='Slice width')
    sahi_parser.add_argument('--overlap-height-ratio', type=float, default=0.5, help='Overlap ratio in height direction')
    sahi_parser.add_argument('--overlap-width-ratio', type=float, default=0.5, help='Overlap ratio in width direction')
    sahi_parser.add_argument('--no-sliced-prediction', action='store_true', help='Disable sliced prediction (use standard inference only)')
    sahi_parser.add_argument('--compare-with-standard', action='store_true', help='Run both SAHI and standard inference for comparison')
    
    # Postprocessing parameters
    sahi_parser.add_argument('--postprocess-type', type=str, default='GREEDYNMM', choices=['GREEDYNMM', 'NMM', 'NMS', 'LSNMS'], help='Postprocess type')
    sahi_parser.add_argument('--postprocess-match-metric', type=str, default='IOS', choices=['IOU', 'IOS'], help='Matching metric')
    sahi_parser.add_argument('--postprocess-match-threshold', type=float, default=0.5, help='Matching threshold')
    sahi_parser.add_argument('--postprocess-class-agnostic', action='store_true', help='Class-agnostic postprocessing')
    
    # Evaluation parameters
    sahi_parser.add_argument('--iou-thrs', type=float, nargs='+', default=None, help='IoU threshold list (default: 0.5-0.95, step 0.05)')
    sahi_parser.add_argument('--max-detections', type=int, default=500, help='Maximum number of detections')
    sahi_parser.add_argument('--classwise', action='store_true', help='Show class-wise evaluation results')
    sahi_parser.add_argument('--areas', type=int, nargs=3, default=[1024, 9216, 10000000000], metavar=('SMALL', 'MEDIUM', 'LARGE'), help='Area ranges [small, medium, large]')
    
    # Visualization parameters
    sahi_parser.add_argument('--export-visuals', action='store_true', help='Export visualization images')
    sahi_parser.add_argument('--visual-export-dir', type=str, default=None, help='Visualization image output directory')
    sahi_parser.add_argument('--visual-bbox-thickness', type=int, default=2, help='Bounding box line thickness')
    sahi_parser.add_argument('--visual-text-size', type=float, default=0.5, help='Label text size')
    sahi_parser.add_argument('--visual-hide-labels', action='store_true', help='Hide label text')
    sahi_parser.add_argument('--visual-hide-conf', action='store_true', help='Hide confidence scores')
    sahi_parser.add_argument('--max-visual-samples', type=int, default=None, help='Maximum number of images to visualize (default: None=all)')
    
    # Analysis parameters
    sahi_parser.add_argument('--error-analysis', action='store_true', help='Generate error analysis plots')
    sahi_parser.add_argument('--predict-only', action='store_true', help='Prediction only mode (skip evaluation, generate COCO format predictions.json)')
    
    # Other parameters
    sahi_parser.add_argument('--verbose', '-v', type=int, default=1, choices=[0, 1, 2], help='Verbosity level (0: none, 1: normal, 2: detailed)')
    sahi_parser.add_argument('--no-progress-bar', action='store_true', help='Hide progress bar')
    
    # ========================================
    # Compare mode
    # ========================================
    compare_parser = subparsers.add_parser('compare', help='Compare Ultralytics vs SAHI')
    compare_parser.add_argument('--model', '-m', type=str, default='runs/train/train12/weights/best.pt', help='Model path')
    compare_parser.add_argument('--yolo-dataset', '-y', type=str, default='Dataset/YOLODataset_test_with_label', help='YOLO dataset directory')
    compare_parser.add_argument('--output', '-o', type=str, default='runs/val', help='Output directory')
    compare_parser.add_argument('--name', type=str, default='comparison', help='Experiment name')
    compare_parser.add_argument('--no-increment', action='store_true', help='Disable auto-increment (overwrite existing directory)')
    
    # Ultralytics parameters
    compare_parser.add_argument('--imgsz', type=int, default=640, help='Ultralytics image size')
    compare_parser.add_argument('--split', type=str, default='val', choices=['train', 'val', 'test'], help='Dataset split for Ultralytics')
    compare_parser.add_argument('--device', type=str, default='0', help='Device (default: 0, examples: cpu, cuda:0)')
    compare_parser.add_argument('--save-json', action='store_true', help='Save Ultralytics results as JSON')
    compare_parser.add_argument('--save-hybrid', action='store_true', help='Save Ultralytics hybrid labels')
    
    # Common model parameters
    compare_parser.add_argument('--conf', type=float, default=0.25, help='Confidence threshold')
    compare_parser.add_argument('--iou', type=float, default=0.6, help='IoU threshold for NMS')
    compare_parser.add_argument('--image-size', type=int, default=640, help='SAHI model input image size')
    
    # SAHI slicing parameters
    compare_parser.add_argument('--slice-height', type=int, default=512, help='Slice height for SAHI')
    compare_parser.add_argument('--slice-width', type=int, default=512, help='Slice width for SAHI')
    compare_parser.add_argument('--overlap-height-ratio', type=float, default=0.5, help='Overlap ratio in height direction')
    compare_parser.add_argument('--overlap-width-ratio', type=float, default=0.5, help='Overlap ratio in width direction')
    
    # SAHI postprocessing parameters
    compare_parser.add_argument('--postprocess-type', type=str, default='GREEDYNMM', choices=['GREEDYNMM', 'NMM', 'NMS', 'LSNMS'], help='Postprocess type')
    compare_parser.add_argument('--postprocess-match-metric', type=str, default='IOS', choices=['IOU', 'IOS'], help='Matching metric')
    compare_parser.add_argument('--postprocess-match-threshold', type=float, default=0.5, help='Matching threshold')
    compare_parser.add_argument('--postprocess-class-agnostic', action='store_true', help='Class-agnostic postprocessing')
    
    # SAHI evaluation parameters
    compare_parser.add_argument('--iou-thrs', type=float, nargs='+', default=None, help='IoU threshold list (default: 0.5-0.95, step 0.05)')
    compare_parser.add_argument('--max-detections', type=int, default=500, help='Maximum number of detections')
    compare_parser.add_argument('--classwise', action='store_true', help='Show class-wise evaluation results')
    compare_parser.add_argument('--areas', type=int, nargs=3, default=[1024, 9216, 10000000000], metavar=('SMALL', 'MEDIUM', 'LARGE'), help='Area ranges [small, medium, large]')
    
    # Visualization parameters
    compare_parser.add_argument('--export-visuals', action='store_true', help='Export visual comparisons')
    compare_parser.add_argument('--visual-export-dir', type=str, default=None, help='Visualization image output directory')
    compare_parser.add_argument('--visual-bbox-thickness', type=int, default=2, help='Bounding box line thickness')
    compare_parser.add_argument('--visual-text-size', type=float, default=0.5, help='Label text size')
    compare_parser.add_argument('--visual-hide-labels', action='store_true', help='Hide label text')
    compare_parser.add_argument('--visual-hide-conf', action='store_true', help='Hide confidence scores')
    compare_parser.add_argument('--max-visual-samples', type=int, default=None, help='Maximum number of images to visualize (default: None=all)')
    
    # Analysis parameters
    compare_parser.add_argument('--error-analysis', action='store_true', help='Enable error analysis')
    
    # Other parameters
    compare_parser.add_argument('--verbose', '-v', type=int, default=1, choices=[0, 1, 2], help='Verbosity level (0: none, 1: normal, 2: detailed)')
    compare_parser.add_argument('--no-progress-bar', action='store_true', help='Hide progress bar')
    
    return parser.parse_args()


def run_ultralytics_validation(args):
    """Run Ultralytics standard validation"""
    config = UltralyticsValidationConfig(
        model_path=args.model,
        data_yaml=args.data,
        output_dir=args.output,
        experiment_name=args.name,
        auto_increment=not args.no_increment,
        imgsz=args.imgsz,
        conf=args.conf,
        iou=args.iou,
        split=args.split,
        device=args.device,
        save_json=args.save_json,
        save_hybrid=args.save_hybrid,
        verbose=args.verbose,
    )
    
    validator = UltralyticsValidator(config)
    results = validator.validate()
    
    print("\n" + "="*70)
    print(f"✅ Validation完了: {results['output_dir']}")
    print("="*70)
    
    return results


def run_sahi_validation(args):
    """Run SAHI validation"""
    config = SAHIValidationConfig(
        model_path=args.model,
        yolo_dataset_dir=getattr(args, 'yolo_dataset', None),
        dataset_json_path=getattr(args, 'dataset', None),
        image_dir=getattr(args, 'images', None),
        output_dir=args.output,
        experiment_name=args.name,
        auto_increment=not getattr(args, 'no_increment', False),
        confidence_threshold=getattr(args, 'confidence_threshold', 0.25),
        device=getattr(args, 'device', '0'),
        image_size=getattr(args, 'image_size', 640),
        use_sliced_prediction=not getattr(args, 'no_sliced_prediction', False),
        compare_with_standard=getattr(args, 'compare_with_standard', False),
        slice_height=getattr(args, 'slice_height', 512),
        slice_width=getattr(args, 'slice_width', 512),
        overlap_height_ratio=getattr(args, 'overlap_height_ratio', 0.5),
        overlap_width_ratio=getattr(args, 'overlap_width_ratio', 0.5),
        postprocess_type=getattr(args, 'postprocess_type', 'GREEDYNMM'),
        postprocess_match_metric=getattr(args, 'postprocess_match_metric', 'IOS'),
        postprocess_match_threshold=getattr(args, 'postprocess_match_threshold', 0.5),
        postprocess_class_agnostic=getattr(args, 'postprocess_class_agnostic', False),
        iou_thrs=getattr(args, 'iou_thrs', None),
        max_detections=getattr(args, 'max_detections', 500),
        classwise=getattr(args, 'classwise', False),
        areas=getattr(args, 'areas', [1024, 9216, 10000000000]),
        export_visuals=getattr(args, 'export_visuals', False),
        visual_export_dir=getattr(args, 'visual_export_dir', None),
        visual_bbox_thickness=getattr(args, 'visual_bbox_thickness', 2),
        visual_text_size=getattr(args, 'visual_text_size', 0.5),
        visual_hide_labels=getattr(args, 'visual_hide_labels', False),
        visual_hide_conf=getattr(args, 'visual_hide_conf', False),
        max_visual_samples=getattr(args, 'max_visual_samples', None),
        error_analysis=getattr(args, 'error_analysis', False),
        predict_only=getattr(args, 'predict_only', False),
        verbose=getattr(args, 'verbose', 1),
    )
    
    validator = SAHIValidator(config)
    results = validator.validate()
    
    print("\n" + "="*70)
    print(f"✅ SAHI Validation完了: {results['output_dir']}")
    print("="*70)
    
    return results


def run_comparison(args):
    """Run comparison between Ultralytics and SAHI"""
    from pathlib import Path
    
    print("\n" + "="*70)
    print("比較モードを実行中...")
    print("="*70)
    
    # Create comparison directory
    if args.name:
        from validation.utils.coco_converter import get_incremental_dir
        comparison_dir = get_incremental_dir(args.output, args.name)
    else:
        comparison_dir = Path(args.output) / "comparison"
    
    comparison_dir = Path(comparison_dir)
    comparison_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Ultralytics validation
    print("\n[1/2] Ultralytics標準評価...")
    ultralytics_config = UltralyticsValidationConfig(
        model_path=args.model,
        data_yaml=f"{args.yolo_dataset}/data.yaml",
        output_dir=str(comparison_dir),
        experiment_name="ultralytics",
        auto_increment=False,  # UltralyticsValidator will add experiment_name to output_dir
        imgsz=getattr(args, 'imgsz', args.image_size),
        conf=args.conf,
        iou=args.iou,
        split=getattr(args, 'split', 'val'),
        device=getattr(args, 'device', '0'),
        save_json=getattr(args, 'save_json', True),
        save_hybrid=getattr(args, 'save_hybrid', False),
        verbose=getattr(args, 'verbose', 1),
    )
    
    ultralytics_validator = UltralyticsValidator(ultralytics_config)
    ultralytics_results = ultralytics_validator.validate()
    
    # 2. SAHI validation
    print("\n[2/2] SAHI評価...")
    sahi_config = SAHIValidationConfig(
        model_path=args.model,
        yolo_dataset_dir=args.yolo_dataset,
        dataset_json_path=None,
        image_dir=None,
        output_dir=str(comparison_dir / "sahi"),  # Explicit subdirectory
        experiment_name="sahi",
        auto_increment=False,  # Use the comparison_dir directly
        confidence_threshold=args.conf,
        device=getattr(args, 'device', '0'),
        image_size=args.image_size,
        use_sliced_prediction=True,
        compare_with_standard=False,
        slice_height=args.slice_height,
        slice_width=args.slice_width,
        overlap_height_ratio=getattr(args, 'overlap_height_ratio', 0.5),
        overlap_width_ratio=getattr(args, 'overlap_width_ratio', 0.5),
        postprocess_type=getattr(args, 'postprocess_type', 'GREEDYNMM'),
        postprocess_match_metric=getattr(args, 'postprocess_match_metric', 'IOS'),
        postprocess_match_threshold=getattr(args, 'postprocess_match_threshold', 0.5),
        postprocess_class_agnostic=getattr(args, 'postprocess_class_agnostic', False),
        iou_thrs=getattr(args, 'iou_thrs', None),
        max_detections=getattr(args, 'max_detections', 500),
        classwise=getattr(args, 'classwise', False),
        areas=getattr(args, 'areas', [1024, 9216, 10000000000]),
        export_visuals=getattr(args, 'export_visuals', False),
        visual_export_dir=getattr(args, 'visual_export_dir', None),
        visual_bbox_thickness=getattr(args, 'visual_bbox_thickness', 2),
        visual_text_size=getattr(args, 'visual_text_size', 0.5),
        visual_hide_labels=getattr(args, 'visual_hide_labels', False),
        visual_hide_conf=getattr(args, 'visual_hide_conf', False),
        max_visual_samples=getattr(args, 'max_visual_samples', None),
        error_analysis=getattr(args, 'error_analysis', False),
        predict_only=False,
        verbose=getattr(args, 'verbose', 1),
    )
    
    sahi_validator = SAHIValidator(sahi_config)
    sahi_results = sahi_validator.validate()
    
    # 3. Compare results
    print("\n" + "="*70)
    print("=== 比較結果 ===")
    print("="*70)
    
    _print_comparison(ultralytics_results, sahi_results, comparison_dir)
    
    print("\n" + "="*70)
    print(f"✅ 比較完了: {comparison_dir}")
    print("="*70)
    
    return {
        "comparison_dir": str(comparison_dir),
        "ultralytics": ultralytics_results,
        "sahi": sahi_results,
    }


def _print_comparison(ultralytics_results, sahi_results, comparison_dir):
    """Print comparison results"""
    ultra_metrics = ultralytics_results.get("metrics", {})
    sahi_metrics = sahi_results.get("eval_results", {})
    
    print(f"\n保存先: {comparison_dir}")
    print("\n[Ultralytics]")
    if "segm_map" in ultra_metrics:
        print(f"  Segm mAP50-95: {ultra_metrics['segm_map']:.4f} ({ultra_metrics['segm_map']*100:.2f}%)")
        print(f"  Segm mAP50:    {ultra_metrics['segm_map50']:.4f}")
        print(f"  Precision:     {ultra_metrics.get('segm_precision', 0):.4f}")
        print(f"  Recall:        {ultra_metrics.get('segm_recall', 0):.4f}")
    
    print("\n[SAHI]")
    if "segm_mAP" in sahi_metrics:
        print(f"  Segm mAP50-95: {sahi_metrics['segm_mAP']:.4f} ({sahi_metrics['segm_mAP']*100:.2f}%)")
        print(f"  Segm mAP50:    {sahi_metrics.get('segm_mAP50', 0):.4f}")
        print(f"  Precision:     {sahi_metrics.get('segm_precision', 0):.4f}")
        print(f"  Recall:        {sahi_metrics.get('segm_recall', 0):.4f}")
    
    # Calculate differences
    if "segm_map" in ultra_metrics and "segm_mAP" in sahi_metrics:
        ultra_map = ultra_metrics['segm_map']
        sahi_map = sahi_metrics['segm_mAP']
        
        print("\n[重要な注意]")
        print(f"  Ultralytics: {ultra_map*100:.2f}% ← 公式ベンチマーク（信頼できる）")
        print(f"  SAHI:        {sahi_map*100:.2f}% ← 参考値（絶対値として使えない）")
        print("\n  💡 SAHIの値が低いのは正常です（内部処理の違い）")
        print("  💡 Ultralyticsの値を精度評価に使用してください")


def main():
    """Main entry point"""
    args = parse_args()
    
    try:
        if args.mode == 'ultralytics':
            run_ultralytics_validation(args)
        elif args.mode == 'sahi':
            run_sahi_validation(args)
        elif args.mode == 'compare':
            run_comparison(args)
        else:
            print(f"未知のモード: {args.mode}")
            sys.exit(1)
    
    except KeyboardInterrupt:
        print("\n\n中断されました")
        sys.exit(130)
    except Exception as e:
        print(f"\nエラー: {e}", file=sys.stderr)
        import traceback
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()

