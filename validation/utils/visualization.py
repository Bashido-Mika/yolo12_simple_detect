"""
Visualization utilities for validation results.

This module provides utilities for creating comparison images and reports.
"""

import json
import time
from PIL import Image, ImageDraw, ImageFont


def create_comparison_image(
    sahi_visual_path: str,
    standard_visual_path: str,
    output_path: str,
    sahi_metrics: dict,
    standard_metrics: dict,
    image_name: str
) -> None:
    """SAHIと標準推論の比較画像を作成
    
    Args:
        sahi_visual_path: SAHI可視化画像のパス
        standard_visual_path: 標準推論可視化画像のパス
        output_path: 出力先パス
        sahi_metrics: SAHIのメトリクス
        standard_metrics: 標準推論のメトリクス
        image_name: 画像名
    """
    try:
        # 画像を読み込み
        sahi_img = Image.open(sahi_visual_path)
        standard_img = Image.open(standard_visual_path)
        
        # 画像サイズを取得
        width, height = sahi_img.size
        
        # テキスト領域の高さ
        text_height = 60
        
        # 新しい画像を作成（横並び + 上部にテキスト）
        comparison_img = Image.new('RGB', (width * 2, height + text_height), color='white')
        
        # 画像を配置
        comparison_img.paste(sahi_img, (0, text_height))
        comparison_img.paste(standard_img, (width, text_height))
        
        # テキストを描画
        draw = ImageDraw.Draw(comparison_img)
        
        try:
            # フォントを設定（システムフォントを試す）
            font_large = ImageFont.truetype("arial.ttf", 24)
            font_small = ImageFont.truetype("arial.ttf", 16)
        except:
            # フォントが見つからない場合はデフォルトを使用
            font_large = ImageFont.load_default()
            font_small = ImageFont.load_default()
        
        # タイトルを描画
        draw.text((10, 5), f"Image: {image_name}", fill='black', font=font_large)
        draw.text((width // 2 - 100, 35), "SAHI (Sliced)", fill='blue', font=font_small)
        draw.text((width + width // 2 - 100, 35), "Standard", fill='red', font=font_small)
        
        # 中央に区切り線
        draw.line([(width, text_height), (width, height + text_height)], fill='gray', width=3)
        
        # 画像を保存
        comparison_img.save(output_path)
        
    except Exception as e:
        print(f"比較画像の作成エラー ({image_name}): {e}")


def save_comparison_report(
    sahi_results: dict,
    standard_results: dict,
    sahi_dir: str,
    standard_dir: str,
    output_path: str
) -> None:
    """比較レポートをJSON形式で保存
    
    Args:
        sahi_results: SAHIの評価結果
        standard_results: 標準推論の評価結果
        sahi_dir: SAHIの出力ディレクトリ
        standard_dir: 標準推論の出力ディレクトリ
        output_path: レポート保存先
    """
    sahi_eval = sahi_results.get("eval_results", {})
    standard_eval = standard_results.get("eval_results", {})
    
    comparison = {
        "comparison_date": time.strftime("%Y-%m-%d %H:%M:%S"),
        "sahi_output_dir": sahi_dir,
        "standard_output_dir": standard_dir,
        "metrics_comparison": {},
        "summary": {}
    }
    
    # メトリクスごとに比較
    key_metrics = [
        "segm_mAP", "segm_mAP50", "segm_mAP75",
        "segm_mAP_s", "segm_mAP_m", "segm_mAP_l",
        "segm_mAP50_s", "segm_mAP50_m", "segm_mAP50_l"
    ]
    
    for key in key_metrics:
        if key in sahi_eval and key in standard_eval:
            sahi_val = sahi_eval[key]
            std_val = standard_eval[key]
            diff = sahi_val - std_val
            diff_pct = (diff / std_val * 100) if std_val != 0 else 0
            
            comparison["metrics_comparison"][key] = {
                "sahi": round(sahi_val, 4),
                "standard": round(std_val, 4),
                "difference": round(diff, 4),
                "difference_percent": round(diff_pct, 2),
                "improvement": diff > 0
            }
    
    # サマリー統計
    improvements = [v["difference_percent"] for v in comparison["metrics_comparison"].values() if v["improvement"]]
    comparison["summary"] = {
        "total_metrics_compared": len(comparison["metrics_comparison"]),
        "metrics_improved": sum(1 for v in comparison["metrics_comparison"].values() if v["improvement"]),
        "metrics_degraded": sum(1 for v in comparison["metrics_comparison"].values() if not v["improvement"]),
        "average_improvement_percent": round(sum(improvements) / len(improvements), 2) if improvements else 0,
        "best_improvement": max(improvements) if improvements else 0,
    }
    
    # JSONファイルに保存
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, indent=2, ensure_ascii=False)
    
    print(f"\n比較レポートを保存しました: {output_path}")

