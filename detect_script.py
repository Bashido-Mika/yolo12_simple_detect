# UltralyticsのYOLOv8を使用した画像推論スクリプト
from ultralytics import YOLO
import os
import glob
from pathlib import Path

def main():
    # パスの設定
    model_path = "runs/train/train6/weights/best.pt"
    source_path = "detect_images"
    
    # YOLOモデルの読み込み
    print(f"モデルを読み込み中: {model_path}")
    model = YOLO(model_path)
    
    print(f"推論対象のディレクトリ: {source_path}")
    
    # 画像推論の実行（指定されたパラメータで）
    print("推論を実行中...")
    results = model.predict(
        source=source_path,
        imgsz=800,           # 画像サイズ
        save_txt=True,       # 結果をテキストファイルで保存
        line_width=1,        # バウンディングボックスの線の太さ
        save=True,           # 推論結果の画像を保存
        conf=0.5,           # 信頼度のしきい値（0.5以上のみ表示）
        device='cpu',        # GPUが利用可能な場合は'0'に変更
        project='runs/detect', # 結果保存ディレクトリ
        name='exp',          # 実験名
        exist_ok=True        # 既存のディレクトリがある場合上書き
    )
    
    print("推論が完了しました！")
    
    # 推論結果の詳細表示
    for i, result in enumerate(results):
        print(f"\n=== 画像 {i+1}: {Path(result.path).name} ===")
        
        # 検出されたオブジェクトの情報を表示
        if result.boxes is not None and len(result.boxes) > 0:
            print(f"検出されたオブジェクト数: {len(result.boxes)}")
            
            for j, box in enumerate(result.boxes):
                # クラス名と信頼度を取得
                class_id = int(box.cls[0])
                confidence = float(box.conf[0])
                class_name = model.names[class_id]
                
                print(f"  オブジェクト {j+1}: {class_name} (信頼度: {confidence:.3f})")
        else:
            print("オブジェクトは検出されませんでした")
    
    # 保存されたファイルの確認
    print("\n=== 保存されたファイル ===")
    detection_dir = "runs/detect/exp"
    if os.path.exists(detection_dir):
        saved_images = glob.glob(os.path.join(detection_dir, "*.jpg"))
        saved_labels = glob.glob(os.path.join(detection_dir, "labels", "*.txt"))
        
        print(f"推論結果画像: {len(saved_images)}枚")
        print(f"ラベルファイル: {len(saved_labels)}個")
        print(f"結果は {detection_dir} に保存されました")
    else:
        print("結果ディレクトリが見つかりません")

if __name__ == "__main__":
    main()
