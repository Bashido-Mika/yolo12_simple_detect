# Ultralyticsの基本テスト
import sys
print(f"Python version: {sys.version}")

try:
    from ultralytics import YOLO
    print("✓ Ultralytics import成功")
    
    # モデルファイルの存在確認
    model_path = "runs/train/train6/weights/best.pt"
    import os
    if os.path.exists(model_path):
        print(f"✓ モデルファイル存在確認: {model_path}")
        
        # モデルの読み込みテスト
        try:
            model = YOLO(model_path)
            print(f"✓ モデル読み込み成功")
            print(f"モデルクラス数: {len(model.names)}")
            print(f"クラス名: {model.names}")
        except Exception as e:
            print(f"✗ モデル読み込みエラー: {e}")
    else:
        print(f"✗ モデルファイルが見つかりません: {model_path}")
        
    # 画像フォルダの確認
    source_path = "detect_images"
    if os.path.exists(source_path):
        images = [f for f in os.listdir(source_path) if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
        print(f"✓ 画像フォルダ存在確認: {source_path}")
        print(f"画像数: {len(images)}")
    else:
        print(f"✗ 画像フォルダが見つかりません: {source_path}")
        
except ImportError as e:
    print(f"✗ Ultralytics import失敗: {e}")
except Exception as e:
    print(f"✗ 予期しないエラー: {e}")
    import traceback
    traceback.print_exc()

