import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

def segment_enclosed_regions(image_path):
    # 画像ファイルの存在を確認
    if not os.path.exists(image_path):
        print(f"エラー: 画像ファイル '{image_path}' が見つかりません。")
        print(f"現在の作業ディレクトリ: {os.getcwd()}")
        
        # 代替画像ファイルを探す
        directory = os.path.dirname(image_path) if os.path.dirname(image_path) else '.'
        image_files = [f for f in os.listdir(directory) if f.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp'))]
        
        if image_files:
            print("利用可能な画像ファイル:")
            for img_file in image_files:
                print(f" - {img_file}")
            
            # 最初の利用可能な画像ファイルを使用するかどうか尋ねる
            print(f"代わりに '{image_files[0]}' を使用しますか？(y/n)")
            response = input()
            if response.lower() == 'y':
                image_path = os.path.join(directory, image_files[0])
                print(f"'{image_path}' を使用します。")
            else:
                return None
        else:
            print("ディレクトリ内に利用可能な画像ファイルが見つかりません。")
            return None
    
    # 画像を読み込む
    image = cv2.imread(image_path)
    if image is None:
        print(f"エラー: 画像 '{image_path}' を読み込めませんでした。")
        print("ファイルが画像フォーマットでない、または破損している可能性があります。")
        return None
    
    print(f"画像 '{image_path}' を読み込みました。サイズ: {image.shape}")
    
    # 作業用にコピーを作成
    original = image.copy()
    
    # 色の階層を定義（RGBとH値）
    # 0が最も階層が低く、5が最も階層が高い
    color_hierarchy = {
        0: {"rgb": [0, 0, 255], "h": 240/2},      # 青
        1: {"rgb": [0, 149, 255], "h": 204.94/2}, # 水色
        2: {"rgb": [0, 238, 255], "h": 184/2},    # 薄い水色
        3: {"rgb": [145, 255, 0], "h": 85.88/2},  # 黄緑
        4: {"rgb": [255, 255, 0], "h": 60/2},     # 黄
        5: {"rgb": [255, 140, 0], "h": 32.94/2}   # オレンジ
    }
    
    # HSV色空間に変換
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    
    # 階層ごとのマスクを作成
    hierarchy_masks = {}
    for level, color_info in color_hierarchy.items():
        h_value = color_info["h"]
        # HSVでの許容範囲を設定（H値の周囲±10度、SとVは比較的広めに）
        lower_bound = np.array([max(0, h_value - 10), 100, 100], dtype=np.uint8)
        upper_bound = np.array([min(179, h_value + 10), 255, 255], dtype=np.uint8)
        
        # マスクを作成
        mask = cv2.inRange(hsv, lower_bound, upper_bound)
        
        # ノイズ除去
        kernel = np.ones((5, 5), np.uint8)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        
        hierarchy_masks[level] = mask
    
    # 低い階層から処理していく
    # 階層0と1のみ囲まれた領域として扱う（この値は要件に応じて調整可能）
    low_hierarchy_levels = [0, 1]
    enclosed_contours = []
    
    for level in low_hierarchy_levels:
        mask = hierarchy_masks[level]
        contours, hierarchy = cv2.findContours(mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
        
        if hierarchy is not None and len(hierarchy) > 0:
            for i, h in enumerate(hierarchy[0]):
                # 親が存在する場合（囲まれている）
                if h[3] != -1:
                    # 小さすぎる輪郭は無視
                    if cv2.contourArea(contours[i]) > 100:
                        enclosed_contours.append(contours[i])
    
    # 結果の可視化
    # 元の画像
    result_original = cv2.cvtColor(original, cv2.COLOR_BGR2RGB)
    
    # 各階層のマスクを可視化
    combined_mask = np.zeros_like(hierarchy_masks[0])
    for level, mask in hierarchy_masks.items():
        combined_mask = cv2.bitwise_or(combined_mask, mask)
    
    # 囲まれた輪郭を描画
    result_enclosed = original.copy()
    cv2.drawContours(result_enclosed, enclosed_contours, -1, (0, 0, 255), 2)
    result_enclosed = cv2.cvtColor(result_enclosed, cv2.COLOR_BGR2RGB)
    
    # セグメンテーション結果
    result_segmentation = np.zeros_like(original)
    for contour in enclosed_contours:
        # 輪郭内を塗りつぶす
        cv2.drawContours(result_segmentation, [contour], 0, (255, 255, 255), -1)
    
    # 元の画像とセグメンテーション結果を合成
    segmented_image = cv2.bitwise_and(original, result_segmentation)
    segmented_image = cv2.cvtColor(segmented_image, cv2.COLOR_BGR2RGB)
    
    # カラーマップで階層を可視化
    hierarchy_visualization = np.zeros((original.shape[0], original.shape[1]), dtype=np.uint8)
    for level, mask in sorted(hierarchy_masks.items()):
        hierarchy_visualization[mask > 0] = (level + 1) * 40  # レベルに応じた明るさ
    
    # 結果を表示
    plt.figure(figsize=(15, 10))
    
    plt.subplot(2, 3, 1)
    plt.title("元の画像")
    plt.imshow(result_original)
    plt.axis('off')
    
    plt.subplot(2, 3, 2)
    plt.title("階層マスク（全結合）")
    plt.imshow(combined_mask, cmap='gray')
    plt.axis('off')
    
    plt.subplot(2, 3, 3)
    plt.title("階層の可視化")
    plt.imshow(hierarchy_visualization, cmap='viridis')
    plt.colorbar(ticks=np.arange(0, 6) * 40 + 40, 
                 label='階層（0-5）', 
                 orientation='horizontal')
    plt.axis('off')
    
    plt.subplot(2, 3, 4)
    plt.title("囲まれた輪郭")
    plt.imshow(result_enclosed)
    plt.axis('off')
    
    plt.subplot(2, 3, 5)
    plt.title("セグメンテーション結果")
    plt.imshow(segmented_image)
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    # 各階層のマスク表示はオプション機能のため、必要な場合のみコメントを外す
    """
    for level, mask in hierarchy_masks.items():
        plt.figure(figsize=(5, 5))
        plt.title(f"階層 {level} のマスク")
        plt.imshow(mask, cmap='gray')
        plt.axis('off')
        plt.show()
    """
    
    return segmented_image

# メイン実行部分
if __name__ == "__main__":
    # フルパスで指定する方法
    # image_path = r'C:\フルパス\0521_test.png'  # Windowsの場合
    
    # 相対パスで指定する方法
    image_path = '0521_test.png'  
    
    # セグメンテーション実行
    segmented_image = segment_enclosed_regions(image_path)
    
    if segmented_image is not None:
        # 結果の保存
        plt.imsave('segmented_result.png', segmented_image)
        print("処理が完了し、結果を保存しました。")