import cv2
import numpy as np
import os

# === 色階層の定義（低 → 高） ===
elevation_colors = {
    0: (0, 0, 255),        # 赤（最も低い）
    1: (0, 149, 255),      # 水色寄りの青
    2: (0, 238, 255),      # 明るいシアン
    3: (145, 255, 0),      # ライム
    4: (255, 255, 0),      # 黄
    5: (255, 140, 0),      # オレンジ（最も高い）
}

output_dir = "segmented_layers"
os.makedirs(output_dir, exist_ok=True)

def create_color_mask(img, target_color, tolerance=10):
    """指定された色に近いピクセルを抽出するマスク"""
    lower = np.array([max(c - tolerance, 0) for c in target_color], dtype=np.uint8)
    upper = np.array([min(c + tolerance, 255) for c in target_color], dtype=np.uint8)
    return cv2.inRange(img, lower, upper)

def segment_enclosed_regions(img, current_level):
    """current_level の色が higher_level に囲まれている領域を抽出"""
    current_color = elevation_colors[current_level]

    # 対象階層のマスク
    current_mask = create_color_mask(img, current_color)

    # 背景マスク（higher_level 以上の色をまとめて1にする）
    background_mask = np.zeros_like(current_mask)
    for level in range(current_level + 1, len(elevation_colors)):
        background_mask = cv2.bitwise_or(background_mask, create_color_mask(img, elevation_colors[level]))

    # 対象階層: 黒、背景: 白の2値画像を生成
    binary = np.full_like(current_mask, 255)
    binary[current_mask > 0] = 0

    # 輪郭検出（RETR_CCOMP：外枠と内枠を取得）
    contours, hierarchy = cv2.findContours(binary, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE)

    result_mask = np.zeros_like(current_mask)
    if hierarchy is not None:
        for i, contour in enumerate(contours):
            # 親が存在 → 内部（囲まれた）領域
            if hierarchy[0][i][3] != -1:
                cv2.drawContours(result_mask, [contour], -1, 255, -1)

    # 色付き画像で出力
    result_img = np.zeros_like(img)
    result_img[result_mask == 255] = current_color
    return result_img, result_mask

def process_all_layers(image_path):
    img = cv2.imread(image_path)
    if img is None:
        raise FileNotFoundError(f"画像が見つかりません: {image_path}")
    h, w, _ = img.shape
    final_result = np.zeros_like(img)

    for level in range(len(elevation_colors) - 1):  # 最上階層は囲まれないので対象外
        print(f"▶ Level {level} 処理中...")
        segmented_img, _ = segment_enclosed_regions(img, level)
        final_result = cv2.add(final_result, segmented_img)

        cv2.imwrite(f"{output_dir}/segmented_level_{level}.png", segmented_img)
        print(f"✓ Level {level} を保存しました。")

    # 合成画像保存
    cv2.imwrite("segmented_combined.png", final_result)
    print("✓ 全セグメント合成画像を保存しました（segmented_combined.png）")

# 実行
if __name__ == "__main__":
    process_all_layers("C:\\Users\\USER\\Desktop\\4color\\250520\\0521_test.png")  # ← 必要に応じてパスを変更


# 実行
#process_all_layers("C:\\Users\\USER\\Desktop\\4color\\250520\\0521_test.png")
