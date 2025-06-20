import cv2
import numpy as np
import os
from typing import Dict, List, Tuple

class ElevationSegmenter:
    def __init__(self):
        # 階層定義 (階層, (R, G, B))
        self.hierarchy_colors = {
            0: (0, 0, 255),      # 青
            1: (0, 149, 255),    # 水色
            2: (0, 238, 255),    # 明るい水色
            3: (145, 255, 0),    # 明るい緑
            4: (255, 255, 0),    # 黄色
            5: (255, 140, 0)     # オレンジ
        }
        
        # BGR形式に変換（OpenCVはBGR）
        self.hierarchy_colors_bgr = {
            level: (b, g, r) for level, (r, g, b) in self.hierarchy_colors.items()
        }
    
    def load_image(self, image_path: str) -> np.ndarray:
        """画像を読み込む"""
        image = cv2.imread(image_path)
        if image is None:
            raise ValueError(f"画像を読み込めませんでした: {image_path}")
        return image
    
    def create_color_mask(self, image: np.ndarray, target_color_bgr: Tuple[int, int, int], 
                         tolerance: int = 5) -> np.ndarray:
        """指定された色のマスクを作成"""
        # 色の範囲を定義（許容誤差を含む）
        lower = np.array([max(0, c - tolerance) for c in target_color_bgr])
        upper = np.array([min(255, c + tolerance) for c in target_color_bgr])
        
        # マスクを作成
        mask = cv2.inRange(image, lower, upper)
        return mask
    
    def classify_pixels_by_hierarchy(self, image: np.ndarray) -> np.ndarray:
        """各ピクセルを階層に分類"""
        height, width = image.shape[:2]
        hierarchy_map = np.full((height, width), -1, dtype=np.int8)  # -1は未分類
        
        # 各階層の色でマスクを作成し、分類
        for level, color_bgr in self.hierarchy_colors_bgr.items():
            mask = self.create_color_mask(image, color_bgr)
            hierarchy_map[mask > 0] = level
        
        return hierarchy_map
    
    def create_binary_image_for_level(self, hierarchy_map: np.ndarray, target_level: int) -> np.ndarray:
        """指定階層用の2値画像を作成
        target_level以下: 255 (白、対象領域)
        target_levelより上: 0 (黒、背景)
        """
        binary_image = np.zeros(hierarchy_map.shape, dtype=np.uint8)
        binary_image[hierarchy_map <= target_level] = 255
        return binary_image
    
    def find_enclosed_regions(self, binary_image: np.ndarray) -> Tuple[np.ndarray, List]:
        """囲まれた領域を検出"""
        # 輪郭を検出
        contours, hierarchy = cv2.findContours(
            binary_image, cv2.RETR_CCOMP, cv2.CHAIN_APPROX_SIMPLE
        )
        
        # 囲まれた領域（穴）を検出
        enclosed_regions = []
        if hierarchy is not None:
            for i, h in enumerate(hierarchy[0]):
                # h[3]が親の輪郭のインデックス（-1でない場合は内部の輪郭）
                if h[3] != -1:  # 親がある場合（穴の場合）
                    enclosed_regions.append(contours[i])
        
        # セグメント画像を作成
        segment_image = np.zeros_like(binary_image)
        if enclosed_regions:
            cv2.fillPoly(segment_image, enclosed_regions, 255)
        
        return segment_image, enclosed_regions
    
    def create_colored_output(self, hierarchy_map: np.ndarray, target_level: int) -> np.ndarray:
        """階層別の色分け画像を作成"""
        height, width = hierarchy_map.shape
        colored_image = np.zeros((height, width, 3), dtype=np.uint8)
        
        # target_level以下を対象色（青）、それより上を背景色（灰色）に設定
        target_color = (255, 0, 0)    # 青 (BGR)
        background_color = (128, 128, 128)  # 灰色 (BGR)
        
        # 色を設定
        colored_image[hierarchy_map <= target_level] = target_color
        colored_image[hierarchy_map > target_level] = background_color
        colored_image[hierarchy_map == -1] = (0, 0, 0)  # 未分類は黒
        
        return colored_image
    
    def process_image(self, image_path: str, output_dir: str = "output"):
        """メイン処理"""
        # 出力ディレクトリを作成
        os.makedirs(output_dir, exist_ok=True)
        
        # 画像を読み込み
        print(f"画像を読み込み中: {image_path}")
        image = self.load_image(image_path)
        
        # ピクセルを階層に分類
        print("ピクセルを階層に分類中...")
        hierarchy_map = self.classify_pixels_by_hierarchy(image)
        
        # 各階層について処理
        for level in range(6):  # 0-5の階層
            print(f"\n階層 {level} を処理中...")
            
            # 2値画像を作成
            binary_image = self.create_binary_image_for_level(hierarchy_map, level)
            
            # 色分け画像を作成
            colored_image = self.create_colored_output(hierarchy_map, level)
            
            # 囲まれた領域を検出
            segment_image, enclosed_regions = self.find_enclosed_regions(binary_image)
            
            # 結果を保存
            binary_filename = os.path.join(output_dir, f"level_{level}_binary.png")
            colored_filename = os.path.join(output_dir, f"level_{level}_colored.png")
            segment_filename = os.path.join(output_dir, f"level_{level}_segments.png")
            
            cv2.imwrite(binary_filename, binary_image)
            cv2.imwrite(colored_filename, colored_image)
            cv2.imwrite(segment_filename, segment_image)
            
            print(f"  2値画像: {binary_filename}")
            print(f"  色分け画像: {colored_filename}")
            print(f"  セグメント画像: {segment_filename}")
            print(f"  検出された囲まれた領域数: {len(enclosed_regions)}")
    
    def visualize_hierarchy_colors(self, output_dir: str = "output"):
        """階層色の見本を作成"""
        os.makedirs(output_dir, exist_ok=True)
        
        # 色見本画像を作成（各色50x50ピクセル）
        sample_height, sample_width = 50, 50
        color_sample = np.zeros((sample_height * 6, sample_width, 3), dtype=np.uint8)
        
        for level, color_bgr in self.hierarchy_colors_bgr.items():
            start_y = level * sample_height
            end_y = start_y + sample_height
            color_sample[start_y:end_y, :] = color_bgr
        
        # ラベルを追加した画像を作成
        labeled_sample = np.zeros((sample_height * 6, sample_width + 100, 3), dtype=np.uint8)
        labeled_sample[:, :sample_width] = color_sample
        
        # テキストを追加
        for level in range(6):
            y_pos = level * sample_height + sample_height // 2
            cv2.putText(labeled_sample, f"Level {level}", 
                       (sample_width + 10, y_pos), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1)
        
        sample_filename = os.path.join(output_dir, "hierarchy_colors.png")
        cv2.imwrite(sample_filename, labeled_sample)
        print(f"階層色見本: {sample_filename}")

def main():
    """使用例"""
    segmenter = ElevationSegmenter()
    
    # 階層色の見本を作成
    segmenter.visualize_hierarchy_colors()
    
    # 画像ファイルのパスを指定
    image_path = "C:\\Users\\USER\\Desktop\\4color\\250520\\0521_test.png"  # ここに実際の画像パスを設定
    
    try:
        # 処理を実行
        segmenter.process_image(image_path)
        print("\n処理が完了しました！")
        print("出力ファイル:")
        print("- level_X_binary.png: X階層以下を白、それより上を黒にした2値画像")
        print("- level_X_colored.png: X階層以下を青、それより上を灰色にした色分け画像")
        print("- level_X_segments.png: X階層で囲まれた領域のセグメント画像")
        
    except FileNotFoundError:
        print(f"画像ファイルが見つかりません: {image_path}")
        print("画像ファイルのパスを確認して、image_path変数を更新してください。")
    except Exception as e:
        print(f"エラーが発生しました: {e}")

if __name__ == "__main__":
    main()

# 実行
#process_all_layers("C:\\Users\\USER\\Desktop\\4color\\250520\\0521_test.png")
