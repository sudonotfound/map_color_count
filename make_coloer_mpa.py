import numpy as np
import matplotlib.pyplot as plt
import rasterio

# DEMデータの読み込み
def load_dem(file_path):
    with rasterio.open(file_path) as dem_file:
        dem_data = dem_file.read(1)  # 1バンド目を取得
        dem_data[dem_data == dem_file.nodata] = np.nan  # NoData値をNaNに置き換え
    return dem_data

# 色別標高図の作成
def create_colormap(dem_data, cmap="terrain"):
    plt.figure(figsize=(10, 8))
    plt.imshow(dem_data, cmap=cmap)
    plt.colorbar(label="Elevation (m)")
    plt.title("Color-coded Elevation Map")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

# メイン処理
if __name__ == "__main__":
    # DEMデータのパス
    dem_file_path = "path_to_your_dem_file.tif"  # あなたのDEMデータファイルのパスに変更してください

    # DEMデータを読み込み
    dem_data = load_dem(dem_file_path)

    # 色別標高図を作成
    create_colormap(dem_data)
