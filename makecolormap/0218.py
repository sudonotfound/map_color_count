import xml.etree.ElementTree as ET
import numpy as np
import matplotlib.pyplot as plt

# === XMLデータのパース関数 ===
def parse_dem_from_gml(xml_file):
    """ GML（XML形式）のDEMデータを解析し、標高データの2D配列を取得 """
    
    tree = ET.parse(xml_file)
    root = tree.getroot()
    
    # 名前空間の定義（必要に応じて修正）
    ns = {
        "gml": "http://www.opengis.net/gml/3.2",
        "default": "http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema"
    }
    
    # <gml:low> と <gml:high> からグリッドサイズを取得
    grid_envelope = root.find(".//gml:GridEnvelope", ns)
    if grid_envelope is None:
        raise ValueError("Error: <gml:GridEnvelope> not found in XML.")

    low_values = list(map(int, grid_envelope.find("gml:low", ns).text.split()))
    high_values = list(map(int, grid_envelope.find("gml:high", ns).text.split()))
    
    width = high_values[0] - low_values[0] + 1
    height = high_values[1] - low_values[1] + 1

    # 標高データを取得（スペース区切りの数値リスト）
    data_block = root.find(".//gml:DataBlock/gml:rangeSet/gml:QuantityList", ns)

    # `None` の場合のエラーハンドリング
    if data_block is None or data_block.text is None:
        raise ValueError("Error: Elevation data <gml:QuantityList> not found or empty.")

    # 標高データを取得し、数値配列に変換
    elevation_values = list(map(float, data_block.text.strip().split()))

    # データサイズと一致するか確認
    if len(elevation_values) != width * height:
        raise ValueError(f"Error: Data size mismatch. Expected {width * height}, but got {len(elevation_values)}.")

    # 標高データを2D numpy配列に変換
    elevation_array = np.array(elevation_values).reshape((height, width))
    
    return elevation_array

# === 標高データの可視化関数 ===
def plot_elevation_map(elevation_data):
    """標高データをカラーマップとしてプロット"""
    plt.figure(figsize=(10, 8))
    plt.imshow(elevation_data, cmap="terrain", origin="upper")
    plt.colorbar(label="Elevation (m)")
    plt.title("Color-coded Elevation Map")
    plt.xlabel("X Coordinate")
    plt.ylabel("Y Coordinate")
    plt.show()

# === メイン処理 ===
if __name__ == "__main__":
    xml_file = "C:/Users/USER/Desktop/4color/dem/FG-GML-4730-66-00-DEM5A-20161001.xml"  

    try:
        # XMLからDEMデータを取得
        elevation_data = parse_dem_from_gml(xml_file)

        # 標高マップをプロット
        plot_elevation_map(elevation_data)
    
    except Exception as e:
        print(f"Error: {e}")
