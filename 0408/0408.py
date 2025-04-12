import numpy as np
import xml.etree.ElementTree as ET
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm
from matplotlib.cm import get_cmap

def parse_xml(filename):
    ns = {
        'ns': 'http://fgd.gsi.go.jp/spec/2008/FGD_GMLSchema',
        'gml': 'http://www.opengis.net/gml/3.2'
    }

    tree = ET.parse(filename)
    root = tree.getroot()

    dem = root.find('ns:DEM', ns)
    coverage = dem.find('ns:coverage', ns)
    envelope = coverage.find('gml:boundedBy//gml:Envelope', ns)
    lower = np.array(envelope.find('gml:lowerCorner', ns).text.split(), dtype=np.float64)
    upper = np.array(envelope.find('gml:upperCorner', ns).text.split(), dtype=np.float64)

    grid = coverage.find('gml:gridDomain//gml:Grid//gml:limits//gml:GridEnvelope', ns)
    low = np.array(grid.find('gml:low', ns).text.split(), dtype=np.int64)
    high = np.array(grid.find('gml:high', ns).text.split(), dtype=np.int64)

    tuple_list = coverage.find('gml:rangeSet//gml:DataBlock//gml:tupleList', ns).text.strip().split('\n')

    return lower, upper, low, high, tuple_list

def create_raster(lower, upper, low, high, tuple_list):
    sizex, sizey = high - low + 1
    raster = np.full((sizey, sizex), np.nan, dtype=np.float32)

    x, y = 0, 0
    for line in tuple_list:
        values = line.split(',')
        if len(values) < 2:
            continue
        value = float(values[1])
        raster[y, x] = value if value > -9998 else np.nan
        x += 1
        if x >= sizex:
            x = 0
            y += 1

    return raster

def save_as_png(raster, output_filename):
    # NaNを除いた標高の最小・最大を整数で取得
    valid_values = raster[~np.isnan(raster)]
    vmin = int(np.floor(np.nanmin(valid_values)))
    vmax = int(np.ceil(np.nanmax(valid_values)))

    # 1mごとの境界リスト
    levels = np.arange(vmin, vmax + 2, 1)

    # 色分けルールを定義（1m刻み）
    norm = BoundaryNorm(levels, ncolors=len(levels) - 1)
    cmap = get_cmap('terrain', len(levels) - 1)

    # 描画
    plt.figure(figsize=(10, 10))
    plt.imshow(raster, cmap=cmap, norm=norm, origin='lower', interpolation='none')
    plt.axis('off')
    plt.savefig(output_filename, dpi=300, bbox_inches='tight', pad_inches=0)
    plt.close()

def main():
    input_file = 'FG-GML-4730-66-00-DEM5A-20161001.xml'
    output_file = 'output0408.png'
    lower, upper, low, high, tuple_list = parse_xml(input_file)
    raster = create_raster(lower, upper, low, high, tuple_list)
    save_as_png(raster, output_file)
    print(f'Saved PNG: {output_file}')

if __name__ == '__main__':
    main()
