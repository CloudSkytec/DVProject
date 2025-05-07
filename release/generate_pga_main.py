import json
from datetime import datetime, timedelta
import math


# 正确的地理边界
MIN_LON = -1.3829758771158791
MAX_LON = -1.0372653124249041
MIN_LAT = 37.2493052788786372
MAX_LAT = 37.5316489524634775

# 计算震中坐标（几何中心）
CENTER_LON = (MIN_LON + MAX_LON) / 2  # -1.2101205947703916
CENTER_LAT = (MIN_LAT + MAX_LAT) / 2  # 37.39047711567105

def simple_decay_model(t, initial_pga):
    """极简衰减模型"""
    if t < 6:  # 前6小时快速衰减
        return initial_pga * (0.9 ** t)
    else:       # 之后慢速衰减
        return initial_pga * (0.9 ** 6) * (0.95 ** (t-6))
    
def radial_contraction(coord, contraction):
    """基于震中的径向坐标收缩"""
    delta_lon = coord[0] - CENTER_LON
    delta_lat = coord[1] - CENTER_LAT
    return [
        CENTER_LON + delta_lon * contraction,
        CENTER_LAT + delta_lat * contraction
    ]
    
def calculate_contraction_factor(t, max_t=64):
    """计算区域收缩因子，范围[0-1]"""
    return 1 - math.log(1 + t) / math.log(1 + max_t)  # 非线性收缩

def generate_contoured_geojson(base_geojson, hour):
    new_features = []
    contraction = 1 - (hour/64)**0.5  # 非线性收缩因子
    
    for feat in base_geojson['features']:
        original_coords = feat['geometry']['coordinates'][0]
        contracted_coords = [
            radial_contraction(coord, contraction)
            for coord in original_coords
        ]
        
        new_feat = {
            "type": "Feature",
            "geometry": {
                "type": "MultiLineString",
                "coordinates": [contracted_coords]
            },
            "properties": feat["properties"]
        }
        new_features.append(new_feat)
    
    return {"type": "FeatureCollection", "features": new_features}
# 参数配置
INPUT_FILE = "mc1-majorquake-shakemap_4levels_v2.geojson"
OUTPUT_DIR = "hourly_pga"
START_TIME = datetime(2020, 4, 8, 8)
END_TIME = datetime(2020, 4, 11, 0)
HOURS = int((END_TIME - START_TIME).total_seconds() / 3600)  # 56小时

# 加载初始数据
with open(INPUT_FILE, encoding='utf-8') as f:
    base_data = json.load(f)

    # 2. 生成每小时数据
    start_time = datetime(2020,4,8,8)
    for hour in range(64):  # 64小时
        current_time = start_time + timedelta(hours=hour)
        
        # 生成收缩后的数据
        output_data = generate_contoured_geojson(base_data, hour)

        if hour == 63:
            output_data['features'] = [
                feat for feat in output_data['features']
                if feat['properties']['value'] <= 0.5  # 仅保留0.5g及以下
            ]
        
        # 保存文件
        filename = f"pga_{current_time.strftime('%Y%m%d%H%M')}.json"
        with open(filename, 'w') as f:
            json.dump(output_data, f)

print(f"生成完成，共{HOURS}个文件")
