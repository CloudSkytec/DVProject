import json
from datetime import datetime, timedelta
import math
import random

# 地理范围约束
PRE_MIN_LON = -1.2943938800558410
PRE_MAX_LON = -1.1685531617945071
PRE_MIN_LAT = 37.3364469832967956
PRE_MAX_LAT = 37.4332849202708857

# 增强版波动参数
LON_WAVE_AMP = 0.002  # ±0.002度 ≈ 220米
LAT_WAVE_AMP = 0.0015 # ±0.0015度 ≈ 170米
NOISE_AMP = 0.0003    # 随机噪声幅度 ≈ 33米

def generate_pre_quake_hourly():
    """生成前震阶段每小时数据（仅坐标波动）"""
    # 加载基础数据
    with open("data/mc1-pre_quake-shakemap_2levels.geojson") as f:
        base_data = json.load(f)
    
    # 验证基础PGA值
    original_values = {f['properties']['value'] for f in base_data['features']}
    assert original_values <= {0.5, 1.0, 2.0}, "发现非法PGA值"

    # 时间参数（55小时）
    start_time = datetime(2020, 4, 6, 0)
    hours = 55
    
    for hour in range(hours+1):
        current_time = start_time + timedelta(hours=hour)
        
        new_features = []
        for feat in base_data['features']:
            # 深拷贝特征（保持属性不变）
            new_feat = json.loads(json.dumps(feat))
            
            # 仅修改坐标
            new_coords = apply_coordinate_wave(
                feat['geometry']['coordinates'], 
                hour
            )
            new_feat['geometry']['coordinates'] = new_coords
            
            new_features.append(new_feat)
        
        # 保存文件
        output_data = {
            "type": "FeatureCollection",
            "features": new_features
        }
        #filename = f"pre_pga_{current_time.strftime('%Y%m%d%H%M')}.json"
        filename = f"pga_{current_time.strftime('%Y%m%d%H%M')}.json"
        with open(filename, 'w') as f:
            json.dump(output_data, f, indent=2)

def apply_coordinate_wave(coords, hour):
    """应用复合坐标波动"""
    # 主波动（6小时周期）
    main_phase = hour * 2 * math.pi / 6
    
    # 次波动（2小时周期） 
    sub_phase = hour * 2 * math.pi / 2
    
    # 计算波动量
    dx_main = LON_WAVE_AMP * math.sin(main_phase)
    dy_main = LAT_WAVE_AMP * math.cos(main_phase)
    
    dx_sub = 0.3 * LON_WAVE_AMP * math.sin(sub_phase)
    dy_sub = 0.3 * LAT_WAVE_AMP * math.cos(sub_phase)
    
    # 随机噪声
    dx_noise = random.uniform(-NOISE_AMP, NOISE_AMP)
    dy_noise = random.uniform(-NOISE_AMP, NOISE_AMP)
    
    return [
        [
            [
                max(PRE_MIN_LON, min(PRE_MAX_LON, x + dx_main + dx_sub + dx_noise)),
                max(PRE_MIN_LAT, min(PRE_MAX_LAT, y + dy_main + dy_sub + dy_noise))
            ]
            for x, y in line
        ]
        for line in coords
    ]

if __name__ == "__main__":
    random.seed(42)  # 固定随机种子
    generate_pre_quake_hourly()
    print("前震数据生成完成")
