# pipeline.py
import os
import math
import pandas as pd
import geopandas as gpd
import numpy as np
import pymc as pm
import xarray as xr
from shapely.geometry import Point
from datetime import datetime, timedelta
import arviz as az
import pytensor as pt  # 关键导入
from pytensor.tensor import as_tensor_variable  # 从子模块导入
from joblib import Parallel, delayed
from pathlib import Path
import colorsys




# --------------------------
# step1：loading
# --------------------------
class Config:
    raw_data_dir = "./data2/raw"
    processed_dir = "./data2/processed"
    model_dir = "./data2/models"
    viz_dir = "./data2/viz"
    
    start_date = "2020-04-06"
    end_date = "2020-04-10"
    communities_geojson = "community.geojson"
    epicenter = (37.39357,-1.21506)  # 震中

    # 37.39357,-1.21506
    magnitude = 6.5  # 假设震级
    
os.makedirs(Config.processed_dir, exist_ok=True)
os.makedirs(Config.model_dir, exist_ok=True)
os.makedirs(Config.viz_dir, exist_ok=True)

# --------------------------
# 步骤2：数据加载与预处理
# --------------------------
def load_reports():
    """加载居民报告数据"""
    df = pd.read_csv(
        os.path.join(Config.raw_data_dir, "mc1-reports-data.csv"),
        dtype={'location': str},  # 强制转换为字符串类型
        parse_dates=['time'],
        date_format='%Y-%m-%d %H:%M:%S'
    )
    
    # 计算综合烈度指标
    def calculate_composite_intensity(row):
        if not pd.isna(row['shake_intensity']):
            return row['shake_intensity']
        weights = {'sewer_and_water':0.3, 'power':0.2, 
                  'roads_and_bridges':0.15, 'medical':0.15, 'buildings':0.2}
        valid = [row[k]*v for k,v in weights.items() if not pd.isna(row[k])]
        return sum(valid)/sum(weights.values()) if valid else None
    
    df['composite_mmi'] = df.apply(calculate_composite_intensity, axis=1)
    
    # 时空聚合
    return df.groupby([
        # pd.Grouper(key='time', freq='H'),
        pd.Grouper(key='time', freq='h'),
        'location'
    ])['composite_mmi'].mean().reset_index()

def process_pga():


    """处理PGA地理数据并返回社区数据"""
    # 加载社区地理数据
    comm = gpd.read_file(os.path.join(Config.raw_data_dir, Config.communities_geojson))
    comm = comm.rename(columns={'id': 'location'})  # 关键修改
    comm['location'] = comm['location'].astype(str)  # 转换为字符串

    
    # 计算社区中心到震中的距离
    comm['centroid'] = comm.geometry.centroid
    comm['distance'] = comm.centroid.apply(
        lambda p: math.dist(Config.epicenter, (p.y, p.x))
    )

    """处理PGA地理数据"""
    # 加载所有PGA文件
    pga_dfs = []
    for t in pd.date_range(Config.start_date, Config.end_date, freq='H'):
        fname = f"pga_{t.strftime('%Y%m%d%H%M')}.json"
        path = os.path.join(Config.raw_data_dir, fname)
        if os.path.exists(path):
            gdf = gpd.read_file(path)
            # gdf['pga'] = gdf['value'] * 9.81  # 添加单位转换
            # gdf['pga'] = gdf['value'] * 0.01 * 9.81  # 即 value * 0.0981
            gdf['pga'] = gdf['value']   # USGS风格


            gdf['time'] = t
            pga_dfs.append(gdf[['time', 'geometry', 'pga']])
    
    all_pga = gpd.GeoDataFrame(pd.concat(pga_dfs), crs="EPSG:4326")

    # 确保使用相同CRS
    comm = comm.to_crs("EPSG:4326")
    all_pga = all_pga.to_crs("EPSG:4326")
    # 将线状PGA数据转换为点
    if 'LineString' in all_pga.geometry.type.unique():
        all_pga['geometry'] = all_pga.geometry.centroid

    # 执行空间连接（添加缓冲解决边界问题）
    comm_buffered = comm.copy()
    comm_buffered['geometry'] = comm.buffer(0.001)  # 约100米缓冲

    # merged = gpd.sjoin(all_pga, comm, predicate='within')
    merged = gpd.sjoin(all_pga, comm, predicate='intersects')
    merged['location'] = merged['location'].astype(str)

    df_pga = merged.groupby(['time', 'location'])['pga'].mean().reset_index()
    
    return df_pga, comm  # 同时返回处理后的PGA数据和社区数据

# --------------------------
# 步骤3：数据融合
# --------------------------
def merge_datasets(df_reports, df_pga,comm):
    """时空对齐与缺失值处理"""

    # 动态获取最大时间
    max_time = max(
        df_reports['time'].max(),
        df_pga['time'].max(),
        pd.to_datetime('2020-04-10 22:00:00')
    )
    # 创建完整时空索引
    full_idx = pd.MultiIndex.from_product([
        # pd.date_range(Config.start_date, Config.end_date, freq='H'),    
        pd.date_range(Config.start_date, max_time, freq='H'),
        df_pga['location'].unique()
    ], names=['time', 'location'])
    
    # 合并数据集
    merged = (
        pd.DataFrame(index=full_idx)
        .reset_index()
        .merge(df_reports, how='left', on=['time', 'location'])
        .merge(df_pga, how='left', on=['time', 'location'])
        # .set_index(['time', 'location'])  # 重建MultiIndex  重置索引以显式保留 time 和 location 列

    )
    
    # # 缺失值处理
    # merged['composite_mmi'] = (
    #     merged.groupby('location')['composite_mmi']
    #     .apply(lambda x: x.ffill().interpolate(method='linear', limit=6))
    # )

    # 修改后（正确）
    merged['composite_mmi'] = (
        merged.groupby('location')['composite_mmi']
        .transform(lambda x: x.ffill().interpolate(method='linear', limit=6))
    )
    
    # 动态计算PGA最大值
    pga_max = merged.groupby('time')['pga'].transform('max')

    # 创建社区距离字典提高查询效率
    distance_dict = comm.set_index('location')['distance'].to_dict()

    # 处理 merged['pga'] 列的缺失值
    # merged['pga'] = merged.apply(
    #     lambda r: (pga_max[r.time] * math.exp(-0.5 * distance_dict[r.location]))
    #     if pd.isna(r.pga) else r.pga,
    #     axis=1
    # )

    merged['pga'] = np.where(
        merged['pga'].isna(),
        pga_max * np.exp(-0.5 * merged['location'].map(distance_dict)),
        merged['pga']
    )

    # PGA标准化
    pga_mean = merged['pga'].mean()
    pga_std = merged['pga'].std()
    merged['pga_z'] = (merged['pga'] - pga_mean) / pga_std

    # 保存原始参数用于逆变换
    pd.DataFrame({'mean':[pga_mean], 'std':[pga_std]}).to_csv(
        os.path.join(Config.processed_dir, "pga_norm_params.csv")
    )
    
    merged.to_parquet(os.path.join(Config.processed_dir, "merged_data.parquet"))

    # 在merge_datasets函数末尾添加验证
    # assert not merged['pga'].isna().any(), "存在未处理的PGA缺失值"
    # print("所有PGA缺失值已成功填补")
    # 标准化后理论均值和标准差
    print("PGA_z均值:", merged['pga_z'].mean())   # 实际: ≈0（理论合理）
    print("PGA_z标准差:", merged['pga_z'].std())  # 实际: 若含极端值可能 >1（需检查）
    return merged

# --------------------------
# 步骤4：贝叶斯建模
# --------------------------
import pymc as pm
import pytensor.tensor as pt
import numpy as np
import pandas as pd

def build_model(data):
    """处理多地理区域的时间序列模型（最终稳定版）"""
    # =====================
    # 数据预处理（严格过滤）
    # =====================
    # 过滤无效数据
    valid_mask = data[['composite_mmi', 'pga_z', 'location']].notna().all(axis=1)
    data_valid = data[valid_mask].reset_index(drop=True)
    
    # 转换地理标识为整型
    data_valid = data_valid[data_valid['location'].apply(lambda x: str(x).isdigit())]
    data_valid['location'] = data_valid['location'].astype(int)
    
    # =====================
    # 维度计算
    # =====================
    n_obs = len(data_valid)  # 总观测数
    locations = data_valid['location'].unique()  # 地理区域列表
    n_locations = len(locations)
    
    # 创建地理索引映射
    loc_id_to_idx = {loc: idx for idx, loc in enumerate(locations)}
    data_valid['loc_idx'] = data_valid['location'].map(loc_id_to_idx)
    
    # 维度验证
    assert data_valid['loc_idx'].max() < n_locations, "地理索引越界"
    assert len(data_valid) == n_obs, "观测数量不一致"
    
    # =====================
    # 模型定义（显式维度控制）
    # =====================
    coords = {
        "time": np.arange(n_obs),
        "location": np.arange(n_locations)
    }
    
    with pm.Model(coords=coords) as model:
        # 注册坐标维度
        model.add_coords(coords)
        
        # =====================
        # 数据容器（强制维度对齐）
        # =====================
        pm.MutableData(
            'loc_idx', 
            data_valid['loc_idx'].values.astype('int32'),  # 降低精度避免类型问题
            dims="time",
            shape=(n_obs,)  # 显式声明维度
        )
        
        # =====================
        # 模型参数（分层结构）
        # =====================
        # 地理层级参数
        sigma_state = pm.HalfNormal(
            'sigma_state', 
            0.5,
            dims="location",
            shape=(n_locations,)  # 每个区域一个参数
        )
        
        # 时间序列参数
        theta = pm.GaussianRandomWalk(
            'theta',
            mu=0,
            sigma=sigma_state[data_valid['loc_idx'].values],  # 动态选择区域参数
            init_dist=pm.Normal.dist(0, 1, shape=(1,)),  # 静态初始维度
            steps=n_obs - 1,  # 总步数控制
            dims="time",
            shape=(n_obs,)  # 显式锁定最终维度
        )
        
        # =====================
        # 观测方程（严格维度控制）
        # =====================
        pm.Normal(
            'obs_report',
            mu=theta,
            sigma=pm.HalfNormal('sigma_report', 1.5),
            observed=data_valid['composite_mmi'].values,
            dims="time"
        )
        
        # =====================
        # 调试验证
        # =====================
        print("关键参数维度验证:")
        print(f"sigma_state 维度: {sigma_state.tag.test_value.shape}")  # 应 (n_locations,)
        print(f"theta 维度: {theta.tag.test_value.shape}")              # 应 (n_obs,)
        
        # =====================
        # 采样配置
        # =====================
        trace = pm.sample(
            draws=100,
            tune=100,
            cores=1,
            target_accept=0.95,
            random_seed=42,
            init='adapt_diag'  # 更稳定的初始化
        )
    
    return trace



def build_multi_obs_model(data_subset, location_id):
    """修复后的多观测变量分区域模型"""
    # =====================
    # 1. 数据预处理（必须保留）
    # =====================
    valid_mask = data_subset[['composite_mmi', 'pga_z']].notna().all(axis=1)
    data_valid = data_subset[valid_mask].reset_index(drop=True)
    n_obs = len(data_valid)
    print(f"区域 {location_id} 有效观测数:", n_obs)

    # 步骤1：确保时间列是datetime类型
    data_valid['time'] = pd.to_datetime(data_valid['time'])
    
    # 步骤2：获取底层int64（单位：纳秒）
    time_ns = data_valid['time'].view('int64')  # shape: (n_obs,)
    
    # 步骤3：转换为秒级浮点数
    time_values = (time_ns / 1e9).astype('float64')  # 得到Unix时间戳

    # =====================
    # 2. 坐标系统（必须保留）
    # =====================
    coords = {"obs": np.arange(n_obs)
            # "time_dim": time_labels       # 新增实际时间坐标（不影响模型运算）
    }
    
    with pm.Model(name=f"location_{location_id}", coords=coords) as model:
        model.add_coords(coords)

        # ===== 核心修正：显式存储时间数据 =====
        pm.ConstantData(
            "time_var",  # 明确命名避免冲突
            time_values,
            dims="obs"   # 绑定到观测维度
        )
        
        # =====================
        # 3. 动态数据容器（必须保留）
        # =====================
        time_idx = pm.MutableData('time_idx', data_valid.index, dims="obs")
        composite_mmi = pm.MutableData(
            'composite_mmi', 
            data_valid['composite_mmi'].values, 
            dims="obs"
        )
        pga_z = pm.MutableData(
            'pga_z', 
            data_valid['pga_z'].values, 
            dims="obs"
        )
        
        # =====================
        # 4. 模型结构（保持原设计）
        # =====================
        sigma_state = pm.HalfNormal("sigma_state", 0.5)
        theta = pm.GaussianRandomWalk(
            'theta',
            # mu=pm.math.log(data_valid['pga'].values + 1e-6),
            mu=0,  # 或使用 pm.Normal("mu_prior", 0, 1)
            sigma=sigma_state,
            shape=(n_obs,),  # 显式定义形状
            dims="obs"  # 绑定到预定义维度
        )
        
        sigma_report = pm.HalfNormal("sigma_report", 1.5)
        pm.Normal(
            "obs_report",
            mu=theta,
            sigma=sigma_report,
            observed=composite_mmi,
            dims="obs"
        )
        
        beta_pga = pm.Normal("beta_pga", 0, 1)
        sigma_pga = pm.HalfNormal("sigma_pga", 0.3)
        pm.Normal(
            'obs_pga',
            mu=theta[time_idx],
            sigma=sigma_pga,
            observed=pga_z,
            dims='obs'
        )
        # pm.ConstantData("time", time_labels, dims="obs")

        # =====================
        # 5. 采样参数优化
        # =====================
        trace = pm.sample(
            draws=500, 
            tune=500,
            cores=1,    # 建议 >=2 核心
            # chains=2,   # 必须 >=2 链
            target_accept=0.95,
            random_seed=42
        )
        
    return trace

# --------------------------
# 步骤5：可视化输出
# --------------------------



def generate_visualization(trace, comm, model_id):
    """生成VSUP可视化文件（修复数组维度问题）"""
    try:
        print(f"\n正在生成模型 {model_id} 的可视化...")
        
        param_prefix = f"location_{model_id}::"
        target_location = int(model_id)
        
        # 验证theta存在性
        if f"{param_prefix}theta" not in trace.posterior:
            print(f"模型 {model_id} 缺少theta参数")
            return
        
        theta = trace.posterior[f"{param_prefix}theta"]
        print(f"Theta维度: {theta.dims}")  # 调试输出
        
        # 计算统计量（处理多维情况）
        theta_mean = theta.mean(("chain", "draw")).values
        theta_std = theta.std(("chain", "draw")).values * 2.58
        
        # 地理数据处理
        comm = comm.rename(columns={'id': 'location'})
        comm['location'] = pd.to_numeric(comm['location'], errors='coerce').astype(int)
        region_comm = comm[comm['location'] == target_location].copy()
        
        if region_comm.empty:
            print(f"区域 {model_id} 无地理数据")
            return
        
        # 假设每个区域有多个观测点，添加索引
        region_comm['obs_index'] = np.arange(len(region_comm))
        
        # 合并统计量
        stats_df = pd.DataFrame({
            'obs_index': np.arange(len(theta_mean)),
            'map': theta_mean,
            'cir': theta_std
        })
        viz_data = region_comm.merge(stats_df, on='obs_index', how='left').fillna(0)
        
        # 保存文件
        viz_data.to_file(f"{Config.viz_dir}/vsup_data_{model_id}.geojson", driver='GeoJSON')
        
        # 生成HTML（略）
            # 生成HTML地图
        html_template = f"""
        <!DOCTYPE html>
        <html>
        <head>
            <link rel="stylesheet" href="https://unpkg.com/leaflet@1.7.1/dist/leaflet.css"/>
            <style>#map {{ height: 600px; }}</style>
        </head>
        <body>
            <div id="map"></div>
            <script src="https://unpkg.com/leaflet@1.7.1/dist/leaflet.js"></script>
            <script>
                var map = L.map('map').setView([{Config.epicenter[0]}, {Config.epicenter[1]}], 11);
                L.tileLayer('https://{{s}}.tile.openstreetmap.org/{{z}}/{{x}}/{{y}}.png').addTo(map);
                
                function getColor(d) {{
                    return d > 8 ? '#800026' : 
                           d > 6 ? '#E31A1C' :
                           d > 4 ? '#FEB24C' :
                           d > 2 ? '#FED976' : '#FFEDA0';
                }}
                
                L.geoJSON({viz_data.to_json()}, {{
                    style: function(feature) {{
                        return {{
                            fillColor: getColor(feature.properties.map),
                            weight: 1,
                            opacity: 1,
                            color: 'white',
                            fillOpacity: 0.7
                        }};
                    }}
                }}).addTo(map);
            </script>
        </body>
        </html>
        """
        with open(os.path.join(Config.viz_dir, f"map_{model_id}.html"), "w") as f:
            f.write(html_template)
        
    except Exception as e:
        print(f"模型 {model_id} 可视化失败: {str(e)}")


# ----------------------------
# 颜色生成函数（需放在调用前）
# ----------------------------
def cir_to_level(cir):
    """CIR分层判断"""
    if cir <= 1.25:
        return 0, 1.25  # 第一阶
    elif cir <= 2.5:
        return 1, 2.5   # 第二阶
    else:
        return 2, 5.0   # 第三阶

def adjust_hue_luminance(hue, cir, rating):
    """非线性色相偏移与亮度补偿"""
    # 色相偏移规则
    if cir >= 3.75:
        if hue < 0.5:   # 蓝色系向青紫偏移
            hue = min(hue + 0.15, 0.7)
        else:           # 红色系向橙黄偏移
            hue = max(hue - 0.2, 0.05)
    
    # 亮度补偿规则（L范围 0-1）
    if cir <= 2.5:
        l = 0.5 + 0.2*(rating/10)
    else:
        l = 0.7 - 0.1*(rating/10)
    return hue, l

def map_to_color(map_val, cir):
    """核心颜色生成函数"""
    # 预处理：处理负值（假设原始评分范围-5~5）
    rating = np.clip((map_val + 5) * 1.0, 0, 10)  # 映射到0~10分
    
    # 确定CIR层级
    level, max_cir = cir_to_level(cir)
    
    # 基础色相映射（蓝→红渐变）
    hue_base = 0.66 * (1 - rating/10)  # 0.66=蓝色，0.0=红色
    
    # 饱和度计算（CIR越大饱和度越低）
    saturation = 0.9 - 0.5 * (cir / max_cir)
    
    # 色相与亮度调整
    hue, lightness = adjust_hue_luminance(hue_base, cir, rating)
    
    # 转换为RGB
    r, g, b = colorsys.hls_to_rgb(hue, lightness, saturation)
    return f"#{int(r*255):02x}{int(g*255):02x}{int(b*255):02x}"



# ----------------------------
def generate_visualization_hourly(trace, comm, model_id):
    """按小时生成VSUP文件（集成颜色与统计）"""
    try:
        print(f"\n正在生成模型 {model_id} 的可视化...")
        target_location = int(model_id)

        # ===== 初始化数据容器 =====
        all_rows = []
        
        # ===== 1. 基础验证 =====
        param_prefix = f"location_{model_id}::"
        if f"{param_prefix}theta" not in trace.posterior:
            print(f"模型 {model_id} 缺少theta参数")
            return
        
        # ===== 2. 获取时间数据 =====
        time_var_name = f"{param_prefix}time_var"
        if time_var_name not in trace.constant_data:
            print(f"模型 {model_id} 缺少时间数据")
            return
            
        time_values = trace.constant_data[time_var_name].values
        time_labels = [pd.to_datetime(t * 1e9).strftime("%m%d%H%M") for t in time_values]
        
        # ===== 3. 地理数据处理 =====
        comm = comm.rename(columns={'id': 'location'})
        comm['location'] = pd.to_numeric(comm['location'], errors='coerce').astype(int)
        region_comm = comm[comm['location'] == target_location].copy()
        
        if region_comm.empty:
            print(f"区域 {model_id} 无地理数据")
            return
            
        region_clean = region_comm[['location', 'geometry']].copy()
        
        # ===== 4. 按小时生成文件 =====
        theta = trace.posterior[f"{param_prefix}theta"]
        min_map, max_map = float('inf'), -float('inf')
        
        for i in range(theta.sizes['obs']):
            try:
                # 计算统计量
                theta_mean = theta.isel(obs=i).mean(("chain", "draw")).item()
                theta_std = theta.isel(obs=i).std(("chain", "draw")).item() * 2.58
                
                # 更新极值
                min_map = min(min_map, theta_mean)
                max_map = max(max_map, theta_mean)
                
                # 生成颜色
                color = map_to_color(theta_mean, theta_std)
                
                # 构建数据
                hour_data = region_clean.copy()
                hour_data['map'] = theta_mean
                hour_data['cir'] = theta_std
                hour_data['color'] = color
                
                # 保存文件
                # hour_data.to_file(
                #     f"{Config.viz_dir}/vsup_data_{model_id}_{time_labels[i]}.geojson",
                #     driver='GeoJSON'
                # )

                # 记录CSV行数据
                all_rows.append({
                    "time": time_labels[i],     # ISO格式时间
                    "location": model_id,       # 区域ID
                    "map": round(theta_mean,4), # 保留4位小数
                    "cir": round(theta_std,4),  # 保留4位小数
                    "color": color              # 颜色代码
                })
                
            except Exception as e:
                print(f"时间点 {time_labels[i]} 处理失败: {str(e)}")
                continue
        
        # 打印统计信息
        print(f"区域 {model_id} MAP范围: 最小值={min_map:.2f} 最大值={max_map:.2f}")

        # ===== 生成CSV文件 =====
        # csv_path = os.path.join(Config.viz_dir, "seismic_results.csv")
        csv_path = os.path.join(Config.viz_dir, f"seismic_{model_id}.csv")  # 按区域ID命名

        
        # 如果文件不存在，写入表头
        header = not os.path.exists(csv_path)
        
        # 转换为DataFrame并保存
        pd.DataFrame(all_rows).to_csv(
            csv_path,
            mode='a' if header else 'w',
            header=header,
            index=False,
            float_format="%.4f"  # 统一浮点数格式
        )
        
        # print(f"成功保存{len(all_rows)}条数据到{csv_path}")
        print(f"成功生成区域{model_id}数据 -> {os.path.basename(csv_path)} (共{len(all_rows)}行)")

                
    except Exception as e:
        print(f"模型 {model_id} 处理失败: {str(e)}")

def generate_html(viz_data, model_id, time_label):
    """生成带时间戳的HTML地图"""
    # 从时间标签解析可读时间
    display_time = f"{time_label[0:2]}-{time_label[2:4]} {time_label[4:6]}:{time_label[6:8]}"
    
    html_template = f"""
    <!DOCTYPE html>
    <html>
    <head>
        <title>区域 {model_id} {display_time}</title>
        <!-- Leaflet样式保持不变 -->
    </head>
    <body>
        <h2>区域 {model_id} - 时间: {display_time}</h2>
        <!-- 地图脚本保持不变 -->
    </body>
    </html>
    """
    with open(f"{Config.viz_dir}/map_{model_id}_{time_label}.html", "w") as f:
        f.write(html_template)
# --------------------------
# 主程序执行
# --------------------------
if __name__ == "__main__b":
    # 数据准备
    df_reports = load_reports()
    df_pga, comm = process_pga()  # 现在返回社区数据
    
    # 数据融合
    merged = merge_datasets(df_reports, df_pga,comm)
    
    # =====================
    # 分区域建模（核心调用）
    # =====================
    # 获取所有区域列表
    locations = merged['location'].unique().tolist()
    
    # 并行建模（自动调用N次，生成N个模型）
    traces = Parallel(n_jobs=4)(
        delayed(build_multi_obs_model)(
            merged[merged['location'] == loc],  # 自动过滤对应区域数据
            loc
        )
        for loc in locations
    )
    
    # =====================
    # 结果整合与保存
    # =====================
    # 将结果存入字典 {区域ID: trace}
    results = {
        loc: trace
        for loc, trace in zip(locations, traces)
    }
    
    for loc, trace in results.items():
        trace.to_netcdf(  # 直接调用 trace 对象的方法
            os.path.join(Config.model_dir, f"location_{loc}_trace.nc")
        )
    
    print(f"成功构建并保存 {len(locations)} 个区域模型")
    
    # 可视化

    # trace = az.from_netcdf(f"{Config.model_dir}/bsts_model.nc")
    # # trace = az.from_netcdf(f"{Config.model_dir}/location_3_trace.nc")

    # communities = gpd.read_file(os.path.join(Config.raw_data_dir, Config.communities_geojson))
    # generate_visualization(trace, communities)


    # test_df = pd.DataFrame({
    #     'time': pd.date_range('2020-04-06', periods=3, freq='h'),
    #     'location': ['A', 'A', 'B'],
    #     'composite_mmi': [5.0, 6.0, 7.0]
    # })

    # grouped = test_df.groupby([pd.Grouper(key='time', freq='h'), 'location']).mean()
    # print(grouped)

def main_pipeline():
    """主执行流程"""
     # 初始化目录
    Path(Config.viz_dir).mkdir(parents=True, exist_ok=True)
    
    # 加载地理数据
    communities = gpd.read_file(os.path.join(Config.raw_data_dir, Config.communities_geojson))
    
    # 遍历所有区域模型
    model_files = list(Path(Config.model_dir).glob("location_*_trace.nc"))
    print(f"发现 {len(model_files)} 个区域模型")
    
    for model_file in model_files:
        try:
            model_id = model_file.stem.split("_")[1]
            print(f"\n处理区域 {model_id}...")
            
            # 加载模型跟踪数据
            trace = az.from_netcdf(model_file)
            
            # 生成可视化
            # generate_visualization(trace, communities.copy(), model_id)
            generate_visualization_hourly(trace, communities.copy(), model_id)

            
            
        except Exception as e:
            print(f"区域 {model_id} 处理失败: {str(e)}")
            continue

if __name__ == "__main__":
    main_pipeline()