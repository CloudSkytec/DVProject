import arviz as az
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import seaborn as sns

class Config:
    model_dir = "./data2/models"
    viz_dir = "./data2/viz"
    report_path = "./data2/reports/model_diagnostics.csv"

def visualize_region(trace, region_id):
    """生成区域专属的可视化文件"""
    viz_dir = Path(Config.viz_dir)/str(region_id)
    viz_dir.mkdir(parents=True, exist_ok=True)
    
    # ======================
    # 1. 合并轨迹图
    # ======================
    param_prefix = f"location_{region_id}::"
    existing_params = [
        var_name for var_name in 
        [f"{param_prefix}sigma_state", f"{param_prefix}sigma_report", f"{param_prefix}sigma_pga"]
        if var_name in trace.posterior
    ]
    
    if existing_params:
        plt.figure(figsize=(12, 3 * len(existing_params)))
        az.plot_trace(trace, var_names=existing_params)
        plt.tight_layout()
        plt.savefig(viz_dir/"trace_plots.png", dpi=150)
        plt.close()
    
    # ======================
    # 2. 区域诊断图（新增）
    # ======================
    summary = az.summary(trace, skipna=True, round_to=3)
    
    plt.figure(figsize=(12, 4))
    
    # 参数后验分布
    if 'mean' in summary.columns:
        plt.subplot(131)
        sns.barplot(x=summary.index, y=summary['mean'], palette="Blues_r")
        plt.xticks(rotation=45)
        plt.title('Posterior Means')
    
    # ESS显示
    if 'ess_bulk' in summary.columns:
        plt.subplot(132)
        plt.bar(summary.index, summary['ess_bulk'], color='steelblue')
        plt.axhline(100, color='red', ls='--')
        plt.title('ESS Bulk')
        plt.xticks(rotation=45)
    
    # R-hat显示
    if 'r_hat' in summary.columns and not summary['r_hat'].isna().all():
        plt.subplot(133)
        sns.scatterplot(x=summary.index, y=summary['r_hat'], color='darkred')
        plt.axhline(1.01, color='grey', ls='--')
        plt.title('R-hat Values')
        plt.xticks(rotation=45)
    
    plt.tight_layout()
    plt.savefig(viz_dir/"diagnostics.png", dpi=150, bbox_inches='tight')
    plt.close()
    
    # 处理统计摘要
    summary.index = summary.index.str.replace(param_prefix, "")
    summary['region_id'] = region_id
    
    return summary

def analyze_all_regions():  # 修复1：移除错误的df参数
    """全区域批量分析（正确版本）"""
    model_files = list(Path(Config.model_dir).glob("location_*_trace.nc"))
    all_summaries = []
    
    for f in model_files:
        try:
            region_id = f.stem.split("_")[1]
            print(f"正在分析区域 {region_id}...")
            
            trace = az.from_netcdf(f)
            region_summary = visualize_region(trace, region_id)
            all_summaries.append(region_summary)
            
        except Exception as e:
            print(f"区域 {region_id} 分析失败: {str(e)}")
            continue
    
    if all_summaries:
        full_report = pd.concat(all_summaries)
        full_report.sort_values(by='r_hat', ascending=False, inplace=True)
        full_report.to_csv(Config.report_path, index=False)
        print(f"全局报告已保存至 {Config.report_path}")
        
        plot_global_diagnostics(full_report)  # 修复2：正确调用
    else:
        print("未找到有效区域分析结果")

def plot_global_diagnostics(df):
    """容错性更强的全局诊断（最终版）"""
    # 修复3：动态获取有效参数
    valid_params = [p for p in ['sigma_state', 'sigma_report'] if p in df.columns]
    has_ess = 'ess_bulk' in df.columns
    has_rhat = 'r_hat' in df.columns
    
    plt.figure(figsize=(14, 6))
    
    # 参数分布
    if valid_params:
        plt.subplot(131)
        sns.boxplot(data=df[valid_params], palette="Blues")
        plt.title(f'Parameters: {", ".join(valid_params)}')
    else:
        plt.subplot(131)
        plt.text(0.5, 0.5, 'No Parameters', ha='center')
        plt.axis('off')
    
    # ESS分布
    if has_ess:
        plt.subplot(132)
        plt.hist(df['ess_bulk'].dropna(), bins=20, color='steelblue')
        plt.axvline(100, color='red', ls='--')
        plt.title('ESS Bulk')
    else:
        plt.subplot(132)
        plt.text(0.5, 0.5, 'No ESS Data', ha='center')
        plt.axis('off')
    
    # R-hat分布
    if has_rhat:
        plt.subplot(133)
        sns.kdeplot(df['r_hat'].dropna(), fill=True, color='darkred')
        plt.axvline(1.01, color='grey', ls='--')
        plt.title('R-hat Distribution')
    else:
        plt.subplot(133)
        plt.text(0.5, 0.5, 'No R-hat Data', ha='center')
        plt.axis('off')
    
    plt.tight_layout()
    plt.savefig(f"{Config.viz_dir}/global_diagnostics.png", dpi=150)
    plt.close()

if __name__ == "__main__":
    analyze_all_regions()  # 修复4：无需参数