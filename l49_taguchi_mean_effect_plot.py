#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L49田口实验均值效应图生成器
使用taguchi_results_20250625_081216目录的数据
仿照MODABC参数Popsize, SN和TN的均值效应图格式
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path
import matplotlib
matplotlib.rcParams['font.sans-serif'] = ['SimHei']
matplotlib.rcParams['axes.unicode_minus'] = False

def load_taguchi_data(results_dir):
    """加载田口实验数据"""
    results_path = Path(results_dir)
    analysis_file = results_path / "taguchi_analysis.json"
    
    if not analysis_file.exists():
        raise FileNotFoundError(f"未找到分析文件: {analysis_file}")
    
    with open(analysis_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def create_individual_mean_effect_plots(data, results_dir):
    """为每个因子创建独立的均值效应图"""
    
    # 提取因子效应数据
    factor_effects = data['factor_effects']
    factor_names = {
        'A': '学习率',
        'B': 'ε衰减率', 
        'C': '分组比例',
        'D': '折扣因子'
    }
    
    # 为每个因子创建独立图表
    for factor_id, factor_data in factor_effects.items():
        plt.figure(figsize=(8, 6))
        
        # 提取水平和均值数据
        levels = list(range(1, 8))  # L49是7水平设计
        means = [factor_data[str(level)] for level in levels]
        
        # 绘制均值效应图 - 使用与整体图一致的样式
        plt.plot(levels, means, 'bo-', linewidth=2, markersize=6, markerfacecolor='blue')
        
        # 添加水平虚线（整体均值）
        overall_mean = np.mean(means)
        plt.axhline(y=overall_mean, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        # 设置标题和标签 - 与整体图一致
        plt.title(factor_names[factor_id], fontsize=14, fontweight='bold')
        plt.xlabel('水平值', fontsize=12)
        plt.ylabel('SNR', fontsize=12)
        
        # 设置网格
        plt.grid(True, alpha=0.3)
        
        # 设置x轴刻度
        plt.xticks(levels)
        plt.xlim(0.5, 7.5)
        
        # 调整y轴范围以突出差异
        y_min, y_max = min(means), max(means)
        y_range = y_max - y_min
        plt.ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        # 格式化坐标轴标签
        plt.tick_params(axis='both', which='major', labelsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图片
        output_path = Path(results_dir) / f'L49_因子{factor_id}_{factor_names[factor_id]}_均值效应图.png'
        plt.savefig(output_path, dpi=300, bbox_inches='tight', 
                    facecolor='white', edgecolor='none')
        plt.close()
        
        print(f"✅ 因子{factor_id}({factor_names[factor_id]})均值效应图已保存: {output_path}")

def print_analysis_summary(data):
    """打印分析摘要"""
    print("\n" + "="*60)
    print("L49田口实验分析摘要")
    print("="*60)
    
    # 最优参数组合
    optimal = data['optimal_combination']
    print(f"\n最优参数组合:")
    for factor, level in optimal.items():
        print(f"  {factor}: 水平 {level}")
    print(f"  预测SNR: {data['predicted_snr']:.4f}")
    
    # 方差分析
    if 'anova_results' in data:
        anova = data['anova_results']
        print(f"\n方差分析结果:")
        print(f"{'因子':<10} {'贡献率(%)':<12} {'F值':<12} {'显著性':<10}")
        print("-" * 40)
        for factor, stats in anova.items():
            contribution = stats.get('contribution', 0)
            f_value = stats.get('f_value', 0)
            print(f"{factor:<10} {contribution:<12.2f} {f_value:<12.3f}")
    
    # 因子重要性排序
    factor_effects = data['factor_effects']
    ranges = []
    for factor_id, factor_data in factor_effects.items():
        range_val = factor_data['range']
        rank = factor_data['rank']
        ranges.append((factor_id, range_val, rank))
    
    ranges.sort(key=lambda x: x[2])  # 按排名排序
    
    factor_names = {
        'A': '学习率',
        'B': 'ε衰减率',
        'C': '分组比例', 
        'D': '折扣因子'
    }
    
    print(f"\n因子重要性排序（按极差）:")
    for factor_id, range_val, rank in ranges:
        print(f"  {rank}. {factor_names[factor_id]} ({factor_id}): 极差={range_val:.3f}, 排名={rank}")

def main():
    # 设置结果目录
    results_dir = "taguchi_results_20250625_081216"
    
    print("🔬 开始生成L49田口实验独立均值效应图...")
    
    try:
        # 加载数据
        data = load_taguchi_data(results_dir)
        
        # 打印分析摘要
        print_analysis_summary(data)
        
        # 创建独立的均值效应图
        create_individual_mean_effect_plots(data, results_dir)
        
        print(f"\n✅ L49田口实验独立均值效应图生成完成！")
        print(f"📁 保存位置: {results_dir}/")
        
    except Exception as e:
        print(f"❌ 错误: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 