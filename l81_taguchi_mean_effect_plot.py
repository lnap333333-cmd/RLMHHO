#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
L81田口实验均值效应图生成器
仿照MODABC参数Popsize, SN和TN的均值效应图格式
"""

import matplotlib.pyplot as plt
import numpy as np
import json
import os
from matplotlib import rcParams

# 设置中文字体
rcParams['font.sans-serif'] = ['SimHei', 'Arial Unicode MS', 'DejaVu Sans']
rcParams['axes.unicode_minus'] = False

def load_l81_results():
    """加载L81田口实验结果"""
    result_dir = "taguchi_l81_results_20250626_084731"
    json_file = os.path.join(result_dir, "taguchi_analysis.json")
    
    if not os.path.exists(json_file):
        print(f"错误：找不到结果文件 {json_file}")
        return None
    
    with open(json_file, 'r', encoding='utf-8') as f:
        data = json.load(f)
    
    return data

def create_mean_effect_plot():
    """创建均值效应图"""
    # 加载数据
    data = load_l81_results()
    if data is None:
        return
    
    factor_effects = data['factor_effects']
    
    # 参数名称映射
    factor_names = {
        'A': '学习率',
        'B': 'ε衰减率', 
        'C': '分组比例',
        'D': '折扣因子'
    }
    
    # 创建图形
    fig, axes = plt.subplots(1, 4, figsize=(16, 4))
    fig.suptitle('L81田口实验参数均值效应图', fontsize=16, fontweight='bold')
    
    # 为每个因子绘制效应图
    for idx, (factor, name) in enumerate(factor_names.items()):
        ax = axes[idx]
        
        # 获取水平和效应值
        levels = list(range(1, 10))  # 1到9水平
        effects = [factor_effects[factor][str(level)] for level in levels]
        
        # 绘制折线图
        ax.plot(levels, effects, 'bo-', linewidth=2, markersize=6, markerfacecolor='blue')
        
        # 添加水平虚线（整体均值）
        overall_mean = np.mean(effects)
        ax.axhline(y=overall_mean, color='gray', linestyle='--', alpha=0.7, linewidth=1)
        
        # 设置标题和标签
        ax.set_title(name, fontsize=14, fontweight='bold')
        ax.set_xlabel('水平值', fontsize=12)
        if idx == 0:
            ax.set_ylabel('SNR', fontsize=12)
        
        # 设置网格
        ax.grid(True, alpha=0.3)
        
        # 设置x轴刻度
        ax.set_xticks(levels)
        ax.set_xlim(0.5, 9.5)
        
        # 调整y轴范围以突出差异
        y_min, y_max = min(effects), max(effects)
        y_range = y_max - y_min
        ax.set_ylim(y_min - 0.1 * y_range, y_max + 0.1 * y_range)
        
        # 格式化y轴标签
        ax.tick_params(axis='both', which='major', labelsize=10)
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图片
    output_file = "L81_田口实验均值效应图.png"
    plt.savefig(output_file, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    print(f"✅ 均值效应图已保存为: {output_file}")
    
    # 显示图片
    plt.show()
    
    return fig

def print_analysis_summary():
    """打印分析摘要"""
    data = load_l81_results()
    if data is None:
        return
    
    print("\n" + "="*60)
    print("L81田口实验分析摘要")
    print("="*60)
    
    # 最优组合
    optimal = data['optimal_combination']
    print(f"\n最优参数组合:")
    print(f"  学习率 (A): 水平 {optimal['A']}")
    print(f"  ε衰减率 (B): 水平 {optimal['B']}")
    print(f"  分组比例 (C): 水平 {optimal['C']}")
    print(f"  折扣因子 (D): 水平 {optimal['D']}")
    print(f"  预测SNR: {data['predicted_snr']:.4f}")
    
    # 方差分析结果
    anova = data['anova_results']
    print(f"\n方差分析结果:")
    print(f"{'因子':<8} {'贡献率(%)':<12} {'F值':<10} {'显著性':<8}")
    print("-" * 40)
    
    for factor in ['A', 'B', 'C', 'D']:
        contribution = anova[factor]['contribution']
        f_value = anova[factor]['f_value']
        significance = "**" if f_value > 2.0 else "*" if f_value > 1.0 else ""
        print(f"{factor:<8} {contribution:<12.2f} {f_value:<10.3f} {significance:<8}")
    
    # 因子重要性排序
    factor_importance = [(factor, anova[factor]['contribution']) 
                        for factor in ['A', 'B', 'C', 'D']]
    factor_importance.sort(key=lambda x: x[1], reverse=True)
    
    print(f"\n因子重要性排序:")
    factor_map = {'A': '学习率', 'B': 'ε衰减率', 'C': '分组比例', 'D': '折扣因子'}
    for i, (factor, contribution) in enumerate(factor_importance, 1):
        print(f"  {i}. {factor_map[factor]} ({factor}): {contribution:.2f}%")

def main():
    """主函数"""
    print("🔬 开始生成L81田口实验均值效应图...")
    
    # 打印分析摘要
    print_analysis_summary()
    
    # 创建均值效应图
    fig = create_mean_effect_plot()
    
    if fig is not None:
        print("\n✅ L81田口实验均值效应图生成完成！")
    else:
        print("\n❌ 图形生成失败，请检查数据文件。")

if __name__ == "__main__":
    main() 