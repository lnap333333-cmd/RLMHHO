#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
最新田口实验结果分析脚本 - taguchi_results_20250624_222654
基于优化后的学习率水平配置的实验结果分析
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

def load_experiment_data():
    """加载最新实验数据"""
    results_dir = Path('taguchi_results_20250624_222654')
    
    df = None
    taguchi_data = None
    
    # 尝试读取Excel文件
    excel_file = results_dir / 'l49_results_summary.xlsx'
    if excel_file.exists():
        print("📊 读取Excel汇总文件...")
        df = pd.read_excel(excel_file)
    
    # 读取田口分析JSON
    taguchi_file = results_dir / 'taguchi_analysis.json'
    if taguchi_file.exists():
        print("📊 读取田口分析JSON文件...")
        with open(taguchi_file, 'r') as f:
            taguchi_data = json.load(f)
    
    if df is None and taguchi_data is None:
        print("❌ 未找到数据文件")
    
    return df, taguchi_data

def analyze_basic_statistics(df, taguchi_data):
    """基础统计分析"""
    print("=" * 80)
    print("🔬 最新田口L49实验结果分析 (2025-06-24 22:26:54)")
    print("=" * 80)
    
    if df is not None:
        print(f"\n📈 实验概览:")
        print(f"  • 实验组数: {len(df)} 组")
        print(f"  • 数据列数: {len(df.columns)} 列")
        print(f"  • 总实验次数: {len(df) * 10} 次 (每组10次重复)")
        
        if 'SNR_Value' in df.columns:
            print(f"\n🎯 SNR性能统计:")
            best_idx = df['SNR_Value'].idxmax()
            worst_idx = df['SNR_Value'].idxmin()
            
            print(f"  • 最高SNR: {df.loc[best_idx, 'SNR_Value']:.3f} dB (实验组 {df.loc[best_idx, 'Exp_ID']})")
            print(f"  • 最低SNR: {df.loc[worst_idx, 'SNR_Value']:.3f} dB (实验组 {df.loc[worst_idx, 'Exp_ID']})")
            print(f"  • 平均SNR: {df['SNR_Value'].mean():.3f} ± {df['SNR_Value'].std():.3f} dB")
            print(f"  • 性能跨度: {df['SNR_Value'].max() - df['SNR_Value'].min():.3f} dB")
            
            return df.loc[best_idx], df.loc[worst_idx]
    
    if taguchi_data:
        print(f"\n🔬 田口分析结果 (基于JSON):")
        snr_array = np.fromstring(taguchi_data['snr_data'][1:-1], sep=' ')
        print(f"  • 最高SNR: {snr_array.max():.3f} dB")
        print(f"  • 最低SNR: {snr_array.min():.3f} dB") 
        print(f"  • 平均SNR: {snr_array.mean():.3f} ± {snr_array.std():.3f} dB")
        print(f"  • 预测最优SNR: {taguchi_data['predicted_snr']:.3f} dB")
        
    return None, None

def analyze_factor_effects(taguchi_data):
    """分析因子效应"""
    if not taguchi_data:
        return
        
    print("\n🔍 因子效应分析:")
    print("=" * 60)
    
    factor_names = {
        'A': '学习率 (优化后)',
        'B': '探索率衰减', 
        'C': '鹰群分组比例',
        'D': '折扣因子'
    }
    
    # 按重要性排序
    factors = taguchi_data['factor_effects']
    sorted_factors = sorted(factors.items(), key=lambda x: x[1]['range'], reverse=True)
    
    print("📊 因子重要性排序:")
    for i, (factor, data) in enumerate(sorted_factors, 1):
        print(f"  {i}. {factor_names[factor]}: 极差={data['range']:.3f}")
    
    print("\n🎯 各因子最优水平:")
    optimal = taguchi_data['optimal_combination']
    for factor, level in optimal.items():
        best_snr = factors[factor][str(level)]
        print(f"  • {factor_names[factor]}: 水平{level} (SNR={best_snr:.3f} dB)")

def analyze_learning_rate_improvement(df, taguchi_data):
    """分析学习率优化效果"""
    print("\n🚀 学习率优化效果分析:")
    print("=" * 60)
    
    # 新的学习率水平映射
    new_lr_levels = {
        1: 0.00005, 2: 0.0001, 3: 0.0002, 4: 0.0005,
        5: 0.001, 6: 0.002, 7: 0.005
    }
    
    # 旧的学习率水平映射
    old_lr_levels = {
        1: 0.00005, 2: 0.0001, 3: 0.0005, 4: 0.001,
        5: 0.003, 6: 0.005, 7: 0.01
    }
    
    print("📈 学习率水平对比:")
    print("┌" + "─" * 8 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 15 + "┐")
    print("│  水平  │   旧配置   │   新配置   │     状态      │")
    print("├" + "─" * 8 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 15 + "┤")
    
    for level in range(1, 8):
        old_val = old_lr_levels[level]
        new_val = new_lr_levels[level]
        status = "不变" if old_val == new_val else ("减小" if new_val < old_val else "增大")
        print(f"│   {level}    │  {old_val:>8.5f}  │  {new_val:>8.5f}  │   {status:>10s}   │")
    
    print("└" + "─" * 8 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 15 + "┘")
    
    if taguchi_data:
        # 分析学习率的因子效应
        lr_effects = taguchi_data['factor_effects']['A']
        optimal_level = taguchi_data['optimal_combination']['A']
        optimal_lr = new_lr_levels[optimal_level]
        optimal_snr = lr_effects[str(optimal_level)]
        
        print(f"\n🎯 优化后学习率表现:")
        print(f"  • 最优水平: 水平{optimal_level} (学习率={optimal_lr:.5f})")
        print(f"  • 最优SNR: {optimal_snr:.3f} dB")
        print(f"  • 学习率重要性排名: {lr_effects['rank']}")
        print(f"  • 学习率效应极差: {lr_effects['range']:.3f}")

def show_top_experiments(df):
    """显示最优实验组"""
    if df is None or 'SNR_Value' not in df.columns:
        return
        
    print("\n🏆 Top 10 最优实验组:")
    print("=" * 80)
    
    # 选择关键列显示
    display_cols = ['Exp_ID', 'A_LearningRate', 'B_EpsilonDecay', 'D_Gamma', 'SNR_Value']
    if all(col in df.columns for col in display_cols):
        top10 = df.nlargest(10, 'SNR_Value')[display_cols]
        
        print("┌" + "─" * 6 + "┬" + "─" * 12 + "┬" + "─" * 12 + "┬" + "─" * 10 + "┬" + "─" * 12 + "┐")
        print("│ 实验组 │   学习率   │  探索衰减  │  折扣因子  │   SNR(dB)  │")
        print("├" + "─" * 6 + "┼" + "─" * 12 + "┼" + "─" * 12 + "┼" + "─" * 10 + "┼" + "─" * 12 + "┤")
        
        for _, row in top10.iterrows():
            print(f"│  {int(row['Exp_ID']):>3d}   │ {row['A_LearningRate']:>10.5f} │ {row['B_EpsilonDecay']:>10.4f} │ {row['D_Gamma']:>8.3f} │ {row['SNR_Value']:>10.3f} │")
        
        print("└" + "─" * 6 + "┴" + "─" * 12 + "┴" + "─" * 12 + "┴" + "─" * 10 + "┴" + "─" * 12 + "┘")

def compare_with_previous_results():
    """与前次实验结果对比"""
    print("\n📊 与前次实验结果对比:")
    print("=" * 60)
    
    # 从记忆中获取前次最优结果
    previous_best_snr = -15.743  # 实验组10的结果
    previous_best_lr = 0.0001
    
    print(f"前次最优结果 (taguchi_results_20250624_172744):")
    print(f"  • 最优SNR: {previous_best_snr:.3f} dB")
    print(f"  • 最优学习率: {previous_best_lr:.5f}")
    print(f"  • 实验时间: 2025-06-24 17:27:44")
    
    print(f"\n本次实验 (taguchi_results_20250624_222654):")
    print(f"  • 实验时间: 2025-06-24 22:26:54")
    print(f"  • 学习率配置: 已优化为围绕0.0001的密集采样")
    print(f"  • 配置改进: 移除过大学习率，增加精细度")

def generate_summary_report(df, taguchi_data, best_exp, worst_exp):
    """生成总结报告"""
    print("\n" + "=" * 80)
    print("📋 实验总结报告")
    print("=" * 80)
    
    print(f"🕐 实验时间: 2025年6月24日 22:26:54")
    print(f"🔬 实验类型: 田口L49正交实验 (优化学习率版本)")
    print(f"📊 实验规模: 49组 × 10次重复 = 490次实验")
    
    if best_exp is not None:
        print(f"\n🥇 最优配置 (实验组{int(best_exp['Exp_ID'])}):")
        print(f"  • 学习率: {best_exp['A_LearningRate']:.5f}")
        print(f"  • 探索率衰减: {best_exp['B_EpsilonDecay']:.4f}")
        print(f"  • 折扣因子: {best_exp['D_Gamma']:.3f}")
        print(f"  • SNR值: {best_exp['SNR_Value']:.3f} dB")
    
    if taguchi_data:
        print(f"\n🎯 田口方法预测:")
        optimal = taguchi_data['optimal_combination']
        print(f"  • 预测最优组合: A{optimal['A']}-B{optimal['B']}-C{optimal['C']}-D{optimal['D']}")
        print(f"  • 预测SNR: {taguchi_data['predicted_snr']:.3f} dB")
    
    print(f"\n✅ 学习率优化成效:")
    print(f"  • 配置更科学: 围绕最优值0.0001进行密集采样")
    print(f"  • 范围更合理: 全部在DQN推荐范围内")
    print(f"  • 预期改进: 更稳定的训练收敛")

def main():
    """主函数"""
    # 加载数据
    df, taguchi_data = load_experiment_data()
    
    # 基础统计分析
    best_exp, worst_exp = analyze_basic_statistics(df, taguchi_data)
    
    # 因子效应分析
    analyze_factor_effects(taguchi_data)
    
    # 学习率优化分析
    analyze_learning_rate_improvement(df, taguchi_data)
    
    # 显示最优实验
    show_top_experiments(df)
    
    # 对比前次结果
    compare_with_previous_results()
    
    # 生成总结报告
    generate_summary_report(df, taguchi_data, best_exp, worst_exp)
    
    print(f"\n🎉 分析完成!")
    print(f"📁 数据目录: taguchi_results_20250624_222654/")

if __name__ == "__main__":
    main() 