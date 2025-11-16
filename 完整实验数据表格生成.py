#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整实验数据表格生成器
生成包含所有49个实验组的详细数据表格
"""

import pandas as pd
import json
import numpy as np
from pathlib import Path

def load_all_experiment_data():
    """加载所有实验数据"""
    results_dir = Path('taguchi_results_20250624_222654')
    
    print("📊 加载实验数据...")
    
    # 定义因子水平映射
    factor_levels = {
        'A_learning_rate': {
            1: 0.00005, 2: 0.0001, 3: 0.0002, 4: 0.0005,
            5: 0.001, 6: 0.002, 7: 0.005
        },
        'B_epsilon_decay': {
            1: 0.988, 2: 0.990, 3: 0.993, 4: 0.995,
            5: 0.997, 6: 0.999, 7: 0.9995
        },
        'C_group_ratios': {
            1: "超级探索主导 [0.70,0.15,0.10,0.05]",
            2: "极端探索主导 [0.60,0.20,0.15,0.05]",
            3: "探索主导 [0.50,0.30,0.15,0.05]",
            4: "基准平衡 [0.45,0.25,0.20,0.10]",
            5: "开发主导 [0.35,0.40,0.20,0.05]",
            6: "极端开发主导 [0.25,0.45,0.20,0.10]",
            7: "超级开发主导 [0.20,0.50,0.20,0.10]"
        },
        'D_gamma': {
            1: 0.90, 2: 0.93, 3: 0.95, 4: 0.97,
            5: 0.98, 6: 0.99, 7: 0.995
        }
    }
    
    # 寻找所有实验汇总文件
    summary_files = list(results_dir.glob('exp_*_summary.json'))
    print(f"找到 {len(summary_files)} 个实验汇总文件")
    
    all_experiments = []
    
    for file_path in sorted(summary_files):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            exp_id = data['exp_id']
            config = data['exp_config']
            stats = data.get('statistics', {})
            
            # 获取因子水平
            A_level = config['A']
            B_level = config['B']
            C_level = config['C']
            D_level = config['D']
            
            # 获取实际参数值
            learning_rate = factor_levels['A_learning_rate'][A_level]
            epsilon_decay = factor_levels['B_epsilon_decay'][B_level]
            group_ratios = factor_levels['C_group_ratios'][C_level]
            gamma = factor_levels['D_gamma'][D_level]
            
            # 获取性能指标
            comprehensive_score = stats.get('comprehensive_mean', 0)
            snr_value = stats.get('snr_value', 0)
            
            experiment_data = {
                '实验组': exp_id,
                'A_学习率水平': A_level,
                'A_学习率值': learning_rate,
                'B_探索率衰减水平': B_level,
                'B_探索率衰减值': epsilon_decay,
                'C_鹰群分组水平': C_level,
                'C_鹰群分组配置': group_ratios,
                'D_折扣因子水平': D_level,
                'D_折扣因子值': gamma,
                '加权得分': comprehensive_score,
                'SNR值': snr_value
            }
            
            all_experiments.append(experiment_data)
            
        except Exception as e:
            print(f"处理文件 {file_path} 时出错: {e}")
    
    # 转换为DataFrame
    df = pd.DataFrame(all_experiments)
    
    # 按SNR值排序（SNR越高越好）
    df = df.sort_values('SNR值', ascending=False).reset_index(drop=True)
    
    # 添加排名列
    df.insert(1, '排名', range(1, len(df) + 1))
    
    return df

def analyze_factor_effects(df):
    """分析因子效应"""
    print("\n📈 因子效应分析")
    print("=" * 50)
    
    factors = ['A_学习率水平', 'B_探索率衰减水平', 'C_鹰群分组水平', 'D_折扣因子水平']
    factor_names = ['学习率', '探索率衰减', '鹰群分组', '折扣因子']
    
    factor_effects = {}
    
    for factor, name in zip(factors, factor_names):
        level_means = df.groupby(factor)['SNR值'].mean()
        effect_range = level_means.max() - level_means.min()
        factor_effects[name] = effect_range
        
        print(f"\n{name} 因子:")
        for level in sorted(level_means.index):
            print(f"  水平 {level}: {level_means[level]:.3f} dB")
        print(f"  效应范围: {effect_range:.3f} dB")
    
    # 按效应范围排序
    sorted_effects = sorted(factor_effects.items(), key=lambda x: x[1], reverse=True)
    
    print(f"\n🎯 因子重要性排序:")
    for i, (factor, effect) in enumerate(sorted_effects, 1):
        print(f"  {i}. {factor}: {effect:.3f} dB")
    
    return factor_effects

def create_summary_table(df):
    """创建汇总表格"""
    print("\n📊 实验结果汇总统计")
    print("=" * 50)
    
    print(f"实验总数: {len(df)}")
    print(f"最高SNR: {df['SNR值'].max():.3f} dB (实验组 {df.loc[df['SNR值'].idxmax(), '实验组']})")
    print(f"最低SNR: {df['SNR值'].min():.3f} dB (实验组 {df.loc[df['SNR值'].idxmin(), '实验组']})")
    print(f"平均SNR: {df['SNR值'].mean():.3f} ± {df['SNR值'].std():.3f} dB")
    print(f"性能跨度: {df['SNR值'].max() - df['SNR值'].min():.3f} dB")
    
    print(f"\n最高加权得分: {df['加权得分'].max():.6f} (实验组 {df.loc[df['加权得分'].idxmax(), '实验组']})")
    print(f"最低加权得分: {df['加权得分'].min():.6f} (实验组 {df.loc[df['加权得分'].idxmin(), '实验组']})")
    print(f"平均加权得分: {df['加权得分'].mean():.6f} ± {df['加权得分'].std():.6f}")

def save_results(df):
    """保存结果到文件"""
    # 保存完整数据表
    excel_file = '田口实验完整数据表_20250624_222654.xlsx'
    with pd.ExcelWriter(excel_file, engine='openpyxl') as writer:
        # 主要数据表
        df.to_excel(writer, sheet_name='完整数据表', index=False)
        
        # Top 10 表
        top10_df = df.head(10).copy()
        top10_df.to_excel(writer, sheet_name='Top10最优组', index=False)
        
        # Bottom 10 表  
        bottom10_df = df.tail(10).copy()
        bottom10_df.to_excel(writer, sheet_name='Bottom10最差组', index=False)
        
        # 简化表格（只包含关键信息）
        simple_df = df[['实验组', '排名', 'A_学习率值', 'B_探索率衰减值', 
                       'C_鹰群分组配置', 'D_折扣因子值', '加权得分', 'SNR值']].copy()
        simple_df.to_excel(writer, sheet_name='简化数据表', index=False)
    
    print(f"\n💾 结果已保存到: {excel_file}")
    
    # 保存CSV格式
    csv_file = '田口实验完整数据表_20250624_222654.csv'
    df.to_csv(csv_file, index=False, encoding='utf-8-sig')
    print(f"💾 CSV格式已保存到: {csv_file}")
    
    return excel_file, csv_file

def main():
    """主函数"""
    print("🔬 田口实验完整数据表格生成器")
    print("=" * 60)
    
    try:
        # 加载数据
        df = load_all_experiment_data()
        
        if df.empty:
            print("❌ 没有找到有效的实验数据")
            return
        
        # 显示数据概览
        print(f"\n✅ 成功加载 {len(df)} 个实验的数据")
        
        # 分析因子效应
        factor_effects = analyze_factor_effects(df)
        
        # 创建汇总统计
        create_summary_table(df)
        
        # 显示Top 10结果
        print("\n🏆 Top 10 最优实验组:")
        print("=" * 50)
        top10_display = df.head(10)[['实验组', '排名', 'A_学习率值', 'B_探索率衰减值', 
                                    'D_折扣因子值', '加权得分', 'SNR值']].copy()
        print(top10_display.to_string(index=False, float_format='%.6f'))
        
        # 保存结果
        excel_file, csv_file = save_results(df)
        
        print("\n🎉 分析完成！")
        print(f"📈 共分析了 {len(df)} 个实验组的数据")
        print(f"🥇 最优实验组: 第{df.iloc[0]['实验组']}组 (SNR: {df.iloc[0]['SNR值']:.3f} dB)")
        
    except Exception as e:
        print(f"❌ 程序执行出错: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    main() 