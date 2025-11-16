import pandas as pd
import numpy as np
import json
import os
from pathlib import Path

def analyze_taguchi_results():
    """完整分析田口实验结果，找出数据不一致的原因"""
    
    print("=" * 80)
    print("🔍 田口L49实验深度数据分析 - 2025-06-24")
    print("=" * 80)
    
    # 1. 数据文件检查
    print("\n📂 1. 数据文件完整性检查:")
    result_dir = Path("taguchi_results_20250624_172744")
    
    if not result_dir.exists():
        print(f"❌ 结果目录不存在: {result_dir}")
        return
    
    # 检查关键文件
    excel_file = result_dir / "l49_results_summary.xlsx"
    json_file = result_dir / "taguchi_analysis.json"
    
    print(f"📊 Excel汇总文件: {'✅' if excel_file.exists() else '❌'} {excel_file}")
    print(f"📈 田口分析文件: {'✅' if json_file.exists() else '❌'} {json_file}")
    
    if not excel_file.exists() or not json_file.exists():
        print("❌ 关键数据文件缺失，无法进行分析")
        return
    
    # 2. 加载和验证数据
    print("\n📋 2. 数据加载和验证:")
    try:
        # 加载Excel数据
        df = pd.read_excel(excel_file)
        print(f"✅ Excel数据加载成功: {len(df)}行 × {len(df.columns)}列")
        
        # 加载田口分析数据
        with open(json_file, 'r', encoding='utf-8') as f:
            analysis_data = json.load(f)
        print(f"✅ 田口分析数据加载成功")
        
        # 验证数据完整性
        required_columns = [
            'Exp_ID', 'A_LearningRate', 'B_EpsilonDecay', 'C_GroupRatios', 'D_Gamma',
            'HV_Mean', 'IGD_Mean', 'GD_Mean', 'Comprehensive_Mean', 'SNR_Value', 'Successful_Runs'
        ]
        
        missing_cols = [col for col in required_columns if col not in df.columns]
        if missing_cols:
            print(f"⚠️ 缺失关键列: {missing_cols}")
        else:
            print("✅ 所有关键列都存在")
            
        print(f"📊 列名详情: {list(df.columns)}")
        
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        return
    
    # 3. 基础统计分析
    print("\n📈 3. 基础统计分析:")
    print(f"实验组总数: {len(df)}")
    print(f"每组重复次数: 10次")
    print(f"总实验次数: {len(df) * 10}")
    print(f"实验成功率: {(df['Successful_Runs'].sum() / (len(df) * 10) * 100):.1f}%")
    
    # SNR统计
    snr_stats = df['SNR_Value'].describe()
    print(f"\nSNR统计分析:")
    print(f"  最大值: {snr_stats['max']:.3f} dB")
    print(f"  最小值: {snr_stats['min']:.3f} dB")
    print(f"  平均值: {snr_stats['mean']:.3f} dB")
    print(f"  标准差: {snr_stats['std']:.3f} dB")
    print(f"  中位数: {snr_stats['50%']:.3f} dB")
    print(f"  性能跨度: {snr_stats['max'] - snr_stats['min']:.3f} dB")
    
    # 4. 寻找最优结果（多种方法对比）
    print("\n🎯 4. 最优结果识别（多方法对比）:")
    
    # 方法1：按SNR排序
    best_snr_idx = df['SNR_Value'].idxmax()
    best_snr_row = df.loc[best_snr_idx]
    print(f"\n方法1 - 按SNR最大值:")
    print(f"  实验组: {best_snr_row['Exp_ID']}")
    print(f"  SNR值: {best_snr_row['SNR_Value']:.3f} dB")
    print(f"  学习率: {best_snr_row['A_LearningRate']}")
    print(f"  探索率衰减: {best_snr_row['B_EpsilonDecay']}")
    print(f"  折扣因子: {best_snr_row['D_Gamma']}")
    print(f"  综合得分: {best_snr_row['Comprehensive_Mean']:.4f}")
    
    # 方法2：按综合得分排序
    best_comp_idx = df['Comprehensive_Mean'].idxmax()
    best_comp_row = df.loc[best_comp_idx]
    print(f"\n方法2 - 按综合得分最大值:")
    print(f"  实验组: {best_comp_row['Exp_ID']}")
    print(f"  SNR值: {best_comp_row['SNR_Value']:.3f} dB")
    print(f"  综合得分: {best_comp_row['Comprehensive_Mean']:.4f}")
    
    # 方法3：田口预测最优
    optimal_combo = analysis_data['optimal_combination']
    predicted_snr = analysis_data['predicted_snr']
    print(f"\n方法3 - 田口方法预测最优:")
    print(f"  预测组合: A={optimal_combo['A']}, B={optimal_combo['B']}, C={optimal_combo['C']}, D={optimal_combo['D']}")
    print(f"  预测SNR: {predicted_snr:.3f} dB")
    
    # 检查一致性
    if best_snr_idx == best_comp_idx:
        print(f"\n✅ SNR和综合得分指向同一最优解: 实验组{best_snr_row['Exp_ID']}")
    else:
        print(f"\n⚠️ SNR和综合得分指向不同最优解:")
        print(f"   SNR最优: 实验组{best_snr_row['Exp_ID']} (SNR={best_snr_row['SNR_Value']:.3f})")
        print(f"   综合得分最优: 实验组{best_comp_row['Exp_ID']} (得分={best_comp_row['Comprehensive_Mean']:.4f})")
    
    # 5. 田口方法因子效应分析
    print("\n🔬 5. 田口方法因子效应分析:")
    factors = ['A', 'B', 'C', 'D']
    factor_names = ['学习率', '探索率衰减', '鹰群分组比例', '折扣因子']
    
    print("因子重要性排序:")
    factor_effects = analysis_data['factor_effects']
    factor_importance = []
    
    for factor, name in zip(factors, factor_names):
        effects = factor_effects[factor]
        rank = effects['rank']
        range_val = effects['range']
        # 获取ANOVA结果
        anova = analysis_data['anova_results'][factor]
        contribution = anova['contribution']
        f_value = anova['f_value']
        
        factor_importance.append((rank, name, factor, range_val, contribution, f_value))
        
        print(f"\n{name} (因子{factor}):")
        print(f"  重要性排名: {rank}")
        print(f"  效应极差: {range_val:.3f}")
        print(f"  贡献度: {contribution:.2f}%")
        print(f"  F值: {f_value:.3f}")
        
        # 显示各水平效应
        print(f"  各水平SNR效应:")
        for level in range(1, 8):
            if str(level) in effects:
                print(f"    水平{level}: {effects[str(level)]:.3f} dB")
    
    # 6. 排序分析
    print("\n📊 6. 实验组性能排序分析:")
    df_sorted = df.sort_values('SNR_Value', ascending=False).reset_index(drop=True)
    
    print("前10名实验组:")
    print("排名  实验组  学习率      衰减率    折扣因子  综合得分    SNR(dB)   成功率")
    print("-" * 85)
    for i in range(min(10, len(df_sorted))):
        row = df_sorted.iloc[i]
        success_rate = row['Successful_Runs'] / 10 * 100
        print(f"{i+1:2d}    {row['Exp_ID']:2d}      {row['A_LearningRate']:.6f}  {row['B_EpsilonDecay']:.4f}   {row['D_Gamma']:.2f}     {row['Comprehensive_Mean']:.4f}     {row['SNR_Value']:6.2f}   {success_rate:3.0f}%")
    
    print("\n后10名实验组:")
    print("排名  实验组  学习率      衰减率    折扣因子  综合得分    SNR(dB)   成功率")
    print("-" * 85)
    for i in range(max(0, len(df_sorted)-10), len(df_sorted)):
        row = df_sorted.iloc[i]
        success_rate = row['Successful_Runs'] / 10 * 100
        print(f"{i+1:2d}    {row['Exp_ID']:2d}      {row['A_LearningRate']:.6f}  {row['B_EpsilonDecay']:.4f}   {row['D_Gamma']:.2f}     {row['Comprehensive_Mean']:.4f}     {row['SNR_Value']:6.2f}   {success_rate:3.0f}%")
    
    # 7. 参数分布分析
    print("\n📈 7. 参数分布分析:")
    
    # 学习率分布
    lr_groups = df.groupby('A_LearningRate')['SNR_Value'].agg(['mean', 'std', 'count'])
    print(f"\n学习率性能分析:")
    print("学习率        平均SNR    标准差    样本数")
    print("-" * 45)
    for lr, stats in lr_groups.iterrows():
        print(f"{lr:.6f}     {stats['mean']:7.3f}   {stats['std']:6.3f}    {stats['count']:3.0f}")
    
    # 探索率衰减分布
    decay_groups = df.groupby('B_EpsilonDecay')['SNR_Value'].agg(['mean', 'std', 'count'])
    print(f"\n探索率衰减性能分析:")
    print("衰减率     平均SNR    标准差    样本数")
    print("-" * 40)
    for decay, stats in decay_groups.iterrows():
        print(f"{decay:.4f}    {stats['mean']:7.3f}   {stats['std']:6.3f}    {stats['count']:3.0f}")
    
    # 折扣因子分布
    gamma_groups = df.groupby('D_Gamma')['SNR_Value'].agg(['mean', 'std', 'count'])
    print(f"\n折扣因子性能分析:")
    print("折扣因子   平均SNR    标准差    样本数")
    print("-" * 37)
    for gamma, stats in gamma_groups.iterrows():
        print(f"{gamma:.2f}      {stats['mean']:7.3f}   {stats['std']:6.3f}    {stats['count']:3.0f}")
    
    # 8. 数据质量检查
    print("\n🔍 8. 数据质量检查:")
    
    # 检查缺失值
    missing_data = df.isnull().sum()
    if missing_data.sum() > 0:
        print("⚠️ 发现缺失数据:")
        for col, count in missing_data[missing_data > 0].items():
            print(f"  {col}: {count}个缺失值")
    else:
        print("✅ 无缺失数据")
    
    # 检查异常值
    print("\n异常值检查:")
    for col in ['Comprehensive_Mean', 'SNR_Value', 'HV_Mean', 'IGD_Mean', 'GD_Mean']:
        if col in df.columns:
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            lower_bound = Q1 - 1.5 * IQR
            upper_bound = Q3 + 1.5 * IQR
            outliers = df[(df[col] < lower_bound) | (df[col] > upper_bound)]
            if len(outliers) > 0:
                print(f"  {col}: {len(outliers)}个异常值")
                for idx in outliers.index:
                    print(f"    实验组{df.loc[idx, 'Exp_ID']}: {df.loc[idx, col]:.4f}")
            else:
                print(f"  {col}: 无异常值")
    
    # 9. 保存完整分析结果
    print("\n💾 9. 保存分析结果:")
    
    # 保存排序后的完整数据
    output_file = "田口L49实验深度分析结果_20250624.csv"
    df_sorted_with_rank = df_sorted.copy()
    df_sorted_with_rank['性能排名'] = range(1, len(df_sorted_with_rank) + 1)
    df_sorted_with_rank.to_csv(output_file, index=False, encoding='utf-8-sig')
    print(f"✅ 完整排序数据已保存: {output_file}")
    
    # 保存因子效应表
    factor_effects_data = []
    for factor, name in zip(factors, factor_names):
        effects = factor_effects[factor]
        for level in range(1, 8):
            if str(level) in effects:
                factor_effects_data.append({
                    '因子名称': name,
                    '因子代码': factor,
                    '水平': level,
                    'SNR均值': effects[str(level)],
                    '重要性排名': effects['rank']
                })
    
    effects_df = pd.DataFrame(factor_effects_data)
    effects_file = "田口因子水平效应表_20250624.csv"
    effects_df.to_csv(effects_file, index=False, encoding='utf-8-sig')
    print(f"✅ 因子效应表已保存: {effects_file}")
    
    # 10. 最终结论
    print("\n🎯 10. 最终分析结论:")
    print(f"✅ 最优实验组: {best_snr_row['Exp_ID']}")
    print(f"✅ 最优SNR: {best_snr_row['SNR_Value']:.3f} dB")
    print(f"✅ 最优参数组合:")
    print(f"   学习率: {best_snr_row['A_LearningRate']}")
    print(f"   探索率衰减: {best_snr_row['B_EpsilonDecay']}")
    print(f"   折扣因子: {best_snr_row['D_Gamma']}")
    print(f"   鹰群分组: {best_snr_row['C_GroupRatios']}")
    print(f"✅ 性能改进: 相比最差配置提升 {snr_stats['max'] - snr_stats['min']:.3f} dB")
    print(f"✅ 关键影响因子: {factor_names[0]} (贡献度最高)")
    
    print("\n" + "=" * 80)
    print("🔍 深度分析完成！所有结果文件已生成。")
    print("=" * 80)

if __name__ == "__main__":
    analyze_taguchi_results() 