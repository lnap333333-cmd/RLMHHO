import pandas as pd
import numpy as np
import json
import os
from pathlib import Path
import glob

def load_experiment_data(result_dir):
    """从JSON文件加载完整的实验数据"""
    
    print("📂 正在加载实验数据...")
    
    # 加载田口分析结果
    analysis_file = result_dir / "taguchi_analysis.json"
    with open(analysis_file, 'r', encoding='utf-8') as f:
        analysis_data = json.load(f)
    
    print(f"✅ 田口分析数据加载完成")
    
    # 从各个实验汇总文件构建DataFrame
    experiments = []
    
    # 找到所有实验汇总文件
    summary_files = list(result_dir.glob("exp_*_summary.json"))
    print(f"📄 找到{len(summary_files)}个实验汇总文件")
    
    for summary_file in summary_files:
        try:
            with open(summary_file, 'r', encoding='utf-8') as f:
                exp_data = json.load(f)
            
            exp_id = exp_data['exp_id']
            exp_config = exp_data['exp_config']
            
            # 从第一个运行获取参数配置
            if exp_data['individual_results']:
                first_result = exp_data['individual_results'][0]
                params = first_result['parameters']
                
                learning_rate = params['learning_rate']
                epsilon_decay = params['epsilon_decay']
                gamma = params['gamma']
                group_ratios = params['group_ratios']
                
                # 收集所有运行的性能数据
                hv_values = []
                igd_values = []
                gd_values = []
                comp_scores = []
                successful_runs = 0
                
                for result in exp_data['individual_results']:
                    if 'metrics' in result:
                        metrics = result['metrics']
                        if all(key in metrics for key in ['hypervolume', 'igd', 'gd']):
                            hv_values.append(metrics['hypervolume'])
                            igd_values.append(metrics['igd'])
                            gd_values.append(metrics['gd'])
                            
                            # 计算5:3:2加权综合得分
                            # 归一化处理 - 这里需要注意IGD和GD越小越好
                            hv_norm = metrics['hypervolume']  # HV越大越好
                            igd_norm = 1.0 / (1.0 + metrics['igd'])  # IGD越小越好，转换为越大越好
                            gd_norm = 1.0 / (1.0 + metrics['gd'])    # GD越小越好，转换为越大越好
                            
                            comp_score = 0.5 * hv_norm + 0.3 * igd_norm + 0.2 * gd_norm
                            comp_scores.append(comp_score)
                            successful_runs += 1
                
                if comp_scores:  # 确保有有效数据
                    # 计算统计量
                    hv_mean = np.mean(hv_values)
                    hv_std = np.std(hv_values) if len(hv_values) > 1 else 0
                    igd_mean = np.mean(igd_values)
                    igd_std = np.std(igd_values) if len(igd_values) > 1 else 0
                    gd_mean = np.mean(gd_values)
                    gd_std = np.std(gd_values) if len(gd_values) > 1 else 0
                    comp_mean = np.mean(comp_scores)
                    comp_std = np.std(comp_scores) if len(comp_scores) > 1 else 0
                    
                    # 计算SNR (信噪比，望大特性)
                    if len(comp_scores) > 0:
                        # 使用田口方法的信噪比计算公式：SNR = -10 * log10(mean(1/yi^2))
                        snr = -10 * np.log10(np.mean([1/(score**2) for score in comp_scores if score > 0]))
                    else:
                        snr = -100  # 如果没有有效数据
                    
                    experiments.append({
                        'Exp_ID': exp_id,
                        'A_LearningRate': learning_rate,
                        'B_EpsilonDecay': epsilon_decay,
                        'D_Gamma': gamma,
                        'C_GroupRatios': str(group_ratios),
                        'HV_Mean': hv_mean,
                        'HV_Std': hv_std,
                        'IGD_Mean': igd_mean,
                        'IGD_Std': igd_std,
                        'GD_Mean': gd_mean,
                        'GD_Std': gd_std,
                        'Comprehensive_Mean': comp_mean,
                        'Comprehensive_Std': comp_std,
                        'SNR_Value': snr,
                        'Successful_Runs': successful_runs
                    })
        
        except Exception as e:
            print(f"⚠️ 实验{summary_file.stem}加载失败: {e}")
            continue
    
    df = pd.DataFrame(experiments)
    df = df.sort_values('Exp_ID').reset_index(drop=True)
    
    print(f"✅ 数据构建完成: {len(df)}个实验组")
    
    return df, analysis_data

def analyze_taguchi_results():
    """完整分析田口实验结果"""
    
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
    analysis_file = result_dir / "taguchi_analysis.json"
    
    print(f"📈 田口分析文件: {'✅' if analysis_file.exists() else '❌'} {analysis_file}")
    
    if not analysis_file.exists():
        print("❌ 关键数据文件缺失，无法进行分析")
        return
    
    # 检查实验文件数量
    exp_files = list(result_dir.glob("exp_*_summary.json"))
    run_files = list(result_dir.glob("exp_*_run_*.json"))
    print(f"📄 实验汇总文件: {len(exp_files)}个")
    print(f"📄 运行详细文件: {len(run_files)}个")
    
    # 2. 加载数据
    print("\n📋 2. 数据加载和处理:")
    try:
        df, analysis_data = load_experiment_data(result_dir)
        print(f"✅ 数据处理完成: {len(df)}行 × {len(df.columns)}列")
        print(f"📊 列名: {list(df.columns)}")
    except Exception as e:
        print(f"❌ 数据加载失败: {e}")
        import traceback
        traceback.print_exc()
        return
    
    # 3. 基础统计分析
    print("\n📈 3. 基础统计分析:")
    print(f"实验组总数: {len(df)}")
    print(f"每组重复次数: 10次")
    print(f"总实验次数: {len(df) * 10}")
    print(f"实验成功率: {(df['Successful_Runs'].sum() / (len(df) * 10) * 100):.1f}%")
    
    # SNR统计
    valid_snr = df[df['SNR_Value'] > -100]['SNR_Value']  # 过滤掉无效数据
    if len(valid_snr) > 0:
        snr_stats = valid_snr.describe()
        print(f"\nSNR统计分析 (有效数据{len(valid_snr)}组):")
        print(f"  最大值: {snr_stats['max']:.3f} dB")
        print(f"  最小值: {snr_stats['min']:.3f} dB")
        print(f"  平均值: {snr_stats['mean']:.3f} dB")
        print(f"  标准差: {snr_stats['std']:.3f} dB")
        print(f"  中位数: {snr_stats['50%']:.3f} dB")
        print(f"  性能跨度: {snr_stats['max'] - snr_stats['min']:.3f} dB")
    else:
        print("⚠️ 没有有效的SNR数据")
        return
    
    # 4. 寻找最优结果
    print("\n🎯 4. 最优结果识别:")
    
    # 按SNR排序（排除无效数据）
    valid_df = df[df['SNR_Value'] > -100]
    if len(valid_df) == 0:
        print("❌ 没有有效的实验数据")
        return
        
    best_snr_idx = valid_df['SNR_Value'].idxmax()
    best_snr_row = valid_df.loc[best_snr_idx]
    
    print(f"\n按SNR最大值找到的最优解:")
    print(f"  实验组: {best_snr_row['Exp_ID']}")
    print(f"  SNR值: {best_snr_row['SNR_Value']:.3f} dB")
    print(f"  学习率: {best_snr_row['A_LearningRate']}")
    print(f"  探索率衰减: {best_snr_row['B_EpsilonDecay']}")
    print(f"  折扣因子: {best_snr_row['D_Gamma']}")
    print(f"  鹰群分组: {best_snr_row['C_GroupRatios']}")
    print(f"  综合得分: {best_snr_row['Comprehensive_Mean']:.4f} ± {best_snr_row['Comprehensive_Std']:.4f}")
    print(f"  HV均值: {best_snr_row['HV_Mean']:.4f} ± {best_snr_row['HV_Std']:.4f}")
    print(f"  IGD均值: {best_snr_row['IGD_Mean']:.4f} ± {best_snr_row['IGD_Std']:.4f}")
    print(f"  GD均值: {best_snr_row['GD_Mean']:.4f} ± {best_snr_row['GD_Std']:.4f}")
    print(f"  成功运行: {best_snr_row['Successful_Runs']}/10次")
    
    # 田口预测最优
    optimal_combo = analysis_data['optimal_combination']
    predicted_snr = analysis_data['predicted_snr']
    print(f"\n田口方法预测最优:")
    print(f"  预测组合: A={optimal_combo['A']}, B={optimal_combo['B']}, C={optimal_combo['C']}, D={optimal_combo['D']}")
    print(f"  预测SNR: {predicted_snr:.3f} dB")
    
    # 5. 田口方法因子效应分析
    print("\n🔬 5. 田口方法因子效应分析:")
    factors = ['A', 'B', 'C', 'D']
    factor_names = ['学习率', '探索率衰减', '鹰群分组比例', '折扣因子']
    
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
    
    # 按重要性排序
    factor_importance.sort(key=lambda x: x[0])
    
    print("因子重要性排序:")
    for rank, name, factor, range_val, contribution, f_value in factor_importance:
        print(f"{rank}. {name}({factor}): 极差={range_val:.3f}, 贡献度={contribution:.2f}%, F值={f_value:.3f}")
    
    # 6. 排序分析
    print("\n📊 6. 实验组性能排序分析:")
    df_sorted = valid_df.sort_values('SNR_Value', ascending=False).reset_index(drop=True)
    
    print("\n前15名实验组:")
    print("排名  实验组  学习率      衰减率    折扣因子  综合得分    SNR(dB)   HV均值   IGD均值  成功率")
    print("-" * 105)
    for i in range(min(15, len(df_sorted))):
        row = df_sorted.iloc[i]
        success_rate = row['Successful_Runs'] / 10 * 100
        print(f"{i+1:2d}    {row['Exp_ID']:2d}      {row['A_LearningRate']:.6f}  {row['B_EpsilonDecay']:.4f}   {row['D_Gamma']:.2f}     {row['Comprehensive_Mean']:.4f}     {row['SNR_Value']:6.2f}   {row['HV_Mean']:.4f}  {row['IGD_Mean']:5.1f}   {success_rate:3.0f}%")
    
    print("\n后10名实验组:")
    print("排名  实验组  学习率      衰减率    折扣因子  综合得分    SNR(dB)   HV均值   IGD均值  成功率")
    print("-" * 105)
    for i in range(max(0, len(df_sorted)-10), len(df_sorted)):
        row = df_sorted.iloc[i]
        success_rate = row['Successful_Runs'] / 10 * 100
        print(f"{i+1:2d}    {row['Exp_ID']:2d}      {row['A_LearningRate']:.6f}  {row['B_EpsilonDecay']:.4f}   {row['D_Gamma']:.2f}     {row['Comprehensive_Mean']:.4f}     {row['SNR_Value']:6.2f}   {row['HV_Mean']:.4f}  {row['IGD_Mean']:5.1f}   {success_rate:3.0f}%")
    
    # 7. 参数分布分析
    print("\n📈 7. 参数分布分析:")
    
    # 学习率分布
    lr_groups = valid_df.groupby('A_LearningRate')['SNR_Value'].agg(['mean', 'std', 'count'])
    print(f"\n学习率性能分析:")
    print("学习率        平均SNR    标准差    样本数")
    print("-" * 45)
    for lr, stats in lr_groups.iterrows():
        std_val = stats['std'] if not pd.isna(stats['std']) else 0.0
        print(f"{lr:.6f}     {stats['mean']:7.3f}   {std_val:6.3f}    {stats['count']:3.0f}")
    
    # 探索率衰减分布
    decay_groups = valid_df.groupby('B_EpsilonDecay')['SNR_Value'].agg(['mean', 'std', 'count'])
    print(f"\n探索率衰减性能分析:")
    print("衰减率     平均SNR    标准差    样本数")
    print("-" * 40)
    for decay, stats in decay_groups.iterrows():
        std_val = stats['std'] if not pd.isna(stats['std']) else 0.0
        print(f"{decay:.4f}    {stats['mean']:7.3f}   {std_val:6.3f}    {stats['count']:3.0f}")
    
    # 折扣因子分布
    gamma_groups = valid_df.groupby('D_Gamma')['SNR_Value'].agg(['mean', 'std', 'count'])
    print(f"\n折扣因子性能分析:")
    print("折扣因子   平均SNR    标准差    样本数")
    print("-" * 37)
    for gamma, stats in gamma_groups.iterrows():
        std_val = stats['std'] if not pd.isna(stats['std']) else 0.0
        print(f"{gamma:.2f}      {stats['mean']:7.3f}   {std_val:6.3f}    {stats['count']:3.0f}")
    
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
    
    # 检查无效SNR
    invalid_snr = len(df[df['SNR_Value'] <= -100])
    if invalid_snr > 0:
        print(f"⚠️ 发现{invalid_snr}个无效SNR值（≤-100 dB）")
        invalid_exps = df[df['SNR_Value'] <= -100]['Exp_ID'].tolist()
        print(f"   无效实验组: {invalid_exps}")
    else:
        print("✅ 所有SNR值都有效")
    
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
    print(f"✅ 性能表现:")
    print(f"   综合得分: {best_snr_row['Comprehensive_Mean']:.4f} ± {best_snr_row['Comprehensive_Std']:.4f}")
    print(f"   超体积(HV): {best_snr_row['HV_Mean']:.4f} ± {best_snr_row['HV_Std']:.4f}")
    print(f"   反向世代距离(IGD): {best_snr_row['IGD_Mean']:.2f} ± {best_snr_row['IGD_Std']:.2f}")
    print(f"   世代距离(GD): {best_snr_row['GD_Mean']:.2f} ± {best_snr_row['GD_Std']:.2f}")
    print(f"✅ 性能改进: 相比最差配置提升 {snr_stats['max'] - snr_stats['min']:.3f} dB")
    print(f"✅ 关键影响因子: {factor_importance[0][1]} (排名第1，贡献度{factor_importance[0][4]:.2f}%)")
    
    print("\n" + "=" * 80)
    print("🔍 深度分析完成！所有结果文件已生成。")
    print("=" * 80)

if __name__ == "__main__":
    analyze_taguchi_results() 