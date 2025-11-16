import pandas as pd
import numpy as np
import json

# 读取分析结果
with open('taguchi_results_20250624_172744/taguchi_analysis.json', 'r') as f:
    analysis = json.load(f)

# 读取Excel文件
df = pd.read_excel('taguchi_results_20250624_172744/l49_results_summary.xlsx')

print('📊 最新田口L49实验结果汇总 (10次重复)')
print('=' * 60)

# 显示基本信息
print(f'实验组数: {len(df)}')
print(f'实验列数: {len(df.columns)}')

# 按SNR排序
df_sorted = df.sort_values('SNR_Value', ascending=False)

print('\n🏆 前10名实验组 (按SNR排序):')
print('排名 | 实验组 | 学习率    | 衰减率  | 折扣因子 | 综合得分 | SNR(dB)')
print('-' * 75)

for i, (_, row) in enumerate(df_sorted.head(10).iterrows(), 1):
    print(f'{i:2d}   | {row["Exp_ID"]:2d}     | {row["A_LearningRate"]:.6f} | {row["B_EpsilonDecay"]:.4f} | {row["D_Gamma"]:.2f}    | {row["Comprehensive_Mean"]:.4f}   | {row["SNR_Value"]:.2f}')

print('\n📈 性能统计:')
print(f'最高SNR: {df["SNR_Value"].max():.3f} dB (实验组{df.loc[df["SNR_Value"].idxmax(), "Exp_ID"]})')
print(f'最低SNR: {df["SNR_Value"].min():.3f} dB (实验组{df.loc[df["SNR_Value"].idxmin(), "Exp_ID"]})')
print(f'平均SNR: {df["SNR_Value"].mean():.3f} ± {df["SNR_Value"].std():.3f} dB')
print(f'性能跨度: {df["SNR_Value"].max() - df["SNR_Value"].min():.3f} dB')

print('\n🎯 最优参数组合:')
best_exp = df_sorted.iloc[0]
print(f'实验组: {best_exp["Exp_ID"]}')
print(f'学习率: {best_exp["A_LearningRate"]:.6f}')
print(f'探索率衰减: {best_exp["B_EpsilonDecay"]:.4f}')
print(f'折扣因子: {best_exp["D_Gamma"]:.2f}')
print(f'综合得分: {best_exp["Comprehensive_Mean"]:.4f} ± {best_exp["Comprehensive_Std"]:.4f}')
print(f'成功运行: {best_exp["Successful_Runs"]}/10次')

print('\n🔍 田口方法分析:')
print(f'最优因子组合 (从田口分析): A={analysis["optimal_combination"]["A"]}, B={analysis["optimal_combination"]["B"]}, C={analysis["optimal_combination"]["C"]}, D={analysis["optimal_combination"]["D"]}')
print(f'预测最优SNR: {analysis["predicted_snr"]:.3f} dB')

print('\n📊 因子重要性排序:')
factors = ['A', 'B', 'C', 'D']
factor_names = ['学习率', '探索率衰减', '鹰群分组比例', '折扣因子']
for factor, name in zip(factors, factor_names):
    rank = analysis['factor_effects'][factor]['rank']
    range_val = analysis['factor_effects'][factor]['range']
    contribution = analysis['anova_results'][factor]['contribution']
    print(f'{rank}. {name}({factor}): 极差={range_val:.3f}, 贡献度={contribution:.2f}%')

print('\n💡 结论:')
print(f'1. 最佳性能: 实验组{best_exp["Exp_ID"]} (SNR = {best_exp["SNR_Value"]:.3f} dB)')
print(f'2. 性能改进: 相比最差配置提升了 {df["SNR_Value"].max() - df["SNR_Value"].min():.3f} dB')
print(f'3. 关键因子: {factor_names[analysis["factor_effects"]["C"]["rank"]-1]} 影响最大')
print(f'4. 实验稳定性: 标准差 = {df["SNR_Value"].std():.3f} dB') 