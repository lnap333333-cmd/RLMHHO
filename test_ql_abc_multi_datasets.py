#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
多数据集QL-ABC测试脚本
对修正版QL-ABC算法在不同规模问题上进行测试
"""

import sys
import os
import time
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from typing import List, Tuple, Dict
from datetime import datetime

# 添加项目路径
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.ql_abc_fixed import QLABC_Optimizer_Fixed


def generate_dataset_configs():
    """生成不同规模的数据集配置"""
    configs = []
    
    # 小规模数据集
    small_configs = [
        {'n_jobs': 10, 'n_factories': 2, 'n_stages': 2, 'name': 'Small-10J2F2S'},
        {'n_jobs': 15, 'n_factories': 2, 'n_stages': 2, 'name': 'Small-15J2F2S'},
        {'n_jobs': 20, 'n_factories': 3, 'n_stages': 2, 'name': 'Small-20J3F2S'},
    ]
    
    # 中规模数据集
    medium_configs = [
        {'n_jobs': 30, 'n_factories': 3, 'n_stages': 3, 'name': 'Medium-30J3F3S'},
        {'n_jobs': 40, 'n_factories': 4, 'n_stages': 3, 'name': 'Medium-40J4F3S'},
        {'n_jobs': 50, 'n_factories': 4, 'n_stages': 3, 'name': 'Medium-50J4F3S'},
    ]
    
    # 大规模数据集
    large_configs = [
        {'n_jobs': 60, 'n_factories': 5, 'n_stages': 4, 'name': 'Large-60J5F4S'},
        {'n_jobs': 80, 'n_factories': 5, 'n_stages': 4, 'name': 'Large-80J5F4S'},
        {'n_jobs': 100, 'n_factories': 6, 'n_stages': 4, 'name': 'Large-100J6F4S'},
    ]
    
    configs.extend(small_configs)
    configs.extend(medium_configs)
    configs.extend(large_configs)
    
    return configs


def generate_problem_data(config: Dict, seed: int = None) -> Dict:
    """根据配置生成问题数据"""
    if seed is not None:
        np.random.seed(seed)
    
    n_jobs = config['n_jobs']
    n_factories = config['n_factories']
    n_stages = config['n_stages']
    
    # 生成处理时间矩阵 [job][stage]
    processing_times = np.random.randint(10, 50, size=(n_jobs, n_stages))
    
    # 生成交货期（基于处理时间的1.5-2倍）
    job_total_times = np.sum(processing_times, axis=1)
    due_dates = np.random.uniform(1.5, 2.0, size=n_jobs) * job_total_times
    
    # 生成机器配置（每阶段2-5台机器）
    machines_per_stage = [np.random.randint(2, 6) for _ in range(n_stages)]
    
    # 构建problem_data字典
    problem_data = {
        'n_jobs': n_jobs,
        'n_factories': n_factories,
        'n_stages': n_stages,
        'machines_per_stage': machines_per_stage,
        'processing_times': processing_times.tolist(),
        'due_dates': due_dates.tolist(),
        'urgencies': [1.0] * n_jobs
    }
    
    return problem_data


def test_single_dataset(config: Dict, algorithm_params: Dict, runs: int = 3) -> Dict:
    """测试单个数据集"""
    print(f"\n测试数据集: {config['name']}")
    print(f"规模: {config['n_jobs']}个作业, {config['n_factories']}个工厂, {config['n_stages']}个阶段")
    
    results = []
    
    for run in range(runs):
        print(f"  运行 {run + 1}/{runs}...")
        
        # 生成问题数据
        problem_data = generate_problem_data(config, seed=42 + run)
        problem = MO_DHFSP_Problem(problem_data)
        
        # 创建优化器
        optimizer = QLABC_Optimizer_Fixed(
            problem=problem,
            **algorithm_params
        )
        
        # 执行优化
        start_time = time.time()
        pareto_front, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        if pareto_front:
            makespans = [sol.makespan for sol in pareto_front]
            tardinesses = [sol.total_tardiness for sol in pareto_front]
            
            result = {
                'run': run + 1,
                'pareto_size': len(pareto_front),
                'best_makespan': min(makespans),
                'worst_makespan': max(makespans),
                'avg_makespan': np.mean(makespans),
                'best_tardiness': min(tardinesses),
                'worst_tardiness': max(tardinesses),
                'avg_tardiness': np.mean(tardinesses),
                'runtime': end_time - start_time,
                'q_table_size': len(optimizer.q_table),
                'final_archive_size': len(optimizer.external_archive) if hasattr(optimizer, 'external_archive') else 0
            }
        else:
            result = {
                'run': run + 1,
                'pareto_size': 0,
                'best_makespan': float('inf'),
                'worst_makespan': float('inf'),
                'avg_makespan': float('inf'),
                'best_tardiness': float('inf'),
                'worst_tardiness': float('inf'),
                'avg_tardiness': float('inf'),
                'runtime': end_time - start_time,
                'q_table_size': len(optimizer.q_table),
                'final_archive_size': 0
            }
        
        results.append(result)
    
    # 计算统计信息
    if results:
        stats = {
            'dataset_name': config['name'],
            'n_jobs': config['n_jobs'],
            'n_factories': config['n_factories'],
            'n_stages': config['n_stages'],
            'avg_pareto_size': np.mean([r['pareto_size'] for r in results]),
            'avg_best_makespan': np.mean([r['best_makespan'] for r in results]),
            'avg_best_tardiness': np.mean([r['best_tardiness'] for r in results]),
            'avg_runtime': np.mean([r['runtime'] for r in results]),
            'avg_q_table_size': np.mean([r['q_table_size'] for r in results]),
            'std_pareto_size': np.std([r['pareto_size'] for r in results]),
            'std_best_makespan': np.std([r['best_makespan'] for r in results]),
            'std_best_tardiness': np.std([r['best_tardiness'] for r in results]),
            'std_runtime': np.std([r['runtime'] for r in results])
        }
    else:
        stats = {
            'dataset_name': config['name'],
            'n_jobs': config['n_jobs'],
            'n_factories': config['n_factories'],
            'n_stages': config['n_stages'],
            'avg_pareto_size': 0,
            'avg_best_makespan': float('inf'),
            'avg_best_tardiness': float('inf'),
            'avg_runtime': 0,
            'avg_q_table_size': 0,
            'std_pareto_size': 0,
            'std_best_makespan': 0,
            'std_best_tardiness': 0,
            'std_runtime': 0
        }
    
    return stats, results


def run_multi_dataset_test():
    """运行多数据集测试"""
    print("=" * 80)
    print("多数据集QL-ABC测试")
    print("=" * 80)
    
    # 生成数据集配置
    configs = generate_dataset_configs()
    
    # 算法参数设置
    algorithm_params = {
        'population_size': 50,
        'max_iterations': 100,
        'learning_rate': 0.4,
        'discount_factor': 0.8,
        'epsilon': 0.1
    }
    
    print(f"算法参数: {algorithm_params}")
    print(f"测试数据集数量: {len(configs)}")
    
    # 存储所有结果
    all_stats = []
    all_detailed_results = {}
    
    # 测试每个数据集
    for i, config in enumerate(configs):
        print(f"\n进度: {i+1}/{len(configs)}")
        
        stats, detailed_results = test_single_dataset(config, algorithm_params, runs=3)
        all_stats.append(stats)
        all_detailed_results[config['name']] = detailed_results
        
        # 打印统计信息
        print(f"  平均帕累托解数量: {stats['avg_pareto_size']:.1f} ± {stats['std_pareto_size']:.1f}")
        print(f"  平均最佳完工时间: {stats['avg_best_makespan']:.2f} ± {stats['std_best_makespan']:.2f}")
        print(f"  平均最佳总拖期: {stats['avg_best_tardiness']:.2f} ± {stats['std_best_tardiness']:.2f}")
        print(f"  平均运行时间: {stats['avg_runtime']:.2f} ± {stats['std_runtime']:.2f}秒")
        print(f"  平均Q表大小: {stats['avg_q_table_size']:.1f}")
    
    # 生成结果报告
    generate_test_report(all_stats, all_detailed_results, algorithm_params)
    
    return all_stats, all_detailed_results


def generate_test_report(all_stats: List[Dict], all_detailed_results: Dict, algorithm_params: Dict):
    """生成测试报告"""
    print("\n" + "=" * 80)
    print("测试报告")
    print("=" * 80)
    
    # 创建DataFrame
    df = pd.DataFrame(all_stats)
    
    # 打印汇总表格
    print("\n数据集性能汇总:")
    print(df.to_string(index=False, float_format='%.2f'))
    
    # 保存结果到CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_filename = f"ql_abc_multi_dataset_results_{timestamp}.csv"
    df.to_csv(csv_filename, index=False)
    print(f"\n结果已保存到: {csv_filename}")
    
    # 生成可视化图表
    generate_visualization_charts(df, all_detailed_results)
    
    # 生成详细分析
    generate_detailed_analysis(df, all_detailed_results)


def generate_visualization_charts(df: pd.DataFrame, all_detailed_results: Dict):
    """生成可视化图表"""
    print("\n生成可视化图表...")
    
    # 设置中文字体
    plt.rcParams['font.sans-serif'] = ['SimHei']
    plt.rcParams['axes.unicode_minus'] = False
    
    # 创建图表
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('QL-ABC多数据集测试结果', fontsize=16, fontweight='bold')
    
    # 1. 帕累托解数量 vs 问题规模
    axes[0, 0].scatter(df['n_jobs'], df['avg_pareto_size'], s=100, alpha=0.7, c='red')
    axes[0, 0].set_xlabel('作业数量')
    axes[0, 0].set_ylabel('平均帕累托解数量')
    axes[0, 0].set_title('帕累托解数量 vs 问题规模')
    axes[0, 0].grid(True, alpha=0.3)
    
    # 2. 最佳完工时间 vs 问题规模
    axes[0, 1].scatter(df['n_jobs'], df['avg_best_makespan'], s=100, alpha=0.7, c='blue')
    axes[0, 1].set_xlabel('作业数量')
    axes[0, 1].set_ylabel('平均最佳完工时间')
    axes[0, 1].set_title('最佳完工时间 vs 问题规模')
    axes[0, 1].grid(True, alpha=0.3)
    
    # 3. 最佳总拖期 vs 问题规模
    axes[0, 2].scatter(df['n_jobs'], df['avg_best_tardiness'], s=100, alpha=0.7, c='green')
    axes[0, 2].set_xlabel('作业数量')
    axes[0, 2].set_ylabel('平均最佳总拖期')
    axes[0, 2].set_title('最佳总拖期 vs 问题规模')
    axes[0, 2].grid(True, alpha=0.3)
    
    # 4. 运行时间 vs 问题规模
    axes[1, 0].scatter(df['n_jobs'], df['avg_runtime'], s=100, alpha=0.7, c='orange')
    axes[1, 0].set_xlabel('作业数量')
    axes[1, 0].set_ylabel('平均运行时间(秒)')
    axes[1, 0].set_title('运行时间 vs 问题规模')
    axes[1, 0].grid(True, alpha=0.3)
    
    # 5. Q表大小 vs 问题规模
    axes[1, 1].scatter(df['n_jobs'], df['avg_q_table_size'], s=100, alpha=0.7, c='purple')
    axes[1, 1].set_xlabel('作业数量')
    axes[1, 1].set_ylabel('平均Q表大小')
    axes[1, 1].set_title('Q表大小 vs 问题规模')
    axes[1, 1].grid(True, alpha=0.3)
    
    # 6. 性能综合对比
    dataset_names = df['dataset_name'].tolist()
    pareto_sizes = df['avg_pareto_size'].tolist()
    runtimes = df['avg_runtime'].tolist()
    
    x = np.arange(len(dataset_names))
    width = 0.35
    
    ax6 = axes[1, 2]
    bars1 = ax6.bar(x - width/2, pareto_sizes, width, label='帕累托解数量', alpha=0.7)
    ax6_twin = ax6.twinx()
    bars2 = ax6_twin.bar(x + width/2, runtimes, width, label='运行时间(秒)', alpha=0.7, color='orange')
    
    ax6.set_xlabel('数据集')
    ax6.set_ylabel('帕累托解数量', color='blue')
    ax6_twin.set_ylabel('运行时间(秒)', color='orange')
    ax6.set_title('性能综合对比')
    ax6.set_xticks(x)
    ax6.set_xticklabels([name.split('-')[1] for name in dataset_names], rotation=45)
    ax6.grid(True, alpha=0.3)
    
    # 添加图例
    lines1, labels1 = ax6.get_legend_handles_labels()
    lines2, labels2 = ax6_twin.get_legend_handles_labels()
    ax6.legend(lines1 + lines2, labels1 + labels2, loc='upper left')
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    chart_filename = f"ql_abc_multi_dataset_charts_{timestamp}.png"
    plt.savefig(chart_filename, dpi=300, bbox_inches='tight')
    print(f"图表已保存到: {chart_filename}")
    plt.show()
    
    # 生成详细性能分析图
    generate_detailed_performance_analysis(df, all_detailed_results)


def generate_detailed_performance_analysis(df: pd.DataFrame, all_detailed_results: Dict):
    """生成详细性能分析图"""
    print("\n生成详细性能分析图...")
    
    # 选择几个代表性数据集进行详细分析
    selected_datasets = ['Small-10J2F2S', 'Medium-30J3F3S', 'Large-60J5F4S']
    
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle('QL-ABC详细性能分析', fontsize=16, fontweight='bold')
    
    for i, dataset_name in enumerate(selected_datasets):
        if dataset_name in all_detailed_results:
            detailed_results = all_detailed_results[dataset_name]
            
            # 提取数据
            runs = [r['run'] for r in detailed_results]
            pareto_sizes = [r['pareto_size'] for r in detailed_results]
            best_makespans = [r['best_makespan'] for r in detailed_results]
            best_tardinesses = [r['best_tardiness'] for r in detailed_results]
            runtimes = [r['runtime'] for r in detailed_results]
            q_table_sizes = [r['q_table_size'] for r in detailed_results]
            
            # 绘制帕累托解数量
            axes[0, i].bar(runs, pareto_sizes, color='red', alpha=0.7)
            axes[0, i].set_xlabel('运行次数')
            axes[0, i].set_ylabel('帕累托解数量')
            axes[0, i].set_title(f'{dataset_name} - 帕累托解数量')
            axes[0, i].grid(True, alpha=0.3)
            
            # 绘制目标函数值
            ax_twin = axes[1, i].twinx()
            line1 = axes[1, i].plot(runs, best_makespans, 'b-o', label='最佳完工时间', linewidth=2)
            line2 = ax_twin.plot(runs, best_tardinesses, 'g-s', label='最佳总拖期', linewidth=2)
            
            axes[1, i].set_xlabel('运行次数')
            axes[1, i].set_ylabel('最佳完工时间', color='blue')
            ax_twin.set_ylabel('最佳总拖期', color='green')
            axes[1, i].set_title(f'{dataset_name} - 目标函数值')
            axes[1, i].grid(True, alpha=0.3)
            
            # 添加图例
            lines = line1 + line2
            labels = [l.get_label() for l in lines]
            axes[1, i].legend(lines, labels, loc='upper right')
    
    plt.tight_layout()
    
    # 保存图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    detail_chart_filename = f"ql_abc_detailed_analysis_{timestamp}.png"
    plt.savefig(detail_chart_filename, dpi=300, bbox_inches='tight')
    print(f"详细分析图表已保存到: {detail_chart_filename}")
    plt.show()


def generate_detailed_analysis(df: pd.DataFrame, all_detailed_results: Dict):
    """生成详细分析报告"""
    print("\n" + "=" * 80)
    print("详细分析报告")
    print("=" * 80)
    
    # 按规模分组分析
    small_datasets = df[df['n_jobs'] <= 20]
    medium_datasets = df[(df['n_jobs'] > 20) & (df['n_jobs'] <= 50)]
    large_datasets = df[df['n_jobs'] > 50]
    
    print(f"\n小规模数据集 ({len(small_datasets)}个):")
    if len(small_datasets) > 0:
        print(f"  平均帕累托解数量: {small_datasets['avg_pareto_size'].mean():.1f}")
        print(f"  平均最佳完工时间: {small_datasets['avg_best_makespan'].mean():.2f}")
        print(f"  平均最佳总拖期: {small_datasets['avg_best_tardiness'].mean():.2f}")
        print(f"  平均运行时间: {small_datasets['avg_runtime'].mean():.2f}秒")
    
    print(f"\n中规模数据集 ({len(medium_datasets)}个):")
    if len(medium_datasets) > 0:
        print(f"  平均帕累托解数量: {medium_datasets['avg_pareto_size'].mean():.1f}")
        print(f"  平均最佳完工时间: {medium_datasets['avg_best_makespan'].mean():.2f}")
        print(f"  平均最佳总拖期: {medium_datasets['avg_best_tardiness'].mean():.2f}")
        print(f"  平均运行时间: {medium_datasets['avg_runtime'].mean():.2f}秒")
    
    print(f"\n大规模数据集 ({len(large_datasets)}个):")
    if len(large_datasets) > 0:
        print(f"  平均帕累托解数量: {large_datasets['avg_pareto_size'].mean():.1f}")
        print(f"  平均最佳完工时间: {large_datasets['avg_best_makespan'].mean():.2f}")
        print(f"  平均最佳总拖期: {large_datasets['avg_best_tardiness'].mean():.2f}")
        print(f"  平均运行时间: {large_datasets['avg_runtime'].mean():.2f}秒")
    
    # 性能趋势分析
    print(f"\n性能趋势分析:")
    correlation_pareto = np.corrcoef(df['n_jobs'], df['avg_pareto_size'])[0, 1]
    correlation_makespan = np.corrcoef(df['n_jobs'], df['avg_best_makespan'])[0, 1]
    correlation_tardiness = np.corrcoef(df['n_jobs'], df['avg_best_tardiness'])[0, 1]
    correlation_runtime = np.corrcoef(df['n_jobs'], df['avg_runtime'])[0, 1]
    
    print(f"  帕累托解数量与问题规模相关性: {correlation_pareto:.3f}")
    print(f"  最佳完工时间与问题规模相关性: {correlation_makespan:.3f}")
    print(f"  最佳总拖期与问题规模相关性: {correlation_tardiness:.3f}")
    print(f"  运行时间与问题规模相关性: {correlation_runtime:.3f}")
    
    # 算法稳定性分析
    print(f"\n算法稳定性分析:")
    avg_std_pareto = df['std_pareto_size'].mean()
    avg_std_makespan = df['std_best_makespan'].mean()
    avg_std_tardiness = df['std_best_tardiness'].mean()
    avg_std_runtime = df['std_runtime'].mean()
    
    print(f"  帕累托解数量标准差平均值: {avg_std_pareto:.2f}")
    print(f"  最佳完工时间标准差平均值: {avg_std_makespan:.2f}")
    print(f"  最佳总拖期标准差平均值: {avg_std_tardiness:.2f}")
    print(f"  运行时间标准差平均值: {avg_std_runtime:.2f}")


if __name__ == "__main__":
    # 运行多数据集测试
    all_stats, all_detailed_results = run_multi_dataset_test()
    
    print("\n" + "=" * 80)
    print("多数据集测试完成！")
    print("=" * 80)
    print("修正版QL-ABC算法已成功在多个数据集上进行测试")
    print("测试结果包括:")
    print("1. 不同规模问题的性能表现")
    print("2. 算法稳定性和一致性")
    print("3. 计算复杂度分析")
    print("4. Q-learning学习效果")
    print("5. 帕累托前沿质量评估") 