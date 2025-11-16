#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速表格格式测试程序
测试增强表格功能和帕累托前沿图
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from utils.data_generator import DataGenerator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_custom_urgencies(n_jobs: int, urgency_range):
    """生成指定范围的紧急度"""
    min_val, avg_val, max_val = urgency_range
    
    # 生成正态分布的紧急度
    std_dev = (max_val - min_val) / 6
    urgencies = np.random.normal(avg_val, std_dev, n_jobs)
    urgencies = np.clip(urgencies, min_val, max_val)
    
    # 确保边界值存在
    urgencies[0] = min_val
    urgencies[1] = max_val
    urgencies[2] = avg_val
    
    return urgencies.tolist()

def run_algorithm_test(problem_data, alg_name, alg_class, alg_params):
    """运行单个算法测试"""
    print(f"正在测试 {alg_name}...")
    
    try:
        # 创建问题实例
        problem = MO_DHFSP_Problem(problem_data)
        
        # 创建优化器
        optimizer = alg_class(problem, **alg_params)
        
        # 运行优化
        start_time = time.time()
        pareto_solutions, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        runtime = end_time - start_time
        
        # 计算统计结果
        if pareto_solutions:
            # 计算各种指标
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            weighted_objs = [0.55 * sol.makespan + 0.45 * sol.total_tardiness for sol in pareto_solutions]
            
            results = {
                'weighted_best': min(weighted_objs),
                'weighted_mean': np.mean(weighted_objs),
                'makespan_best': min(makespans),
                'makespan_mean': np.mean(makespans),
                'tardiness_best': min(tardiness),
                'tardiness_mean': np.mean(tardiness),
                'runtime': runtime,
                'pareto_solutions': pareto_solutions
            }
        else:
            results = {
                'weighted_best': float('inf'),
                'weighted_mean': float('inf'),
                'makespan_best': float('inf'),
                'makespan_mean': float('inf'),
                'tardiness_best': float('inf'),
                'tardiness_mean': float('inf'),
                'runtime': runtime,
                'pareto_solutions': []
            }
        
        print(f"  {alg_name} 完成:")
        print(f"    加权目标值: 最优={results['weighted_best']:.2f}, 均值={results['weighted_mean']:.2f}")
        print(f"    完工时间: 最优={results['makespan_best']:.2f}, 均值={results['makespan_mean']:.2f}")
        print(f"    总拖期: 最优={results['tardiness_best']:.2f}, 均值={results['tardiness_mean']:.2f}")
        print(f"    运行时间: {results['runtime']:.2f}s")
        print(f"    帕累托解数: {len(pareto_solutions) if pareto_solutions else 0}")
        
        return results
        
    except Exception as e:
        print(f"  {alg_name} 运行失败: {str(e)}")
        return {
            'weighted_best': float('inf'),
            'weighted_mean': float('inf'),
            'makespan_best': float('inf'),
            'makespan_mean': float('inf'),
            'tardiness_best': float('inf'),
            'tardiness_mean': float('inf'),
            'runtime': 0.0,
            'pareto_solutions': []
        }

def plot_pareto_comparison(results, scale):
    """绘制帕累托前沿对比图"""
    
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green']
    markers = ['o', 's', '^']
    algorithm_names = ['RL-Chaotic-HHO', 'NSGA-II', 'MOEA/D']
    
    for i, alg_name in enumerate(algorithm_names):
        if alg_name in results and results[alg_name]['pareto_solutions']:
            pareto_solutions = results[alg_name]['pareto_solutions']
            
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            
            plt.scatter(makespans, tardiness, 
                      c=colors[i], marker=markers[i], 
                      label=f'{alg_name} ({len(pareto_solutions)}个解)',
                      alpha=0.7, s=50)
    
    plt.xlabel('完工时间 (Makespan)', fontsize=12)
    plt.ylabel('总拖期 (Total Tardiness)', fontsize=12)
    plt.title(f'{scale}规模 - 帕累托前沿对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/{scale}规模_帕累托前沿对比_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"帕累托前沿对比图已保存: {filename}")
    return filename

def generate_tables(results, scale, urgency_ddt):
    """生成三个对比表格"""
    
    print("\n" + "=" * 100)
    print("实验结果汇总表格")
    print("=" * 100)
    
    # 1. 加权目标函数表格
    print(f"\n1. 加权目标函数对比 (F = 0.55*F1 + 0.45*F2)")
    print("结果格式: (最优解, 均值, 收敛时间)")
    print("-" * 100)
    print(f"{'规模':^10s} | {'紧急度DDT':^15s} | {'RL-Chaotic-HHO':^20s} | {'NSGA-II':^20s} | {'MOEA/D':^20s}")
    print("-" * 100)
    
    rl_result = results.get('RL-Chaotic-HHO', {})
    rl_str = f"({rl_result.get('weighted_best', 0):.1f},{rl_result.get('weighted_mean', 0):.1f},{rl_result.get('runtime', 0):.2f})"
    
    nsga_result = results.get('NSGA-II', {})
    nsga_str = f"({nsga_result.get('weighted_best', 0):.1f},{nsga_result.get('weighted_mean', 0):.1f},{nsga_result.get('runtime', 0):.2f})"
    
    moead_result = results.get('MOEA/D', {})
    moead_str = f"({moead_result.get('weighted_best', 0):.1f},{moead_result.get('weighted_mean', 0):.1f},{moead_result.get('runtime', 0):.2f})"
    
    print(f"{scale:^10s} | {str(urgency_ddt):^15s} | {rl_str:^20s} | {nsga_str:^20s} | {moead_str:^20s}")
    
    # 2. 完工时间表格
    print(f"\n2. 完工时间对比")
    print("-" * 100)
    print(f"{'规模':^10s} | {'紧急度DDT':^15s} | {'RL-Chaotic-HHO':^20s} | {'NSGA-II':^20s} | {'MOEA/D':^20s}")
    print("-" * 100)
    
    rl_str = f"({rl_result.get('makespan_best', 0):.1f},{rl_result.get('makespan_mean', 0):.1f},{rl_result.get('runtime', 0):.2f})"
    nsga_str = f"({nsga_result.get('makespan_best', 0):.1f},{nsga_result.get('makespan_mean', 0):.1f},{nsga_result.get('runtime', 0):.2f})"
    moead_str = f"({moead_result.get('makespan_best', 0):.1f},{moead_result.get('makespan_mean', 0):.1f},{moead_result.get('runtime', 0):.2f})"
    
    print(f"{scale:^10s} | {str(urgency_ddt):^15s} | {rl_str:^20s} | {nsga_str:^20s} | {moead_str:^20s}")
    
    # 3. 总拖期表格
    print(f"\n3. 总拖期对比")
    print("-" * 100)
    print(f"{'规模':^10s} | {'紧急度DDT':^15s} | {'RL-Chaotic-HHO':^20s} | {'NSGA-II':^20s} | {'MOEA/D':^20s}")
    print("-" * 100)
    
    rl_str = f"({rl_result.get('tardiness_best', 0):.1f},{rl_result.get('tardiness_mean', 0):.1f},{rl_result.get('runtime', 0):.2f})"
    nsga_str = f"({nsga_result.get('tardiness_best', 0):.1f},{nsga_result.get('tardiness_mean', 0):.1f},{nsga_result.get('runtime', 0):.2f})"
    moead_str = f"({moead_result.get('tardiness_best', 0):.1f},{moead_result.get('tardiness_mean', 0):.1f},{moead_result.get('runtime', 0):.2f})"
    
    print(f"{scale:^10s} | {str(urgency_ddt):^15s} | {rl_str:^20s} | {nsga_str:^20s} | {moead_str:^20s}")

def quick_test():
    """快速测试增强表格功能"""
    
    print("快速测试增强表格格式实验")
    print("=" * 60)
    
    # 测试配置
    config = {
        'scale': '20×5×3',
        'n_jobs': 20,
        'n_factories': 5,
        'n_stages': 3,
        'machines_per_stage': [3, 3, 3],
        'urgency_ddt': [0.9, 1.9, 2.9],
        'processing_time_range': (1, 20)
    }
    
    # 生成问题数据
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=config['n_jobs'],
        n_factories=config['n_factories'],
        n_stages=config['n_stages'],
        machines_per_stage=config['machines_per_stage'],
        processing_time_range=config['processing_time_range'],
        due_date_tightness=1.5
    )
    
    # 使用自定义紧急度
    problem_data['urgencies'] = generate_custom_urgencies(
        config['n_jobs'], 
        config['urgency_ddt']
    )
    
    # 验证紧急度
    urgencies = np.array(problem_data['urgencies'])
    print(f"实验规模: {config['scale']}")
    print(f"紧急度DDT: {config['urgency_ddt']}")
    print(f"实际紧急度范围: [{urgencies.min():.2f}, {urgencies.max():.2f}]")
    print(f"紧急度均值: {urgencies.mean():.2f}")
    
    # 算法配置 - 快速测试
    algorithms = {
        'RL-Chaotic-HHO': {
            'class': RL_ChaoticHHO_Optimizer,
            'params': {'max_iterations': 30}
        },
        'NSGA-II': {
            'class': NSGA2_Optimizer,
            'params': {
                'population_size': 30,
                'max_generations': 30,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1
            }
        },
        'MOEA/D': {
            'class': MOEAD_Optimizer,
            'params': {
                'population_size': 30,
                'max_generations': 30,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1,
                'neighbor_size': 10,
                'delta': 0.9,
                'nr': 2
            }
        }
    }
    
    # 运行算法测试
    results = {}
    print(f"\n{'='*60}")
    
    for alg_name, alg_config in algorithms.items():
        results[alg_name] = run_algorithm_test(
            problem_data,
            alg_name,
            alg_config['class'],
            alg_config['params']
        )
        print()
    
    # 生成表格
    generate_tables(results, config['scale'], config['urgency_ddt'])
    
    # 绘制帕累托前沿对比图
    print(f"\n绘制帕累托前沿对比图...")
    plot_filename = plot_pareto_comparison(results, config['scale'])
    
    # 保存结果到文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_filename = f"results/快速测试报告_{timestamp}.txt"
    
    with open(report_filename, 'w', encoding='utf-8') as f:
        f.write("快速测试增强表格格式实验报告\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"实验规模: {config['scale']}\n")
        f.write(f"紧急度DDT: {config['urgency_ddt']}\n")
        f.write(f"实际紧急度范围: [{urgencies.min():.2f}, {urgencies.max():.2f}]\n")
        f.write(f"紧急度均值: {urgencies.mean():.2f}\n\n")
        
        f.write("算法结果详情:\n")
        f.write("-" * 40 + "\n")
        
        for alg_name, result in results.items():
            f.write(f"{alg_name}:\n")
            f.write(f"  加权目标值: 最优={result['weighted_best']:.2f}, 均值={result['weighted_mean']:.2f}\n")
            f.write(f"  完工时间: 最优={result['makespan_best']:.2f}, 均值={result['makespan_mean']:.2f}\n")
            f.write(f"  总拖期: 最优={result['tardiness_best']:.2f}, 均值={result['tardiness_mean']:.2f}\n")
            f.write(f"  运行时间: {result['runtime']:.2f}s\n")
            f.write(f"  帕累托解数: {len(result['pareto_solutions'])}\n\n")
        
        f.write(f"帕累托前沿对比图: {plot_filename}\n")
        f.write(f"报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n快速测试报告已保存: {report_filename}")
    print("\n快速测试完成!")

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 运行快速测试
    quick_test() 