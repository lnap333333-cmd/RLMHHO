#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
完整表格格式算法对比实验程序
按照图片中的格式进行多规模对比实验
包含加权目标函数、完工时间、总拖期三个表格和帕累托前沿对比图
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from utils.data_generator import DataGenerator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_custom_urgencies(n_jobs: int, urgency_range: List[float]) -> List[float]:
    """生成指定范围的紧急度"""
    min_val, avg_val, max_val = urgency_range
    
    # 生成正态分布的紧急度，均值为avg_val
    std_dev = (max_val - min_val) / 6
    urgencies = np.random.normal(avg_val, std_dev, n_jobs)
    urgencies = np.clip(urgencies, min_val, max_val)
    
    # 确保边界值存在
    urgencies[0] = min_val
    urgencies[1] = max_val
    urgencies[2] = avg_val
    
    return urgencies.tolist()

def run_algorithm_experiment(problem_data, alg_name, alg_class, alg_params, runs=3):
    """运行算法实验，多次运行取平均"""
    print(f"  正在运行 {alg_name} ({runs}次运行)...")
    
    weighted_values = []
    makespan_values = []
    tardiness_values = []
    runtimes = []
    all_pareto_solutions = []
    
    for run in range(runs):
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
            runtimes.append(runtime)
            
            # 计算指标
            if pareto_solutions:
                # 计算各种指标的最优值
                makespans = [sol.makespan for sol in pareto_solutions]
                tardiness = [sol.total_tardiness for sol in pareto_solutions]
                weighted_objs = [0.55 * sol.makespan + 0.45 * sol.total_tardiness for sol in pareto_solutions]
                
                weighted_values.append(min(weighted_objs))
                makespan_values.append(min(makespans))
                tardiness_values.append(min(tardiness))
                
                # 收集帕累托解
                if run == 0:  # 只收集第一次运行的解用于绘图
                    all_pareto_solutions = pareto_solutions
            else:
                weighted_values.append(float('inf'))
                makespan_values.append(float('inf'))
                tardiness_values.append(float('inf'))
            
            print(f"    运行 {run+1}/{runs}: 加权目标={weighted_values[-1]:.2f}, 完工时间={makespan_values[-1]:.2f}, 拖期={tardiness_values[-1]:.2f}, 时间={runtime:.2f}s")
            
        except Exception as e:
            print(f"    运行 {run+1}/{runs} 失败: {str(e)}")
            weighted_values.append(float('inf'))
            makespan_values.append(float('inf'))
            tardiness_values.append(float('inf'))
            runtimes.append(0.0)
    
    # 计算统计结果
    valid_weighted = [v for v in weighted_values if v != float('inf')]
    valid_makespan = [v for v in makespan_values if v != float('inf')]
    valid_tardiness = [v for v in tardiness_values if v != float('inf')]
    
    if valid_weighted:
        results = {
            'weighted_best': min(valid_weighted),
            'weighted_mean': np.mean(valid_weighted),
            'makespan_best': min(valid_makespan),
            'makespan_mean': np.mean(valid_makespan),
            'tardiness_best': min(valid_tardiness),
            'tardiness_mean': np.mean(valid_tardiness),
            'runtime': np.mean(runtimes),
            'pareto_solutions': all_pareto_solutions
        }
    else:
        results = {
            'weighted_best': float('inf'),
            'weighted_mean': float('inf'),
            'makespan_best': float('inf'),
            'makespan_mean': float('inf'),
            'tardiness_best': float('inf'),
            'tardiness_mean': float('inf'),
            'runtime': 0.0,
            'pareto_solutions': []
        }
    
    return results

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
    
    print(f"  帕累托前沿对比图已保存: {filename}")
    return filename

def run_complete_table_experiments():
    """运行完整的表格格式对比实验"""
    
    print("完整表格格式算法对比实验")
    print("=" * 80)
    
    # 实验配置 - 按照图片中的规模
    experiment_configs = [
        {
            'scale': '20×5×3',
            'n_jobs': 20,
            'n_factories': 5,
            'n_stages': 3,
            'machines_per_stage': [3, 3, 3],
            'urgency_ddt': [0.9, 1.9, 2.9],
            'processing_time_range': (1, 20)
        },
        {
            'scale': '20×5×4',
            'n_jobs': 20,
            'n_factories': 5,
            'n_stages': 4,
            'machines_per_stage': [3, 3, 3, 3],
            'urgency_ddt': [0.8, 1.8, 2.8],
            'processing_time_range': (1, 20)
        },
        {
            'scale': '50×5×3',
            'n_jobs': 50,
            'n_factories': 5,
            'n_stages': 3,
            'machines_per_stage': [4, 5, 6],
            'urgency_ddt': [2.45, 3.45, 4.45],
            'processing_time_range': (1, 25)
        }
    ]
    
    # 算法配置
    algorithms = {
        'RL-Chaotic-HHO': {
            'class': RL_ChaoticHHO_Optimizer,
            'params': {'max_iterations': 50}
        },
        'NSGA-II': {
            'class': NSGA2_Optimizer,
            'params': {
                'population_size': 50,
                'max_generations': 50,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1
            }
        },
        'MOEA/D': {
            'class': MOEAD_Optimizer,
            'params': {
                'population_size': 50,
                'max_generations': 50,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1,
                'neighbor_size': 10,
                'delta': 0.9,
                'nr': 2
            }
        }
    }
    
    # 存储所有实验结果
    all_results = {}
    plot_files = []
    
    # 为每个规模配置运行实验
    for config in experiment_configs:
        scale = config['scale']
        print(f"\n{'='*60}")
        print(f"实验规模: {scale}")
        print(f"紧急度DDT: {config['urgency_ddt']}")
        print(f"{'='*60}")
        
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
        
        # 验证紧急度范围
        urgencies = np.array(problem_data['urgencies'])
        print(f"实际紧急度范围: [{urgencies.min():.2f}, {urgencies.max():.2f}]")
        print(f"紧急度均值: {urgencies.mean():.2f}")
        
        # 存储该规模的结果
        all_results[scale] = {}
        
        # 为每个算法运行实验
        for alg_name, alg_config in algorithms.items():
            print(f"\n运行 {alg_name}...")
            
            result = run_algorithm_experiment(
                problem_data,
                alg_name,
                alg_config['class'],
                alg_config['params'],
                runs=3
            )
            
            all_results[scale][alg_name] = result
            
            print(f"  {alg_name} 结果:")
            print(f"    加权目标值: 最优={result['weighted_best']:.2f}, 均值={result['weighted_mean']:.2f}")
            print(f"    完工时间: 最优={result['makespan_best']:.2f}, 均值={result['makespan_mean']:.2f}")
            print(f"    总拖期: 最优={result['tardiness_best']:.2f}, 均值={result['tardiness_mean']:.2f}")
            print(f"    运行时间: {result['runtime']:.2f}s")
        
        # 绘制该规模的帕累托前沿对比图
        print(f"\n绘制 {scale} 规模的帕累托前沿对比图...")
        plot_file = plot_pareto_comparison(all_results[scale], scale)
        plot_files.append(plot_file)
    
    # 生成完整的表格报告
    generate_complete_table_report(all_results, experiment_configs, plot_files)

def generate_complete_table_report(all_results, configs, plot_files):
    """生成完整的表格格式报告"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/完整表格格式对比报告_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("完整表格格式算法对比实验报告\n")
        f.write("=" * 120 + "\n\n")
        
        f.write("实验说明:\n")
        f.write("- 加权目标函数: F = 0.55*F1 + 0.45*F2 (F1=完工时间, F2=总拖期)\n")
        f.write("- 结果格式: (最优解, 均值, 收敛时间)\n")
        f.write("- 每个算法运行3次取统计结果\n")
        f.write("- 算法: RL-Chaotic-HHO, NSGA-II, MOEA/D\n\n")
        
        # 1. 加权目标函数表格
        f.write("1. 加权目标函数对比表格\n")
        f.write("+" + "-" * 15 + "+" + "-" * 20 + "+" + "-" * 25 + "+" + "-" * 25 + "+" + "-" * 25 + "+\n")
        f.write(f"| {'规模':^13s} | {'紧急度DDT':^18s} | {'RL-Chaotic-HHO':^23s} | {'NSGA-II':^23s} | {'MOEA/D':^23s} |\n")
        f.write("+" + "-" * 15 + "+" + "-" * 20 + "+" + "-" * 25 + "+" + "-" * 25 + "+" + "-" * 25 + "+\n")
        
        for config in configs:
            scale = config['scale']
            urgency_str = f"{config['urgency_ddt']}"
            
            if scale in all_results:
                rl_result = all_results[scale].get('RL-Chaotic-HHO', {})
                rl_str = f"({rl_result.get('weighted_best', 0):.1f},{rl_result.get('weighted_mean', 0):.1f},{rl_result.get('runtime', 0):.2f})"
                
                nsga_result = all_results[scale].get('NSGA-II', {})
                nsga_str = f"({nsga_result.get('weighted_best', 0):.1f},{nsga_result.get('weighted_mean', 0):.1f},{nsga_result.get('runtime', 0):.2f})"
                
                moead_result = all_results[scale].get('MOEA/D', {})
                moead_str = f"({moead_result.get('weighted_best', 0):.1f},{moead_result.get('weighted_mean', 0):.1f},{moead_result.get('runtime', 0):.2f})"
            else:
                rl_str = nsga_str = moead_str = "(--,--,--)"
            
            f.write(f"| {scale:^13s} | {urgency_str:^18s} | {rl_str:^23s} | {nsga_str:^23s} | {moead_str:^23s} |\n")
        
        f.write("+" + "-" * 15 + "+" + "-" * 20 + "+" + "-" * 25 + "+" + "-" * 25 + "+" + "-" * 25 + "+\n\n")
        
        # 2. 完工时间表格
        f.write("2. 完工时间对比表格\n")
        f.write("+" + "-" * 15 + "+" + "-" * 20 + "+" + "-" * 25 + "+" + "-" * 25 + "+" + "-" * 25 + "+\n")
        f.write(f"| {'规模':^13s} | {'紧急度DDT':^18s} | {'RL-Chaotic-HHO':^23s} | {'NSGA-II':^23s} | {'MOEA/D':^23s} |\n")
        f.write("+" + "-" * 15 + "+" + "-" * 20 + "+" + "-" * 25 + "+" + "-" * 25 + "+" + "-" * 25 + "+\n")
        
        for config in configs:
            scale = config['scale']
            urgency_str = f"{config['urgency_ddt']}"
            
            if scale in all_results:
                rl_result = all_results[scale].get('RL-Chaotic-HHO', {})
                rl_str = f"({rl_result.get('makespan_best', 0):.1f},{rl_result.get('makespan_mean', 0):.1f},{rl_result.get('runtime', 0):.2f})"
                
                nsga_result = all_results[scale].get('NSGA-II', {})
                nsga_str = f"({nsga_result.get('makespan_best', 0):.1f},{nsga_result.get('makespan_mean', 0):.1f},{nsga_result.get('runtime', 0):.2f})"
                
                moead_result = all_results[scale].get('MOEA/D', {})
                moead_str = f"({moead_result.get('makespan_best', 0):.1f},{moead_result.get('makespan_mean', 0):.1f},{moead_result.get('runtime', 0):.2f})"
            else:
                rl_str = nsga_str = moead_str = "(--,--,--)"
            
            f.write(f"| {scale:^13s} | {urgency_str:^18s} | {rl_str:^23s} | {nsga_str:^23s} | {moead_str:^23s} |\n")
        
        f.write("+" + "-" * 15 + "+" + "-" * 20 + "+" + "-" * 25 + "+" + "-" * 25 + "+" + "-" * 25 + "+\n\n")
        
        # 3. 总拖期表格
        f.write("3. 总拖期对比表格\n")
        f.write("+" + "-" * 15 + "+" + "-" * 20 + "+" + "-" * 25 + "+" + "-" * 25 + "+" + "-" * 25 + "+\n")
        f.write(f"| {'规模':^13s} | {'紧急度DDT':^18s} | {'RL-Chaotic-HHO':^23s} | {'NSGA-II':^23s} | {'MOEA/D':^23s} |\n")
        f.write("+" + "-" * 15 + "+" + "-" * 20 + "+" + "-" * 25 + "+" + "-" * 25 + "+" + "-" * 25 + "+\n")
        
        for config in configs:
            scale = config['scale']
            urgency_str = f"{config['urgency_ddt']}"
            
            if scale in all_results:
                rl_result = all_results[scale].get('RL-Chaotic-HHO', {})
                rl_str = f"({rl_result.get('tardiness_best', 0):.1f},{rl_result.get('tardiness_mean', 0):.1f},{rl_result.get('runtime', 0):.2f})"
                
                nsga_result = all_results[scale].get('NSGA-II', {})
                nsga_str = f"({nsga_result.get('tardiness_best', 0):.1f},{nsga_result.get('tardiness_mean', 0):.1f},{nsga_result.get('runtime', 0):.2f})"
                
                moead_result = all_results[scale].get('MOEA/D', {})
                moead_str = f"({moead_result.get('tardiness_best', 0):.1f},{moead_result.get('tardiness_mean', 0):.1f},{moead_result.get('runtime', 0):.2f})"
            else:
                rl_str = nsga_str = moead_str = "(--,--,--)"
            
            f.write(f"| {scale:^13s} | {urgency_str:^18s} | {rl_str:^23s} | {nsga_str:^23s} | {moead_str:^23s} |\n")
        
        f.write("+" + "-" * 15 + "+" + "-" * 20 + "+" + "-" * 25 + "+" + "-" * 25 + "+" + "-" * 25 + "+\n\n")
        
        # 帕累托前沿对比图列表
        f.write("4. 帕累托前沿对比图\n")
        f.write("-" * 40 + "\n")
        for i, plot_file in enumerate(plot_files):
            f.write(f"{i+1}. {plot_file}\n")
        
        f.write(f"\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n完整表格格式报告已保存: {filename}")
    
    # 在控制台输出汇总表格
    print("\n" + "=" * 120)
    print("实验结果汇总")
    print("=" * 120)
    
    print("\n1. 加权目标函数对比 (F = 0.55*F1 + 0.45*F2)")
    print("结果格式: (最优解, 均值, 收敛时间)")
    print("-" * 120)
    print(f"{'规模':^10s} | {'紧急度DDT':^15s} | {'RL-Chaotic-HHO':^20s} | {'NSGA-II':^20s} | {'MOEA/D':^20s}")
    print("-" * 120)
    
    for config in configs:
        scale = config['scale']
        urgency_str = f"{config['urgency_ddt']}"
        
        if scale in all_results:
            rl_result = all_results[scale].get('RL-Chaotic-HHO', {})
            rl_str = f"({rl_result.get('weighted_best', 0):.1f},{rl_result.get('weighted_mean', 0):.1f},{rl_result.get('runtime', 0):.2f})"
            
            nsga_result = all_results[scale].get('NSGA-II', {})
            nsga_str = f"({nsga_result.get('weighted_best', 0):.1f},{nsga_result.get('weighted_mean', 0):.1f},{nsga_result.get('runtime', 0):.2f})"
            
            moead_result = all_results[scale].get('MOEA/D', {})
            moead_str = f"({moead_result.get('weighted_best', 0):.1f},{moead_result.get('weighted_mean', 0):.1f},{moead_result.get('runtime', 0):.2f})"
        else:
            rl_str = nsga_str = moead_str = "(--,--,--)"
        
        print(f"{scale:^10s} | {urgency_str:^15s} | {rl_str:^20s} | {nsga_str:^20s} | {moead_str:^20s}")

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 运行完整表格格式实验
    run_complete_table_experiments()
    
    print("\n完整表格格式对比实验完成!") 