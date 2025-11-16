#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
单一规模算法对比实验程序 - 100×5×4规模专用版本
对比六种算法：RL-Chaotic-HHO、NSGA-II、I-NSGA-II、MOEA/D、MOPSO、MODE
专门针对大规模100×5×4配置进行深度对比分析
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
from algorithm.improved_nsga2 import ImprovedNSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from utils.data_generator import DataGenerator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def generate_custom_urgencies(n_jobs: int, urgency_range: List[float]) -> List[float]:
    """
    生成指定范围的紧急度
    
    Args:
        n_jobs: 作业数量
        urgency_range: [最小值, 平均值, 最大值]
    
    Returns:
        紧急度列表
    """
    min_val, avg_val, max_val = urgency_range
    
    # 生成正态分布的紧急度，均值为avg_val
    std_dev = (max_val - min_val) / 6  # 6个标准差覆盖范围
    urgencies = np.random.normal(avg_val, std_dev, n_jobs)
    
    # 限制在指定范围内
    urgencies = np.clip(urgencies, min_val, max_val)
    
    # 确保边界值的存在
    urgencies[0] = min_val
    urgencies[1] = max_val
    urgencies[2] = avg_val
    
    return urgencies.tolist()

def generate_problem_data() -> Dict:
    """
    生成100×5×4规模的问题数据
    
    Returns:
        问题数据字典
    """
    generator = DataGenerator(seed=42)
    
    # 问题配置
    config = {
        'n_jobs': 100,
        'n_factories': 5,
        'n_stages': 4,
        'machines_per_stage': [2, 3, 3, 2],  # 平均配置
        'urgency_ddt': [4.95, 5.95, 6.95],
        'processing_time_range': (1, 30),
        'heterogeneous_machines': {
            # 总机器数: 6+10+11+12+11=50台
            0: [1, 2, 2, 1],  # 工厂0: 6台机器
            1: [2, 3, 3, 2],  # 工厂1: 10台机器
            2: [2, 3, 4, 2],  # 工厂2: 11台机器
            3: [3, 4, 3, 2],  # 工厂3: 12台机器
            4: [2, 3, 4, 2]   # 工厂4: 11台机器
        }
    }
    
    # 生成基础问题数据
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
    
    # 添加异构机器配置信息
    problem_data['heterogeneous_machines'] = config['heterogeneous_machines']
    
    return problem_data, config

def run_single_algorithm(algorithm_name: str, algorithm_class, algorithm_params: Dict, 
                        problem_data: Dict, runs: int = 3) -> Dict:
    """
    运行单个算法的多次实验
    
    Args:
        algorithm_name: 算法名称
        algorithm_class: 算法类
        algorithm_params: 算法参数
        problem_data: 问题数据
        runs: 运行次数
        
    Returns:
        统计结果字典
    """
    print(f"\n🔬 正在运行 {algorithm_name} ({runs}次运行)...")
    
    weighted_values = []    # 存储加权目标函数值
    makespan_values = []   # 存储完工时间
    tardiness_values = []  # 存储总拖期
    runtimes = []
    all_pareto_solutions = []  # 存储所有帕累托解
    
    for run in range(runs):
        try:
            print(f"  📊 第 {run+1}/{runs} 次运行...")
            
            # 创建问题实例
            problem = MO_DHFSP_Problem(problem_data)
            
            # 创建优化器
            optimizer = algorithm_class(problem, **algorithm_params)
            
            # 记录运行时间
            start_time = time.time()
            
            # 运行优化
            pareto_solutions, convergence_data = optimizer.optimize()
            
            end_time = time.time()
            runtime = end_time - start_time
            runtimes.append(runtime)
            
            # 计算目标函数值
            if pareto_solutions:
                # 计算各种指标的最优值
                makespans = [sol.makespan for sol in pareto_solutions]
                tardiness = [sol.total_tardiness for sol in pareto_solutions]
                weighted_objs = [0.55 * sol.makespan + 0.45 * sol.total_tardiness for sol in pareto_solutions]
                
                weighted_values.append(min(weighted_objs))
                makespan_values.append(min(makespans))
                tardiness_values.append(min(tardiness))
                
                # 收集帕累托解用于绘图 (只收集第一次运行的)
                if run == 0:
                    all_pareto_solutions = pareto_solutions
                    
                print(f"    ✅ 完成: 帕累托解={len(pareto_solutions)}, 加权目标={weighted_values[-1]:.2f}, 完工时间={makespan_values[-1]:.2f}, 拖期={tardiness_values[-1]:.2f}, 时间={runtime:.2f}s")
            else:
                weighted_values.append(float('inf'))
                makespan_values.append(float('inf'))
                tardiness_values.append(float('inf'))
                print(f"    ❌ 第 {run+1} 次运行失败")
                
        except Exception as e:
            print(f"    ❌ 第 {run+1} 次运行出错: {str(e)}")
            weighted_values.append(float('inf'))
            makespan_values.append(float('inf'))
            tardiness_values.append(float('inf'))
            runtimes.append(0.0)
    
    # 计算统计结果
    valid_weighted = [v for v in weighted_values if v != float('inf')]
    valid_makespans = [v for v in makespan_values if v != float('inf')]
    valid_tardiness = [v for v in tardiness_values if v != float('inf')]
    
    if valid_weighted:
        results = {
            'weighted_best': min(valid_weighted),
            'weighted_mean': np.mean(valid_weighted),
            'weighted_std': np.std(valid_weighted),
            'makespan_best': min(valid_makespans),
            'makespan_mean': np.mean(valid_makespans),
            'makespan_std': np.std(valid_makespans),
            'tardiness_best': min(valid_tardiness),
            'tardiness_mean': np.mean(valid_tardiness),
            'tardiness_std': np.std(valid_tardiness),
            'runtime': np.mean(runtimes),
            'runtime_std': np.std(runtimes),
            'pareto_solutions': all_pareto_solutions,
            'success_rate': len(valid_weighted) / runs
        }
    else:
        results = {
            'weighted_best': float('inf'),
            'weighted_mean': float('inf'),
            'weighted_std': 0.0,
            'makespan_best': float('inf'),
            'makespan_mean': float('inf'),
            'makespan_std': 0.0,
            'tardiness_best': float('inf'),
            'tardiness_mean': float('inf'),
            'tardiness_std': 0.0,
            'runtime': 0.0,
            'runtime_std': 0.0,
            'pareto_solutions': [],
            'success_rate': 0.0
        }
    
    print(f"  🎯 {algorithm_name} 汇总结果:")
    print(f"    加权目标值: 最优={results['weighted_best']:.2f}, 均值={results['weighted_mean']:.2f}±{results['weighted_std']:.2f}")
    print(f"    完工时间: 最优={results['makespan_best']:.2f}, 均值={results['makespan_mean']:.2f}±{results['makespan_std']:.2f}")
    print(f"    总拖期: 最优={results['tardiness_best']:.2f}, 均值={results['tardiness_mean']:.2f}±{results['tardiness_std']:.2f}")
    print(f"    运行时间: {results['runtime']:.2f}±{results['runtime_std']:.2f}s")
    print(f"    成功率: {results['success_rate']*100:.1f}%")
    
    return results

def plot_pareto_comparison(all_results: Dict):
    """绘制帕累托前沿对比图"""
    
    plt.figure(figsize=(14, 10))
    
    colors = ['red', 'blue', 'cyan', 'green', 'orange', 'purple']
    markers = ['o', 's', 'p', '^', 'D', 'v']
    algorithm_names = ['RL-Chaotic-HHO', 'NSGA-II', 'I-NSGA-II', 'MOEA/D', 'MOPSO', 'MODE']
    
    for i, alg_name in enumerate(algorithm_names):
        if alg_name in all_results and 'pareto_solutions' in all_results[alg_name]:
            pareto_solutions = all_results[alg_name]['pareto_solutions']
            
            if pareto_solutions:
                makespans = [sol.makespan for sol in pareto_solutions]
                tardiness = [sol.total_tardiness for sol in pareto_solutions]
                
                plt.scatter(makespans, tardiness, 
                          c=colors[i], marker=markers[i], 
                          label=f'{alg_name} ({len(pareto_solutions)}个解)',
                          alpha=0.7, s=60)
    
    plt.xlabel('完工时间 (Makespan)', fontsize=14)
    plt.ylabel('总拖期 (Total Tardiness)', fontsize=14)
    plt.title('100×5×4规模 - 六算法帕累托前沿对比', fontsize=16, fontweight='bold')
    plt.legend(fontsize=12, loc='upper right')
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/100x5x4规模_六算法帕累托对比_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"📊 帕累托前沿对比图已保存: {filename}")

def generate_detailed_report(all_results: Dict, config: Dict):
    """生成详细对比报告"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/100x5x4规模_详细对比报告_{timestamp}.txt"
    
    algorithms = ['RL-Chaotic-HHO', 'NSGA-II', 'I-NSGA-II', 'MOEA/D', 'MOPSO', 'MODE']
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("100×5×4规模六算法详细对比报告\n")
        f.write("=" * 120 + "\n\n")
        
        f.write("实验配置:\n")
        f.write(f"- 问题规模: 100作业 × 5工厂 × 4阶段\n")
        f.write(f"- 异构机器配置: 工厂0(6台) + 工厂1(10台) + 工厂2(11台) + 工厂3(12台) + 工厂4(11台) = 50台总机器\n")
        f.write(f"- 紧急度范围: {config.get('urgency_ddt', 'N/A')}\n")
        f.write(f"- 处理时间范围: {config.get('processing_time_range', 'N/A')}\n")
        f.write(f"- 加权目标函数: F = 0.55×完工时间 + 0.45×总拖期\n")
        f.write(f"- 每个算法运行3次取统计结果\n\n")
        
        # 算法参数配置
        f.write("算法参数配置:\n")
        f.write("- RL-Chaotic-HHO: 种群100, 迭代100, 学习率0.015, ε衰减0.997, 组比例[0.5, 0.25, 0.15, 0.1], 折扣因子0.9\n")
        f.write("- NSGA-II: 种群100, 代数100, 交叉0.9, 变异0.1\n")
        f.write("- I-NSGA-II: 种群100, 代数100, 交叉0.9, 变异0.1 (改进拥挤距离)\n")
        f.write("- MOEA/D: 种群100, 代数100, 邻居10, δ=0.9\n")
        f.write("- MOPSO: 群体100, 迭代100, w=0.5, c1=c2=2.0\n")
        f.write("- MODE: 种群100, 代数100, F=0.5, CR=0.9\n\n")
        
        # 详细结果表格
        f.write("详细结果对比:\n")
        f.write("=" * 120 + "\n")
        f.write(f"| {'算法':^16s} | {'加权最优':^10s} | {'加权均值':^10s} | {'完工最优':^10s} | {'完工均值':^10s} | {'拖期最优':^10s} | {'拖期均值':^10s} | {'运行时间':^10s} | {'成功率':^8s} |\n")
        f.write("+" + "-" * 18 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 10 + "+\n")
        
        for alg in algorithms:
            if alg in all_results:
                result = all_results[alg]
                f.write(f"| {alg:^16s} | {result['weighted_best']:^10.2f} | {result['weighted_mean']:^10.2f} | {result['makespan_best']:^10.2f} | {result['makespan_mean']:^10.2f} | {result['tardiness_best']:^10.2f} | {result['tardiness_mean']:^10.2f} | {result['runtime']:^10.2f} | {result['success_rate']*100:^6.1f}% |\n")
            else:
                f.write(f"| {alg:^16s} | {'N/A':^10s} | {'N/A':^10s} | {'N/A':^10s} | {'N/A':^10s} | {'N/A':^10s} | {'N/A':^10s} | {'N/A':^10s} | {'N/A':^8s} |\n")
        
        f.write("+" + "-" * 18 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 10 + "+\n\n")
        
        # 算法排名分析
        f.write("算法性能排名分析:\n")
        f.write("-" * 60 + "\n")
        
        # 按加权目标函数排名
        valid_results = {alg: result for alg, result in all_results.items() 
                        if result['weighted_best'] != float('inf')}
        
        if valid_results:
            weighted_ranking = sorted(valid_results.items(), key=lambda x: x[1]['weighted_best'])
            f.write("按加权目标函数最优值排名:\n")
            for i, (alg, result) in enumerate(weighted_ranking, 1):
                f.write(f"  {i}. {alg}: {result['weighted_best']:.2f}\n")
            
            makespan_ranking = sorted(valid_results.items(), key=lambda x: x[1]['makespan_best'])
            f.write("\n按完工时间最优值排名:\n")
            for i, (alg, result) in enumerate(makespan_ranking, 1):
                f.write(f"  {i}. {alg}: {result['makespan_best']:.2f}\n")
                
            tardiness_ranking = sorted(valid_results.items(), key=lambda x: x[1]['tardiness_best'])
            f.write("\n按总拖期最优值排名:\n")
            for i, (alg, result) in enumerate(tardiness_ranking, 1):
                f.write(f"  {i}. {alg}: {result['tardiness_best']:.2f}\n")
                
            runtime_ranking = sorted(valid_results.items(), key=lambda x: x[1]['runtime'])
            f.write("\n按运行时间排名:\n")
            for i, (alg, result) in enumerate(runtime_ranking, 1):
                f.write(f"  {i}. {alg}: {result['runtime']:.2f}s\n")
        
        # 帕累托解数量统计
        f.write(f"\n帕累托解数量统计:\n")
        f.write("-" * 40 + "\n")
        for alg in algorithms:
            if alg in all_results and all_results[alg]['pareto_solutions']:
                count = len(all_results[alg]['pareto_solutions'])
                f.write(f"  {alg}: {count}个解\n")
            else:
                f.write(f"  {alg}: 0个解\n")
        
        f.write(f"\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"📄 详细对比报告已保存: {filename}")

def print_console_summary(all_results: Dict):
    """在控制台输出汇总结果"""
    
    algorithms = ['RL-Chaotic-HHO', 'NSGA-II', 'I-NSGA-II', 'MOEA/D', 'MOPSO', 'MODE']
    
    print("\n" + "=" * 120)
    print("🎯 100×5×4规模六算法对比实验结果汇总")
    print("=" * 120)
    
    print(f"| {'算法':^16s} | {'加权最优':^10s} | {'加权均值':^10s} | {'完工最优':^10s} | {'完工均值':^10s} | {'拖期最优':^10s} | {'拖期均值':^10s} | {'运行时间':^10s} |")
    print("+" + "-" * 18 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    
    for alg in algorithms:
        if alg in all_results:
            result = all_results[alg]
            print(f"| {alg:^16s} | {result['weighted_best']:^10.2f} | {result['weighted_mean']:^10.2f} | {result['makespan_best']:^10.2f} | {result['makespan_mean']:^10.2f} | {result['tardiness_best']:^10.2f} | {result['tardiness_mean']:^10.2f} | {result['runtime']:^10.2f} |")
        else:
            print(f"| {alg:^16s} | {'N/A':^10s} | {'N/A':^10s} | {'N/A':^10s} | {'N/A':^10s} | {'N/A':^10s} | {'N/A':^10s} | {'N/A':^10s} |")
    
    print("+" + "-" * 18 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+" + "-" * 12 + "+")
    
    # 找出最优算法
    valid_results = {alg: result for alg, result in all_results.items() 
                    if result['weighted_best'] != float('inf')}
    
    if valid_results:
        best_weighted = min(valid_results.items(), key=lambda x: x[1]['weighted_best'])
        best_makespan = min(valid_results.items(), key=lambda x: x[1]['makespan_best'])
        best_tardiness = min(valid_results.items(), key=lambda x: x[1]['tardiness_best'])
        fastest = min(valid_results.items(), key=lambda x: x[1]['runtime'])
        
        print(f"\n🏆 最优性能:")
        print(f"  加权目标函数最优: {best_weighted[0]} ({best_weighted[1]['weighted_best']:.2f})")
        print(f"  完工时间最优: {best_makespan[0]} ({best_makespan[1]['makespan_best']:.2f})")
        print(f"  总拖期最优: {best_tardiness[0]} ({best_tardiness[1]['tardiness_best']:.2f})")
        print(f"  运行速度最快: {fastest[0]} ({fastest[1]['runtime']:.2f}s)")
    
    print("=" * 120)

def run_comparison_experiment():
    """运行100×5×4规模的六算法对比实验"""
    
    print("🚀 启动100×5×4规模六算法对比实验")
    print("=" * 80)
    
    # 生成问题数据
    print("📊 生成问题数据...")
    problem_data, config = generate_problem_data()
    
    print(f"✅ 问题规模: {config['n_jobs']}作业 × {config['n_factories']}工厂 × {config['n_stages']}阶段")
    print(f"✅ 总机器数: 50台 (异构配置)")
    print(f"✅ 紧急度范围: {config['urgency_ddt']}")
    
    # 算法配置
    algorithms = {
        'RL-Chaotic-HHO': {
            'class': RL_ChaoticHHO_Optimizer,
            'params': {
                'max_iterations': 100,
                'population_size_override': 100,
                'learning_rate': 0.015,
                'epsilon_decay': 0.997,
                'group_ratios': [0.5, 0.25, 0.15, 0.1],
                'discount_factor': 0.9
            }
        },
        'NSGA-II': {
            'class': NSGA2_Optimizer,
            'params': {
                'population_size': 100,
                'max_generations': 100,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1
            }
        },
        'I-NSGA-II': {
            'class': ImprovedNSGA2_Optimizer,
            'params': {
                'population_size': 100,
                'max_generations': 100,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1
            }
        },
        'MOEA/D': {
            'class': MOEAD_Optimizer,
            'params': {
                'population_size': 100,
                'max_generations': 100,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1,
                'neighbor_size': 10,
                'delta': 0.9,
                'nr': 2
            }
        },
        'MOPSO': {
            'class': MOPSO_Optimizer,
            'params': {
                'swarm_size': 100,
                'max_iterations': 100,
                'w': 0.5,
                'c1': 2.0,
                'c2': 2.0,
                'archive_size': 100
            }
        },
        'MODE': {
            'class': MODE_Optimizer,
            'params': {
                'population_size': 100,
                'max_generations': 100,
                'F': 0.5,
                'CR': 0.9,
                'mutation_prob': 0.1
            }
        }
    }
    
    # 运行所有算法
    all_results = {}
    total_algorithms = len(algorithms)
    
    for i, (alg_name, alg_config) in enumerate(algorithms.items(), 1):
        print(f"\n🔄 进度: {i}/{total_algorithms} - 运行 {alg_name}")
        print("-" * 60)
        
        result = run_single_algorithm(
            alg_name,
            alg_config['class'],
            alg_config['params'],
            problem_data,
            runs=3
        )
        
        all_results[alg_name] = result
    
    # 生成报告和图表
    print(f"\n📊 生成对比报告和图表...")
    
    # 控制台汇总
    print_console_summary(all_results)
    
    # 绘制图表
    plot_pareto_comparison(all_results)
    
    # 生成详细报告
    generate_detailed_report(all_results, config)
    
    print(f"\n✅ 100×5×4规模六算法对比实验完成!")
    print(f"📁 所有结果文件已保存到 results/ 目录")

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 运行对比实验
    run_comparison_experiment() 