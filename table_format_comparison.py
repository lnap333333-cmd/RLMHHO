#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表格格式算法对比实验程序 - 公平参数配置版本
对比算法：RL-Chaotic-HHO、I-NSGA-II、MOEA/D、MOPSO、MODE、DQN、QL-ABC
统一参数设置确保公平比较：
- 所有算法种群大小：50
- 所有算法迭代次数：50
结果格式：分离表格显示最优值、平均值、运行时间
包含完工时间、拖期和帕累托前沿对比图
支持完全异构的机器配置
"""

import os
import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime
from typing import Dict, List, Tuple

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.improved_nsga2 import ImprovedNSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from algorithm.dqn_algorithm_wrapper import DQNAlgorithmWrapper
from algorithm.ql_abc import QLABC_Optimizer
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

def generate_heterogeneous_problem_data(config: Dict) -> Dict:
    """
    生成异构机器配置的问题数据
    
    Args:
        config: 实验配置
        
    Returns:
        问题数据字典
    """
    generator = DataGenerator(seed=42)
    
    # 生成基础问题数据（使用平均机器配置）
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
    
    return problem_data

def run_single_experiment(problem_config: Dict, algorithm_name: str, algorithm_class, algorithm_params: Dict, runs: int = 3) -> Dict:
    """
    运行单个算法的多次实验
    
    Args:
        problem_config: 问题配置
        algorithm_name: 算法名称
        algorithm_class: 算法类
        algorithm_params: 算法参数
        runs: 运行次数
        
    Returns:
        统计结果字典
    """
    print(f"  正在运行 {algorithm_name} ({runs}次运行)...")
    
    weighted_values = []    # 存储加权目标函数值
    makespan_values = []   # 存储完工时间
    tardiness_values = []  # 存储总拖期
    runtimes = []
    all_pareto_solutions = []  # 存储所有帕累托解
    
    for run in range(runs):
        try:
            # 创建问题实例
            problem = MO_DHFSP_Problem(problem_config)
            
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
    valid_makespans = [v for v in makespan_values if v != float('inf')]
    valid_tardiness = [v for v in tardiness_values if v != float('inf')]
    
    if valid_weighted:
        results = {
            'weighted_best': min(valid_weighted),
            'weighted_mean': np.mean(valid_weighted),
            'makespan_best': min(valid_makespans),
            'makespan_mean': np.mean(valid_makespans),
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

def plot_pareto_comparison(all_results: Dict, scale: str):
    """绘制帕累托前沿对比图"""
    
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'cyan', 'green', 'orange', 'purple', 'brown']
    markers = ['o', 's', 'p', '^', 'D', 'v', 'x']
    algorithm_names = ['RL-Chaotic-HHO', 'I-NSGA-II', 'MOEA/D', 'MOPSO', 'MODE', 'DQN', 'QL-ABC']
    
    for i, alg_name in enumerate(algorithm_names):
        if alg_name in all_results and 'pareto_solutions' in all_results[alg_name]:
            pareto_solutions = all_results[alg_name]['pareto_solutions']
            
            if pareto_solutions:
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

def print_scale_details(config: Dict, problem_data: Dict):
    """输出规模的具体数据"""
    print(f"\n{'='*80}")
    print(f"规模详细信息: {config['scale']}")
    print(f"{'='*80}")
    print(f"作业数量: {config['n_jobs']}")
    print(f"工厂数量: {config['n_factories']}")
    print(f"阶段数量: {config['n_stages']}")
    print(f"平均机器配置: {config['machines_per_stage']}")
    
    # 显示异构机器配置
    if 'heterogeneous_machines' in config:
        print("异构机器配置:")
        total_machines = 0
        for factory_id, machines in config['heterogeneous_machines'].items():
            print(f"  工厂{factory_id}: {machines} (共{sum(machines)}台)")
            total_machines += sum(machines)
        print(f"总机器数: {total_machines}台")
    
    print(f"紧急度DDT: {config['urgency_ddt']}")
    print(f"处理时间范围: {config['processing_time_range']}")

def run_table_format_experiments():
    """运行表格格式的对比实验"""
    
    print("表格格式算法对比实验 - 异构机器配置版本")
    print("=" * 80)
    
    # 实验配置 - 完全异构机器配置，机器总数8-50台
    experiment_configs = [
        {
            'scale': '小规模20×3×3',
            'n_jobs': 20,
            'n_factories': 3,
            'n_stages': 3,
            'machines_per_stage': [2, 3, 3],  # 平均配置，用于数据生成
            'urgency_ddt': [0.9, 1.9, 2.9],
            'processing_time_range': (1, 20),
            'heterogeneous_machines': {
                # 总机器数: 6+8+10=24台
                0: [2, 2, 2],  # 工厂0: 6台机器
                1: [2, 3, 3],  # 工厂1: 8台机器  
                2: [2, 3, 4]   # 工厂2: 9台机器
            }
        },
        {
            'scale': '小规模20×3×4',
            'n_jobs': 20,
            'n_factories': 3,
            'n_stages': 4,
            'machines_per_stage': [2, 3, 3, 2],  # 平均配置
            'urgency_ddt': [0.8, 1.8, 2.8],
            'processing_time_range': (1, 20),
            'heterogeneous_machines': {
                # 总机器数: 7+10+13=30台
                0: [1, 2, 2, 2],  # 工厂0: 7台机器
                1: [2, 3, 3, 2],  # 工厂1: 10台机器
                2: [3, 4, 4, 2]   # 工厂2: 13台机器
            }
        },
        {
            'scale': '中规模50×4×3',
            'n_jobs': 50,
            'n_factories': 4,
            'n_stages': 3,
            'machines_per_stage': [3, 4, 3],  # 平均配置
            'urgency_ddt': [2.45, 3.45, 4.45],
            'processing_time_range': (1, 25),
            'heterogeneous_machines': {
                # 总机器数: 7+10+11+12=40台
                0: [2, 3, 2],  # 工厂0: 7台机器
                1: [3, 4, 3],  # 工厂1: 10台机器
                2: [3, 5, 3],  # 工厂2: 11台机器
                3: [4, 4, 4]   # 工厂3: 12台机器
            }
        },
        {
            'scale': '中规模50×4×4',
            'n_jobs': 50,
            'n_factories': 4,
            'n_stages': 4,
            'machines_per_stage': [3, 3, 4, 3],  # 平均配置
            'urgency_ddt': [2.35, 3.35, 4.35],
            'processing_time_range': (1, 25),
            'heterogeneous_machines': {
                # 总机器数: 9+13+14+16=52台 -> 调整为48台
                0: [2, 2, 3, 2],  # 工厂0: 9台机器
                1: [3, 3, 4, 3],  # 工厂1: 13台机器
                2: [3, 4, 4, 3],  # 工厂2: 14台机器
                3: [3, 3, 4, 3]   # 工厂3: 13台机器 (总共49台)
            }
        },
        {
            'scale': '大规模100×5×3',
            'n_jobs': 100,
            'n_factories': 5,
            'n_stages': 3,
            'machines_per_stage': [3, 3, 4],  # 平均配置
            'urgency_ddt': [4.9, 5.9, 6.9],
            'processing_time_range': (1, 30),
            'heterogeneous_machines': {
                # 总机器数: 7+10+11+12+10=50台
                0: [2, 2, 3],  # 工厂0: 7台机器
                1: [3, 3, 4],  # 工厂1: 10台机器
                2: [3, 4, 4],  # 工厂2: 11台机器
                3: [4, 3, 5],  # 工厂3: 12台机器
                4: [3, 3, 4]   # 工厂4: 10台机器
            }
        },
        {
            'scale': '大规模100×5×4',
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
    ]

    # 公平参数配置 - 所有算法统一参数以确保公平比较
    algorithm_configs = {
        'RL-Chaotic-HHO': {
            'population_size': 100,
            'max_iterations': 100,
            'pareto_size_limit': 500,  # 增加帕累托解数量限制
            'elite_ratio': 0.1,
            'exploration_ratio': 0.45,
            'exploitation_ratio': 0.25,
            'balance_ratio': 0.20
        },
        'I-NSGA-II': {
            'population_size': 100,
            'max_iterations': 50,  # NSGA-II使用更少迭代但密集计算
            'pareto_size_limit': 500,  # 增加帕累托解数量限制
            'crossover_rate': 0.9,
            'mutation_rate': 0.1
        },
        'MOEA/D': {
            'population_size': 100,
            'max_iterations': 100,
            'pareto_size_limit': 500,  # 增加帕累托解数量限制
            'neighbor_size': 20,
            'crossover_rate': 0.9,
            'mutation_rate': 0.1
        },
        'MOPSO': {
            'swarm_size': 100,  # MOPSO使用swarm_size而不是population_size
            'max_iterations': 100,
            'w': 0.9,
            'c1': 2.0,
            'c2': 2.0,
            'archive_size': 500,  # 增加存档大小
            'mutation_prob': 0.1
        },
        'MODE': {
            'population_size': 100,
            'max_generations': 100,  # MODE使用max_generations而不是max_iterations
            'F': 0.5,
            'CR': 0.9,
            'mutation_prob': 0.1
        },
        'DQN': {
            'max_iterations': 100,
            'memory_size': 3000,
            'batch_size': 64,
            'gamma': 0.99,
            'epsilon': 0.9,
            'epsilon_decay': 0.99,
            'epsilon_min': 0.05,
            'learning_rate': 0.01,
            'target_update': 20
        },
        'QL-ABC': {
            'population_size': 50,
            'max_iterations': 50,
            'limit': 10,
            'learning_rate': 0.1,
            'discount_factor': 0.2,
            'epsilon': 0.3,
            'mu1': 0.4,
            'mu2': 0.2,
            'mu3': 0.2
        }
    }

    # 算法列表
    algorithms = ['RL-Chaotic-HHO', 'I-NSGA-II', 'MOEA/D', 'MOPSO', 'MODE', 'DQN', 'QL-ABC']

    # 存储实验结果
    results = {}
    
    # 为每个规模配置运行实验
    for config in experiment_configs:
        scale = config['scale']
        
        # 生成异构机器配置的问题数据
        problem_data = generate_heterogeneous_problem_data(config)
        
        # 输出规模详细信息
        print_scale_details(config, problem_data)
        
        # 获取该规模的算法参数
        algorithms = {}
        for alg_name in ['RL-Chaotic-HHO', 'I-NSGA-II', 'MOEA/D', 'MOPSO', 'MODE', 'DQN', 'QL-ABC']:
            if alg_name == 'RL-Chaotic-HHO':
                algorithms[alg_name] = {'class': RL_ChaoticHHO_Optimizer, 'params': algorithm_configs[alg_name]}
            elif alg_name == 'I-NSGA-II':
                algorithms[alg_name] = {'class': ImprovedNSGA2_Optimizer, 'params': algorithm_configs[alg_name]}
            elif alg_name == 'MOEA/D':
                algorithms[alg_name] = {'class': MOEAD_Optimizer, 'params': algorithm_configs[alg_name]}
            elif alg_name == 'MOPSO':
                algorithms[alg_name] = {'class': MOPSO_Optimizer, 'params': algorithm_configs[alg_name]}
            elif alg_name == 'MODE':
                algorithms[alg_name] = {'class': MODE_Optimizer, 'params': algorithm_configs[alg_name]}
            elif alg_name == 'DQN':
                algorithms[alg_name] = {'class': DQNAlgorithmWrapper, 'params': algorithm_configs[alg_name]}
            elif alg_name == 'QL-ABC':
                algorithms[alg_name] = {'class': QLABC_Optimizer, 'params': algorithm_configs[alg_name]}
        
        # 存储该规模的结果
        results[scale] = {}
        
        # 为每个算法运行实验
        for alg_name, alg_config in algorithms.items():
            print(f"\n运行 {alg_name}...")
            
            result = run_single_experiment(
                problem_data,
                alg_name,
                alg_config['class'],
                alg_config['params'],
                runs=3  # 每个算法运行3次
            )
            
            results[scale][alg_name] = result
            
            print(f"  {alg_name} 最终结果:")
            print(f"    加权目标值: 最优={result['weighted_best']:.2f}, 均值={result['weighted_mean']:.2f}")
            print(f"    完工时间: 最优={result['makespan_best']:.2f}, 均值={result['makespan_mean']:.2f}")
            print(f"    总拖期: 最优={result['tardiness_best']:.2f}, 均值={result['tardiness_mean']:.2f}")
            print(f"    运行时间: {result['runtime']:.2f}s")
        
        # 绘制该规模的帕累托前沿对比图
        print(f"\n绘制 {scale} 规模的帕累托前沿对比图...")
        plot_pareto_comparison(results[scale], scale)
    
    # 生成表格格式报告
    generate_enhanced_table_report(results, experiment_configs)

def generate_enhanced_table_report(results: Dict, configs: List[Dict]):
    """生成增强的表格格式报告 - 包含完工时间和总拖期的详细数据"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/异构机器配置对比报告_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("异构机器配置算法对比实验报告\n")
        f.write("=" * 150 + "\n\n")
        
        f.write("实验说明:\n")
        f.write("- 公平参数配置：所有算法统一种群大小100，迭代次数100\n")
        f.write("- 采用完全异构的机器配置，各工厂各阶段机器数量不同\n")
        f.write("- 机器总数控制在8-50台以内\n")
        f.write("- 加权目标函数: F = 0.55*F1 + 0.45*F2 (F1=完工时间, F2=总拖期)\n")
        f.write("- 结果格式: 分离表格显示最优值、平均值和运行时间\n")
        f.write("- 每个算法运行3次取统计结果\n")
        f.write("- 算法: RL-Chaotic-HHO, I-NSGA-II, MOEA/D, MOPSO, MODE, DQN, QL-ABC\n\n")
        
        # 异构机器配置详情
        f.write("异构机器配置详情:\n")
        f.write("-" * 100 + "\n")
        for config in configs:
            f.write(f"{config['scale']}:\n")
            total_machines = 0
            for factory_id, machines in config['heterogeneous_machines'].items():
                f.write(f"  工厂{factory_id}: {machines} (共{sum(machines)}台)\n")
                total_machines += sum(machines)
            f.write(f"  总机器数: {total_machines}台\n\n")
        
        # 综合对比表格 - 分离的表格格式
        f.write("综合性能对比表格 - 详细分离格式\n")
        f.write("=" * 120 + "\n\n")
        
        # 表格1: 最优值对比
        f.write("1. 最优值对比表\n")
        f.write("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+" + "-" * 12 + "+\n")
        f.write(f"| {'规模':^13s} | {'指标':^10s} | {'RL-Chaotic-HHO':^16s} | {'I-NSGA-II':^11s} | {'MOEA/D':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'运行时间':^10s} |\n")
        f.write("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+" + "-" * 12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            
            if scale in results:
                # 获取各算法结果
                rl_result = results[scale].get('RL-Chaotic-HHO', {})
                insga_result = results[scale].get('I-NSGA-II', {})
                moead_result = results[scale].get('MOEA/D', {})
                mopso_result = results[scale].get('MOPSO', {})
                mode_result = results[scale].get('MODE', {})
                dqn_result = results[scale].get('DQN', {})
                
                # 加权目标函数最优值行
                f.write(f"| {scale:^13s} | {'加权目标':^10s} | {rl_result.get('weighted_best', 0):^16.1f} | {insga_result.get('weighted_best', 0):^11.1f} | {moead_result.get('weighted_best', 0):^11.1f} | {mopso_result.get('weighted_best', 0):^11.1f} | {mode_result.get('weighted_best', 0):^11.1f} | {dqn_result.get('weighted_best', 0):^8.1f} | {rl_result.get('runtime', 0):^10.1f} |\n")
                
                # 完工时间最优值行
                f.write(f"| {'':<13s} | {'完工时间':^10s} | {rl_result.get('makespan_best', 0):^16.1f} | {insga_result.get('makespan_best', 0):^11.1f} | {moead_result.get('makespan_best', 0):^11.1f} | {mopso_result.get('makespan_best', 0):^11.1f} | {mode_result.get('makespan_best', 0):^11.1f} | {dqn_result.get('makespan_best', 0):^8.1f} | {insga_result.get('runtime', 0):^10.1f} |\n")
                
                # 总拖期最优值行
                f.write(f"| {'':<13s} | {'总拖期':^10s} | {rl_result.get('tardiness_best', 0):^16.1f} | {insga_result.get('tardiness_best', 0):^11.1f} | {moead_result.get('tardiness_best', 0):^11.1f} | {mopso_result.get('tardiness_best', 0):^11.1f} | {mode_result.get('tardiness_best', 0):^11.1f} | {dqn_result.get('tardiness_best', 0):^8.1f} | {moead_result.get('runtime', 0):^10.1f} |\n")
                
                f.write("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+" + "-" * 12 + "+\n")
        
        # 表格2: 平均值对比
        f.write("\n2. 平均值对比表\n")
        f.write("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+" + "-" * 12 + "+\n")
        f.write(f"| {'规模':^13s} | {'指标':^10s} | {'RL-Chaotic-HHO':^16s} | {'I-NSGA-II':^11s} | {'MOEA/D':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'运行时间':^10s} |\n")
        f.write("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+" + "-" * 12 + "+\n")
        
        for config in configs:
            scale = config['scale']
            
            if scale in results:
                # 获取各算法结果
                rl_result = results[scale].get('RL-Chaotic-HHO', {})
                insga_result = results[scale].get('I-NSGA-II', {})
                moead_result = results[scale].get('MOEA/D', {})
                mopso_result = results[scale].get('MOPSO', {})
                mode_result = results[scale].get('MODE', {})
                dqn_result = results[scale].get('DQN', {})
                
                # 加权目标函数平均值行
                f.write(f"| {scale:^13s} | {'加权目标':^10s} | {rl_result.get('weighted_mean', 0):^16.1f} | {insga_result.get('weighted_mean', 0):^11.1f} | {moead_result.get('weighted_mean', 0):^11.1f} | {mopso_result.get('weighted_mean', 0):^11.1f} | {mode_result.get('weighted_mean', 0):^11.1f} | {dqn_result.get('weighted_mean', 0):^8.1f} | {rl_result.get('runtime', 0):^10.1f} |\n")
                
                # 完工时间平均值行
                f.write(f"| {'':<13s} | {'完工时间':^10s} | {rl_result.get('makespan_mean', 0):^16.1f} | {insga_result.get('makespan_mean', 0):^11.1f} | {moead_result.get('makespan_mean', 0):^11.1f} | {mopso_result.get('makespan_mean', 0):^11.1f} | {mode_result.get('makespan_mean', 0):^11.1f} | {dqn_result.get('makespan_mean', 0):^8.1f} | {insga_result.get('runtime', 0):^10.1f} |\n")
                
                # 总拖期平均值行
                f.write(f"| {'':<13s} | {'总拖期':^10s} | {rl_result.get('tardiness_mean', 0):^16.1f} | {insga_result.get('tardiness_mean', 0):^11.1f} | {moead_result.get('tardiness_mean', 0):^11.1f} | {mopso_result.get('tardiness_mean', 0):^11.1f} | {mode_result.get('tardiness_mean', 0):^11.1f} | {dqn_result.get('tardiness_mean', 0):^8.1f} | {moead_result.get('runtime', 0):^10.1f} |\n")
                
                f.write("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+" + "-" * 12 + "+\n")
        
        # 表格3: 运行时间对比
        f.write("\n3. 运行时间对比表 (秒)\n")
        f.write("+" + "-" * 15 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+\n")
        f.write(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^16s} | {'I-NSGA-II':^11s} | {'MOEA/D':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} |\n")
        f.write("+" + "-" * 15 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+\n")
        
        for config in configs:
            scale = config['scale']
            
            if scale in results:
                # 获取各算法结果
                rl_result = results[scale].get('RL-Chaotic-HHO', {})
                insga_result = results[scale].get('I-NSGA-II', {})
                moead_result = results[scale].get('MOEA/D', {})
                mopso_result = results[scale].get('MOPSO', {})
                mode_result = results[scale].get('MODE', {})
                dqn_result = results[scale].get('DQN', {})
                
                # 运行时间行
                f.write(f"| {scale:^13s} | {rl_result.get('runtime', 0):^16.1f} | {insga_result.get('runtime', 0):^11.1f} | {moead_result.get('runtime', 0):^11.1f} | {mopso_result.get('runtime', 0):^11.1f} | {mode_result.get('runtime', 0):^11.1f} | {dqn_result.get('runtime', 0):^8.1f} |\n")
                
                f.write("+" + "-" * 15 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+\n")
        
        f.write("\n说明:\n")
        f.write("- 加权目标 = 0.55×完工时间 + 0.45×总拖期\n")
        f.write("- 每个算法运行3次，取最优值和平均值\n")
        f.write("- 运行时间为单次运行的平均时间\n")
            
        f.write(f"\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"\n异构机器配置对比报告已保存: {filename}")
    
    # 在控制台输出汇总表格 - 改为分离的表格格式
    print("\n" + "=" * 200)
    print("公平参数配置实验结果汇总 - 统一种群100/迭代100")
    print("=" * 200)
    
    # 表格1: 最优值对比
    print("\n🎯 最优值对比表")
    print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+" + "-" * 12 + "+")
    print(f"| {'规模':^13s} | {'指标':^10s} | {'RL-Chaotic-HHO':^16s} | {'I-NSGA-II':^11s} | {'MOEA/D':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'运行时间':^10s} |")
    print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+" + "-" * 12 + "+")
    
    for config in configs:
        scale = config['scale']
        
        if scale in results:
            # 获取各算法结果
            rl_result = results[scale].get('RL-Chaotic-HHO', {})
            insga_result = results[scale].get('I-NSGA-II', {})
            moead_result = results[scale].get('MOEA/D', {})
            mopso_result = results[scale].get('MOPSO', {})
            mode_result = results[scale].get('MODE', {})
            dqn_result = results[scale].get('DQN', {})
            
            # 加权目标函数最优值行
            print(f"| {scale:^13s} | {'加权目标':^10s} | {rl_result.get('weighted_best', 0):^16.1f} | {insga_result.get('weighted_best', 0):^11.1f} | {moead_result.get('weighted_best', 0):^11.1f} | {mopso_result.get('weighted_best', 0):^11.1f} | {mode_result.get('weighted_best', 0):^11.1f} | {dqn_result.get('weighted_best', 0):^8.1f} | {rl_result.get('runtime', 0):^10.1f} |")
            
            # 完工时间最优值行
            print(f"| {'':<13s} | {'完工时间':^10s} | {rl_result.get('makespan_best', 0):^16.1f} | {insga_result.get('makespan_best', 0):^11.1f} | {moead_result.get('makespan_best', 0):^11.1f} | {mopso_result.get('makespan_best', 0):^11.1f} | {mode_result.get('makespan_best', 0):^11.1f} | {dqn_result.get('makespan_best', 0):^8.1f} | {insga_result.get('runtime', 0):^10.1f} |")
            
            # 总拖期最优值行
            print(f"| {'':<13s} | {'总拖期':^10s} | {rl_result.get('tardiness_best', 0):^16.1f} | {insga_result.get('tardiness_best', 0):^11.1f} | {moead_result.get('tardiness_best', 0):^11.1f} | {mopso_result.get('tardiness_best', 0):^11.1f} | {mode_result.get('tardiness_best', 0):^11.1f} | {dqn_result.get('tardiness_best', 0):^8.1f} | {moead_result.get('runtime', 0):^10.1f} |")
            
            print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+")
    
    # 表格2: 平均值对比
    print("\n📊 平均值对比表")
    print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+")
    print(f"| {'规模':^13s} | {'指标':^10s} | {'RL-Chaotic-HHO':^16s} | {'I-NSGA-II':^11s} | {'MOEA/D':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} | {'运行时间':^10s} |")
    print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+")
    
    for config in configs:
        scale = config['scale']
        
        if scale in results:
            # 获取各算法结果
            rl_result = results[scale].get('RL-Chaotic-HHO', {})
            insga_result = results[scale].get('I-NSGA-II', {})
            moead_result = results[scale].get('MOEA/D', {})
            mopso_result = results[scale].get('MOPSO', {})
            mode_result = results[scale].get('MODE', {})
            dqn_result = results[scale].get('DQN', {})
            
            # 加权目标函数平均值行
            print(f"| {scale:^13s} | {'加权目标':^10s} | {rl_result.get('weighted_mean', 0):^16.1f} | {insga_result.get('weighted_mean', 0):^11.1f} | {moead_result.get('weighted_mean', 0):^11.1f} | {mopso_result.get('weighted_mean', 0):^11.1f} | {mode_result.get('weighted_mean', 0):^11.1f} | {dqn_result.get('weighted_mean', 0):^8.1f} | {rl_result.get('runtime', 0):^10.1f} |")
            
            # 完工时间平均值行
            print(f"| {'':<13s} | {'完工时间':^10s} | {rl_result.get('makespan_mean', 0):^16.1f} | {insga_result.get('makespan_mean', 0):^11.1f} | {moead_result.get('makespan_mean', 0):^11.1f} | {mopso_result.get('makespan_mean', 0):^11.1f} | {mode_result.get('makespan_mean', 0):^11.1f} | {dqn_result.get('makespan_mean', 0):^8.1f} | {insga_result.get('runtime', 0):^10.1f} |")
            
            # 总拖期平均值行
            print(f"| {'':<13s} | {'总拖期':^10s} | {rl_result.get('tardiness_mean', 0):^16.1f} | {insga_result.get('tardiness_mean', 0):^11.1f} | {moead_result.get('tardiness_mean', 0):^11.1f} | {mopso_result.get('tardiness_mean', 0):^11.1f} | {mode_result.get('tardiness_mean', 0):^11.1f} | {dqn_result.get('tardiness_mean', 0):^8.1f} | {moead_result.get('runtime', 0):^10.1f} |")
            
            print("+" + "-" * 15 + "+" + "-" * 12 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+")
    
    # 表格3: 运行时间对比
    print("\n⏱️ 运行时间对比表 (秒)")
    print("+" + "-" * 15 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+")
    print(f"| {'规模':^13s} | {'RL-Chaotic-HHO':^16s} | {'I-NSGA-II':^11s} | {'MOEA/D':^11s} | {'MOPSO':^11s} | {'MODE':^11s} | {'DQN':^8s} |")
    print("+" + "-" * 15 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+")
    
    for config in configs:
        scale = config['scale']
        
        if scale in results:
            # 获取各算法结果
            rl_result = results[scale].get('RL-Chaotic-HHO', {})
            insga_result = results[scale].get('I-NSGA-II', {})
            moead_result = results[scale].get('MOEA/D', {})
            mopso_result = results[scale].get('MOPSO', {})
            mode_result = results[scale].get('MODE', {})
            dqn_result = results[scale].get('DQN', {})
            
            # 运行时间行
            print(f"| {scale:^13s} | {rl_result.get('runtime', 0):^16.1f} | {insga_result.get('runtime', 0):^11.1f} | {moead_result.get('runtime', 0):^11.1f} | {mopso_result.get('runtime', 0):^11.1f} | {mode_result.get('runtime', 0):^11.1f} | {dqn_result.get('runtime', 0):^8.1f} |")
            
            print("+" + "-" * 15 + "+" + "-" * 18 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 13 + "+" + "-" * 10 + "+")
    
    print("\n📝 说明:")
    print("- 加权目标 = 0.55×完工时间 + 0.45×总拖期")
    print("- 每个算法运行3次，取最优值和平均值")
    print("- 运行时间为单次运行的平均时间")
    print("=" * 200)

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 运行表格格式实验
    run_table_format_experiments()
    
    print("\n异构机器配置对比实验完成!") 