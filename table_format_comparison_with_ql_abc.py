#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
表格格式算法对比实验程序 - 包含QL-ABC算法
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
    
    for i, (alg_name, results) in enumerate(all_results.items()):
        if results['pareto_solutions']:
            makespans = [sol.makespan for sol in results['pareto_solutions']]
            tardiness = [sol.total_tardiness for sol in results['pareto_solutions']]
            
            plt.scatter(makespans, tardiness, 
                       c=colors[i % len(colors)], 
                       marker=markers[i % len(markers)], 
                       s=50, alpha=0.7, label=alg_name)
    
    plt.xlabel('完工时间 (Makespan)', fontsize=12)
    plt.ylabel('总拖期 (Total Tardiness)', fontsize=12)
    plt.title(f'帕累托前沿对比图 - {scale}规模问题\n(包含QL-ABC算法)', fontsize=14)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, alpha=0.3)
    
    plt.tight_layout()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(f'pareto_comparison_with_ql_abc_{scale}_{timestamp}.png', dpi=300, bbox_inches='tight')
    plt.show()

def print_scale_details(config: Dict, problem_data: Dict):
    """打印问题规模详细信息"""
    print(f"\n问题配置详情:")
    print(f"  作业数: {config['n_jobs']}")
    print(f"  工厂数: {config['n_factories']}")
    print(f"  阶段数: {config['n_stages']}")
    print(f"  平均机器配置: {config['machines_per_stage']}")
    print(f"  处理时间范围: {config['processing_time_range']}")
    print(f"  紧急度范围: {config['urgency_ddt']}")
    
    # 计算总机器数
    if 'heterogeneous_machines' in config:
        total_machines = sum(sum(stages) for stages in config['heterogeneous_machines'].values())
        print(f"  总机器数: {total_machines}")
        print(f"  各工厂机器分布:")
        for factory_id, machines in config['heterogeneous_machines'].items():
            print(f"    工厂 {factory_id}: {machines} (合计: {sum(machines)})")

def run_table_format_experiments():
    """运行表格格式的算法对比实验（包含QL-ABC）"""
    
    print("🚀 开始表格格式算法对比实验 (包含QL-ABC算法)")
    print("=" * 80)
    
    # 实验配置
    configs = [
        {
            'name': '小规模',
            'n_jobs': 20,
            'n_factories': 3,
            'n_stages': 2,
            'machines_per_stage': [3, 4],
            'processing_time_range': (1, 15),
            'urgency_ddt': [0.8, 1.5, 2.2],
            'heterogeneous_machines': {
                0: [3, 4],  # 工厂1: 7台机器
                1: [2, 5],  # 工厂2: 7台机器
                2: [4, 3]   # 工厂3: 7台机器
            }
        },
        {
            'name': '中规模',
            'n_jobs': 50,
            'n_factories': 4,
            'n_stages': 3,
            'machines_per_stage': [4, 4, 4],
            'processing_time_range': (1, 25),
            'urgency_ddt': [1.0, 1.8, 2.8],
            'heterogeneous_machines': {
                0: [4, 4, 4],  # 工厂1: 12台机器
                1: [3, 5, 3],  # 工厂2: 11台机器
                2: [5, 3, 4],  # 工厂3: 12台机器
                3: [3, 4, 5]   # 工厂4: 12台机器
            }
        },
        {
            'name': '大规模',
            'n_jobs': 100,
            'n_factories': 5,
            'n_stages': 3,
            'machines_per_stage': [6, 6, 6],
            'processing_time_range': (1, 30),
            'urgency_ddt': [1.2, 2.0, 3.0],
            'heterogeneous_machines': {
                0: [6, 6, 6],  # 工厂1: 18台机器
                1: [5, 7, 6],  # 工厂2: 18台机器
                2: [7, 5, 6],  # 工厂3: 18台机器
                3: [6, 6, 6],  # 工厂4: 18台机器
                4: [4, 8, 6]   # 工厂5: 18台机器
            }
        }
    ]
    
    # 算法配置（统一参数）
    common_params = {
        'population_size': 50,
        'max_iterations': 50
    }
    
    algorithms = {
        'RL-Chaotic-HHO': (RL_ChaoticHHO_Optimizer, {
            **common_params,
            'archive_size': 100,
            'learning_rate': 0.001,
            'epsilon': 0.1
        }),
        'I-NSGA-II': (ImprovedNSGA2_Optimizer, {
            **common_params,
            'crossover_prob': 0.9,
            'mutation_prob': 0.1
        }),
        'MOEA/D': (MOEAD_Optimizer, {
            **common_params,
            'crossover_prob': 0.9,
            'mutation_prob': 0.1
        }),
        'MOPSO': (MOPSO_Optimizer, {
            **common_params,
            'w': 0.4,
            'c1': 2.0,
            'c2': 2.0
        }),
        'MODE': (MODE_Optimizer, {
            **common_params,
            'crossover_prob': 0.9,
            'mutation_factor': 0.5
        }),
        'DQN': (DQNAlgorithmWrapper, {
            **common_params,
            'learning_rate': 0.001,
            'epsilon': 0.1
        }),
        'QL-ABC': (QLABC_Optimizer, {
            **common_params,
            'learning_rate': 0.1,
            'epsilon': 0.05,
            'limit': 10
        })
    }
    
    all_scale_results = {}
    
    # 对每个规模运行实验
    for config in configs:
        scale_name = config['name']
        print(f"\n🔬 运行{scale_name}实验")
        print("=" * 60)
        
        # 打印规模详情
        print_scale_details(config, {})
        
        # 生成问题数据
        problem_data = generate_heterogeneous_problem_data(config)
        
        # 存储该规模的结果
        scale_results = {}
        
        # 运行每个算法
        for alg_name, (alg_class, alg_params) in algorithms.items():
            results = run_single_experiment(
                problem_data, alg_name, alg_class, alg_params, runs=3
            )
            scale_results[alg_name] = results
        
        all_scale_results[scale_name] = scale_results
        
        # 绘制该规模的帕累托前沿对比图
        print(f"\n📊 绘制{scale_name}帕累托前沿对比图...")
        plot_pareto_comparison(scale_results, scale_name)
    
    # 生成综合对比报告
    generate_enhanced_table_report(all_scale_results, configs)

def generate_enhanced_table_report(results: Dict, configs: List[Dict]):
    """生成增强版表格对比报告（包含QL-ABC）"""
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    report_file = f"algorithm_comparison_with_ql_abc_{timestamp}.txt"
    
    with open(report_file, 'w', encoding='utf-8') as f:
        f.write("多目标分布式混合流水车间调度算法对比报告 (包含QL-ABC)\n")
        f.write("=" * 80 + "\n")
        f.write(f"生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"对比算法: RL-Chaotic-HHO, I-NSGA-II, MOEA/D, MOPSO, MODE, DQN, QL-ABC\n")
        f.write(f"统一参数: 种群大小=50, 迭代次数=50\n\n")
        
        # 为每个规模生成表格
        for scale_name, scale_results in results.items():
            f.write(f"{scale_name}问题对比结果\n")
            f.write("-" * 60 + "\n")
            
            # 最优值表格
            f.write("最优值对比:\n")
            f.write(f"{'算法':<15} {'完工时间':<12} {'拖期':<12} {'加权目标':<12} {'帕累托解':<10} {'运行时间(s)':<12}\n")
            f.write("-" * 80 + "\n")
            
            for alg_name, alg_results in scale_results.items():
                f.write(f"{alg_name:<15} {alg_results['makespan_best']:<12.2f} "
                       f"{alg_results['tardiness_best']:<12.2f} {alg_results['weighted_best']:<12.2f} "
                       f"{len(alg_results['pareto_solutions']):<10} {alg_results['runtime']:<12.2f}\n")
            
            f.write("\n")
            
            # 平均值表格
            f.write("平均值对比:\n")
            f.write(f"{'算法':<15} {'完工时间':<12} {'拖期':<12} {'加权目标':<12}\n")
            f.write("-" * 60 + "\n")
            
            for alg_name, alg_results in scale_results.items():
                f.write(f"{alg_name:<15} {alg_results['makespan_mean']:<12.2f} "
                       f"{alg_results['tardiness_mean']:<12.2f} {alg_results['weighted_mean']:<12.2f}\n")
            
            f.write("\n" + "="*80 + "\n")
        
        # 算法排名分析
        f.write("算法性能排名分析 (基于加权目标函数)\n")
        f.write("-" * 50 + "\n")
        
        for scale_name, scale_results in results.items():
            f.write(f"\n{scale_name}排名:\n")
            
            # 按加权目标函数排序
            sorted_algorithms = sorted(scale_results.items(), 
                                     key=lambda x: x[1]['weighted_best'])
            
            for rank, (alg_name, alg_results) in enumerate(sorted_algorithms, 1):
                f.write(f"  {rank}. {alg_name}: {alg_results['weighted_best']:.2f}\n")
        
        # QL-ABC性能评价
        f.write("\nQL-ABC算法性能评价:\n")
        f.write("-" * 30 + "\n")
        
        for scale_name, scale_results in results.items():
            ql_abc_results = scale_results.get('QL-ABC', {})
            hho_results = scale_results.get('RL-Chaotic-HHO', {})
            
            if ql_abc_results and hho_results:
                ql_abc_weighted = ql_abc_results['weighted_best']
                hho_weighted = hho_results['weighted_best']
                
                if ql_abc_weighted != float('inf') and hho_weighted != float('inf'):
                    improvement = (hho_weighted - ql_abc_weighted) / hho_weighted * 100
                    f.write(f"{scale_name}: QL-ABC vs RL-Chaotic-HHO = {improvement:+.2f}%\n")
    
    print(f"\n📄 综合对比报告已保存: {report_file}")

def main():
    """主函数"""
    try:
        run_table_format_experiments()
        print("\n✅ 表格格式算法对比实验完成！")
        return True
    except Exception as e:
        print(f"\n❌ 实验失败: {str(e)}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = main()
    if success:
        print("\n🎉 包含QL-ABC的算法对比实验成功完成！")
    else:
        print("\n💥 实验失败，请检查配置。") 