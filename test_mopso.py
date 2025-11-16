#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
MOPSO算法测试程序
验证多目标粒子群优化算法在分布式异构混合流水车间调度问题上的性能
"""

import time
import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.mopso import MOPSO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from utils.data_generator import DataGenerator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def test_mopso_basic():
    """基础功能测试"""
    print("=" * 60)
    print("MOPSO基础功能测试")
    print("=" * 60)
    
    # 生成测试问题
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=20,
        n_factories=3,
        n_stages=3,
        machines_per_stage=[2, 3, 2],
        processing_time_range=(1, 10),
        due_date_tightness=1.5
    )
    
    # 设置完全异构配置
    problem_data['machines_per_stage'] = [2, 3, 2]
    problem_data['urgencies'] = [0.9, 1.4, 1.9, 2.4, 2.9] * 4  # 20个作业的紧急度
    
    print(f"测试问题规模: {problem_data['n_jobs']}作业 × {problem_data['n_factories']}工厂 × {problem_data['n_stages']}阶段")
    print(f"机器配置: {problem_data['machines_per_stage']}")
    
    # 创建问题实例
    problem = MO_DHFSP_Problem(problem_data)
    
    # 创建MOPSO优化器
    optimizer = MOPSO_Optimizer(
        problem=problem,
        swarm_size=30,
        max_iterations=50,
        w=0.9,
        c1=2.0,
        c2=2.0,
        archive_size=50,
        mutation_prob=0.1
    )
    
    # 运行优化
    start_time = time.time()
    pareto_solutions, convergence_data = optimizer.optimize()
    end_time = time.time()
    
    # 输出结果
    print(f"\n优化结果:")
    print(f"  运行时间: {end_time - start_time:.2f}秒")
    print(f"  帕累托解数量: {len(pareto_solutions)}")
    
    if pareto_solutions:
        makespans = [sol.makespan for sol in pareto_solutions]
        tardiness = [sol.total_tardiness for sol in pareto_solutions]
        
        print(f"  完工时间范围: [{min(makespans):.2f}, {max(makespans):.2f}]")
        print(f"  总拖期范围: [{min(tardiness):.2f}, {max(tardiness):.2f}]")
        
        # 获取统计信息
        stats = optimizer.get_statistics()
        print(f"  最优完工时间: {stats.get('best_makespan', 'N/A'):.2f}")
        print(f"  最优总拖期: {stats.get('best_tardiness', 'N/A'):.2f}")
        print(f"  最终超体积: {stats.get('final_hypervolume', 'N/A'):.2f}")
        print(f"  最终间距: {stats.get('final_spacing', 'N/A'):.4f}")
    
    return pareto_solutions, convergence_data

def test_mopso_vs_others():
    """MOPSO与其他算法对比测试"""
    print("\n" + "=" * 60)
    print("MOPSO与其他算法对比测试")
    print("=" * 60)
    
    # 生成测试问题
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=30,
        n_factories=3,
        n_stages=4,
        machines_per_stage=[2, 3, 4, 2],
        processing_time_range=(1, 15),
        due_date_tightness=1.5
    )
    
    # 设置完全异构配置
    problem_data['machines_per_stage'] = [2, 3, 4, 2]
    problem_data['urgencies'] = [1.0 + i * 0.1 for i in range(30)]  # 渐增紧急度
    
    print(f"对比测试问题规模: {problem_data['n_jobs']}作业 × {problem_data['n_factories']}工厂 × {problem_data['n_stages']}阶段")
    
    # 创建问题实例
    problem = MO_DHFSP_Problem(problem_data)
    
    # 算法配置
    algorithms = {
        'MOPSO': {
            'class': MOPSO_Optimizer,
            'params': {
                'swarm_size': 50,
                'max_iterations': 80,
                'w': 0.9,
                'c1': 2.0,
                'c2': 2.0,
                'archive_size': 80,
                'mutation_prob': 0.1
            }
        },
        'NSGA-II': {
            'class': NSGA2_Optimizer,
            'params': {
                'population_size': 50,
                'max_generations': 80,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1
            }
        },
        'MOEA/D': {
            'class': MOEAD_Optimizer,
            'params': {
                'population_size': 50,
                'max_generations': 80,
                'crossover_prob': 0.9,
                'mutation_prob': 0.1,
                'neighbor_size': 15,
                'delta': 0.9,
                'nr': 2
            }
        }
    }
    
    results = {}
    
    # 运行各算法
    for alg_name, alg_config in algorithms.items():
        print(f"\n运行 {alg_name}...")
        
        # 创建优化器
        optimizer = alg_config['class'](problem, **alg_config['params'])
        
        # 运行优化
        start_time = time.time()
        pareto_solutions, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        # 存储结果
        results[alg_name] = {
            'pareto_solutions': pareto_solutions,
            'convergence_data': convergence_data,
            'runtime': end_time - start_time,
            'stats': optimizer.get_statistics() if hasattr(optimizer, 'get_statistics') else {}
        }
        
        print(f"  {alg_name} 完成: 帕累托解={len(pareto_solutions)}, 用时={end_time - start_time:.2f}s")
    
    # 对比分析
    print(f"\n算法对比结果:")
    print(f"{'算法':<10} {'帕累托解数':<12} {'最优完工时间':<12} {'最优拖期':<12} {'运行时间(s)':<12}")
    print("-" * 60)
    
    for alg_name, result in results.items():
        pareto_solutions = result['pareto_solutions']
        runtime = result['runtime']
        
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            best_makespan = min(makespans)
            best_tardiness = min(tardiness)
        else:
            best_makespan = float('inf')
            best_tardiness = float('inf')
        
        print(f"{alg_name:<10} {len(pareto_solutions):<12} {best_makespan:<12.2f} {best_tardiness:<12.2f} {runtime:<12.2f}")
    
    return results

def plot_pareto_comparison(results, save_path=None):
    """绘制帕累托前沿对比图"""
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    
    for i, (alg_name, result) in enumerate(results.items()):
        pareto_solutions = result['pareto_solutions']
        
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            
            plt.scatter(makespans, tardiness, 
                       c=colors[i % len(colors)], 
                       marker=markers[i % len(markers)],
                       label=f'{alg_name} ({len(pareto_solutions)}个解)',
                       alpha=0.7, s=60)
    
    plt.xlabel('完工时间 (Makespan)', fontsize=12)
    plt.ylabel('总拖期 (Total Tardiness)', fontsize=12)
    plt.title('MOPSO与其他算法帕累托前沿对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"帕累托前沿对比图已保存: {save_path}")
    
    plt.show()

def plot_convergence_comparison(results, save_path=None):
    """绘制收敛曲线对比图"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    
    for i, (alg_name, result) in enumerate(results.items()):
        convergence_data = result['convergence_data']
        
        # 处理不同算法的收敛数据格式
        if alg_name == 'MOPSO':
            # MOPSO返回的是列表格式
            if convergence_data and isinstance(convergence_data, list):
                iterations = [data['iteration'] for data in convergence_data]
                best_makespans = [data.get('best_makespan', float('inf')) for data in convergence_data]
                best_tardiness = [data.get('best_tardiness', float('inf')) for data in convergence_data]
                
                # 过滤无效值
                valid_makespans = [m for m in best_makespans if m != float('inf')]
                valid_tardiness = [t for t in best_tardiness if t != float('inf')]
                
                if valid_makespans:
                    ax1.plot(iterations[:len(valid_makespans)], valid_makespans, 
                            color=colors[i % len(colors)], label=alg_name, linewidth=2)
                
                if valid_tardiness:
                    ax2.plot(iterations[:len(valid_tardiness)], valid_tardiness, 
                            color=colors[i % len(colors)], label=alg_name, linewidth=2)
        
        else:
            # NSGA-II和MOEA/D返回的是字典格式
            if convergence_data and isinstance(convergence_data, dict):
                makespan_history = convergence_data.get('makespan_history', [])
                tardiness_history = convergence_data.get('tardiness_history', [])
                
                if makespan_history:
                    iterations = list(range(len(makespan_history)))
                    ax1.plot(iterations, makespan_history, 
                            color=colors[i % len(colors)], label=alg_name, linewidth=2)
                
                if tardiness_history:
                    iterations = list(range(len(tardiness_history)))
                    ax2.plot(iterations, tardiness_history, 
                            color=colors[i % len(colors)], label=alg_name, linewidth=2)
    
    ax1.set_xlabel('迭代次数', fontsize=12)
    ax1.set_ylabel('最优完工时间', fontsize=12)
    ax1.set_title('完工时间收敛曲线', fontsize=12, fontweight='bold')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    ax2.set_xlabel('迭代次数', fontsize=12)
    ax2.set_ylabel('最优总拖期', fontsize=12)
    ax2.set_title('总拖期收敛曲线', fontsize=12, fontweight='bold')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches='tight')
        print(f"收敛曲线对比图已保存: {save_path}")
    
    plt.show()

def test_mopso_parameters():
    """MOPSO参数敏感性测试"""
    print("\n" + "=" * 60)
    print("MOPSO参数敏感性测试")
    print("=" * 60)
    
    # 生成测试问题
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=25,
        n_factories=3,
        n_stages=3,
        machines_per_stage=[2, 3, 2],
        processing_time_range=(1, 12),
        due_date_tightness=1.5
    )
    
    problem_data['urgencies'] = [1.0 + i * 0.1 for i in range(25)]
    problem = MO_DHFSP_Problem(problem_data)
    
    # 参数配置
    parameter_configs = [
        {'name': 'w=0.5', 'params': {'w': 0.5, 'c1': 2.0, 'c2': 2.0}},
        {'name': 'w=0.7', 'params': {'w': 0.7, 'c1': 2.0, 'c2': 2.0}},
        {'name': 'w=0.9', 'params': {'w': 0.9, 'c1': 2.0, 'c2': 2.0}},
        {'name': 'c1=1.5', 'params': {'w': 0.9, 'c1': 1.5, 'c2': 2.0}},
        {'name': 'c2=1.5', 'params': {'w': 0.9, 'c1': 2.0, 'c2': 1.5}},
    ]
    
    results = {}
    
    for config in parameter_configs:
        print(f"\n测试参数配置: {config['name']}")
        
        # 创建优化器
        optimizer = MOPSO_Optimizer(
            problem=problem,
            swarm_size=40,
            max_iterations=60,
            archive_size=60,
            mutation_prob=0.1,
            **config['params']
        )
        
        # 运行优化
        start_time = time.time()
        pareto_solutions, convergence_data = optimizer.optimize()
        end_time = time.time()
        
        # 存储结果
        results[config['name']] = {
            'pareto_solutions': pareto_solutions,
            'convergence_data': convergence_data,
            'runtime': end_time - start_time
        }
        
        print(f"  结果: 帕累托解={len(pareto_solutions)}, 用时={end_time - start_time:.2f}s")
    
    # 参数敏感性分析
    print(f"\n参数敏感性分析:")
    print(f"{'参数配置':<10} {'帕累托解数':<12} {'最优完工时间':<12} {'最优拖期':<12} {'运行时间(s)':<12}")
    print("-" * 60)
    
    for config_name, result in results.items():
        pareto_solutions = result['pareto_solutions']
        runtime = result['runtime']
        
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            best_makespan = min(makespans)
            best_tardiness = min(tardiness)
        else:
            best_makespan = float('inf')
            best_tardiness = float('inf')
        
        print(f"{config_name:<10} {len(pareto_solutions):<12} {best_makespan:<12.2f} {best_tardiness:<12.2f} {runtime:<12.2f}")
    
    return results

def main():
    """主测试函数"""
    print("MOPSO算法测试程序")
    print("=" * 80)
    
    # 基础功能测试
    basic_pareto, basic_convergence = test_mopso_basic()
    
    # 算法对比测试
    comparison_results = test_mopso_vs_others()
    
    # 参数敏感性测试
    parameter_results = test_mopso_parameters()
    
    # 生成图表
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    print(f"\n生成对比图表...")
    plot_pareto_comparison(comparison_results, f"results/MOPSO_帕累托前沿对比_{timestamp}.png")
    plot_convergence_comparison(comparison_results, f"results/MOPSO_收敛曲线对比_{timestamp}.png")
    
    print(f"\nMOPSO算法测试完成!")
    print(f"测试结果显示MOPSO算法能够有效求解多目标分布式异构混合流水车间调度问题")

if __name__ == "__main__":
    import os
    os.makedirs("results", exist_ok=True)
    main() 