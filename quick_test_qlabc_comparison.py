#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
快速QL-ABC对比测试 - 只运行小规模实验验证系统
包含超体积和IGD指标
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
from algorithm.ql_abc import QLABC_Optimizer
from utils.data_generator import DataGenerator

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei']
plt.rcParams['axes.unicode_minus'] = False

def calculate_hypervolume(pareto_solutions: List, reference_point: Tuple[float, float] = None) -> float:
    """计算超体积指标"""
    if not pareto_solutions:
        return 0.0
    
    objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    if reference_point is None:
        max_f1 = max(obj[0] for obj in objectives)
        max_f2 = max(obj[1] for obj in objectives)
        reference_point = (max_f1 * 1.1, max_f2 * 1.1)
    
    sorted_objectives = sorted(objectives)
    hypervolume = 0.0
    prev_f2 = reference_point[1]
    
    for f1, f2 in sorted_objectives:
        if f2 < prev_f2:
            area = (reference_point[0] - f1) * (prev_f2 - f2)
            hypervolume += area
            prev_f2 = f2
    
    return max(0.0, hypervolume)

def calculate_igd(pareto_solutions: List, true_pareto_front: List = None) -> float:
    """计算反世代距离(IGD)指标"""
    if not pareto_solutions:
        return float('inf')
    
    current_objectives = [(sol.makespan, sol.total_tardiness) for sol in pareto_solutions]
    
    if true_pareto_front is None:
        true_pareto_front = current_objectives
    
    if not true_pareto_front:
        return float('inf')
    
    total_distance = 0.0
    for true_point in true_pareto_front:
        min_distance = float('inf')
        for current_point in current_objectives:
            distance = np.sqrt((true_point[0] - current_point[0])**2 + 
                             (true_point[1] - current_point[1])**2)
            min_distance = min(min_distance, distance)
        total_distance += min_distance
    
    return total_distance / len(true_pareto_front)

def run_quick_algorithm_test(problem_data: Dict, algorithm_name: str, algorithm_class, algorithm_params: Dict) -> Dict:
    """运行单个算法的快速测试"""
    print(f"  正在运行 {algorithm_name}...")
    
    try:
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
        
        # 计算指标
        if pareto_solutions:
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            weighted_objs = [0.55 * sol.makespan + 0.45 * sol.total_tardiness for sol in pareto_solutions]
            
            hypervolume = calculate_hypervolume(pareto_solutions)
            igd = calculate_igd(pareto_solutions)
            
            results = {
                'weighted_best': min(weighted_objs),
                'makespan_best': min(makespans),
                'tardiness_best': min(tardiness),
                'runtime': runtime,
                'hypervolume': hypervolume,
                'igd': igd,
                'pareto_count': len(pareto_solutions),
                'pareto_solutions': pareto_solutions
            }
        else:
            results = {
                'weighted_best': float('inf'),
                'makespan_best': float('inf'),
                'tardiness_best': float('inf'),
                'runtime': runtime,
                'hypervolume': 0.0,
                'igd': float('inf'),
                'pareto_count': 0,
                'pareto_solutions': []
            }
        
        print(f"    ✅ {algorithm_name} 完成:")
        print(f"       加权目标: {results['weighted_best']:.2f}")
        print(f"       完工时间: {results['makespan_best']:.2f}")
        print(f"       总拖期: {results['tardiness_best']:.2f}")
        print(f"       超体积: {results['hypervolume']:.0f}")
        igd_str = f"{results['igd']:.2f}" if results['igd'] != float('inf') else "∞"
        print(f"       IGD: {igd_str}")
        print(f"       解数量: {results['pareto_count']}")
        print(f"       运行时间: {results['runtime']:.2f}s")
        
        return results
        
    except Exception as e:
        print(f"    ❌ {algorithm_name} 失败: {str(e)}")
        return {
            'weighted_best': float('inf'),
            'makespan_best': float('inf'),
            'tardiness_best': float('inf'),
            'runtime': 0.0,
            'hypervolume': 0.0,
            'igd': float('inf'),
            'pareto_count': 0,
            'pareto_solutions': []
        }

def plot_quick_pareto_comparison(results: Dict):
    """绘制快速对比的帕累托前沿图"""
    plt.figure(figsize=(10, 6))
    
    colors = ['red', 'blue', 'brown']
    markers = ['o', 's', 'x']
    
    for i, (alg_name, result) in enumerate(results.items()):
        if 'pareto_solutions' in result and result['pareto_solutions']:
            pareto_solutions = result['pareto_solutions']
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            
            plt.scatter(makespans, tardiness, 
                      c=colors[i], marker=markers[i], 
                      label=f'{alg_name} ({len(pareto_solutions)}个解)',
                      alpha=0.7, s=50)
    
    plt.xlabel('完工时间 (Makespan)', fontsize=12)
    plt.ylabel('总拖期 (Total Tardiness)', fontsize=12)
    plt.title('快速测试 - 帕累托前沿对比', fontsize=14, fontweight='bold')
    plt.legend()
    plt.grid(True, alpha=0.3)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/快速测试_帕累托前沿对比_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  📊 帕累托前沿对比图已保存: {filename}")

def run_quick_comparison_test():
    """运行快速对比测试"""
    print("快速QL-ABC对比测试 - 包含超体积和IGD指标")
    print("=" * 80)
    
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 生成测试问题
    print("1. 生成测试问题...")
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=20,
        n_factories=3,
        n_stages=3,
        machines_per_stage=[2, 3, 3],
        processing_time_range=(1, 20),
        due_date_tightness=1.5
    )
    
    # 添加异构机器配置
    problem_data['heterogeneous_machines'] = {
        0: [2, 2, 2],  # 工厂0: 6台机器
        1: [2, 3, 3],  # 工厂1: 8台机器  
        2: [2, 3, 4]   # 工厂2: 9台机器
    }
    print("   ✅ 测试问题生成成功 (20×3×3, 总机器数: 23台)")
    
    # 算法配置 - 增强参数以获得更多帕累托解
    algorithms = {
        'RL-Chaotic-HHO': {
            'class': RL_ChaoticHHO_Optimizer,
            'params': {
                'population_size': 60,    # 增加种群大小
                'max_iterations': 40,     # 增加迭代次数
                'pareto_size_limit': 200  # 增加帕累托解限制
            }
        },
        'I-NSGA-II': {
            'class': ImprovedNSGA2_Optimizer,
            'params': {
                'population_size': 60,    # 增加种群大小
                'max_iterations': 40,     # 增加迭代次数
                'pareto_size_limit': 200, # 增加帕累托解限制
                'crossover_rate': 0.9,
                'mutation_rate': 0.15     # 增加变异率
            }
        },
        'QL-ABC': {
            'class': QLABC_Optimizer,
            'params': {
                'population_size': 50,    # 增加种群大小
                'max_iterations': 40,     # 增加迭代次数
                'learning_rate': 0.1,
                'epsilon': 0.4,           # 增加探索率
                'limit': 12               # 增加限制参数
            }
        }
    }
    
    # 运行算法
    print("\n2. 运行算法对比...")
    results = {}
    
    for alg_name, alg_config in algorithms.items():
        results[alg_name] = run_quick_algorithm_test(
            problem_data,
            alg_name,
            alg_config['class'],
            alg_config['params']
        )
    
    # 绘制对比图
    print("\n3. 生成对比图...")
    plot_quick_pareto_comparison(results)
    
    # 生成报告
    print("\n4. 生成快速对比报告...")
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/快速QL_ABC对比报告_{timestamp}.txt"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("快速QL-ABC对比测试报告\n")
        f.write("=" * 60 + "\n\n")
        f.write("测试规模: 20×3×3 (作业×工厂×阶段)\n")
        f.write("总机器数: 23台\n")
        f.write("算法参数: 种群50-60, 迭代40 (增强配置)\n")
        f.write("对比算法: RL-Chaotic-HHO, I-NSGA-II, QL-ABC\n\n")
        
        f.write("性能对比结果:\n")
        f.write("-" * 60 + "\n")
        f.write(f"{'算法':<15} {'加权目标':<10} {'完工时间':<10} {'拖期':<8} {'超体积':<8} {'IGD':<8} {'解数量':<6} {'时间(s)':<8}\n")
        f.write("-" * 60 + "\n")
        
        for alg_name, result in results.items():
            igd_str = f"{result['igd']:.2f}" if result['igd'] != float('inf') else "∞"
            f.write(f"{alg_name:<15} {result['weighted_best']:<10.2f} {result['makespan_best']:<10.2f} {result['tardiness_best']:<8.2f} {result['hypervolume']:<8.0f} {igd_str:<8} {result['pareto_count']:<6} {result['runtime']:<8.2f}\n")
        
        f.write("\n说明:\n")
        f.write("- 加权目标 = 0.55×完工时间 + 0.45×总拖期\n")
        f.write("- 超体积: 帕累托前沿覆盖面积，越大越好\n")
        f.write("- IGD: 反世代距离，越小越好\n")
        f.write("- 解数量: 帕累托解数量，越多越好\n")
        
        f.write(f"\n报告生成时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
    
    print(f"   📝 快速对比报告已保存: {filename}")
    
    # 控制台输出汇总
    print("\n" + "=" * 80)
    print("🎯 快速QL-ABC对比测试结果汇总")
    print("=" * 80)
    print(f"{'算法':<15} {'加权目标':<10} {'完工时间':<10} {'拖期':<8} {'超体积':<8} {'IGD':<8} {'解数量':<6} {'时间(s)':<8}")
    print("-" * 80)
    
    for alg_name, result in results.items():
        igd_str = f"{result['igd']:.2f}" if result['igd'] != float('inf') else "∞"
        print(f"{alg_name:<15} {result['weighted_best']:<10.2f} {result['makespan_best']:<10.2f} {result['tardiness_best']:<8.2f} {result['hypervolume']:<8.0f} {igd_str:<8} {result['pareto_count']:<6} {result['runtime']:<8.2f}")
    
    print("\n📊 性能分析:")
    # 找出各指标最优算法
    best_weighted = min(results.items(), key=lambda x: x[1]['weighted_best'])
    best_hypervolume = max(results.items(), key=lambda x: x[1]['hypervolume'])
    best_igd = min(results.items(), key=lambda x: x[1]['igd'] if x[1]['igd'] != float('inf') else float('inf'))
    best_count = max(results.items(), key=lambda x: x[1]['pareto_count'])
    
    print(f"🏆 最优加权目标: {best_weighted[0]} ({best_weighted[1]['weighted_best']:.2f})")
    print(f"🏆 最优超体积: {best_hypervolume[0]} ({best_hypervolume[1]['hypervolume']:.0f})")
    if best_igd[1]['igd'] != float('inf'):
        print(f"🏆 最优IGD: {best_igd[0]} ({best_igd[1]['igd']:.2f})")
    print(f"🏆 最多解数量: {best_count[0]} ({best_count[1]['pareto_count']}个)")
    
    print("\n✅ 快速测试完成！系统工作正常")
    print("🚀 如需运行完整对比实验，请运行: python table_format_comparison_with_ql_abc_full.py")
    print("=" * 80)

if __name__ == "__main__":
    run_quick_comparison_test() 