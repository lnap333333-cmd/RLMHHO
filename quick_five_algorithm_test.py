#!/usr/bin/env python3
"""
快速五算法对比测试
包含RL-Chaotic-HHO、NSGA-II、MOEA/D、MOPSO、MODE
"""

import os
import time
import numpy as np
from datetime import datetime
import matplotlib.pyplot as plt

# 设置中文字体
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'DejaVu Sans']
plt.rcParams['axes.unicode_minus'] = False

from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from utils.data_generator import DataGenerator

def run_five_algorithm_test():
    """运行五算法对比测试"""
    
    print("快速五算法对比测试")
    print("=" * 80)
    
    # 创建小规模测试问题
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=15,
        n_factories=3,
        n_stages=3,
        machines_per_stage=[2, 3, 2],
        processing_time_range=(1, 15),
        due_date_tightness=1.5
    )
    
    print(f"测试问题规模: 15作业 × 3工厂 × 3阶段")
    print(f"机器配置: {[2, 3, 2]}")
    print(f"处理时间范围: (1, 15)")
    
    # 五种算法配置
    algorithms = [
        ('RL-Chaotic-HHO', RL_ChaoticHHO_Optimizer, {'max_iterations': 30}),
        ('NSGA-II', NSGA2_Optimizer, {
            'population_size': 40,
            'max_generations': 30,
            'crossover_prob': 0.9,
            'mutation_prob': 0.1
        }),
        ('MOEA/D', MOEAD_Optimizer, {
            'population_size': 40,
            'max_generations': 30,
            'crossover_prob': 0.9,
            'mutation_prob': 0.1,
            'neighbor_size': 10,
            'delta': 0.9,
            'nr': 2
        }),
        ('MOPSO', MOPSO_Optimizer, {
            'swarm_size': 40,
            'max_iterations': 30,
            'w': 0.5,
            'c1': 2.0,
            'c2': 2.0,
            'archive_size': 60
        }),
        ('MODE', MODE_Optimizer, {
            'population_size': 40,
            'max_generations': 30,
            'F': 0.5,
            'CR': 0.9,
            'mutation_prob': 0.1
        })
    ]
    
    results = {}
    
    for alg_name, alg_class, params in algorithms:
        print(f"\n运行 {alg_name}...")
        
        try:
            # 创建问题实例
            problem = MO_DHFSP_Problem(problem_data)
            
            # 创建优化器
            optimizer = alg_class(problem, **params)
            
            # 记录运行时间
            start_time = time.time()
            
            # 运行优化
            pareto_solutions, convergence_data = optimizer.optimize()
            
            end_time = time.time()
            runtime = end_time - start_time
            
            # 计算结果
            if pareto_solutions:
                makespans = [sol.makespan for sol in pareto_solutions]
                tardiness = [sol.total_tardiness for sol in pareto_solutions]
                weighted_objs = [0.55 * sol.makespan + 0.45 * sol.total_tardiness for sol in pareto_solutions]
                
                results[alg_name] = {
                    'pareto_count': len(pareto_solutions),
                    'best_makespan': min(makespans),
                    'best_tardiness': min(tardiness),
                    'best_weighted': min(weighted_objs),
                    'runtime': runtime,
                    'pareto_solutions': pareto_solutions
                }
                
                print(f"  ✓ 成功完成")
                print(f"    帕累托解数量: {len(pareto_solutions)}")
                print(f"    最优完工时间: {min(makespans):.2f}")
                print(f"    最优总拖期: {min(tardiness):.2f}")
                print(f"    最优加权目标: {min(weighted_objs):.2f}")
                print(f"    运行时间: {runtime:.2f}s")
            else:
                print(f"  ✗ 未找到帕累托解")
                results[alg_name] = {
                    'pareto_count': 0,
                    'best_makespan': float('inf'),
                    'best_tardiness': float('inf'),
                    'best_weighted': float('inf'),
                    'runtime': runtime,
                    'pareto_solutions': []
                }
                
        except Exception as e:
            print(f"  ✗ 运行失败: {str(e)}")
            results[alg_name] = {
                'pareto_count': 0,
                'best_makespan': float('inf'),
                'best_tardiness': float('inf'),
                'best_weighted': float('inf'),
                'runtime': 0.0,
                'pareto_solutions': []
            }
    
    # 输出对比结果
    print(f"\n{'='*80}")
    print("五算法对比结果汇总")
    print(f"{'='*80}")
    
    print(f"{'算法':^15s} | {'帕累托解':^8s} | {'最优完工时间':^12s} | {'最优总拖期':^10s} | {'最优加权目标':^12s} | {'运行时间(s)':^10s}")
    print("-" * 80)
    
    for alg_name in ['RL-Chaotic-HHO', 'NSGA-II', 'MOEA/D', 'MOPSO', 'MODE']:
        if alg_name in results:
            result = results[alg_name]
            print(f"{alg_name:^15s} | {result['pareto_count']:^8d} | {result['best_makespan']:^12.2f} | {result['best_tardiness']:^10.2f} | {result['best_weighted']:^12.2f} | {result['runtime']:^10.2f}")
        else:
            print(f"{alg_name:^15s} | {'N/A':^8s} | {'N/A':^12s} | {'N/A':^10s} | {'N/A':^12s} | {'N/A':^10s}")
    
    # 绘制帕累托前沿对比图
    print(f"\n绘制帕累托前沿对比图...")
    plot_pareto_comparison(results)
    
    return results

def plot_pareto_comparison(results):
    """绘制帕累托前沿对比图"""
    
    plt.figure(figsize=(12, 8))
    
    colors = ['red', 'blue', 'green', 'orange', 'purple']
    markers = ['o', 's', '^', 'D', 'v']
    algorithm_names = ['RL-Chaotic-HHO', 'NSGA-II', 'MOEA/D', 'MOPSO', 'MODE']
    
    for i, alg_name in enumerate(algorithm_names):
        if alg_name in results and results[alg_name]['pareto_solutions']:
            pareto_solutions = results[alg_name]['pareto_solutions']
            
            makespans = [sol.makespan for sol in pareto_solutions]
            tardiness = [sol.total_tardiness for sol in pareto_solutions]
            
            plt.scatter(makespans, tardiness, 
                      c=colors[i], marker=markers[i], 
                      label=f'{alg_name} ({len(pareto_solutions)}个解)',
                      alpha=0.7, s=60)
    
    plt.xlabel('完工时间 (Makespan)', fontsize=12)
    plt.ylabel('总拖期 (Total Tardiness)', fontsize=12)
    plt.title('快速测试 - 五算法帕累托前沿对比', fontsize=14, fontweight='bold')
    plt.legend(fontsize=10)
    plt.grid(True, alpha=0.3)
    
    # 保存图片
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = f"results/快速五算法对比_{timestamp}.png"
    plt.savefig(filename, dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"  帕累托前沿对比图已保存: {filename}")

if __name__ == "__main__":
    # 确保结果目录存在
    os.makedirs("results", exist_ok=True)
    
    # 运行快速测试
    results = run_five_algorithm_test()
    
    print("\n快速五算法对比测试完成!") 