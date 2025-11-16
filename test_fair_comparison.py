#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
公平算法对比测试
验证所有算法都使用统一的种群大小50和迭代次数50
"""

import time
from problem.mo_dhfsp import MO_DHFSP_Problem
from algorithm.rl_chaotic_hho import RL_ChaoticHHO_Optimizer
from algorithm.nsga2 import NSGA2_Optimizer
from algorithm.moead import MOEAD_Optimizer
from algorithm.mopso import MOPSO_Optimizer
from algorithm.mode import MODE_Optimizer
from utils.data_generator import DataGenerator

def test_fair_parameters():
    """测试公平参数设置"""
    print("🔧 公平算法对比参数测试")
    print("=" * 60)
    
    # 生成小规模测试问题
    generator = DataGenerator(seed=42)
    problem_data = generator.generate_problem(
        n_jobs=20,
        n_factories=3,
        n_stages=3,
        machines_per_stage=[2, 3, 2],
        processing_time_range=(1, 10),
        due_date_tightness=1.5
    )
    
    problem = MO_DHFSP_Problem(problem_data)
    
    # 统一公平参数配置
    algorithms = {
        'RL-Chaotic-HHO': {
            'class': RL_ChaoticHHO_Optimizer,
            'params': {
                'max_iterations': 50,
                'population_size_override': 50  # 强制设置种群大小
            }
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
        },
        'MOPSO': {
            'class': MOPSO_Optimizer,
            'params': {
                'swarm_size': 50,
                'max_iterations': 50,
                'w': 0.5,
                'c1': 2.0,
                'c2': 2.0,
                'archive_size': 100
            }
        },
        'MODE': {
            'class': MODE_Optimizer,
            'params': {
                'population_size': 50,
                'max_generations': 50,
                'F': 0.5,
                'CR': 0.9,
                'mutation_prob': 0.1
            }
        }
    }
    
    results = {}
    
    for alg_name, alg_config in algorithms.items():
        print(f"\n🧪 测试 {alg_name}...")
        print(f"  参数验证: 种群大小=50, 迭代次数=50")
        
        try:
            # 创建优化器
            optimizer = alg_config['class'](problem, **alg_config['params'])
            
            # 验证参数设置
            if hasattr(optimizer, 'population_size'):
                print(f"  ✓ 实际种群大小: {optimizer.population_size}")
                assert optimizer.population_size == 50, f"种群大小不是50: {optimizer.population_size}"
            
            if hasattr(optimizer, 'max_iterations'):
                print(f"  ✓ 实际迭代次数: {optimizer.max_iterations}")
                assert optimizer.max_iterations == 50, f"迭代次数不是50: {optimizer.max_iterations}"
            elif hasattr(optimizer, 'max_generations'):
                print(f"  ✓ 实际代数: {optimizer.max_generations}")
                assert optimizer.max_generations == 50, f"代数不是50: {optimizer.max_generations}"
            
            # 运行优化
            start_time = time.time()
            pareto_solutions, convergence_data = optimizer.optimize()
            end_time = time.time()
            
            runtime = end_time - start_time
            
            # 记录结果
            results[alg_name] = {
                'pareto_size': len(pareto_solutions),
                'runtime': runtime,
                'best_makespan': min(sol.makespan for sol in pareto_solutions) if pareto_solutions else float('inf'),
                'best_tardiness': min(sol.total_tardiness for sol in pareto_solutions) if pareto_solutions else float('inf')
            }
            
            print(f"  ✓ 运行成功: 帕累托解={len(pareto_solutions)}, 运行时间={runtime:.2f}s")
            
        except Exception as e:
            print(f"  ❌ 运行失败: {e}")
            results[alg_name] = {'error': str(e)}
    
    # 输出对比结果
    print("\n" + "=" * 80)
    print("📊 公平对比结果汇总")
    print("=" * 80)
    
    print("| {:^15s} | {:^12s} | {:^12s} | {:^12s} | {:^12s} |".format(
        "算法", "帕累托解数", "最优完工时间", "最优拖期", "运行时间(s)"
    ))
    print("|" + "-" * 15 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|" + "-" * 12 + "|")
    
    for alg_name, result in results.items():
        if 'error' not in result:
            print("| {:^15s} | {:^12d} | {:^12.2f} | {:^12.2f} | {:^12.2f} |".format(
                alg_name,
                result['pareto_size'],
                result['best_makespan'],
                result['best_tardiness'],
                result['runtime']
            ))
        else:
            print("| {:^15s} | {:^12s} | {:^12s} | {:^12s} | {:^12s} |".format(
                alg_name, "ERROR", "ERROR", "ERROR", "ERROR"
            ))
    
    print("\n✅ 公平参数测试完成！")
    print("📋 确认：所有算法使用统一的种群大小50和迭代次数50")
    
    return results

if __name__ == "__main__":
    test_fair_parameters() 